"""DinoSeqWMPolicy — Phase 2 bimanual sequential world model.

Two ViTPredictors share a frozen DINOv3 visual encoder. Helper predicts
first; its output's action token is replaced with the leader's action and
fed to the leader predictor to produce a joint next-state estimate.

Training: phase-aware weighted loss using `batch["phase"]` (an int8 column
embedded in the dataset by drop_bad_episodes.py).

Inference: phase detector based on left-arm joint speed selects which
predictor to use; CEM optimizes the active arm's 6D action while the
other arm is frozen at its current observed pose.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .cem_planner import CEMPlanner
from .configuration_dino_seqwm import DinoSeqWMConfig
from .dino_encoder import DINOv3Encoder
from .objectives import create_objective_fn
from .proprio_embedding import ProprioceptiveEmbedding
from .vit_predictor import ViTPredictor


class DinoSeqWMPolicy(PreTrainedPolicy):
    """DINO-SeqWM bimanual world model.

    Architecture:
        DINOv3 (frozen) → patch tokens for each camera (concatenated)
        + proprio embedding (12D state → 768D token)
        + helper-action embedding (6D → 768D token, used as the action slot
          in the per-frame token sequence during training and history-encoding
          at inference)
        → ViTPredictor_helper → predicted next-frame tokens
        → swap action slot with leader-action embedding (6D → 768D)
        → ViTPredictor_leader → predicted JOINT next-frame tokens

    Loss:
        z_loss_joint  + alpha * z_loss_helper
        Each per-sample loss is weighted by phase (helper down-weighted in B,
        leader down-weighted in A).
    """

    config_class = DinoSeqWMConfig
    name = "dino_seqwm"

    def __init__(self, config: DinoSeqWMConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        self.config = config

        # Frozen visual encoder
        self.dino_encoder = DINOv3Encoder(img_size=config.dino_img_size)

        # Shared proprio (12D) + per-arm action encoders (6D each)
        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=config.num_hist + config.num_pred,
            tubelet_size=1,
            in_chans=config.state_dim,
            emb_dim=config.dino_embed_dim,
        )
        self.helper_action_encoder = ProprioceptiveEmbedding(
            num_frames=config.num_hist,
            tubelet_size=1,
            in_chans=config.helper_action_dim,
            emb_dim=config.dino_embed_dim,
        )
        self.leader_action_encoder = ProprioceptiveEmbedding(
            num_frames=config.num_hist,
            tubelet_size=1,
            in_chans=config.leader_action_dim,
            emb_dim=config.dino_embed_dim,
        )

        # Two causal ViT predictors (independent weights)
        predictor_kwargs = dict(
            num_patches=config.num_patches_per_frame,
            num_frames=config.num_hist,
            dim=config.dino_embed_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            mlp_dim=config.predictor_mlp_dim,
            dim_head=config.predictor_dim_head,
            dropout=config.predictor_dropout,
            emb_dropout=config.predictor_emb_dropout,
        )
        self.helper_predictor = ViTPredictor(**predictor_kwargs)
        self.leader_predictor = ViTPredictor(**predictor_kwargs)

        # CEM planner (lazy)
        self._planner = None
        self._objective_fn = None

        # Inference state
        self._action_queue = deque()
        self._left_state_history: deque = deque(
            maxlen=config.phase_speed_history_frames
        )
        self._left_peak_speed = 0.0
        self._still_count = 0
        self._in_phase_B = False

    def get_optim_params(self) -> list[dict]:
        params = (
            list(self.helper_predictor.parameters())
            + list(self.leader_predictor.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.helper_action_encoder.parameters())
            + list(self.leader_action_encoder.parameters())
        )
        return [{"params": params}]

    def reset(self):
        self._action_queue.clear()
        self._left_state_history.clear()
        self._left_peak_speed = 0.0
        self._still_count = 0
        self._in_phase_B = False

    # -------------------------------------------------------------------------
    # Encoding helpers (mirror dino_wm_test layout for token-sequence ops)
    # -------------------------------------------------------------------------

    def _extract_image_features(self, batch: dict) -> dict:
        images = {}
        for key, value in batch.items():
            if key.startswith("observation.images."):
                cam_name = key.split("observation.images.")[-1]
                if cam_name in self.config.camera_names:
                    images[cam_name] = value
        return images

    def _encode_images(self, images_dict: dict, batch_size: int, num_frames: int) -> Tensor:
        """Encode all cameras with frozen DINOv3 and concatenate along token axis.

        Returns: (B, T, num_cameras * 256, 768)
        """
        all_patches = []
        # Sort camera names so concatenation order is deterministic across batches.
        for cam_name in sorted(images_dict.keys()):
            imgs = images_dict[cam_name]                  # (B, T, 3, H, W)
            imgs_flat = rearrange(imgs, "b t c h w -> (b t) c h w")
            patches = self.dino_encoder(imgs_flat)         # (B*T, 256, 768)
            patches = rearrange(
                patches, "(b t) p d -> b t p d",
                b=batch_size, t=num_frames,
            )
            all_patches.append(patches)
        return torch.cat(all_patches, dim=2)               # (B, T, K*256, 768)

    def _encode_frame_tokens(self, visual: Tensor, proprio: Tensor, action_token: Tensor) -> Tensor:
        """[visual_patches | proprio | action] → (B, T, P, 768)."""
        return torch.cat(
            [visual, proprio.unsqueeze(2), action_token.unsqueeze(2)], dim=2
        )

    def _run_predictor(self, predictor: ViTPredictor, z: Tensor, T_hist: int) -> Tensor:
        """Run a ViTPredictor on (B, T_hist, P, D) and reshape back."""
        z_flat = rearrange(z, "b t p d -> b (t p) d")
        z_pred_flat = predictor(z_flat)
        return rearrange(
            z_pred_flat, "b (t p) d -> b t p d",
            t=T_hist, p=self.config.num_patches_per_frame,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Phase-aware weighted SeqWM forward.

        Dataloader provides (with delta_indices=[-10,-5,0,+5] for obs and
        [-10,-5,0] for action):
            observation.images.{left_wrist,right_wrist,top}: (B, 4, 3, H, W)
            observation.state:                                (B, 4, 12)
            action:                                           (B, 3, 12)
            phase:                                            (B, 1) int8 (0=A, 1=B)
        """
        images_dict = self._extract_image_features(batch)
        state = batch["observation.state"]                 # (B, 4, 12)
        action = batch["action"]                            # (B, 3, 12)

        B = state.shape[0]
        T_total = state.shape[1]
        T_hist = action.shape[1]

        # 1. DINOv3 encode all cameras → concat across token axis
        visual = self._encode_images(images_dict, B, T_total)        # (B, 4, K*256, 768)

        # 2. Proprio (full 12D state → 768D)
        proprio_emb = self.proprio_encoder(state)                    # (B, 4, 768)

        # 3. Per-arm action embeddings (6D each → 768D)
        a_helper = action[..., : self.config.helper_action_dim]      # (B, 3, 6)
        a_leader = action[..., self.config.helper_action_dim :]      # (B, 3, 6)
        a_helper_emb = self.helper_action_encoder(a_helper)          # (B, 3, 768)
        a_leader_emb = self.leader_action_encoder(a_leader)          # (B, 3, 768)

        # 4. Source / target split
        visual_src = visual[:, :T_hist]                              # (B, 3, ...)
        visual_tgt = visual[:, 1:]                                    # (B, 3, ...)
        proprio_src = proprio_emb[:, :T_hist]
        proprio_tgt = proprio_emb[:, 1:]

        # 5. Stage 1: helper predictor (action slot = helper action)
        z_src_h = self._encode_frame_tokens(visual_src, proprio_src, a_helper_emb)
        z_pred_helper = self._run_predictor(self.helper_predictor, z_src_h, T_hist)

        # 6. Stage 2: replace action slot with leader action; leader predicts
        z_swap = z_pred_helper.clone()
        z_swap[:, :, -1, :] = a_leader_emb
        z_pred_joint = self._run_predictor(self.leader_predictor, z_swap, T_hist)

        # 7. Targets (visual + proprio only — action token excluded from loss)
        z_tgt = torch.cat([visual_tgt, proprio_tgt.unsqueeze(2)], dim=2)
        z_tgt = z_tgt.detach()

        # 8. Per-sample MSE (mean over time / patches / dim)
        err_h = (z_pred_helper[:, :, :-1, :] - z_tgt) ** 2
        err_j = (z_pred_joint[:, :, :-1, :] - z_tgt) ** 2
        z_loss_helper_per_b = err_h.mean(dim=(1, 2, 3))               # (B,)
        z_loss_joint_per_b = err_j.mean(dim=(1, 2, 3))                # (B,)

        # 9. Phase-aware weighting
        if "phase" not in batch:
            raise RuntimeError(
                "DinoSeqWMPolicy requires a `phase` column (int8, 0=A / 1=B) in "
                "the batch, but it is missing.\n"
                f"  Available batch keys: {sorted(batch.keys())}\n"
                "Likely causes (in order of likelihood):\n"
                "  1. The dataset uploaded to Drive predates "
                "`scripts_analyze/embed_phase_into_dataset.py`. Re-tar the local "
                "`bimanual_cooperate/` after running embed_phase + drop_bad_episodes "
                "and re-upload.\n"
                "  2. HF datasets cache is stale. Delete the HF cache dir "
                "(`HF_HOME/datasets/...`) and re-run.\n"
                "  3. Some processor step is stripping unknown keys (unlikely)."
            )
        phase = batch["phase"].to(z_loss_helper_per_b.dtype).view(B)  # (B,) 0 or 1
        in_A = 1.0 - phase
        in_B = phase
        helper_w = (
            in_A * self.config.phase_helper_weight_A
            + in_B * self.config.phase_helper_weight_B
        )
        leader_w = (
            in_A * self.config.phase_leader_weight_A
            + in_B * self.config.phase_leader_weight_B
        )
        loss_joint = (leader_w * z_loss_joint_per_b).mean()
        loss_helper = (helper_w * z_loss_helper_per_b).mean()
        loss = loss_joint + self.config.helper_aux_loss_weight * loss_helper

        return loss, {
            "z_loss": loss.item(),
            "z_loss_joint": loss_joint.item(),
            "z_loss_helper": loss_helper.item(),
            "phase_A_frac": float(in_A.mean().item()),
        }

    # -------------------------------------------------------------------------
    # Inference: CEM planning with phase routing
    # -------------------------------------------------------------------------

    def _get_planner(self, action_dim: int) -> CEMPlanner:
        # Re-init if action_dim changes between phases (it doesn't — both 6 — but
        # keep the check in case someone tunes asymmetric dims).
        if self._planner is None or self._planner.action_dim != action_dim:
            self._planner = CEMPlanner(
                horizon=self.config.cem_horizon,
                topk=self.config.cem_topk,
                num_samples=self.config.cem_num_samples,
                opt_steps=self.config.cem_opt_steps,
                action_dim=action_dim,
                var_scale=self.config.cem_var_scale,
            )
            self._objective_fn = create_objective_fn(
                alpha=self.config.cem_objective_alpha,
                mode=self.config.cem_objective_mode,
            )
        return self._planner

    def _detect_phase(self, current_left_state: Tensor) -> str:
        """Adaptive phase detector: 'B' once left arm has been still for ~1s."""
        # Append latest 6D left-arm state to history
        self._left_state_history.append(current_left_state.detach().cpu().numpy())

        if self._in_phase_B:
            return "B"
        if len(self._left_state_history) < self.config.phase_speed_window_frames:
            return "A"

        arr = np.stack(list(self._left_state_history)[-self.config.phase_speed_window_frames :])
        diffs = np.diff(arr[:, :5], axis=0)        # exclude gripper
        speed = float(np.linalg.norm(diffs, axis=1).mean())

        self._left_peak_speed = max(self._left_peak_speed, speed)
        thr = max(
            self._left_peak_speed * self.config.phase_still_threshold_frac,
            self.config.phase_still_threshold_floor,
        )

        if speed < thr:
            self._still_count += 1
            if self._still_count >= self.config.phase_still_required_frames:
                self._in_phase_B = True
        else:
            self._still_count = 0
        return "B" if self._in_phase_B else "A"

    def _seq_rollout(
        self,
        visual_hist: Tensor,            # (N, T_hist, K*256, 768)
        proprio_hist: Tensor,           # (N, T_hist, 768)
        a_helper_hist: Tensor,          # (N, T_hist, 6) — real history actions
        a_helper_seq: Tensor,           # (N, H, 6)
        a_leader_seq: Tensor,           # (N, H, 6)
    ) -> dict[str, Tensor]:
        """Iterative two-stage rollout.

        Returns z_obses dict {visual, proprio} with T_hist + H frames each.
        """
        T_hist = self.config.num_hist
        H = a_helper_seq.shape[1]

        a_helper_hist_emb = self.helper_action_encoder(a_helper_hist)
        z = self._encode_frame_tokens(visual_hist, proprio_hist, a_helper_hist_emb)

        for t in range(H):
            window = z[:, -T_hist:]
            # Stage 1: helper
            z_pred_h = self._run_predictor(self.helper_predictor, window, T_hist)
            # Replace action slot in last predicted frame with leader action
            a_l_emb = self.leader_action_encoder(a_leader_seq[:, t : t + 1])  # (N, 1, 768)
            z_swap = z_pred_h.clone()
            z_swap[:, -1, -1, :] = a_l_emb.squeeze(1)
            # Stage 2: leader → joint prediction
            z_pred_j = self._run_predictor(self.leader_predictor, z_swap, T_hist)

            # Take the last predicted frame as the new history step.
            z_new = z_pred_j[:, -1:, ...].clone()                    # (N, 1, P, D)
            # Replace its action slot with helper action at this step (matches
            # training convention: action token in z[t] = action taken at frame t).
            a_h_emb = self.helper_action_encoder(a_helper_seq[:, t : t + 1])  # (N, 1, D)
            z_new[:, :, -1, :] = a_h_emb                             # (N, 1, D) ← (N, 1, D)
            z = torch.cat([z, z_new], dim=1)

        # Return full latent sequence (history + rollout)
        visual_all = z[:, :, :-2, :]
        proprio_all = z[:, :, -2, :]
        return {"visual": visual_all, "proprio": proprio_all}

    def _build_goal_latent(self, batch: dict, phase: str) -> dict[str, Tensor]:
        """Goal images are passed in via:
            phase A: batch['subgoal.images.<cam>'] for each camera (lid moved)
            phase B: batch['goal.images.<cam>']    for each camera (cube in box)
        Optional: 'subgoal.state' / 'goal.state' for proprio goal (defaults to
        last observed state).
        """
        prefix = "subgoal.images." if phase == "A" else "goal.images."
        goal_images = {
            k.split(prefix)[-1]: v
            for k, v in batch.items()
            if k.startswith(prefix) and k.split(prefix)[-1] in self.config.camera_names
        }
        if not goal_images:
            # No goal provided — fall back to last observation (CEM won't converge
            # but training-time .forward() doesn't hit this path).
            goal_images = {
                k.split("observation.images.")[-1]: v
                for k, v in batch.items()
                if k.startswith("observation.images.")
                and k.split("observation.images.")[-1] in self.config.camera_names
            }
            # If they have a time axis, take last frame.
            goal_images = {
                k: (v[:, -1:] if v.dim() == 5 else v.unsqueeze(1))
                for k, v in goal_images.items()
            }

        B = next(iter(goal_images.values())).shape[0]
        # Ensure (B, 1, 3, H, W)
        for k in list(goal_images.keys()):
            if goal_images[k].dim() == 4:
                goal_images[k] = goal_images[k].unsqueeze(1)
        goal_visual = self._encode_images(goal_images, B, 1)   # (B, 1, K*256, 768)

        state_key = "subgoal.state" if phase == "A" else "goal.state"
        if state_key in batch:
            goal_state = batch[state_key]
        else:
            goal_state = batch["observation.state"]
            if goal_state.dim() == 3:
                goal_state = goal_state[:, -1:]
            elif goal_state.dim() == 2:
                goal_state = goal_state.unsqueeze(1)
        goal_proprio = self.proprio_encoder(goal_state)         # (B, 1, 768)

        # Keep the time axis (size 1) so it matches z_obs_pred[:, -1:] shape.
        # CEM expects goal with leading dim 1 (the planner expands to N samples).
        return {"visual": goal_visual[:1], "proprio": goal_proprio[:1]}

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Plan a 12D action chunk by running CEM on the active arm only."""
        self.eval()
        device = next(self.parameters()).device

        state = batch["observation.state"]
        if state.dim() == 2:
            state = state.unsqueeze(1)
        B, T_avail = state.shape[:2]
        assert B == 1, "DinoSeqWMPolicy.predict_action_chunk currently supports B=1."

        # Phase routing using current left-arm state
        phase = self._detect_phase(state[0, -1, :6])

        # Pad observations to T_hist if needed
        T_hist = self.config.num_hist
        if T_avail < T_hist:
            pad = T_hist - T_avail
            state = torch.cat([state[:, :1].repeat(1, pad, 1), state], dim=1)

        images_dict = self._extract_image_features(batch)
        for k, v in images_dict.items():
            if v.dim() == 4:                              # (B, 3, H, W) → (B, 1, 3, H, W)
                images_dict[k] = v.unsqueeze(1)
            if images_dict[k].shape[1] < T_hist:
                pad = T_hist - images_dict[k].shape[1]
                images_dict[k] = torch.cat(
                    [images_dict[k][:, :1].repeat(1, pad, 1, 1, 1), images_dict[k]], dim=1
                )

        # History actions (used to seed the action token in initial z)
        a_hist_full = batch.get("action", None)
        if a_hist_full is not None and a_hist_full.dim() == 3 and a_hist_full.shape[1] >= T_hist:
            a_helper_hist = a_hist_full[:, -T_hist:, : self.config.helper_action_dim]
        else:
            # Fallback: use the helper part of state as a stand-in (same shape).
            a_helper_hist = state[:, -T_hist:, : self.config.helper_action_dim]

        with torch.no_grad():
            visual_hist = self._encode_images(images_dict, B, T_hist)
            proprio_hist = self.proprio_encoder(state[:, -T_hist:])
            z_obs_goal = self._build_goal_latent(batch, phase)

            H = self.config.cem_horizon
            current_helper = state[0, -1, : self.config.helper_action_dim]   # (6,)
            current_leader = state[0, -1, self.config.helper_action_dim :]   # (6,)

            if phase == "A":
                # Plan helper; freeze leader at its current observed pose.
                leader_frozen = current_leader[None, None, :].expand(1, H, 6)

                def rollout_fn(action_helper_samples: Tensor) -> dict[str, Tensor]:
                    N = action_helper_samples.shape[0]
                    v = repeat(visual_hist, "b t p d -> (b n) t p d", n=N)
                    p = repeat(proprio_hist, "b t d -> (b n) t d", n=N)
                    a_h_hist = repeat(a_helper_hist, "b t d -> (b n) t d", n=N)
                    a_l = repeat(leader_frozen, "b h d -> (b n) h d", n=N)
                    return self._seq_rollout(v, p, a_h_hist, action_helper_samples, a_l)

                planner = self._get_planner(self.config.helper_action_dim)
                best_helper = planner.plan(
                    rollout_fn=rollout_fn,
                    objective_fn=self._objective_fn,
                    z_obs_goal=z_obs_goal,
                    device=device,
                )                                                 # (H, 6)
                best_actions = torch.cat(
                    [best_helper, current_leader[None, :].expand(H, -1)], dim=-1
                )                                                 # (H, 12)

            else:  # phase B
                helper_frozen = current_helper[None, None, :].expand(1, H, 6)

                def rollout_fn(action_leader_samples: Tensor) -> dict[str, Tensor]:
                    N = action_leader_samples.shape[0]
                    v = repeat(visual_hist, "b t p d -> (b n) t p d", n=N)
                    p = repeat(proprio_hist, "b t d -> (b n) t d", n=N)
                    a_h_hist = repeat(a_helper_hist, "b t d -> (b n) t d", n=N)
                    a_h = repeat(helper_frozen, "b h d -> (b n) h d", n=N)
                    return self._seq_rollout(v, p, a_h_hist, a_h, action_leader_samples)

                planner = self._get_planner(self.config.leader_action_dim)
                best_leader = planner.plan(
                    rollout_fn=rollout_fn,
                    objective_fn=self._objective_fn,
                    z_obs_goal=z_obs_goal,
                    device=device,
                )                                                 # (H, 6)
                best_actions = torch.cat(
                    [current_helper[None, :].expand(H, -1), best_leader], dim=-1
                )                                                 # (H, 12)

        return best_actions.unsqueeze(0)                         # (1, H, 12)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)             # (1, H, 12)
            for t in range(chunk.shape[1]):
                self._action_queue.append(chunk[:, t])
        return self._action_queue.popleft()
