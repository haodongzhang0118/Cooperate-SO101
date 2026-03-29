"""DinoWMTestPolicy — DINO-WM single-arm world model policy for LeRobot (Phase 1).

Predicts dynamics in frozen DINOv3 patch-token space using a causal ViTPredictor.
Plans actions at inference time via CEM in latent space.
"""

import torch
import torch.nn as nn
from collections import deque
from einops import rearrange, repeat
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_dino_wm_test import DinoWMTestConfig
from .dino_encoder import DINOv3Encoder
from .vit_predictor import ViTPredictor
from .proprio_embedding import ProprioceptiveEmbedding
from .cem_planner import CEMPlanner
from .objectives import create_objective_fn


class DinoWMTestPolicy(PreTrainedPolicy):
    """DINO-WM world model policy for a single SO101 arm.

    Architecture:
    - Frozen DINOv3 ViT-B/16 encodes raw images into patch tokens (256 tokens x 768D per camera)
    - ProprioceptiveEmbedding projects state/action into 768D tokens
    - Tokens are concatenated: [visual_cam1, visual_cam2, proprio, action] per frame
    - ViTPredictor (causal transformer) predicts next-frame tokens
    - Training loss: MSE on visual + proprio tokens (action token excluded)
    - Inference: CEM planner optimizes action sequences in latent space
    """

    config_class = DinoWMTestConfig
    name = "dino_wm_test"

    def __init__(self, config: DinoWMTestConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        self.config = config

        # Frozen visual encoder
        self.dino_encoder = DINOv3Encoder(img_size=config.dino_img_size)

        # Proprioceptive and action embeddings (trainable)
        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=config.num_hist + config.num_pred,
            tubelet_size=1,
            in_chans=config.state_dim,
            emb_dim=config.dino_embed_dim,
        )
        self.action_encoder = ProprioceptiveEmbedding(
            num_frames=config.num_hist,
            tubelet_size=1,
            in_chans=config.action_dim,
            emb_dim=config.dino_embed_dim,
        )

        # Causal ViT predictor (trainable)
        self.predictor = ViTPredictor(
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

        # Loss
        self.loss_fn = nn.MSELoss()

        # CEM planner (lazy init on first inference call)
        self._planner = None
        self._objective_fn = None

        # Action queue for deployment
        self._action_queue = deque()

    def get_optim_params(self) -> list[dict]:
        params = (
            list(self.predictor.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        return [{"params": params}]

    def reset(self):
        self._action_queue.clear()

    # -------------------------------------------------------------------------
    # Encoding helpers
    # -------------------------------------------------------------------------

    def _encode_images(self, images_dict: dict, batch_size: int, num_frames: int) -> Tensor:
        """Encode all camera images with frozen DINOv3 and concatenate.

        Args:
            images_dict: Dict of camera_name -> (B, T, 3, H, W).
            batch_size: B
            num_frames: T

        Returns:
            (B, T, num_cameras * 256, 768)
        """
        all_patches = []
        for cam_name in sorted(images_dict.keys()):
            imgs = images_dict[cam_name]  # (B, T, 3, H, W)
            # Flatten batch and time for encoder
            imgs_flat = rearrange(imgs, "b t c h w -> (b t) c h w")
            patches = self.dino_encoder(imgs_flat)  # (B*T, 256, 768)
            patches = rearrange(patches, "(b t) p d -> b t p d", b=batch_size, t=num_frames)
            all_patches.append(patches)
        # Concatenate across cameras: (B, T, num_cameras*256, 768)
        return torch.cat(all_patches, dim=2)

    def _encode_frame_tokens(self, visual: Tensor, proprio: Tensor, action: Tensor) -> Tensor:
        """Combine visual, proprio, and action into per-frame token sequences.

        Args:
            visual: (B, T, num_visual_patches, 768)
            proprio: (B, T, 768)  — already embedded
            action: (B, T, 768)   — already embedded

        Returns:
            (B, T, num_patches_per_frame, 768)
        """
        # Add token dimension to proprio and action
        return torch.cat(
            [visual, proprio.unsqueeze(2), action.unsqueeze(2)], dim=2
        )

    def _separate_tokens(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split combined tokens back into visual, proprio, action.

        Args:
            z: (B, T, num_patches_per_frame, 768)

        Returns:
            visual: (B, T, num_visual_patches, 768)
            proprio: (B, T, 768)
            action: (B, T, 768)
        """
        return z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]

    def _extract_image_features(self, batch: dict) -> dict:
        """Extract image tensors from a batch dict, filtered by config.camera_names."""
        images = {}
        for key, value in batch.items():
            if key.startswith("observation.images."):
                cam_name = key.split("observation.images.")[-1]
                if cam_name in self.config.camera_names:
                    images[cam_name] = value
        return images

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        The dataloader provides:
        - observation.images.*: (B, 4, 3, H, W)  — 3 history + 1 target
        - observation.state:    (B, 4, 6)
        - action:               (B, 3, 6)         — actions for history frames

        We predict the target frame tokens from the history frames.
        """
        images_dict = self._extract_image_features(batch)
        state = batch["observation.state"]  # (B, T_total, state_dim)
        action = batch["action"]            # (B, T_hist, action_dim)

        B = state.shape[0]
        T_total = state.shape[1]  # num_hist + num_pred = 4
        T_hist = action.shape[1]  # num_hist = 3

        # 1. Encode all images with frozen DINOv3
        visual = self._encode_images(images_dict, B, T_total)  # (B, 4, num_cam*256, 768)

        # 2. Embed proprioceptive state and action
        proprio_emb = self.proprio_encoder(state)   # (B, 4, 768)
        action_emb = self.action_encoder(action)    # (B, 3, 768)

        # 3. Split into source (history) and target
        visual_src = visual[:, :T_hist]           # (B, 3, num_cam*256, 768)
        visual_tgt = visual[:, 1:]                # (B, 3, num_cam*256, 768) — shifted by 1
        proprio_src = proprio_emb[:, :T_hist]     # (B, 3, 768)
        proprio_tgt = proprio_emb[:, 1:]          # (B, 3, 768)

        # 4. Combine source tokens: [visual, proprio, action]
        z_src = self._encode_frame_tokens(visual_src, proprio_src, action_emb)
        # (B, 3, num_patches_per_frame, 768)

        # 5. Predict with ViTPredictor
        z_flat = rearrange(z_src, "b t p d -> b (t p) d")
        z_pred_flat = self.predictor(z_flat)
        z_pred = rearrange(
            z_pred_flat, "b (t p) d -> b t p d",
            t=T_hist, p=self.config.num_patches_per_frame,
        )

        # 6. Build target tokens (visual + proprio only, no action)
        # Target = next frame's visual + proprio tokens
        z_tgt_visual = visual_tgt  # (B, 3, num_cam*256, 768)
        z_tgt_proprio = proprio_tgt  # (B, 3, 768)
        z_tgt = torch.cat(
            [z_tgt_visual, z_tgt_proprio.unsqueeze(2)], dim=2
        )  # (B, 3, num_cam*256 + 1, 768)

        # 7. Compute loss: MSE on visual + proprio tokens (exclude action token at -1)
        z_pred_no_action = z_pred[:, :, :-1, :]  # exclude action token
        loss = self.loss_fn(z_pred_no_action, z_tgt.detach())

        # Detailed losses for logging
        z_pred_visual, z_pred_proprio, _ = self._separate_tokens(z_pred)
        z_visual_loss = self.loss_fn(z_pred_visual, z_tgt_visual.detach())
        z_proprio_loss = self.loss_fn(z_pred_proprio, z_tgt_proprio.detach())

        loss_dict = {
            "z_loss": loss.item(),
            "z_visual_loss": z_visual_loss.item(),
            "z_proprio_loss": z_proprio_loss.item(),
        }

        return loss, loss_dict

    # -------------------------------------------------------------------------
    # Inference (CEM planning)
    # -------------------------------------------------------------------------

    def _get_planner(self) -> CEMPlanner:
        if self._planner is None:
            self._planner = CEMPlanner(
                horizon=self.config.cem_horizon,
                topk=self.config.cem_topk,
                num_samples=self.config.cem_num_samples,
                opt_steps=self.config.cem_opt_steps,
                action_dim=self.config.action_dim,
                var_scale=self.config.cem_var_scale,
            )
            self._objective_fn = create_objective_fn(
                alpha=self.config.cem_objective_alpha,
                mode=self.config.cem_objective_mode,
            )
        return self._planner

    def _rollout(self, obs_visual: Tensor, obs_proprio: Tensor, actions: Tensor) -> dict:
        """Roll out the world model given initial observation and action sequence.

        Args:
            obs_visual: (B, num_hist, num_cam*256, 768) — initial visual tokens.
            obs_proprio: (B, num_hist, 768) — initial proprio embeddings.
            actions: (B, horizon, action_dim) — raw action sequences.

        Returns:
            Dict with "visual" and "proprio" predicted latents.
        """
        B = obs_visual.shape[0]
        T_hist = self.config.num_hist

        # Encode initial actions (use zeros for the initial window)
        init_action = torch.zeros(B, T_hist, self.config.action_dim, device=actions.device)
        init_action_emb = self.action_encoder(init_action)  # (B, T_hist, 768)

        # Build initial z
        z = self._encode_frame_tokens(obs_visual, obs_proprio, init_action_emb)
        # (B, T_hist, num_patches_per_frame, 768)

        # Iterative rollout
        for t in range(actions.shape[1]):
            # Predict next frame from last num_hist frames
            z_window = z[:, -T_hist:]
            z_flat = rearrange(z_window, "b t p d -> b (t p) d")
            z_pred_flat = self.predictor(z_flat)
            z_pred = rearrange(
                z_pred_flat, "b (t p) d -> b t p d",
                t=T_hist, p=self.config.num_patches_per_frame,
            )

            # Take only the last predicted frame
            z_new = z_pred[:, -1:, ...]  # (B, 1, num_patches_per_frame, 768)

            # Replace action token with the current action
            act_t = actions[:, t:t+1, :]  # (B, 1, action_dim)
            act_emb = self.action_encoder(act_t)  # (B, 1, 768)
            z_new[:, :, -1, :] = act_emb.squeeze(2) if act_emb.dim() > 3 else act_emb

            z = torch.cat([z, z_new], dim=1)

        # Final prediction (one more step without action)
        z_window = z[:, -T_hist:]
        z_flat = rearrange(z_window, "b t p d -> b (t p) d")
        z_pred_flat = self.predictor(z_flat)
        z_pred = rearrange(
            z_pred_flat, "b (t p) d -> b t p d",
            t=T_hist, p=self.config.num_patches_per_frame,
        )
        z_final = z_pred[:, -1:, ...]
        z = torch.cat([z, z_final], dim=1)

        # Separate and return
        visual_all = z[:, :, :-2, :]  # all visual tokens
        proprio_all = z[:, :, -2, :]  # proprio token

        return {"visual": visual_all, "proprio": proprio_all}

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Plan an action chunk using CEM.

        Args:
            batch: Dict with observation images and state (single timestep, with batch dim).

        Returns:
            (B, horizon, action_dim) — planned action chunk.
        """
        self.eval()
        planner = self._get_planner()
        device = next(self.parameters()).device

        images_dict = self._extract_image_features(batch)
        state = batch["observation.state"]  # (B, 1, state_dim) or (B, state_dim)

        B = state.shape[0]

        # For CEM we need a goal image. During deployment this should be
        # provided in the batch. For now, use a placeholder approach:
        # The goal can be passed as batch["goal.images.*"] or we skip planning.
        goal_images = {}
        for key, value in batch.items():
            if key.startswith("goal.images."):
                cam_name = key.split("goal.images.")[-1]
                goal_images[cam_name] = value

        if not goal_images:
            # No goal provided — return zeros as fallback
            return torch.zeros(B, self.config.cem_horizon, self.config.action_dim, device=device)

        # Encode current observation
        # We need num_hist frames; replicate current frame if only 1 is available
        if state.dim() == 2:
            state = state.unsqueeze(1)
        T_avail = state.shape[1]

        with torch.no_grad():
            # Encode current visual observation
            visual_curr = self._encode_images(images_dict, B, T_avail)
            proprio_curr = self.proprio_encoder(state)

            # Pad to num_hist frames if needed
            if T_avail < self.config.num_hist:
                pad = self.config.num_hist - T_avail
                visual_curr = torch.cat(
                    [visual_curr[:, :1].repeat(1, pad, 1, 1), visual_curr], dim=1
                )
                proprio_curr = torch.cat(
                    [proprio_curr[:, :1].repeat(1, pad, 1), proprio_curr], dim=1
                )

            # Encode goal
            goal_visual = self._encode_images(goal_images, B, 1)  # (B, 1, num_cam*256, 768)
            goal_state = batch.get("goal.state", state[:, -1:])
            goal_proprio = self.proprio_encoder(goal_state)  # (B, 1, 768)

            z_obs_goal = {"visual": goal_visual[:, 0], "proprio": goal_proprio[:, 0]}

            # Create rollout function for CEM
            def rollout_fn(action_batch):
                # action_batch: (num_samples, horizon, action_dim)
                N = action_batch.shape[0]
                # Expand observations to match num_samples
                v_exp = repeat(visual_curr, "b t p d -> (b n) t p d", n=N)
                p_exp = repeat(proprio_curr, "b t d -> (b n) t d", n=N)
                # For B=1 (typical inference), flatten
                if B == 1:
                    result = self._rollout(v_exp, p_exp, action_batch)
                else:
                    a_exp = repeat(action_batch, "n t d -> (b n) t d", b=B)
                    result = self._rollout(v_exp, p_exp, a_exp)
                return result

            # Expand goal for B=1
            z_goal = {
                "visual": z_obs_goal["visual"][:1],
                "proprio": z_obs_goal["proprio"][:1],
            }

            best_actions = planner.plan(
                rollout_fn=rollout_fn,
                objective_fn=self._objective_fn,
                z_obs_goal=z_goal,
                device=device,
            )

        return best_actions.unsqueeze(0)  # (1, horizon, action_dim)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for deployment.

        Uses action chunking: plans a full chunk and caches remaining actions.
        """
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch)  # (B, horizon, action_dim)
            for t in range(action_chunk.shape[1]):
                self._action_queue.append(action_chunk[:, t])

        return self._action_queue.popleft()
