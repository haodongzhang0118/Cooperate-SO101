"""Local DINO-WM policy for Isaac Sim evaluation via CEM planning."""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

from collections import deque

import numpy as np
import torch
from einops import rearrange, repeat

from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

from .base import Policy


def _setup_lerobot_imports(lerobot_src: str):
    """Register a stub ``lerobot.policies`` in *sys.modules* so that
    subpackages (e.g. ``dino_wm_test``, ``pretrained``) can be imported
    without executing the real ``__init__.py`` which eagerly pulls in every
    policy (GR00T, SmolVLA …) and their heavy / incompatible dependencies.

    All other lerobot subpackages (common, configs, …) are left untouched
    so they import normally.
    """
    if lerobot_src not in sys.path:
        sys.path.insert(0, lerobot_src)

    # Import the top-level lerobot package normally so that subpackages
    # like lerobot.common remain fully functional.
    import lerobot as _lr

    policies_dir = os.path.join(os.path.dirname(_lr.__file__), "policies")

    # Only replace lerobot.policies with a thin stub so Python can resolve
    # subpackages (dino_wm_test, pretrained) without executing __init__.py.
    if "lerobot.policies" not in sys.modules:
        stub = types.ModuleType("lerobot.policies")
        stub.__path__ = [policies_dir]
        stub.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = stub


class DinoWMLocalPolicy(Policy):
    """DINO-WM world model policy that runs CEM planning locally.

    Loads a trained DinoWMTestPolicy checkpoint, buffers observation history,
    extracts goal images from a LeRobot dataset, and plans actions via CEM
    in DINO latent space.

    Expected observation dict keys (from Isaac Sim):
        - camera image keys (e.g. "wrist", "top"): (num_envs, H, W, 3) uint8
        - "joint_pos": (num_envs, action_dim) float — current joint positions
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_repo_id: str,
        camera_names: list[str] = None,
        device: str = "cuda",
        goal_episode_idx: int = 0,
        goal_frame: int = -1,
    ):
        super().__init__("dino_wm")
        self.device = torch.device(device)
        self.camera_names = camera_names or ["wrist", "top"]
        self.dataset_repo_id = dataset_repo_id
        self.goal_episode_idx = goal_episode_idx
        self.goal_frame = goal_frame

        lerobot_src = self._find_lerobot_path(checkpoint_path)
        _setup_lerobot_imports(lerobot_src)

        from lerobot.policies.dino_wm_test.modeling_dino_wm_test import DinoWMTestPolicy

        self.policy = DinoWMTestPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()

        self.config = self.policy.config
        self.config.cem_num_samples = 20
        self.config.cem_opt_steps = 10
        self.config.cem_topk = 5
        self.camera_names = self.config.camera_names
        print(f"[DinoWM] Using cameras from trained model: {self.camera_names}")
        self._obs_buffer = deque(maxlen=self.config.num_hist)
        self._goal_latent = None
        self._prev_plan = None

        self._load_norm_stats(checkpoint_path)

    def _load_norm_stats(self, checkpoint_path: str):
        """Load action and state normalization stats from the checkpoint.

        Training uses MEAN_STD normalization, so the model operates in
        normalized space. We need these stats to normalize inputs and
        un-normalize CEM outputs.
        """
        import json
        import struct

        norm_file = os.path.join(
            checkpoint_path,
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
        )
        with open(norm_file, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            data_start = 8 + header_size

            def _read(name):
                info = header[name]
                f.seek(data_start + info["data_offsets"][0])
                n_bytes = info["data_offsets"][1] - info["data_offsets"][0]
                buf = np.frombuffer(f.read(n_bytes), dtype=np.float32).copy()
                return torch.from_numpy(buf).to(self.device)

            self._action_mean = _read("action.mean")
            self._action_std = _read("action.std")
            self._state_mean = _read("observation.state.mean")
            self._state_std = _read("observation.state.std")

        print(f"[DinoWM] Loaded norm stats — action mean: {self._action_mean.cpu().numpy()}")
        print(f"[DinoWM]                      action std:  {self._action_std.cpu().numpy()}")
        print(f"[DinoWM]                      state mean:  {self._state_mean.cpu().numpy()}")
        print(f"[DinoWM]                      state std:   {self._state_std.cpu().numpy()}")

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state from raw lerobot motor units to zero-mean unit-std."""
        return (state - self._state_mean) / (self._state_std + 1e-8)

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Un-normalize CEM output from zero-mean unit-std to raw lerobot motor units."""
        return action * self._action_std + self._action_mean

    @staticmethod
    def _find_lerobot_path(checkpoint_path: str) -> str:
        """Walk up from checkpoint to find the lerobot src directory."""
        path = Path(checkpoint_path).resolve()
        for parent in path.parents:
            candidate = parent / "lerobot" / "src"
            if candidate.is_dir():
                return str(candidate)
        candidate = Path(checkpoint_path).resolve().parents[5] / "lerobot" / "src"
        if candidate.is_dir():
            return str(candidate)
        raise FileNotFoundError(
            "Could not locate lerobot/src relative to checkpoint path. "
            "Ensure the checkpoint is inside the Cooperate-SO101 project tree."
        )

    def _load_goal_from_dataset(self):
        """Load a specific frame of a demonstration episode as goal."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(self.dataset_repo_id)
        ep = ds.meta.episodes[self.goal_episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        if self.goal_frame < 0:
            goal_idx = ep_end - 1
        else:
            goal_idx = min(ep_start + self.goal_frame, ep_end - 1)

        print(f"[DinoWM] Goal: episode {self.goal_episode_idx}, "
              f"frame {goal_idx} (offset {goal_idx - ep_start}/{ep_end - ep_start})")
        sample = ds[goal_idx]

        goal_images = {}
        for cam_name in self.camera_names:
            key = f"observation.images.{cam_name}"
            img = sample[key]  # (3, H, W) float [0,1] or uint8
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            goal_images[cam_name] = img.unsqueeze(0).unsqueeze(0).to(self.device)

        goal_state = sample["observation.state"]
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0).unsqueeze(0)
        goal_state = goal_state.to(self.device)
        goal_state = self._normalize_state(goal_state)

        with torch.no_grad():
            goal_visual = self.policy._encode_images(goal_images, 1, 1)
            goal_proprio = self.policy.proprio_encoder(goal_state)

        self._goal_latent = {
            "visual": goal_visual[:, 0],
            "proprio": goal_proprio[:, 0],
        }
        print(f"[DinoWM] Goal loaded from episode {self.goal_episode_idx}, frame {goal_idx}")

    def _obs_dict_to_tensors(self, obs_dict: dict) -> dict:
        """Convert Isaac Sim observation dict to tensors suitable for the policy.

        Isaac Sim provides images as (num_envs, H, W, 3) uint8 and joint_pos as
        (num_envs, D) float.
        """
        images = {}
        for cam_name in self.camera_names:
            if cam_name not in obs_dict:
                continue
            img = obs_dict[cam_name]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            if img.dim() == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
            images[cam_name] = img.to(self.device)

        joint_pos = obs_dict["joint_pos"]
        if isinstance(joint_pos, np.ndarray):
            joint_pos = torch.from_numpy(joint_pos)
        state_np = convert_leisaac_action_to_lerobot(joint_pos)
        state = torch.from_numpy(state_np).float().to(self.device)
        state = self._normalize_state(state)

        return {"images": images, "state": state}

    def _build_history_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack buffered observations into (1, num_hist, ...) tensors."""
        buf = list(self._obs_buffer)
        while len(buf) < self.config.num_hist:
            buf.insert(0, buf[0])

        images_stacked = {}
        for cam_name in self.camera_names:
            imgs = torch.stack([b["images"][cam_name][0] for b in buf], dim=0)
            images_stacked[cam_name] = imgs.unsqueeze(0)

        state = torch.stack([b["state"][0] for b in buf], dim=0).unsqueeze(0)

        with torch.no_grad():
            visual = self.policy._encode_images(
                images_stacked, 1, self.config.num_hist
            )
            proprio = self.policy.proprio_encoder(state)

        return visual, proprio

    def get_action(self, obs_dict: dict) -> torch.Tensor:
        """Plan actions via CEM given current Isaac Sim observation.

        Args:
            obs_dict: Observation from Isaac Sim containing camera images and joint_pos.

        Returns:
            Tensor of shape (horizon, num_envs, action_dim) in leisaac convention.
        """
        import time as _time
        from pathlib import Path
        from torchvision.utils import save_image

        if not hasattr(self, "_step_count"):
            self._step_count = 0
            self._save_root = Path("/home/haodong/Desktop/Cooperate-SO101/eval_frames")
            self._save_cameras = list(self.camera_names) + [c for c in ["front"] if c not in self.camera_names]
            for cam_name in self._save_cameras:
                (self._save_root / cam_name).mkdir(parents=True, exist_ok=True)
            print(f"[DinoWM] Saving frames to {self._save_root}")

        for cam_name in self._save_cameras:
            if cam_name in obs_dict:
                img = obs_dict[cam_name]
                if isinstance(img, torch.Tensor):
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0
                    if img.dim() == 4 and img.shape[-1] == 3:
                        img = img.permute(0, 3, 1, 2)
                    save_image(img[0], self._save_root / cam_name / f"{self._step_count:04d}.png")
        self._step_count += 1

        if self._goal_latent is None:
            print("[DinoWM] Loading goal from dataset...")
            t0 = _time.time()
            self._load_goal_from_dataset()
            print(f"[DinoWM] Goal loaded in {_time.time() - t0:.1f}s")

        print(f"[DinoWM] Planning (step {self._step_count - 1}, warm_start={'yes' if self._prev_plan is not None else 'no'})...")
        t0 = _time.time()

        tensors = self._obs_dict_to_tensors(obs_dict)
        self._obs_buffer.append(tensors)

        visual_curr, proprio_curr = self._build_history_batch()
        print(f"[DinoWM]   Encoding took {_time.time() - t0:.1f}s")

        planner = self.policy._get_planner()
        objective_fn = self.policy._objective_fn

        z_goal = {
            "visual": self._goal_latent["visual"][:1],
            "proprio": self._goal_latent["proprio"][:1],
        }

        t1 = _time.time()
        mini_batch = 5
        with torch.no_grad():
            def rollout_fn(action_batch):
                N = action_batch.shape[0]
                results = {"visual": [], "proprio": []}
                for start in range(0, N, mini_batch):
                    end = min(start + mini_batch, N)
                    chunk_size = end - start
                    v_exp = repeat(visual_curr, "b t p d -> (b n) t p d", n=chunk_size)
                    p_exp = repeat(proprio_curr, "b t d -> (b n) t d", n=chunk_size)
                    out = self.policy._rollout(v_exp, p_exp, action_batch[start:end])
                    results["visual"].append(out["visual"])
                    results["proprio"].append(out["proprio"])
                return {
                    "visual": torch.cat(results["visual"], dim=0),
                    "proprio": torch.cat(results["proprio"], dim=0),
                }

            best_actions = planner.plan(
                rollout_fn=rollout_fn,
                objective_fn=objective_fn,
                z_obs_goal=z_goal,
                device=self.device,
                actions=self._prev_plan,
            )
        print(f"[DinoWM]   CEM planning took {_time.time() - t1:.1f}s")

        # Warm-start for next call: shift plan by 1 step, append zero
        self._prev_plan = torch.cat([
            best_actions[1:],
            torch.zeros(1, self.config.action_dim, device=self.device),
        ])

        best_actions = self._unnormalize_action(best_actions)
        action_np = convert_lerobot_action_to_leisaac(best_actions.cpu().numpy())
        action_chunk = torch.from_numpy(action_np).float()
        print(f"[DinoWM]   Actions (lerobot): {best_actions[0].cpu().numpy()}")
        print(f"[DinoWM]   Actions (leisaac): {action_chunk[0].numpy()}")
        print(f"[DinoWM]   Total: {_time.time() - t0:.1f}s")

        # MPC: only execute the first action, discard the rest
        num_envs = tensors["state"].shape[0]
        return action_chunk[0].unsqueeze(0).unsqueeze(0).expand(-1, num_envs, -1)
