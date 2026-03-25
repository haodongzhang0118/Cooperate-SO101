"""Configuration for the DINO-WM single-arm test policy (Phase 1).

Single SO101 arm, 2 cameras (wrist + global), 6D state/action,
DINO-WM world model with CEM planning.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("dino_wm_test")
@dataclass
class DinoWMTestConfig(PreTrainedConfig):
    """Configuration for DinoWMTestPolicy.

    Designed for a single SO101 arm with 2 cameras (wrist + global).
    Uses a frozen DINOv3 ViT-B/16 encoder and a causal ViTPredictor
    to predict dynamics in DINO latent space.
    """

    # Input / output structure
    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- DINOv3 encoder ---
    dino_img_size: int = 256
    dino_embed_dim: int = 768
    dino_num_patches: int = 256  # (256/16)^2

    # --- ViTPredictor ---
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.0
    predictor_emb_dropout: float = 0.0

    # --- World model ---
    num_cameras: int = 2
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    state_dim: int = 6
    action_dim: int = 6

    # --- CEM planner (inference) ---
    cem_horizon: int = 5
    cem_topk: int = 10
    cem_num_samples: int = 100
    cem_opt_steps: int = 30
    cem_var_scale: float = 1.0
    cem_objective_alpha: float = 0.1
    cem_objective_mode: str = "last"

    # --- Optimizer ---
    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 1e-4

    @property
    def num_patches_per_frame(self) -> int:
        """Total tokens per frame: num_cameras * dino_num_patches + proprio + action."""
        return self.num_cameras * self.dino_num_patches + 2  # +2 for proprio, action tokens

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.state_dim,)
            )
        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )
        image_features = {
            k: v for k, v in self.input_features.items() if v.type == FeatureType.VISUAL
        }
        if not image_features:
            raise ValueError(
                "DinoWMTestPolicy requires at least one visual input feature "
                "(e.g., observation.images.wrist)."
            )

    @property
    def observation_delta_indices(self) -> list:
        # frameskip=5, num_hist=3: [-10, -5, 0] (history) + [5] (target)
        indices = [
            -self.frameskip * i for i in range(self.num_hist - 1, -1, -1)
        ]
        indices.append(self.frameskip)
        return indices

    @property
    def action_delta_indices(self) -> list:
        # Actions correspond to the history frames: [-10, -5, 0]
        return [
            -self.frameskip * i for i in range(self.num_hist - 1, -1, -1)
        ]

    @property
    def reward_delta_indices(self) -> None:
        return None
