"""Configuration for DINO-SeqWM (Phase 2): bimanual SO101 sequential world model.

Two SO101 arms (helper=left, leader=right), 12D total state/action.
3 cameras (left_wrist, right_wrist, top). Two ViTPredictors with sequential
conditioning via action token replacement. Phase-aware loss weighting using
the `phase` int8 column embedded in the bimanual_cooperate dataset.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("dino_seqwm")
@dataclass
class DinoSeqWMConfig(PreTrainedConfig):
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
    dino_num_patches: int = 256

    # --- ViTPredictor (per agent — helper and leader each get their own) ---
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.0
    predictor_emb_dropout: float = 0.0

    # --- World model: bimanual ---
    num_cameras: int = 3
    camera_names: list[str] = field(
        default_factory=lambda: ["left_wrist", "right_wrist", "top"]
    )
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    state_dim: int = 12              # 6 helper + 6 leader joint positions
    action_dim: int = 12             # 6 helper + 6 leader action targets
    helper_action_dim: int = 6       # left-arm action width
    leader_action_dim: int = 6       # right-arm action width

    # --- Loss ---
    helper_aux_loss_weight: float = 0.5      # alpha in: loss = joint + alpha * helper
    # Phase-aware per-sample weights. Idea: helper_predictor specializes on
    # phase A (left arm working), leader_predictor on phase B (right arm working).
    # Down-weight, do NOT zero, so cross-phase gradients still flow.
    phase_helper_weight_A: float = 1.0
    phase_helper_weight_B: float = 0.1
    phase_leader_weight_A: float = 0.1
    phase_leader_weight_B: float = 1.0

    # --- CEM planner (single-arm, 6D per stage; the OTHER arm is frozen) ---
    cem_horizon: int = 5
    cem_topk: int = 10
    cem_num_samples: int = 100
    cem_opt_steps: int = 30
    cem_var_scale: float = 1.0
    cem_objective_alpha: float = 0.1
    cem_objective_mode: str = "last"

    # --- Phase detection at inference (per-episode adaptive on left-arm speed) ---
    phase_speed_history_frames: int = 30
    phase_speed_window_frames: int = 15      # smoothing/diff window
    phase_still_threshold_frac: float = 0.05  # still if speed < 5% of episode peak
    phase_still_threshold_floor: float = 0.005
    phase_still_required_frames: int = 30    # ~1s of continuous stillness to enter B
    # Once we've entered phase B in an episode, we never go back. Re-enables on reset().

    # --- Optimizer ---
    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 1e-4

    @property
    def num_patches_per_frame(self) -> int:
        # num_cameras * 256 visual patches + 1 proprio + 1 action token
        return self.num_cameras * self.dino_num_patches + 2

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
                "DinoSeqWMPolicy requires at least one visual input feature "
                "(e.g., observation.images.left_wrist)."
            )

    @property
    def observation_delta_indices(self) -> list:
        # frameskip=5, num_hist=3: history [-10, -5, 0] + target [+5]
        indices = [-self.frameskip * i for i in range(self.num_hist - 1, -1, -1)]
        indices.append(self.frameskip)
        return indices

    @property
    def action_delta_indices(self) -> list:
        # Actions correspond to the history frames only
        return [-self.frameskip * i for i in range(self.num_hist - 1, -1, -1)]

    @property
    def reward_delta_indices(self) -> None:
        return None
