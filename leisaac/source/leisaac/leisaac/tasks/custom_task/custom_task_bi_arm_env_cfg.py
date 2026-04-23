import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.custom_scene import CUSTOM_SCENE_CFG, CUSTOM_SCENE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform

from ..template import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)
from ..template.bi_arm_env_cfg import BiArmEventCfg

from .custom_task_env_cfg import ROBOT_INIT_JOINT_POS, cube_in_box


def reset_arm_to_init_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos: dict[str, float] = ROBOT_INIT_JOINT_POS,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
):
    """Write specific joint positions to one arm after the default scene reset."""
    asset: Articulation = env.scene[asset_cfg.name]
    positions = asset.data.default_joint_pos[env_ids].clone()
    for name, val in joint_pos.items():
        joint_ids = asset.find_joints(name)[0]
        positions[:, joint_ids] = val
    zeros = torch.zeros_like(positions)
    asset.write_joint_state_to_sim(positions, zeros, env_ids=env_ids)


@configclass
class CustomTaskBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the bimanual custom task."""

    scene: AssetBaseCfg = CUSTOM_SCENE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=320,
        height=240,
        update_period=1 / 30.0,
    )

    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=320,
        height=240,
        update_period=1 / 30.0,
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.27, 1.0),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=320,
        height=240,
        update_period=1 / 30.0,
    )


@configclass
class BiArmTerminationsCfg_(BiArmTerminationsCfg):
    """Termination configuration for the bimanual custom task."""

    success = DoneTerm(
        func=cube_in_box,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "box_cfg": SceneEntityCfg("box"),
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "height_threshold": 0.10,
        },
    )


@configclass
class CustomTaskBiArmEventCfg(BiArmEventCfg):
    """Event configuration that resets both arms to training-matched poses."""

    reset_right_arm_pose = EventTerm(
        func=reset_arm_to_init_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("right_arm")},
    )

    reset_left_arm_pose = EventTerm(
        func=reset_arm_to_init_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("left_arm")},
    )


@configclass
class CustomTaskBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the bimanual custom task environment."""

    scene: CustomTaskBiArmSceneCfg = CustomTaskBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    events: CustomTaskBiArmEventCfg = CustomTaskBiArmEventCfg()

    terminations: BiArmTerminationsCfg_ = BiArmTerminationsCfg_()

    task_description: str = "cooperatively pick up the red cube and place it into the box."

    def use_teleop_device(self, teleop_device) -> None:
        if teleop_device in ["keyboard", "gamepad"]:
            teleop_device = "bi-so101leader"
        super().use_teleop_device(teleop_device)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.2, -1.0, 0.5)
        self.viewer.lookat = (0.6, 0.0, -0.2)

        # Right arm (leader): same position as the original single arm
        self.scene.right_arm.init_state.pos = (0.35, -0.64, 0.01)
        self.scene.right_arm.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # wxyz, no rotation

        # Left arm (helper): opposite side, rotated 180 deg relative to right arm
        self.scene.left_arm.init_state.pos = (0.60, -0.04, 0.01)
        self.scene.left_arm.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # wxyz, facing right arm

        parse_usd_and_create_subassets(CUSTOM_SCENE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
            ],
        )
