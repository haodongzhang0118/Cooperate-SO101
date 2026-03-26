import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.custom_scene import CUSTOM_SCENE_CFG, CUSTOM_SCENE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from ..template import mdp


@configclass
class CustomTaskSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the custom task."""

    scene: AssetBaseCfg = CUSTOM_SCENE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/top_camera",
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
        width=640,
        height=480,
        update_period=1 / 30.0,
    )


def cube_in_box(env, cube_cfg: SceneEntityCfg, box_cfg: SceneEntityCfg, x_range: tuple[float, float], y_range: tuple[float, float], height_threshold: float):
    """Termination condition for the object in the box."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    box: RigidObject = env.scene[box_cfg.name]
    box_x = box.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    box_y = box.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]

    cube: RigidObject = env.scene[cube_cfg.name]
    cube_x = cube.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    cube_y = cube.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    done = torch.logical_and(done, cube_x < box_x + x_range[1])
    done = torch.logical_and(done, cube_x > box_x + x_range[0])
    done = torch.logical_and(done, cube_y < box_y + y_range[1])
    done = torch.logical_and(done, cube_y > box_y + y_range[0])
    done = torch.logical_and(done, cube_z < height_threshold)

    return done


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Termination configuration for the custom task."""
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
class ObservationsCfg(SingleArmObservationsCfg):
    """Observation configuration with top camera added."""

    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        """Observations for policy group, with top camera."""

        top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CustomTaskEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the custom task environment."""

    scene: CustomTaskSceneCfg = CustomTaskSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "pick up the red cube and place it into the box."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.2, -1.0, 0.5)
        self.viewer.lookat = (0.6, 0.0, -0.2)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

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
                randomize_object_uniform(
                    "box",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
            ],
        )
