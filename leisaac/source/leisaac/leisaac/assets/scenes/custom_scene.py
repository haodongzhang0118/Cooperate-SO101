from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Custom Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

CUSTOM_SCENE_USD_PATH = str(SCENES_ROOT / "custom_scene" / "scene.usd")

CUSTOM_SCENE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CUSTOM_SCENE_USD_PATH,
    )
)
