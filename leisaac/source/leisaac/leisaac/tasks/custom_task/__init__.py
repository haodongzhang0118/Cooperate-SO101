import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-CustomTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_task_env_cfg:CustomTaskEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-CustomTask-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_task_bi_arm_env_cfg:CustomTaskBiArmEnvCfg",
    },
)
