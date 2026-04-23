"""Launch an environment and let the user look around in the viewer.
No teleop device needed — just opens the sim and steps indefinitely.
"""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View an environment without teleop.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import leisaac.tasks  # noqa: F401 — register envs


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Fill action fields with JointPosition defaults so validation passes
    from leisaac.devices.action_process import init_action_cfg
    try:
        init_action_cfg(env_cfg.actions, device="bi-so101leader")
    except Exception:
        try:
            init_action_cfg(env_cfg.actions, device="so101leader")
        except Exception:
            pass

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    print("\n" + "=" * 60)
    print("Environment loaded. Drag the viewer to look around.")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    action_dim = env.action_manager.total_action_dim
    while simulation_app.is_running():
        zero_action = torch.zeros(env.num_envs, action_dim, device=env.device)
        env.step(zero_action)

    env.close()


if __name__ == "__main__":
    main()
