#!/bin/bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CustomTask-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id=haodoz0118/PickAndPlace \
    --lerobot_dataset_fps=30
