#!/bin/bash

lerobot-train \
    --policy.type=dino_wm_test \
    --policy.push_to_hub=true \
    --policy.repo_id=haodoz0118/dino_wm_test \
    --dataset.repo_id=haodoz0118/PickAndPlace \
    --batch_size=2 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --num_workers=4 \
    --seed=1000
