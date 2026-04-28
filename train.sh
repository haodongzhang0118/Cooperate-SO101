#!/bin/bash

lerobot-train \
    --policy.type=dino_seqwm \
    --policy.push_to_hub=true \
    --policy.repo_id=haodoz0118/dino_seqwm \
    --dataset.repo_id=haodoz0118/bimanual_cooperate \
    --batch_size=256 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --num_workers=4 \
    --seed=1000
