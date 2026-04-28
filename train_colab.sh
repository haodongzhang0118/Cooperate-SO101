#!/bin/bash
# Colab launcher for dino_seqwm.
# Assumes:
#   /content/data/bimanual_cooperate/   ← dataset (untar'd from Drive)
#   /content/drive/MyDrive/dino_seqwm/  ← Drive folder for checkpoints
# Pass any extra flags through (e.g. `bash train_colab.sh --steps=100`).

set -e

DATASET_ROOT="${DATASET_ROOT:-/content/data/bimanual_cooperate}"
DRIVE_RUNS_DIR="${DRIVE_RUNS_DIR:-/content/drive/MyDrive/dino_seqwm/runs}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_RUNS_DIR}/${RUN_NAME}}"

# Sanity: dataset present
if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo "ERROR: ${DATASET_ROOT}/meta/info.json not found."
    echo "Did you tar -xzf bimanual_cooperate.tar.gz -C /content/data/  ?"
    exit 1
fi

# Cache HF assets (DINOv3 weights, ~340MB) on Drive so they survive runtime restarts
export HF_HOME="${HF_HOME:-/content/drive/MyDrive/dino_seqwm/hf_cache}"

mkdir -p "${OUTPUT_DIR}"
echo "Run name:    ${RUN_NAME}"
echo "Dataset:     ${DATASET_ROOT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "HF cache:    ${HF_HOME}"
echo

lerobot-train \
    --policy.type=dino_seqwm \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/bimanual_cooperate \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=256 \
    --steps=100000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=8 \
    --seed=1000 \
    "$@"
