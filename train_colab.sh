#!/bin/bash
# Colab launcher for dino_seqwm.
# Assumes:
#   /content/data/bimanual_cooperate/   ← dataset (untar'd from Drive)
#   /content/drive/MyDrive/dino_seqwm/  ← Drive folder for checkpoints
# Override via env vars, e.g.:
#   STEPS=100 bash train_colab.sh
#   BATCH_SIZE=128 bash train_colab.sh

set -e

DATASET_ROOT="${DATASET_ROOT:-/content/data/bimanual_cooperate}"
DRIVE_RUNS_DIR="${DRIVE_RUNS_DIR:-/content/drive/MyDrive/dino_seqwm/runs}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_RUNS_DIR}/${RUN_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-64}"   # 256 + T=2310 + depth=6×2 predictors will OOM on A100 80GB
STEPS="${STEPS:-200000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SEED="${SEED:-1000}"
USE_AMP="${USE_AMP:-true}"      # BF16 mixed precision; A100 native

if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo "ERROR: ${DATASET_ROOT}/meta/info.json not found." >&2
    echo "Did you tar -xzf bimanual_cooperate.tar.gz -C /content/data/  ?" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/content/drive/MyDrive/dino_seqwm/hf_cache}"

# Force HuggingFace Accelerate to use BF16. LeRobot's Accelerator() ctor
# does NOT pass mixed_precision, so --policy.use_amp on its own is a no-op;
# we have to enable autocast via this env var.
if [ "${USE_AMP}" = "true" ]; then
    export ACCELERATE_MIXED_PRECISION=bf16
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Run name:                   ${RUN_NAME}"
echo "Dataset:                    ${DATASET_ROOT}"
echo "Output dir:                 ${OUTPUT_DIR}"
echo "HF cache:                   ${HF_HOME}"
echo "Batch size:                 ${BATCH_SIZE}"
echo "Steps:                      ${STEPS}"
echo "Use AMP:                    ${USE_AMP}"
echo "ACCELERATE_MIXED_PRECISION: ${ACCELERATE_MIXED_PRECISION:-no}"
echo

lerobot-train \
    --policy.type=dino_seqwm \
    --policy.push_to_hub=false \
    --policy.use_amp="${USE_AMP}" \
    --dataset.repo_id=local/bimanual_cooperate \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --num_workers="${NUM_WORKERS}" \
    --seed="${SEED}"
