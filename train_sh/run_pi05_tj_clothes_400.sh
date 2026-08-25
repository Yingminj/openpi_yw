#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/ssd/hhw/openpi-hzh"
CONFIG_NAME="pi05_yw_tidy_up"
EXPERIMENT_NAME="pi0.5fine_tuning_tidy_up"
DATASET_PATH="${TIDY_UP_DATASET_ROOT:-/ssd/ying/lerobot_dataset/gripper/tidy_up_stationery_le}"
export TIDY_UP_DATASET_ROOT="${DATASET_PATH}"
LOG_PATH="${PROJECT_ROOT}/training_logs/${EXPERIMENT_NAME}.log"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/training_logs"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
export PYTHONUNBUFFERED="1"

printf '[%s] Starting pi0.5 full fine-tuning on GPUs %s\n' \
    "$(date '+%F %T')" "${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_PATH}"
printf '[%s] Config=%s exp_name=%s dataset=%s global_batch=12 per_device_batch=2 steps=150000 keyframe_sampling=disabled prompt_from_task=true\n' \
    "$(date '+%F %T')" "${CONFIG_NAME}" "${EXPERIMENT_NAME}" "${DATASET_PATH}" \
    | tee -a "${LOG_PATH}"

/root/.local/bin/uv run --no-sync scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXPERIMENT_NAME}" 2>&1 | tee -a "${LOG_PATH}"

printf '[%s] Training exited with code %s\n' \
    "$(date '+%F %T')" "${PIPESTATUS[0]}" | tee -a "${LOG_PATH}"
