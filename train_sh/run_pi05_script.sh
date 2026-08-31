#!/usr/bin/env bash
# pi0.5 full fine-tuning on the tidy-up dataset (EEF by default, joint via MODE=joint).
set -euo pipefail

MODE="${MODE:-eef}"
case "${MODE}" in
    eef)   CONFIG_NAME="pi05_yw_tidy_up_eef"; DEFAULT_ROOT="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_505_eef" ;;
    joint) CONFIG_NAME="pi05_yw_tidy_up";     DEFAULT_ROOT="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_505" ;;
    *) echo "MODE must be 'eef' or 'joint', got '${MODE}'" >&2; exit 1 ;;
esac

PROJECT_ROOT="${OPENPI_ROOT:-/home/kewei/YING/openpi_yw}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
DATASET_PATH="${TIDY_UP_DATASET_ROOT:-${DEFAULT_ROOT}}"
export TIDY_UP_DATASET_ROOT="${DATASET_PATH}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pi0.5fine_tuning_tidy_up_${MODE}}"
LOG_PATH="${PROJECT_ROOT}/training_logs/${EXPERIMENT_NAME}.log"

# One entry per visible GPU; BATCH_SIZE must stay divisible by that count.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
BATCH_SIZE="${BATCH_SIZE:-16}"
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "BATCH_SIZE=${BATCH_SIZE} is not divisible by ${NUM_GPUS} visible GPU(s)" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/training_logs"

# Drop any inherited PYTHONPATH (e.g. /opt/ros) before prepending this checkout.
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src"
# Base weights: prefer a local copy, else download from the public bucket.
LOCAL_PI05_BASE="${PROJECT_ROOT}/checkpoint/pi05_base/params"
if [[ -z "${PI05_BASE_CHECKPOINT:-}" && -d "${LOCAL_PI05_BASE}" ]]; then
    export PI05_BASE_CHECKPOINT="${LOCAL_PI05_BASE}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED="1"

if [[ ! -s "${DATASET_PATH}/norm_stats.json" ]]; then
    echo "Missing ${DATASET_PATH}/norm_stats.json - run 'MODE=${MODE} train_sh/prepare_data.sh' first." >&2
    exit 1
fi

printf '[%s] Starting pi0.5 full fine-tuning on GPUs %s\n' \
    "$(date '+%F %T')" "${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_PATH}"
printf '[%s] mode=%s config=%s exp_name=%s dataset=%s global_batch=%s per_device_batch=%s steps=150000 prompt_from_task=false\n' \
    "$(date '+%F %T')" "${MODE}" "${CONFIG_NAME}" "${EXPERIMENT_NAME}" "${DATASET_PATH}" \
    "${BATCH_SIZE}" "$(( BATCH_SIZE / NUM_GPUS ))" | tee -a "${LOG_PATH}"

"${UV_BIN}" run --no-sync scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXPERIMENT_NAME}" \
    --batch-size="${BATCH_SIZE}" 2>&1 | tee -a "${LOG_PATH}"

printf '[%s] Training exited with code %s\n' \
    "$(date '+%F %T')" "${PIPESTATUS[0]}" | tee -a "${LOG_PATH}"
