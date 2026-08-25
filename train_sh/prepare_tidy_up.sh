#!/usr/bin/env bash
# Prepare the tidy_up_stationery_le dataset: compute normalization stats for pi05_yw_tidy_up.
set -euo pipefail

DATASET_ROOT="${TIDY_UP_DATASET_ROOT:-/ssd/ying/lerobot_dataset/gripper/tidy_up_stationery_le}"
export TIDY_UP_DATASET_ROOT="${DATASET_ROOT}"
CONFIG_NAME="pi05_yw_tidy_up"

if [[ ! -s "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Not a LeRobot dataset (no meta/info.json): ${DATASET_ROOT}" >&2
    exit 1
fi

OPENPI_ROOT="/ssd/hhw/openpi-hzh"
cd "${OPENPI_ROOT}"

# Guard against a stale venv / editable install pointing at another openpi checkout (e.g. /ssd/hzh/openpi-hzh).
unset VIRTUAL_ENV || true
export PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${DATASET_ROOT}/norm_stats.json" ]]; then
    /root/.local/bin/uv run --no-sync scripts/compute_norm_stats_low_mem.py \
        --config-name "${CONFIG_NAME}" \
        --direct-lerobot \
        --direct-chunk-size 1024
fi

test -s "${DATASET_ROOT}/norm_stats.json"
echo "PREPARE_TIDY_UP_OK"
