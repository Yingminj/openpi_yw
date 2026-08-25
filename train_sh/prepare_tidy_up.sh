#!/usr/bin/env bash
# Prepare the tidy_up_stationery_le dataset: compute normalization stats for pi05_yw_tidy_up.
set -euo pipefail

DATASET_ROOT="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361"
CONFIG_NAME="pi05_yw_tidy_up"

if [[ ! -e "${DATASET_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATASET_ROOT}" >&2
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
