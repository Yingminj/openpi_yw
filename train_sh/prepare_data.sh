#!/usr/bin/env bash
# Compute normalization stats for the tidy-up dataset (EEF by default, joint via MODE=joint).
set -euo pipefail

MODE="${MODE:-eef}"
case "${MODE}" in
    eef)   CONFIG_NAME="pi05_yw_tidy_up_eef"; DEFAULT_ROOT="/ssd/ying/lerobot_test_yingminj/gripper/tidy_up_505_eef" ;;
    # joint) CONFIG_NAME="pi05_yw_tidy_up";     DEFAULT_ROOT="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_505" ;;
    *) echo "MODE must be 'eef' or 'joint', got '${MODE}'" >&2; exit 1 ;;
esac

DATASET_ROOT="${TIDY_UP_DATASET_ROOT:-${DEFAULT_ROOT}}"
export TIDY_UP_DATASET_ROOT="${DATASET_ROOT}"

if [[ ! -s "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Not a LeRobot dataset (no meta/info.json): ${DATASET_ROOT}" >&2
    exit 1
fi

OPENPI_ROOT="${OPENPI_ROOT:-/ssd/hhw/openpi-hzh}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
cd "${OPENPI_ROOT}"

# Guard against a stale venv / editable install pointing at another openpi checkout.
unset VIRTUAL_ENV || true
# Drop any inherited PYTHONPATH (e.g. /opt/ros) before prepending this checkout.
export PYTHONPATH="${OPENPI_ROOT}/src"

# Stats are written to `assets_dirs / repo_id`; repo_id is absolute, so they land in the dataset dir.
if [[ ! -s "${DATASET_ROOT}/norm_stats.json" ]]; then
    "${UV_BIN}" run --no-sync scripts/compute_norm_stats_low_mem.py \
        --config-name "${CONFIG_NAME}" \
        --direct-lerobot \
        --direct-chunk-size 1024
fi

test -s "${DATASET_ROOT}/norm_stats.json"
echo "PREPARE_TIDY_UP_OK mode=${MODE} config=${CONFIG_NAME} dataset=${DATASET_ROOT}"
