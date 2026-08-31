#!/usr/bin/env bash
set -euo pipefail

cd /ssd/hhw/openpi-hzh
export PYTHONPATH=/ssd/hhw/openpi-hzh/src
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec /root/.local/bin/uv run --no-sync examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /ssd/hhw/openpi-hzh/checkpoint/pi05_base \
  --config-name pi05_hhw_tj_zadai_200_pytorch_uniform \
  --output-path /ssd/hhw/openpi-hzh/checkpoint/pi05_base_pytorch_bfloat16 \
  --precision bfloat16
