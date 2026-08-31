#!/usr/bin/env bash
set -euo pipefail

cd /ssd/hhw/openpi-hzh
export PYTHONPATH=/ssd/hhw/openpi-hzh/src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29627
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec /root/.local/bin/uv run --no-sync python -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  scripts/train_pytorch.py pi05_hhw_tj_zadai_200_pytorch_uniform
