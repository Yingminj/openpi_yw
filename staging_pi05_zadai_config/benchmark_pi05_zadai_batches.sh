#!/usr/bin/env bash
set -u

cd /ssd/hhw/openpi-hzh
export PYTHONPATH=/ssd/hhw/openpi-hzh/src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p /ssd/hhw/openpi-hzh/logs/zadai_pytorch_batch_benchmark
run_id="$(date +%Y%m%d_%H%M%S)"

for per_device_batch in 1 2 4 8 12 16; do
  global_batch=$((per_device_batch * 4))
  master_port=$((29700 + per_device_batch))
  exp_name="benchmark_zadai_b${per_device_batch}_${run_id}"
  log_path="/ssd/hhw/openpi-hzh/logs/zadai_pytorch_batch_benchmark/batch_${per_device_batch}.log"

  printf 'START batch=%s global_batch=%s time=%s\n' \
    "$per_device_batch" "$global_batch" "$(date --iso-8601=seconds)" | tee "$log_path"

  timeout --signal=TERM --kill-after=20s 150s \
    /root/.local/bin/uv run --no-sync python -m torch.distributed.run \
      --nnodes=1 \
      --nproc-per-node=4 \
      --master-addr=127.0.0.1 \
      --master-port="$master_port" \
      scripts/train_pytorch.py pi05_hhw_tj_zadai_200_pytorch_uniform \
      --exp-name="$exp_name" \
      --checkpoint-base-dir=/ssd/hhw/openpi-hzh/checkpoints_batch_benchmark \
      --batch-size="$global_batch" \
      --num-train-steps=1000000 \
      --log-interval=20 \
      --save-interval=1000000 \
      --no-wandb-enabled >>"$log_path" 2>&1
  status=$?

  printf 'END batch=%s status=%s time=%s\n' \
    "$per_device_batch" "$status" "$(date --iso-8601=seconds)" | tee -a "$log_path"

  for _ in $(seq 1 20); do
    if ! nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
      break
    fi
    sleep 2
  done
done

echo "BENCHMARK_COMPLETE run_id=$run_id"
