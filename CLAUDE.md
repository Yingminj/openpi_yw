# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [openpi](https://github.com/Physical-Intelligence/openpi) (π₀ / π₀-FAST / π₀.₅ vision-language-action models) with local additions for bimanual/single-arm robots on `hhw-dev`. Upstream README covers checkpoints, DROID/ALOHA/LIBERO examples, PyTorch support, and troubleshooting; this file covers what is specific here.

## Commands

Everything runs through `uv` (workspace: root + `packages/openpi-client`). Setup requires `GIT_LFS_SKIP_SMUDGE=1 uv sync` and submodules (`git submodule update --init --recursive`).

```bash
# Tests — CI runs exactly this (markers: "manual" tests are excluded)
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/policies/bimanual_eef_rot6d_policy_test.py -k rot6d   # single test

# Lint / format (also enforced by pre-commit; `pre-commit install` once)
uv run ruff check . && uv run ruff format .

# Norm stats — required before training, writes into assets_dirs/asset_id
uv run scripts/compute_norm_stats.py --config-name <config>
uv run scripts/compute_norm_stats_low_mem.py --config-name <config>   # skips image decode; local LeRobot dirs only

# Train (JAX)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<run> [--overwrite|--resume] [--fsdp-devices N]

# Train (PyTorch) — needs the transformers patch, see below
uv run scripts/train_pytorch.py <config> --exp_name <run>
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config> --exp_name <run>

# Serve a checkpoint over websocket (port 8000)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=checkpoints/<config>/<exp>/<step>
```

Two patch scripts must be re-run after any `uv sync` that reinstalls deps:
- PyTorch models: `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/` (AdaRMS, activation precision, non-updating KV cache). With uv's hardlink mode this mutates the shared uv cache — undo with `uv cache clean transformers`.
- LeRobot dataset loader: `uv run scripts/patch_lerobot_dataset.py` (pinned LeRobot calls `torch.stack` on `datasets.Column`, which `datasets==4.7.0` no longer returns as a tensor list).

## Architecture

**One config drives everything.** `TrainConfig` in `src/openpi/training/config.py` is the single entry point — training, norm stats, and serving all resolve a config by name from the `_CONFIGS` list at the bottom of that file. Adding a robot/dataset means appending a `TrainConfig` there, not touching the scripts.

**The data pipeline is three transform groups**, applied in order (`DataConfig` in `config.py`, primitives in `src/openpi/transforms.py`):
1. `repack_transforms` — rename raw LeRobot dataset keys into the canonical `image` / `*_wrist_image` / `state` / `actions` / `prompt` dict.
2. `data_transforms` — robot-specific `*Inputs`/`*Outputs` pairs from `src/openpi/policies/` (e.g. `bimanual_joint_policy.py`), plus optional `DeltaActions`/`AbsoluteActions` masks. Runs *before* normalization.
3. `model_transforms` — `ModelTransformFactory` picks resize/tokenize/pad by `ModelType` (PI0 / PI05 / PI0_FAST).

`Outputs` transforms are the exact inverse of `Inputs` and run at inference, so a change to one almost always needs the mirrored change in the other. `DataConfigFactory.create()` assembles the three groups; the `LeRobot*DataConfig` classes are the concrete factories.

**Inputs/Outputs pairs are the per-robot contract.** Each policy module owns its state/action layout and dimensionality (e.g. `bimanual_eef_rot6d_policy.py`: 20D = 2 arms × [xyz(3) + rot6d(6) + gripper(1)], rotation-6D in PyTorch3D *first-two-rows* order — not the first-two-columns convention some Zhou-6D code uses). Each has a `*_test.py` beside it; those are the fastest check that a layout change is consistent.

**Models exist twice.** `src/openpi/models/` is JAX/flax-nnx (`pi0.py`, `pi0_fast.py`, `gemma.py`, `siglip.py`); `src/openpi/models_pytorch/` mirrors it on top of a patched HF transformers. PyTorch weights come from `examples/convert_jax_model_to_pytorch.py` and are wired in via `pytorch_weight_path` in the config. `policy_config.create_trained_policy` auto-detects which format a checkpoint dir holds. PyTorch does not support pi0-FAST, LoRA, FSDP, EMA, or mixed precision.

**Serving** wraps a `Policy` (`src/openpi/policies/policy.py`) in `websocket_policy_server.py`; robot-side clients live in the separate `packages/openpi-client` package (deliberately dependency-light so it installs on the robot).

## Fork-specific pieces

- `src/openpi/training/v3_lerobot_dataset.py` — minimal reader for LeRobot **v3** datasets (parquet + video decode via polars), auto-selected in `data_loader.create_torch_dataset` when `meta/info.json` says `codebase_version: 3.x`, because the pinned LeRobot runtime can't read them. It implements only what training needs — no HF hub, no full LeRobotDataset API.
- `src/openpi/training/keyframe_sampler.py` — per-epoch oversampling around labeled keyframes (`KeyframeSamplingConfig` on `DataConfig`). Requires a dataset exposing `episode_lengths`, and is not supported under PyTorch DDP.
- `src/openpi/policies/{bimanual_eef,bimanual_eef_rot6d,bimanual_joint,single_joint}_policy.py` — local robot layouts (10D/arm EEF, 8D/arm joint).
- `Pi0Config.image_resolution` — configurable input resolution (upstream hardcodes 224×224).
- Local configs (`pi05_hhw_*`, `pi05_yw_*`, `pi05_marvin_*`) point at absolute dataset/checkpoint paths under `/ssd/...`; they will not run elsewhere without editing those paths.
- `src/openpi/training/config.py.bak_*` and `scripts/*.bak_*` are dated snapshots checked into the fork. Grep hits there are stale — always edit `config.py` itself.

## Conventions

- Ruff, line length 120, `py311`. Excluded from lint: `third_party/`, `docker/`, `models_pytorch/transformers_replace/` (the last is a vendored transformers copy — keep it diff-minimal against upstream transformers 4.53.2).
- Checkpoints land in `checkpoints/<config_name>/<exp_name>/<step>`; norm stats in `assets/<asset_id>/norm_stats.json` and are copied into each checkpoint.
- Diverging loss usually means bad norm stats — check `q01`/`q99`/`std` for rarely-moving dimensions before suspecting the model.
