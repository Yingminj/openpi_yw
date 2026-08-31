#!/usr/bin/env python3
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as config_module


CONFIG_NAME = "pi05_hhw_tj_zadai_200_pytorch_uniform"
MODEL_PATH = Path("/ssd/hhw/openpi-hzh/checkpoint/pi05_base_pytorch_bfloat16/model.safetensors")
CONVERTER_PATH = Path("/ssd/hhw/openpi-hzh/examples/convert_jax_model_to_pytorch.py")
JAX_CHECKPOINT = "/ssd/hhw/openpi-hzh/checkpoint/pi05_base"
EXPECTED_TIED_ALIASES = {
    "paligemma_with_expert.paligemma.lm_head.weight",
    "paligemma_with_expert.gemma_expert.lm_head.weight",
}


def load_converter_module():
    spec = importlib.util.spec_from_file_location("openpi_conversion_audit", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_saved_model(model_config) -> tuple[int, int]:
    model = PI0Pytorch(model_config)
    missing, unexpected = safetensors.torch.load_model(model, MODEL_PATH, strict=True, device="cpu")
    assert not missing, f"Strict load missing keys: {sorted(missing)}"
    assert not unexpected, f"Strict load unexpected keys: {sorted(unexpected)}"

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total_parameters == trainable_parameters

    for name, parameter in model.named_parameters():
        flattened = parameter.detach().reshape(-1)
        if flattened.numel() == 0:
            continue
        sample_count = min(257, flattened.numel())
        indices = torch.arange(sample_count, dtype=torch.int64)
        if sample_count > 1:
            indices = indices * (flattened.numel() - 1) // (sample_count - 1)
        assert torch.isfinite(flattened[indices].float()).all(), f"Non-finite values in {name}"

    del model
    gc.collect()
    return total_parameters, trainable_parameters


def audit_conversion_mapping(model_config) -> tuple[list[str], list[str]]:
    converter = load_converter_module()
    original_load_state_dict = PI0Pytorch.load_state_dict
    audit_result: dict[str, list[str]] = {}

    def audited_load_state_dict(self, state_dict, strict=True, assign=False):
        result = original_load_state_dict(self, state_dict, strict=False, assign=assign)
        audit_result["missing"] = list(result.missing_keys)
        audit_result["unexpected"] = list(result.unexpected_keys)
        return result

    PI0Pytorch.load_state_dict = audited_load_state_dict
    converter.safetensors.torch.save_model = lambda *args, **kwargs: None
    try:
        converter.convert_pi0_checkpoint(
            JAX_CHECKPOINT,
            "bfloat16",
            "/tmp/openpi_pi05_conversion_audit",
            model_config,
        )
    finally:
        PI0Pytorch.load_state_dict = original_load_state_dict

    missing = audit_result.get("missing", [])
    unexpected = audit_result.get("unexpected", [])
    assert set(missing) == EXPECTED_TIED_ALIASES, f"Unexpected converter missing keys: {missing}"
    assert not unexpected, f"Converter mapping unexpected keys: {unexpected}"
    return missing, unexpected


def main() -> None:
    config = config_module.get_config(CONFIG_NAME)
    model_config = config.model
    object.__setattr__(model_config, "dtype", config.pytorch_training_precision)

    total_parameters, trainable_parameters = verify_saved_model(model_config)
    missing, unexpected = audit_conversion_mapping(model_config)

    print(f"model_path={MODEL_PATH}")
    print(f"total_parameters={total_parameters}")
    print(f"trainable_parameters={trainable_parameters}")
    print(f"converter_missing_keys={missing}")
    print(f"converter_unexpected_keys={unexpected}")
    print("weight_verification=PASS")


if __name__ == "__main__":
    main()
