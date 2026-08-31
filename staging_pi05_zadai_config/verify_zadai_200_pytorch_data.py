#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as config_module
from openpi.training import data_loader


CONFIG_NAME = "pi05_hhw_tj_zadai_200_pytorch_uniform"
EXPECTED_PROMPT = "Insert the tail of the yellow cable tie into its head to fasten it."


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    config = config_module.get_config(CONFIG_NAME)
    resolved_data = config.data.create(config.assets_dirs, config.model)

    assert config.batch_size == 48
    assert config.num_train_steps == 100_000
    assert resolved_data.keyframe_sampling is None
    assert config.data.default_prompt == EXPECTED_PROMPT

    loader = data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        framework="pytorch",
    )
    observation, actions = next(iter(loader))

    tokens = to_numpy(observation.tokenized_prompt)[0]
    token_mask = to_numpy(observation.tokenized_prompt_mask)[0].astype(bool)
    tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    decoded = tokenizer._tokenizer.decode(tokens[token_mask].tolist())

    assert EXPECTED_PROMPT in decoded, decoded
    assert to_numpy(actions).shape == (
        config.batch_size,
        config.model.action_horizon,
        config.model.action_dim,
    )

    print(f"config={config.name}")
    print(f"repo_id={resolved_data.repo_id}")
    print(f"keyframe_sampling={resolved_data.keyframe_sampling}")
    print(f"batch_shape={to_numpy(actions).shape}")
    print(f"decoded_prompt={decoded}")


if __name__ == "__main__":
    main()
