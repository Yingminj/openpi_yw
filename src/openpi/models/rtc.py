"""Real-Time Chunking utilities for OpenPI flow-matching policies.

The guidance follows Physical Intelligence's Kinetix implementation, adapted
to OpenPI's reverse time convention (t=1 noise, t=0 action).  Schedule values
are numeric JAX inputs so changing a deployment knob does not require mutating
the model or baking a Python string into a compiled function.
"""

from __future__ import annotations

import jax.numpy as jnp

_SCHEDULE_CODES = {"ZEROS": 0, "ONES": 1, "LINEAR": 2, "EXP": 3}


def schedule_code(value: str) -> int:
    """Return the stable integer code for an RTC prefix schedule."""
    normalized = str(value).upper()
    try:
        return _SCHEDULE_CODES[normalized]
    except KeyError as exc:
        raise ValueError(f"prefix_attention_schedule must be one of {sorted(_SCHEDULE_CODES)}, got {value!r}") from exc


def get_prefix_weights(start, end, total: int, schedule):
    """Build RTC prefix weights for JAX scalar or Python scalar arguments."""
    start = jnp.minimum(jnp.maximum(start, 0), total)
    end = jnp.minimum(jnp.maximum(end, 0), total)
    start = jnp.minimum(start, end)
    positions = jnp.arange(total, dtype=jnp.float32)

    zeros = (positions < start).astype(jnp.float32)
    ones = (positions < end).astype(jnp.float32)
    denominator = jnp.maximum(end - start + 1, 1)
    linear = jnp.clip((start - 1 - positions) / denominator + 1, 0, 1)
    linear = jnp.where(positions >= end, 0, linear)
    exponential = linear * jnp.expm1(linear) / (jnp.e - 1)

    schedule = jnp.asarray(schedule, dtype=jnp.int32)
    return jnp.where(
        schedule == 0,
        zeros,
        jnp.where(schedule == 1, ones, jnp.where(schedule == 2, linear, exponential)),
    )
