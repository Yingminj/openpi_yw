import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import rtc


@pytest.mark.parametrize("schedule", ["ZEROS", "ONES", "LINEAR", "EXP"])
def test_prefix_weights_are_bounded_and_end_at_horizon(schedule):
    weights = np.asarray(rtc.get_prefix_weights(2, 6, 10, rtc.schedule_code(schedule)))

    assert weights.shape == (10,)
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    assert np.all(weights[:2] == 1.0)
    assert np.all(weights[6:] == 0.0)
    assert np.all(np.diff(weights) <= 0.0)


def test_prefix_weights_accept_jax_scalar_arguments():
    weights = rtc.get_prefix_weights(jnp.asarray(3), jnp.asarray(8), 10, jnp.asarray(rtc.schedule_code("EXP")))

    assert weights.shape == (10,)
    assert np.asarray(weights)[0] == pytest.approx(1.0)


def test_schedule_code_rejects_unknown_value():
    with pytest.raises(ValueError, match="prefix_attention_schedule"):
        rtc.schedule_code("unknown")
