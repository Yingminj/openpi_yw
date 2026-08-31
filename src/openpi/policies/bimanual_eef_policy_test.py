import numpy as np
import pytest

from openpi.policies import bimanual_eef_policy


def _quat_example(*, state_dim: int = 18, action_dim: int = 16) -> dict:
    return {
        "image": np.zeros((3, 12, 10), dtype=np.float32),
        "left_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
        "right_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
        "state": np.arange(state_dim, dtype=np.float32),
        "actions": np.zeros((15, action_dim), dtype=np.float32),
        "prompt": "move both arms",
    }


def test_bimanual_eef_quat_inputs_preserve_robot_dimensions():
    result = bimanual_eef_policy.BimanualEEFQuatInputs()(_quat_example())

    assert result["state"].shape == (18,)
    assert result["actions"].shape == (15, 16)
    assert result["image"]["base_0_rgb"].shape == (12, 10, 3)
    assert all(result["image_mask"].values())


def test_bimanual_eef_quat_outputs_remove_model_padding():
    actions = np.arange(15 * 32, dtype=np.float32).reshape(15, 32)
    result = bimanual_eef_policy.BimanualEEFQuatOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :16])


@pytest.mark.parametrize("field", ["state", "actions"])
def test_bimanual_eef_quat_inputs_reject_wrong_dimensions(field: str):
    example = _quat_example()
    example[field] = np.zeros((17,), dtype=np.float32) if field == "state" else np.zeros((15, 15), dtype=np.float32)

    with pytest.raises(ValueError, match="quat"):
        bimanual_eef_policy.BimanualEEFQuatInputs()(example)
