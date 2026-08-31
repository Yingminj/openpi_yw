import numpy as np
import pytest

from openpi.policies import single_joint_policy


def _example(*, state_dim: int = 8, action_dim: int = 8) -> dict:
    return {
        "image": np.zeros((3, 12, 10), dtype=np.float32),
        "right_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
        "state": np.arange(state_dim, dtype=np.float32),
        "actions": np.zeros((15, action_dim), dtype=np.float32),
        "prompt": "move the right arm",
    }


def test_single_joint_inputs_preserve_robot_dimensions_and_camera_layout():
    result = single_joint_policy.SingleJointInputs()(_example())

    assert result["state"].shape == (8,)
    assert result["actions"].shape == (15, 8)
    assert result["image"]["base_0_rgb"].shape == (12, 10, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.True_,
    }


def test_single_joint_outputs_remove_model_padding():
    actions = np.arange(15 * 32, dtype=np.float32).reshape(15, 32)
    result = single_joint_policy.SingleJointOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :8])


@pytest.mark.parametrize("field", ["state", "actions"])
def test_single_joint_inputs_reject_wrong_dimensions(field: str):
    example = _example()
    example[field] = np.zeros((7,), dtype=np.float32) if field == "state" else np.zeros((15, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="8 dims"):
        single_joint_policy.SingleJointInputs()(example)
