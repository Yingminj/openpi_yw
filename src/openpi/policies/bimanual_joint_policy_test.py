import numpy as np
import pytest

from openpi.policies import bimanual_joint_policy


def _example(*, state_dim: int = 16, action_dim: int = 16) -> dict:
    return {
        "image": np.zeros((3, 12, 10), dtype=np.float32),
        "left_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
        "right_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
        "state": np.arange(state_dim, dtype=np.float32),
        "actions": np.zeros((15, action_dim), dtype=np.float32),
        "prompt": "move both arms",
    }


def test_bimanual_joint_inputs_preserve_all_robot_dimensions():
    result = bimanual_joint_policy.BimanualJointInputs()(_example())

    assert result["state"].shape == (16,)
    assert result["actions"].shape == (15, 16)
    assert result["image"]["base_0_rgb"].shape == (12, 10, 3)
    assert all(result["image_mask"].values())


def test_bimanual_joint_outputs_remove_model_padding():
    actions = np.arange(15 * 32, dtype=np.float32).reshape(15, 32)
    result = bimanual_joint_policy.BimanualJointOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :16])


def test_bimanual_eef_roundtrip_at_14_dims():
    inputs = bimanual_joint_policy.BimanualJointInputs(action_dim=14)(_example(state_dim=14, action_dim=14))
    assert inputs["state"].shape == (14,)
    assert inputs["actions"].shape == (15, 14)

    actions = np.arange(15 * 32, dtype=np.float32).reshape(15, 32)
    result = bimanual_joint_policy.BimanualJointOutputs(action_dim=14)({"actions": actions})
    np.testing.assert_array_equal(result["actions"], actions[:, :14])


def test_bimanual_eef_inputs_reject_joint_dimensions():
    with pytest.raises(ValueError, match="14 dims"):
        bimanual_joint_policy.BimanualJointInputs(action_dim=14)(_example())


@pytest.mark.parametrize("field", ["state", "actions"])
def test_bimanual_joint_inputs_reject_wrong_dimensions(field: str):
    example = _example()
    example[field] = np.zeros((15,), dtype=np.float32) if field == "state" else np.zeros((15, 15), dtype=np.float32)

    with pytest.raises(ValueError, match="16 dims"):
        bimanual_joint_policy.BimanualJointInputs()(example)
