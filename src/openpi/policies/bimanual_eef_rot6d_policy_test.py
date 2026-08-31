import numpy as np
import pytest

from openpi.policies import bimanual_eef_rot6d_policy as policy

IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _pose(position, rotation_6d, gripper):
    return np.concatenate(
        [np.asarray(position, dtype=np.float32), np.asarray(rotation_6d, dtype=np.float32), [gripper]]
    )


def _example():
    left = _pose([1, 2, 3], IDENTITY_6D, 0.25)
    right = _pose([-1, -2, -3], IDENTITY_6D, 0.75)
    state = np.concatenate([left, right])

    z_90 = policy.matrix_to_rotation_6d(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    )
    target = np.concatenate(
        [
            _pose([1.4, 2.0, 2.8], z_90, 0.5),
            _pose([-0.9, -2.2, -2.7], IDENTITY_6D, 0.6),
        ]
    )
    return state, np.stack([state, target])


def test_chunk_is_relative_to_first_state_and_starts_at_identity():
    state, absolute = _example()
    relative = policy.absolute_pose_chunk_to_relative(state, absolute)

    np.testing.assert_allclose(relative[0, 0:3], 0, atol=1e-6)
    np.testing.assert_allclose(relative[0, 3:9], IDENTITY_6D, atol=1e-6)
    np.testing.assert_allclose(relative[0, 10:13], 0, atol=1e-6)
    np.testing.assert_allclose(relative[0, 13:19], IDENTITY_6D, atol=1e-6)
    np.testing.assert_allclose(relative[1, 0:3], [0.4, 0, -0.2], atol=1e-6)
    np.testing.assert_allclose(relative[:, [9, 19]], absolute[:, [9, 19]], atol=1e-6)


def test_relative_and_absolute_pose_chunks_round_trip():
    state, absolute = _example()
    relative = policy.absolute_pose_chunk_to_relative(state, absolute)
    recovered = policy.relative_pose_chunk_to_absolute(state, relative)
    np.testing.assert_allclose(recovered, absolute, atol=1e-6)


def test_rotation_6d_matrix_round_trip():
    matrices = np.array(
        [
            np.eye(3),
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        ],
        dtype=np.float32,
    )
    recovered = policy.rotation_6d_to_matrix(policy.matrix_to_rotation_6d(matrices))
    np.testing.assert_allclose(recovered, matrices, atol=1e-6)


def test_input_transform_builds_relative_action_chunk_and_images():
    state, absolute = _example()
    transformed = policy.BimanualEEFRot6DInputs()(
        {
            "image": np.zeros((3, 12, 10), dtype=np.float32),
            "left_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
            "right_wrist_image": np.zeros((12, 10, 3), dtype=np.uint8),
            "state": state,
            "actions": absolute,
            "prompt": "move both arms",
        }
    )
    assert transformed["state"].shape == (20,)
    assert transformed["actions"].shape == (2, 20)
    assert transformed["image"]["base_0_rgb"].shape == (12, 10, 3)
    np.testing.assert_allclose(transformed["actions"][0, 3:9], IDENTITY_6D, atol=1e-6)


def test_rejects_wrong_robot_dimension():
    with pytest.raises(ValueError, match="20 dims"):
        policy.absolute_pose_chunk_to_relative(np.zeros(19), np.zeros((2, 20)))
