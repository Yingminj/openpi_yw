"""Bimanual EEF transforms using PyTorch3D's first-two-ROWS rotation 6D layout.

The six values are ``[R00, R01, R02, R10, R11, R12]``. They must not be
decoded as the first two columns used by some Zhou-6D implementations.
"""

import dataclasses

import numpy as np

from openpi import transforms

PER_ARM_DIM = 10
BIMANUAL_DIM = 2 * PER_ARM_DIM


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _require_last_dim(value, name: str, expected: int = BIMANUAL_DIM) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-1] != expected:
        raise ValueError(f"Expected {name} with {expected} dims, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode a rotation matrix using its first two rows (PyTorch3D convention)."""
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [..., 3, 3], got {matrix.shape}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Decode the continuous 6D rotation representation with Gram--Schmidt."""
    rotation_6d = np.asarray(rotation_6d)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation 6D values [..., 6], got {rotation_6d.shape}")

    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(a1_norm <= 1e-8):
        raise ValueError("Rotation 6D first vector has near-zero norm")
    b1 = a1 / a1_norm

    a2_orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    a2_norm = np.linalg.norm(a2_orthogonal, axis=-1, keepdims=True)
    if np.any(a2_norm <= 1e-8):
        raise ValueError("Rotation 6D vectors are nearly collinear")
    b2 = a2_orthogonal / a2_norm
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def absolute_pose_chunk_to_relative(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Make every absolute pose in a chunk relative to its observation state.

    Per-arm layout is ``[x, y, z, rotation_6d(6), gripper]``. Translation
    and rotation remain expressed in the parent frame of the source EEF pose:

        delta_p = target_p - reference_p
        delta_R = target_R @ reference_R.T

    Gripper values remain absolute. If the first action equals the current
    state, its translation is zero and its rotation is the identity encoding.
    """
    state = _require_last_dim(state, "state")
    actions = _require_last_dim(actions, "actions")
    if state.ndim != 1:
        raise ValueError(f"Expected unbatched state [20], got shape {state.shape}")
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk [H, 20], got shape {actions.shape}")

    relative = np.empty_like(actions)
    for arm_start in (0, PER_ARM_DIM):
        position = slice(arm_start, arm_start + 3)
        rotation = slice(arm_start + 3, arm_start + 9)
        gripper = arm_start + 9

        reference_rotation = rotation_6d_to_matrix(state[rotation])
        target_rotations = rotation_6d_to_matrix(actions[:, rotation])
        delta_rotations = target_rotations @ reference_rotation.T

        relative[:, position] = actions[:, position] - state[position]
        relative[:, rotation] = matrix_to_rotation_6d(delta_rotations)
        relative[:, gripper] = actions[:, gripper]
    return relative


def relative_pose_chunk_to_absolute(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`absolute_pose_chunk_to_relative` for validation/deployment."""
    state = _require_last_dim(state, "state")
    actions = _require_last_dim(actions, "actions")
    if state.ndim != 1 or actions.ndim != 2:
        raise ValueError(f"Expected state [20] and actions [H, 20], got {state.shape} and {actions.shape}")

    absolute = np.empty_like(actions)
    for arm_start in (0, PER_ARM_DIM):
        position = slice(arm_start, arm_start + 3)
        rotation = slice(arm_start + 3, arm_start + 9)
        gripper = arm_start + 9

        reference_rotation = rotation_6d_to_matrix(state[rotation])
        delta_rotations = rotation_6d_to_matrix(actions[:, rotation])
        target_rotations = delta_rotations @ reference_rotation

        absolute[:, position] = actions[:, position] + state[position]
        absolute[:, rotation] = matrix_to_rotation_6d(target_rotations)
        absolute[:, gripper] = actions[:, gripper]
    return absolute


@dataclasses.dataclass(frozen=True)
class BimanualEEFRot6DInputs(transforms.DataTransformFn):
    """Prepare absolute-pose LeRobot rows for chunk-relative OpenPI training."""

    def __call__(self, data: dict) -> dict:
        state = _require_last_dim(data["state"], "state")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["image"]),
                "left_wrist_0_rgb": _parse_image(data["left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = absolute_pose_chunk_to_relative(state, data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualEEFRot6DOutputs(transforms.DataTransformFn):
    """Return the 20 robot dimensions; actions remain chunk-relative."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < BIMANUAL_DIM:
            raise ValueError(f"Expected model actions [H, >=20], got shape {actions.shape}")
        return {"actions": actions[:, :BIMANUAL_DIM]}
