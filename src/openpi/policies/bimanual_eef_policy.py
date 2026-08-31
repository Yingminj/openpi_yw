import dataclasses

import numpy as np

from openpi import transforms

EEF_ACTION_DIM = 14
EEF_QUAT_STATE_DIM = 18
EEF_QUAT_ACTION_DIM = 16


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _parse_state(state) -> np.ndarray:
    state = np.asarray(state)
    if state.shape[-1] == 14:
        return state
    if state.shape[-1] == 16:
        # Dataset state is [left eef 6, left gripper, left neg_gripper,
        # right eef 6, right gripper, right neg_gripper].
        return np.concatenate([state[..., :7], state[..., 8:15]], axis=-1)
    raise ValueError(f"Expected bimanual EEF state with 14 or 16 dims, got shape {state.shape}")


def _parse_quat_state(state) -> np.ndarray:
    state = np.asarray(state)
    if state.shape[-1] != EEF_QUAT_STATE_DIM:
        raise ValueError(f"Expected bimanual EEF quat state with {EEF_QUAT_STATE_DIM} dims, got shape {state.shape}")
    return state


@dataclasses.dataclass(frozen=True)
class BimanualEEFInputs(transforms.DataTransformFn):
    """Inputs for bimanual EEF policies.

    Expected repacked inputs:
    - image: base RGB image
    - left_wrist_image: left wrist RGB image
    - right_wrist_image: right wrist RGB image
    - state: [14] or [16]
    - actions: [action_horizon, 14], only during training
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        left_wrist_image = _parse_image(data["left_wrist_image"])
        right_wrist_image = _parse_image(data["right_wrist_image"])

        inputs = {
            "state": _parse_state(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualEEFOutputs(transforms.DataTransformFn):
    """Outputs for bimanual EEF policies."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :EEF_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class BimanualEEFQuatInputs(transforms.DataTransformFn):
    """Inputs for bimanual EEF policies with quaternion state/actions.

    Expected repacked inputs:
    - image: base RGB image
    - left_wrist_image: left wrist RGB image
    - right_wrist_image: right wrist RGB image
    - state: [18], left/right [xyz, quat_xyzw, gripper, -gripper]
    - actions: [action_horizon, 16], left/right [dxyz, dquat_xyzw, gripper], only during training
    """

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _parse_quat_state(data["state"]),
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
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != EEF_QUAT_ACTION_DIM:
                raise ValueError(
                    f"Expected bimanual EEF quat actions with {EEF_QUAT_ACTION_DIM} dims, got shape {actions.shape}"
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualEEFQuatOutputs(transforms.DataTransformFn):
    """Remove model padding and return the 16 quaternion EEF action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :EEF_QUAT_ACTION_DIM])}
