import dataclasses

import numpy as np

from openpi import transforms

JOINT_ACTION_DIM = 16


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _parse_state(state) -> np.ndarray:
    state = np.asarray(state)
    if state.shape[-1] != JOINT_ACTION_DIM:
        raise ValueError(f"Expected bimanual joint state with {JOINT_ACTION_DIM} dims, got shape {state.shape}")
    return state


@dataclasses.dataclass(frozen=True)
class BimanualJointInputs(transforms.DataTransformFn):
    """Build model inputs from a 16-dim bimanual joint observation.

    State/action order is:
    [left joints (7), left gripper, right joints (7), right gripper].
    """

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _parse_state(data["state"]),
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
            if actions.shape[-1] != JOINT_ACTION_DIM:
                raise ValueError(
                    f"Expected bimanual joint actions with {JOINT_ACTION_DIM} dims, got shape {actions.shape}"
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualJointOutputs(transforms.DataTransformFn):
    """Remove model padding and return the 16 robot action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :JOINT_ACTION_DIM])}
