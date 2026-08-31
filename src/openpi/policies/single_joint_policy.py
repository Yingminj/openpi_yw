import dataclasses

import numpy as np

from openpi import transforms

JOINT_ACTION_DIM = 8


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
        raise ValueError(f"Expected single-arm joint state with {JOINT_ACTION_DIM} dims, got shape {state.shape}")
    return state


@dataclasses.dataclass(frozen=True)
class SingleJointInputs(transforms.DataTransformFn):
    """Build model inputs from a single-arm joint observation.

    State/action order is seven right-arm joints followed by the right gripper.
    The model's unused left-wrist image slot is filled with a masked black image.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        right_wrist_image = _parse_image(data["right_wrist_image"])
        inputs = {
            "state": _parse_state(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != JOINT_ACTION_DIM:
                raise ValueError(
                    f"Expected single-arm joint actions with {JOINT_ACTION_DIM} dims, got shape {actions.shape}"
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SingleJointOutputs(transforms.DataTransformFn):
    """Remove model padding and return the eight robot action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :JOINT_ACTION_DIM])}
