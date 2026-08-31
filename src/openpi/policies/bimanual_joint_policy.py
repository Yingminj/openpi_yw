import dataclasses

import numpy as np

from openpi import transforms

# 16 = 14 joints + 2 grippers. EEF datasets use 14 = 12 pose dims + 2 grippers.
JOINT_ACTION_DIM = 16


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _parse_state(state, action_dim: int) -> np.ndarray:
    state = np.asarray(state)
    if state.shape[-1] != action_dim:
        raise ValueError(f"Expected bimanual state with {action_dim} dims, got shape {state.shape}")
    return state


@dataclasses.dataclass(frozen=True)
class BimanualJointInputs(transforms.DataTransformFn):
    """Build model inputs from a bimanual joint or end-effector observation.

    State/action order follows the dataset's `meta/info.json` names:
    joint (action_dim=16): [left joints (7), right joints (7), left gripper, right gripper];
    EEF (action_dim=14): [left xyz+rpy (6), right xyz+rpy (6), left gripper, right gripper].
    """

    action_dim: int = JOINT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _parse_state(data["state"], self.action_dim),
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
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected bimanual actions with {self.action_dim} dims, got shape {actions.shape}"
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualJointOutputs(transforms.DataTransformFn):
    """Remove model padding and return the robot action dimensions."""

    action_dim: int = JOINT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
