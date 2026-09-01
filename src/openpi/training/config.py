"""Training configs for OpenPI.

Customized for pi0.5 bimanual joint-space fine-tuning (hhw / yw datasets).
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.bimanual_joint_policy as bimanual_joint_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Assets config for loading asset files (e.g., norm stats)."""

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Data config for the data pipeline."""

    repo_id: str | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    norm_stats: dict[str, _normalize.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    rlds_data_dir: str | None = None
    action_space: Any | None = None
    datasets: Sequence[Any] = dataclasses.field(default_factory=tuple)


class GroupFactory(Protocol):
    """Interface for creating a group of transforms."""

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group: ...


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory:
    """Creates the default model transforms for a given model type."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Creates the model transforms for the given model."""
        if model_config.model_type == _model.ModelType.PI0:
            return _transforms.Group(
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages((224, 224)),
                _transforms.TokenizePrompt(model_config.model_type),
                _transforms.PadStatesAndActions(model_config.model_type),
            )
        if model_config.model_type == _model.ModelType.PI05:
            return _transforms.Group(
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages((224, 224)),
                _transforms.TokenizePrompt(model_config.model_type),
                _transforms.PadStatesAndActions(
                    model_config.model_type,
                    discrete_state_input=model_config.discrete_state_input,
                ),
            )
        if model_config.model_type == _model.ModelType.PI0_FAST:
            return _transforms.Group(
                _transforms.TokenizeFASTInputs(
                    model_config,
                    is_eval=False,
                    default_prompt=self.default_prompt,
                ),
                _transforms.ExtractFASTActions(model_config),
            )
        raise ValueError(f"Unsupported model type: {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """Interface for creating data configs."""

    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs, model_config) -> DataConfig: ...

    def create_base_config(
        self,
        assets_dirs,
        model_config,
        repo_id: str,
        action_sequence_keys: Sequence[str],
    ) -> DataConfig:
        """Loads norm stats and creates a base data config."""
        norm_stats = self._load_norm_stats(
            assets_dirs, self.assets.asset_id, model_config.model_type
        )
        if self.base_config is not None:
            base_config = self.base_config
            if base_config.repo_id is not None:
                raise ValueError("Cannot specify repo_id in both the factory and base config.")
            return dataclasses.replace(
                base_config,
                repo_id=repo_id,
                norm_stats=norm_stats,
                action_sequence_keys=action_sequence_keys,
            )
        return DataConfig(
            repo_id=repo_id,
            assets=self.assets,
            norm_stats=norm_stats,
            action_sequence_keys=action_sequence_keys,
        )

    def _load_norm_stats(
        self,
        assets_dirs,
        asset_id: str | None,
        model_type: ModelType,
    ) -> dict[str, _normalize.NormStats]:
        """Loads norm stats from the assets directory."""
        if asset_id is not None:
            assets_dir = pathlib.Path(assets_dirs) / asset_id
        else:
            assets_dir = pathlib.Path(assets_dirs)
        if not (assets_dir / "norm_stats.json").exists():
            logging.info("No norm stats found at %s.", assets_dir)
            return {}
        logging.info("Loading norm stats from %s.", assets_dir)
        return _normalize.load_norm_stats(assets_dir, model_type)


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """A fake data config for testing."""

    repo_id: str = "fake"

    def create(self, assets_dirs, model_config) -> DataConfig:
        return DataConfig(
            repo_id=self.repo_id,
            assets=self.assets,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """A simple data config that only modifies the data/model transforms."""

    data_transforms: GroupFactory = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: GroupFactory = dataclasses.field(default_factory=ModelTransformFactory)

    def create(self, assets_dirs, model_config) -> DataConfig:
        base_config = self.create_base_config(
            assets_dirs, model_config, self.repo_id, self.action_sequence_keys
        )
        return dataclasses.replace(
            base_config,
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotBimanualJointDataConfig(DataConfigFactory):
    """LeRobot data config for bimanual joint-space datasets (pi0.5).

    Maps the dataset-specific camera/state/action keys into the canonical
    observation/action space consumed by the bimanual joint policy.
    """

    image_key: str = "observation.images.top"
    left_wrist_image_key: str = "observation.images.wrist_L"
    right_wrist_image_key: str = "observation.images.wrist_R"
    state_key: str = "observation.state"
    actions_key: str = "action"
    action_sequence_keys: Sequence[str] = ("action",)
    action_dim: int = bimanual_joint_policy.JOINT_ACTION_DIM
    default_prompt: str | None = None

    def create(self, assets_dirs, model_config) -> DataConfig:
        base_config = self.create_base_config(
            assets_dirs, model_config, self.repo_id, self.action_sequence_keys
        )
        repack_transforms = _transforms.Group(
            _transforms.SelectTransform(
                {
                    self.image_key: "observation.images.top",
                    self.left_wrist_image_key: "observation.images.wrist_L",
                    self.right_wrist_image_key: "observation.images.wrist_R",
                    self.state_key: "observation.state",
                    self.actions_key: "action",
                }
            )
        )
        model_transforms = _transforms.Group(
            _transforms.InjectDefaultPrompt(self.default_prompt),
            _transforms.ResizeImages((224, 224)),
            _transforms.TokenizePrompt(model_config.model_type),
            _transforms.PadStatesAndActions(
                model_config.model_type,
                discrete_state_input=model_config.discrete_state_input,
                action_horizon=model_config.action_horizon,
                action_dim=self.action_dim,
            ),
        )
        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Top-level config that defines a training run."""

    name: str
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    pytorch_weight_path: str | None = None
    pytorch_training_precision: str = "bfloat16"
    lr_schedule: _optimizer.LRSchedule = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )
    optimizer: _optimizer.Optimizer = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: Filter = nnx.Nothing
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int = 5_000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> epath.Path:
        return epath.Path(self.assets_base_dir) / self.name

    @property
    def checkpoint_dir(self) -> epath.Path:
        return epath.Path(self.checkpoint_base_dir) / self.name / self.exp_name

    @property
    def trainable_filter(self) -> Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.overwrite and self.resume:
            raise ValueError("Cannot set both overwrite and resume.")

    def create_data_config(self, model_config) -> DataConfig:
        """Creates a data config by calling the data factory."""
        return self.data.create(self.assets_dirs, model_config)


# --- pi0.5 bimanual joint-space fine-tuning defaults -------------------------
# Cloud checkpoint (override via the PI05_BASE_CHECKPOINT env var if desired).
_PI05_BASE_CHECKPOINT = os.environ.get(
    "PI05_BASE_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_base/params"
)
# Local checkpoint used by default for offline fine-tuning.
_LOCAL_PI05_BASE_CHECKPOINT = "/ssd/hhw/openpi-hzh/checkpoint/pi05_base/params"


def _pi05_joint_config(
    name: str,
    repo_id: str,
    exp_name: str,
    default_prompt: str | None = None,
    *,
    model: _model.BaseModelConfig | None = None,
    action_dim: int = bimanual_joint_policy.JOINT_ACTION_DIM,
    batch_size: int = 16,
    num_workers: int = 24,
    num_train_steps: int = 150_000,
    peak_lr: float = 2.5e-5,
    decay_lr: float = 1.0e-6,
    prompt_from_task: bool = False,
    image_key: str = "observation.images.top",
    base_checkpoint: str = _LOCAL_PI05_BASE_CHECKPOINT,
    save_interval: int = 10_000,
    keep_period: int = 50_000,
    wandb_enabled: bool = False,
    freeze_filter: Filter | None = None,
    ema_decay: float | None = 0.99,
    **overrides: Any,
) -> TrainConfig:
    """Builds a pi0.5 bimanual joint-space fine-tuning config.

    Args:
        name: Unique config name (also used for the checkpoint directory).
        repo_id: LeRobot dataset id.
        exp_name: Experiment name (checkpoint sub-directory).
        default_prompt: Optional default prompt injected into the model.
        model: Optional model config (e.g. LoRA variants). Defaults to pi0.5 full fine-tune.
        action_dim: Action dimension (defaults to the bimanual joint dim).
        batch_size / num_workers / num_train_steps: Training hyperparameters.
        peak_lr / decay_lr: Cosine schedule learning-rate bounds.
        prompt_from_task: Whether to use the task description as the prompt.
        image_key: Dataset key of the main (top) camera image.
        base_checkpoint: Checkpoint path used as the weight init.
        save_interval / keep_period: Checkpointing schedule.
        wandb_enabled: Whether to log to Weights & Biases.
        freeze_filter: Optional filter of parameters to freeze (e.g. for LoRA).
        ema_decay: EMA decay; set to None when LoRA / frozen-part training.
        **overrides: Any remaining TrainConfig fields (e.g. pytorch_weight_path).
    """
    return TrainConfig(
        name=name,
        exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True) if model is None else model,
        data=LeRobotBimanualJointDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(prompt_from_task=prompt_from_task),
            image_key=image_key,
            left_wrist_image_key="observation.images.wrist_L",
            right_wrist_image_key="observation.images.wrist_R",
            state_key="observation.state",
            actions_key="action",
            action_sequence_keys=("action",),
            action_dim=action_dim,
            default_prompt=default_prompt,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
            decay_lr=decay_lr,
        ),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        save_interval=save_interval,
        keep_period=keep_period,
        wandb_enabled=wandb_enabled,
        **overrides,
    )


_CONFIGS: list[TrainConfig] = [
    # --- hhw / tj: 方块 (uniform) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_fangkuai_uniform",
        repo_id="hhw/tj_fangkuai_uniform",
        exp_name="hhw_tj_fangkuai_uniform",
        image_key="observation.images.right_eye",
        batch_size=32,
    ),
    # LoRA: freeze base weights, only the LoRA adapters are trainable.
    _pi05_joint_config(
        name="pi05_hhw_tj_fangkuai_lora",
        repo_id="hhw/tj_fangkuai_uniform",
        exp_name="hhw_tj_fangkuai_lora",
        image_key="observation.images.right_eye",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        peak_lr=2.0e-4,
        decay_lr=1.0e-5,
        batch_size=32,
    ),
    # --- hhw / tj: 衣物 1 (uniform) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_clothes1_uniform",
        repo_id="hhw/tj_clothes1_uniform",
        exp_name="hhw_tj_clothes1_uniform",
        batch_size=16,
    ),
    # --- hhw / tj: 摊开 200 (uniform) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_tankai_200_uniform",
        repo_id="hhw/tj_tankai_200_uniform",
        exp_name="hhw_tj_tankai_200_uniform",
        batch_size=12,
    ),
    # --- hhw / tj: 衣物 400 (uniform, task-prompted) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_clothes_400_uniform",
        repo_id="hhw/tj_clothes_400_uniform",
        exp_name="hhw_tj_clothes_400_uniform",
        prompt_from_task=True,
        batch_size=12,
    ),
    # --- yw: tidy up (EEF-space actions) ---
    _pi05_joint_config(
        name="pi05_yw_tidy_up_eef",
        repo_id="yw/tidy_up",
        exp_name="yw_tidy_up_eef",
        action_dim=14,
        base_checkpoint=_PI05_BASE_CHECKPOINT,
        wandb_enabled=True,
    ),
    # --- yw: tidy up (joint-space actions) ---
    _pi05_joint_config(
        name="pi05_yw_tidy_up",
        repo_id="yw/tidy_up",
        exp_name="yw_tidy_up",
        base_checkpoint=_PI05_BASE_CHECKPOINT,
        wandb_enabled=True,
    ),
    # --- hhw / tj: 扎带 200 (uniform) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_zadai_200_uniform",
        repo_id="hhw/tj_zadai_200_uniform",
        exp_name="hhw_tj_zadai_200_uniform",
        batch_size=12,
    ),
    # --- hhw / tj: 扎带 200 (uniform, PyTorch checkpoint init) ---
    _pi05_joint_config(
        name="pi05_hhw_tj_zadai_200_pytorch_uniform",
        repo_id="hhw/tj_zadai_200_uniform",
        exp_name="hhw_tj_zadai_200_pytorch_uniform",
        batch_size=48,
        num_workers=8,
        num_train_steps=100_000,
        pytorch_weight_path="/ssd/hhw/openpi-hzh/checkpoint/pi05_base_pytorch_bfloat16",
        pytorch_training_precision="bfloat16",
    ),
]

# Expand with auxiliary configs from the roboarena / polaris recipes.
_CONFIGS += [
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    """CLI to override any config field from the command line."""
    return tyro.extras.overridable_config_cli(_CONFIGS_DICT)


def get_config(config_name: str) -> TrainConfig:
    """Returns the config for the given name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
