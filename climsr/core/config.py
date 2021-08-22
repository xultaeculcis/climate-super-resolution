# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from omegaconf import MISSING

import climsr.consts as consts
from climsr.data import normalization


@dataclass
class SuperResolutionDataConfig:
    data_path: Optional[str] = "datasets"
    world_clim_variable: Optional[str] = consts.world_clim.temp
    world_clim_multiplier: Optional[str] = consts.world_clim.resolution_multipliers[2][
        0
    ]
    generator_type: Optional[str] = consts.models.rcan
    batch_size: Optional[int] = 192
    num_workers: Optional[int] = 8
    scale_factor: Optional[int] = 4
    hr_size: Optional[int] = 128
    seed: Optional[int] = 42
    normalization_method: Optional[str] = normalization.minmax
    normalization_range: Optional[Tuple[float, float]] = (-1.0, 1.0)
    pin_memory: Optional[bool] = True
    use_elevation: Optional[bool] = True
    use_mask_as_3rd_channel: Optional[bool] = True
    use_global_min_max: Optional[bool] = True


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = (
        True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    )
    checkpoint_callback: bool = True
    callbacks: Any = None  # Optional[List[Callback]]
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Any = None  # Union[int, str, List[int], NoneType]
    auto_select_gpus: bool = False
    tpu_cores: Any = None  # Union[int, str, List[int], NoneType]
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Any = False  # Union[str, bool]
    prepare_data_per_node: bool = True
    plugins: Any = None  # Union[str, list, NoneType]
    amp_backend: str = "native"
    amp_level: str = "O2"
    distributed_backend: Optional[str] = None
    automatic_optimization: Optional[bool] = None
    move_metrics_to_cpu: bool = False
    enable_pl_optimizer: bool = False


@dataclass
class GeneratorConfig:
    in_channels: Optional[int] = 3
    out_channels: Optional[int] = 1
    scaling_factor: Optional[int] = 4


@dataclass
class TaskConfig:
    generator: GeneratorConfig = GeneratorConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


@dataclass
class OneCycleLRConf:
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    optimizer: Any = MISSING
    max_lr: Any = MISSING
    total_steps: Any = None
    epochs: Any = None
    steps_per_epoch: Any = None
    pct_start: Any = 0.3
    anneal_strategy: Any = "cos"
    cycle_momentum: Any = True
    base_momentum: Any = 0.85
    max_momentum: Any = 0.95
    div_factor: Any = 25.0
    final_div_factor: Any = 10000.0
    last_epoch: Any = -1


@dataclass
class EarlyStoppingConf:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "early_stop_on"
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "auto"
    strict: bool = True


@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = None
    save_weights_only: bool = False
    mode: str = "auto"
    period: int = 1
    prefix: str = ""
    dirpath: Any = None  # Union[str, Path, NoneType]
    filename: Optional[str] = None


@dataclass
class ProgressBarConf:
    _target_: str = "pytorch_lightning.callbacks.ProgressBar"
    refresh_rate: int = 1
    process_position: int = 0
