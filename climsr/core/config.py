# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import MISSING

import climsr.consts as consts
from climsr.data import normalization


def _default_resolution_list():
    return [consts.world_clim.resolution_5m, consts.world_clim.resolution_2_5m]


@dataclass
class DataDownloadConfig:
    download_path: Optional[str] = "./datasets"
    parallel_downloads: Optional[int] = 8


@dataclass
class PreProcessingConfig:
    data_dir_cruts: Optional[str] = MISSING
    data_dir_world_clim: Optional[str] = MISSING

    output_path: Optional[str] = MISSING

    world_clim_elevation_fp: Optional[str] = MISSING
    elevation_file: Optional[str] = MISSING
    land_mask_file: Optional[str] = MISSING

    run_cruts_to_tiff: Optional[bool] = False
    run_tavg_rasters_generation: Optional[bool] = False
    run_statistics_computation: Optional[bool] = False
    run_world_clim_resize: Optional[bool] = False
    run_world_clim_tiling: Optional[bool] = False
    run_train_val_test_split: Optional[bool] = True
    run_extent_extraction: Optional[bool] = False
    run_z_score_stats_computation: Optional[bool] = False
    run_min_max_stats_computation: Optional[bool] = False

    patch_size: Optional[Tuple[int, int]] = (128, 128)
    patch_stride: Optional[int] = 64
    n_workers: Optional[int] = 8
    threads_per_worker: Optional[int] = 1

    train_years: Optional[Tuple[int, int]] = (1961, 1999)
    val_years: Optional[Tuple[int, int]] = (2000, 2005)
    test_years: Optional[Tuple[int, int]] = (2006, 2020)


@dataclass
class TransformsCfg:
    v_flip: Optional[bool] = True
    h_flip: Optional[bool] = True
    random_90_rotation: Optional[bool] = True


@dataclass
class SuperResolutionDataConfig:
    data_path: Optional[str] = MISSING
    world_clim_variable: Optional[str] = consts.world_clim.temp
    generator_type: Optional[str] = consts.models.rcan
    resolutions: Optional[List[str]] = field(default_factory=_default_resolution_list)
    batch_size: Optional[int] = 192
    validation_batch_size: Optional[int] = 192
    num_workers: Optional[int] = 8
    scale_factor: Optional[int] = 4
    hr_size: Optional[int] = 128
    seed: Optional[int] = 42
    normalization_method: Optional[str] = normalization.minmax
    normalization_range: Optional[Tuple[float, float]] = (-1.0, 1.0)
    pin_memory: Optional[bool] = False
    use_elevation: Optional[bool] = True
    use_mask: Optional[bool] = True
    use_global_min_max: Optional[bool] = True
    use_extra_data: Optional[bool] = False
    transforms: Optional[TransformsCfg] = TransformsCfg()


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class SchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
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
class DiscriminatorConfig:
    in_channels: Optional[int] = 1


@dataclass
class TaskConfig:
    generator: GeneratorConfig = None
    optimizers: Dict[str, OptimizerConfig] = None
    schedulers: Dict[str, SchedulerConfig] = None
    discriminator: DiscriminatorConfig = None
    initial_hp_metric_val: float = 5e-3


@dataclass
class InferenceConfig:
    # Paths to CRU-TS NetCDF files
    ds_path_tmn: Optional[str] = MISSING
    ds_path_tmp: Optional[str] = MISSING
    ds_path_tmx: Optional[str] = MISSING
    ds_path_pre: Optional[str] = MISSING

    # Paths to tiff data dirs
    data_dir: Optional[str] = MISSING
    original_full_res_cruts_data_path: Optional[str] = MISSING
    inference_out_path: Optional[str] = MISSING

    # Europe extent dirs
    tiff_dir: Optional[str] = MISSING
    extent_out_path_lr: Optional[str] = MISSING
    extent_out_path_sr: Optional[str] = MISSING
    extent_out_path_sr_nc: Optional[str] = MISSING

    # Pretrained models
    pretrained_model_tmn: Optional[str] = MISSING
    pretrained_model_tmp: Optional[str] = MISSING
    pretrained_model_tmx: Optional[str] = MISSING
    pretrained_model_pre: Optional[str] = MISSING

    # Misc
    use_netcdf_datasets: Optional[bool] = False  # This defines the dataset type
    temp_only: Optional[bool] = True  # Use model trained on combined temp data? Or models trained on individual files.
    generator_type: Optional[str] = MISSING  # To be used as prefix for generated SR files

    # Dataset stuff
    elevation_file: Optional[str] = MISSING
    land_mask_file: Optional[str] = MISSING
    use_elevation: Optional[bool] = True
    use_mask: Optional[bool] = True
    use_global_min_max: Optional[bool] = True
    cruts_variable: Optional[str] = "tmp"  # Select null if you want to run in a loop for all variables
    scaling_factor: Optional[int] = 4
    normalize: Optional[bool] = True
    normalization_range: Optional[Tuple[int, int]] = (-1.0, 1.0)
    min_max_lookup: Optional[str] = MISSING

    # Run steps
    run_inference: Optional[bool] = True
    extract_polygon_extent: Optional[bool] = True
    to_netcdf: Optional[bool] = True


@dataclass
class ResultInspectionConfig:
    ds_temp_nn_path: Optional[str] = MISSING
    ds_temp_cru_path: Optional[str] = MISSING
    peaks_feather: Optional[str] = MISSING


def infer_generator_config(generator_cfg: GeneratorConfig, data_config: SuperResolutionDataConfig) -> GeneratorConfig:
    in_channels = 3
    if not data_config.use_elevation:
        in_channels = in_channels - 1
    if not data_config.use_mask:
        in_channels = in_channels - 1

    generator_cfg.in_channels = in_channels

    return generator_cfg
