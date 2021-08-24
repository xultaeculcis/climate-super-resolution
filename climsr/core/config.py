# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple

from omegaconf import MISSING

import climsr.consts as consts
from climsr.data import normalization


@dataclass
class PreProcessingConfig:
    data_dir_cruts: Optional[str] = MISSING
    data_dir_world_clim: Optional[str] = MISSING
    out_dir_cruts: Optional[str] = MISSING
    out_dir_world_clim: Optional[str] = MISSING

    world_clim_elevation_fp: Optional[str] = MISSING
    dataframe_output_path: Optional[str] = MISSING
    elevation_file: Optional[str] = MISSING
    land_mask_file: Optional[str] = MISSING

    run_cruts_to_cog: Optional[bool] = True
    run_temp_rasters_generation: Optional[bool] = True
    run_statistics_computation: Optional[bool] = True
    run_world_clim_resize: Optional[bool] = True
    run_world_clim_tiling: Optional[bool] = True
    run_world_clim_elevation_resize: Optional[bool] = True
    run_train_val_test_split: Optional[bool] = True
    run_extent_extraction: Optional[bool] = True
    run_z_score_stats_computation: Optional[bool] = True
    run_min_max_stats_computation: Optional[bool] = True

    patch_size: Optional[Tuple[int, int]] = (128, 128)
    patch_stride: Optional[int] = 64
    normalize_patches: Optional[bool] = False
    n_workers: Optional[int] = 8
    threads_per_worker: Optional[int] = 1
    res_mult_inx: Optional[int] = 2

    train_years: Optional[Tuple[int, int]] = (1961, 2004)
    val_years: Optional[Tuple[int, int]] = (2005, 2017)
    test_years: Optional[Tuple[int, int]] = (2018, 2019)


@dataclass
class SuperResolutionDataConfig:
    data_path: Optional[str] = MISSING
    world_clim_variable: Optional[str] = consts.world_clim.temp
    world_clim_multiplier: Optional[str] = consts.world_clim.resolution_multipliers[2][0]
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
    ...


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
    optimizers: List[OptimizerConfig] = None
    schedulers: List[SchedulerConfig] = None
    discriminator: DiscriminatorConfig = None


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
    use_mask_as_3rd_channel: Optional[bool] = True
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
    peaks_csv: Optional[str] = MISSING
