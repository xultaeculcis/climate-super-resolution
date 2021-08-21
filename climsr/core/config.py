# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple

import climsr.consts as consts
from climsr.data import normalization


@dataclass
class SuperResolutionDataConfig:
    data_path: Optional[str] = "datasets"
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
class TaskConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
