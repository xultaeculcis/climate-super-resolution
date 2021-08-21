# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 0


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
