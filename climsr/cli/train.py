# -*- coding: utf-8 -*-
import gc
import logging
from typing import Any, List, Optional, Union

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from torch import Tensor

from climsr.core import utils
from climsr.core.config import (
    SuperResolutionDataConfig,
    SuperResolutionDataModuleConfig,
    TaskConfig,
    TrainerConfig,
    infer_generator_config,
)
from climsr.core.instantiator import HydraInstantiator, Instantiator
from climsr.core.task import TaskSuperResolutionModule
from climsr.data.super_resolution_data_module import SuperResolutionDataModule

default_sr_dm_config = SuperResolutionDataConfig()
default_task_config = TaskConfig()
default_trainer_config = TrainerConfig()


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    run_fit: bool = True,
    run_test_after_fit: bool = True,
    lr_find_only: bool = False,
    datamodule_cfg: Optional[SuperResolutionDataModuleConfig] = default_sr_dm_config,
    task_cfg: Optional[TaskConfig] = default_task_config,
    trainer_cfg: Optional[TrainerConfig] = default_trainer_config,
    logger_cfgs: Optional[Any] = None,
    callback_cfgs: Optional[Any] = None,
    profiler_cfg: Optional[Any] = None,
    optimized_metric: Optional[str] = None,
) -> Optional[Union[Tensor, float]]:
    if ignore_warnings:
        utils.set_ignore_warnings()

    # Limit RTX 3090 power draw if possible to stabilize PSU usage
    utils.set_gpu_power_limit_if_needed()

    # Init data module
    data_module: SuperResolutionDataModule = instantiator.data_module(datamodule_cfg)
    if data_module is None:
        raise ValueError("No datamodule found. Hydra hint: did you set `datamodule=...`?")
    if not isinstance(data_module, pl.LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance." " Hydra hint: is `datamodule._target_` defined?`"
        )

    model_data_kwargs = data_module.model_data_kwargs
    model_data_kwargs.update(
        {
            "accumulate_grad_batches": trainer_cfg.accumulate_grad_batches,
        }
    )

    # Init lightning module
    model: TaskSuperResolutionModule = instantiator.model(task_cfg, model_data_kwargs=data_module.model_data_kwargs)

    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    for _, lg_conf in logger_cfgs.items():
        if "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: List[Callback] = []
    for _, cb_conf in callback_cfgs.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init profiler
    if profiler_cfg is not None:
        profiler = hydra.utils.instantiate(profiler_cfg)
    else:
        profiler = None

    # Init lightning trainer
    if trainer_cfg.resume_from_checkpoint is not None:
        trainer_cfg.resume_from_checkpoint = to_absolute_path(trainer_cfg.resume_from_checkpoint)
        model = type(model).load_from_checkpoint(checkpoint_path=trainer_cfg.resume_from_checkpoint)
    trainer: pl.Trainer = hydra.utils.instantiate(
        trainer_cfg, logger=loggers, callbacks=callbacks, profiler=profiler, _convert_="partial"
    )

    if lr_find_only:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=data_module, max_lr=8e-4)

        # Plot lr find results
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion, and exit
        new_lr = lr_finder.suggestion()
        logging.info(f"LR Finder suggestion: {new_lr}")
        return

    # Train & Test
    if run_fit:
        trainer.fit(model, datamodule=data_module)
    if run_test_after_fit:
        trainer.test(model, dataloaders=data_module.test_dataloader())

    # Make sure everything closed properly
    logging.info("Finalizing!")
    utils.finish(
        model=model,
        datamodule=data_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    del model, data_module, loggers, callbacks, instantiator
    gc.collect()

    # Return metric score for hyperparameter optimization
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


def main(cfg: DictConfig) -> Optional[Union[float, Tensor]]:
    # Pretty print config using Rich library
    if cfg.get("print_config"):
        utils.print_config(cfg, resolve=True)

    # Reproducibility
    if "seed" in cfg.training:
        seed_everything(cfg.training.seed)

    instantiator = HydraInstantiator()

    data_cfg = cfg.get("datamodule")
    task_cfg = cfg.get("task")
    task_cfg.optimizers = cfg.get("optimizers")
    task_cfg.schedulers = cfg.get("schedulers")
    task_cfg.generator = cfg.get("generator")
    task_cfg.generator = infer_generator_config(task_cfg.generator, data_cfg.get("cfg"))

    return run(
        instantiator,
        ignore_warnings=cfg.get("ignore_warnings"),
        run_fit=cfg.get("training").get("run_fit"),
        run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
        lr_find_only=cfg.get("training").get("lr_find_only"),
        datamodule_cfg=data_cfg,
        task_cfg=task_cfg,
        trainer_cfg=cfg.get("trainer"),
        logger_cfgs=cfg.get("logger"),
        callback_cfgs=cfg.get("callbacks"),
        profiler_cfg=cfg.get("profiler"),
        optimized_metric=cfg.get("optimized_metric"),
    )


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
