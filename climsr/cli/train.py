# -*- coding: utf-8 -*-
import logging
from typing import Any, List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_info

from climsr.core.config import SuperResolutionDataConfig, TaskConfig, TrainerConfig
from climsr.core.instantiator import HydraInstantiator, Instantiator
from climsr.core.task import TaskSuperResolutionModule
from climsr.core.utils import set_gpu_power_limit_if_needed, set_ignore_warnings
from climsr.data.super_resolution_data_module import SuperResolutionDataModule

default_sr_dm_config = SuperResolutionDataConfig()
default_task_config = TaskConfig()
default_trainer_config = TrainerConfig()


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    run_test_after_fit: bool = True,
    lr_find_only: bool = False,
    datamodule_cfg: Optional[SuperResolutionDataConfig] = default_sr_dm_config,
    task_cfg: Optional[TaskConfig] = default_task_config,
    trainer_cfg: Optional[TrainerConfig] = default_trainer_config,
    logger_cfgs: Optional[Any] = None,
    callback_cfgs: Optional[Any] = None,
) -> None:
    if ignore_warnings:
        set_ignore_warnings()

    # Limit RTX 3090 power draw if possible to stabilize PSU usage
    set_gpu_power_limit_if_needed()

    # Init data module
    data_module: SuperResolutionDataModule = instantiator.data_module(datamodule_cfg)
    if data_module is None:
        raise ValueError("No datamodule found. Hydra hint: did you set `datamodule=...`?")
    if not isinstance(data_module, pl.LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance." " Hydra hint: is `datamodule._target_` defined?`"
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

    # Init lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(trainer_cfg, logger=loggers, callbacks=callbacks, _convert_="partial")

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
    trainer.fit(model, datamodule=data_module)
    if run_test_after_fit:
        trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig) -> None:
    # Reproducibility
    if "seed" in cfg.training:
        seed_everything(cfg.training.seed, workers=True)

    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()

    run(
        instantiator,
        ignore_warnings=cfg.get("ignore_warnings"),
        run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
        lr_find_only=cfg.get("training").get("lr_find_only"),
        datamodule_cfg=cfg.get("datamodule"),
        task_cfg=cfg.get("task"),
        trainer_cfg=cfg.get("trainer"),
        logger_cfgs=cfg.get("logger"),
        callback_cfgs=cfg.get("callbacks"),
    )


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
