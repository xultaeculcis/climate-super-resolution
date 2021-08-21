# -*- coding: utf-8 -*-
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.distributed import rank_zero_info

from climsr.core.datamodules import SuperResolutionDataModule
from climsr.core.config import TaskConfig, TrainerConfig, SuperResolutionDataConfig
from climsr.core.instantiator import HydraInstantiator, Instantiator
from climsr.core.model import TaskSuperResolutionModule
from climsr.core.utils import set_ignore_warnings


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    run_test_after_fit: bool = True,
    dataset: SuperResolutionDataConfig = SuperResolutionDataConfig(),
    task: TaskConfig = TaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    logger: Optional[Any] = None,
) -> None:
    if ignore_warnings:
        set_ignore_warnings()

    print(dataset)
    data_module: SuperResolutionDataModule = instantiator.data_module(dataset)
    if data_module is None:
        raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
    if not isinstance(data_module, LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance."
            " Hydra hint: is `dataset._target_` defined?`"
        )
    # data_module.setup("fit")

    # model: TaskSuperResolutionModule = instantiator.model(task, model_data_kwargs=getattr(data_module, "model_data_kwargs", None))
    # trainer = instantiator.trainer(
    #     trainer,
    #     logger=logger,
    # )
    #
    # trainer.fit(model, datamodule=data_module)
    # if run_test_after_fit:
    #     trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    logger = instantiator.logger(cfg)
    run(
        instantiator,
        ignore_warnings=cfg.get("ignore_warnings"),
        run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
        dataset=cfg.get("dataset"),
        task=cfg.get("task"),
        trainer=cfg.get("trainer"),
        logger=logger,
    )


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
