# -*- coding: utf-8 -*-
import logging
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from climsr.data.super_resolution_data_module import SuperResolutionDataModule


class Instantiator:
    def model(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def optimizer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def scheduler(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def data_module(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def logger(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def trainer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")


class HydraInstantiator(Instantiator):
    def model(
        self,
        cfg: DictConfig,
        model_data_kwargs: Optional[DictConfig] = None,
    ) -> pl.LightningModule:
        if model_data_kwargs is None:
            model_data_kwargs = {}

        return self.instantiate(cfg, instantiator=self, **model_data_kwargs)

    def optimizer(self, model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        return self.instantiate(cfg, grouped_parameters)

    def scheduler(self, cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return self.instantiate(cfg, optimizer=optimizer)

    def data_module(
        self,
        cfg: DictConfig,
    ) -> SuperResolutionDataModule:
        return self.instantiate(cfg)

    def logger(self, cfg: DictConfig) -> Optional[logging.Logger]:
        if cfg.get("log"):
            if isinstance(cfg.trainer.logger, bool):
                return cfg.trainer.logger
            return self.instantiate(cfg.trainer.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return self.instantiate(cfg, **kwargs)

    def instantiate(self, *args, **kwargs):
        return hydra.utils.instantiate(*args, **kwargs)
