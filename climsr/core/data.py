# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import climsr.consts as consts
from climsr.core.config import SuperResolutionDataConfig

default_cfg = SuperResolutionDataConfig()


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, cfg: SuperResolutionDataConfig = default_cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["val"],
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> List[DataLoader]:
        if self.cfg.world_clim_variable == consts.world_clim.temp:
            return [
                DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                )
                for test_dataset in self.ds["test"]
            ]
        else:
            return [
                DataLoader(
                    self.ds["test"][0],
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                )
            ]

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}
