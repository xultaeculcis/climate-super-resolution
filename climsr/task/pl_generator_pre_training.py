# -*- coding: utf-8 -*-
import dataclasses
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from climsr.core.task import TaskSuperResolutionModule


class GeneratorPreTrainingLightningModule(TaskSuperResolutionModule):
    """
    LightningModule for pre-training the Generator Network.
    """

    def __init__(self, *args, **kwargs):
        super(GeneratorPreTrainingLightningModule, self).__init__(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Runs training step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.

        Returns (Any): Loss score for further processing.

        """
        hr, sr = self.common_step(batch)
        loss = self.loss(sr, hr)
        self.log("train/pixel_level_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[float, int, Tensor]]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """
        metrics = self.common_val_test_step(batch)

        log_dict = dict(list((f"val/{k}", v) for k, v in dataclasses.asdict(metrics).items() if k != "sr"))

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return log_dict

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log hp_metric at the epoch level."""
        pixel_level_loss_mean = torch.stack([output["val/pixel_level_loss"] for output in outputs]).mean()

        self.log("hp_metric", pixel_level_loss_mean)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Union[float, int, Tensor]:
        """
        Run test step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """

        metrics = self.common_val_test_step(batch)

        log_dict = dict(list((f"test/{k}", v) for k, v in dataclasses.asdict(metrics).items()))

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return metrics.pixel_level_loss
