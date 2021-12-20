# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

from torch import Tensor

from climsr import consts
from climsr.core.task import TaskSuperResolutionModule


class SuperResolutionLightningModule(TaskSuperResolutionModule):
    """
    LightningModule for training the SR networks or pre-training generator networks in GAN setting.
    """

    def __init__(self, *args, **kwargs):
        super(SuperResolutionLightningModule, self).__init__(*args, **kwargs)

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
        self.log("train/loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Union[float, int, Tensor]): Validation loss.

        """
        metric_dict = self.common_val_test_step(batch, prefix=consts.stages.val)
        metric_dict.pop("sr", None)
        self.log_dict(metric_dict, prog_bar=False, on_step=False, on_epoch=True)
        return metric_dict

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Run test step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Union[float, int, Tensor]): Test loss.

        """
        return self.common_val_test_step(batch, prefix=consts.stages.test)
