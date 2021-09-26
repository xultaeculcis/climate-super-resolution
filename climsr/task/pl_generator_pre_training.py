# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from climsr import consts
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
        return self.common_val_test_step(batch, prefix=consts.stages.val)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log hp_metric at the epoch level."""
        hp_metric = torch.stack([output[f"{consts.stages.val}/rmse"] for output in outputs]).mean()
        self.log("hp_metric", hp_metric)

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
