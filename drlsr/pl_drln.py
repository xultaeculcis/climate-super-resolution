import argparse
from typing import List, Any, Tuple, Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from models.drln import DRLN


class DRLNLightningModule(pl.LightningModule):
    def __init__(self, args, kwargs):
        super(DRLNLightningModule, self).__init__()

        self.save_hyperparameters()

        self.model = DRLN(self.hparams.scaling_factor)

    def forward(self, x: Tensor):
        return self.model(x)

    def loss(self, input: Tensor, target: Tensor):
        return F.l1_loss(input=input, target=target)

    def training_step(self, batch: Any, batch_idx: int):
        lr, hr, _ = batch["lr"], batch["hr"], batch["bicubic"]

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The L1 of the generated fake high-resolution image and real high-resolution image is calculated.
        loss = self.loss(sr, hr)

        self.log("train/pixel_level_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        lr, hr, sr_bicubic = batch["lr"], batch["hr"], batch["bicubic"]

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The L1 of the generated fake high-resolution image and real high-resolution image is calculated.
        l1_loss = self.loss(sr, hr)

        return {
            "val/pixel_level_loss": l1_loss,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log validation losses at the epoch level."""

        l1_loss_mean = torch.stack([output["val/pixel_level_loss"] for output in outputs]).mean()
        log_dict = {
            "val/pixel_level_loss": l1_loss_mean,
            "hp_metric": l1_loss_mean,
        }
        self.log_dict(log_dict)

        with torch.no_grad():
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
            lr, hr, sr_bicubic = batch["lr"], batch["hr"], batch["bicubic"]

            lr = lr.to(self.device)

            self.logger.experiment.add_images('hr_images', hr, self.global_step)
            self.logger.experiment.add_images('lr_images', lr, self.global_step)
            self.logger.experiment.add_images('sr_bicubic', sr_bicubic, self.global_step)

            sr = self(lr)
            self.logger.experiment.add_images('sr_images', sr, self.global_step)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, Any]]]]:
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=len(self.trainer.datamodule.train_dataloader()) * self.hparams.max_epochs,
            pct_start=self.hparams.pct_start
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument('--max_lr', default=1e-4, type=float)
        parser.add_argument('--pct_start', default=0.1, type=float)
        return parser
