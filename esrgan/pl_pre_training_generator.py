import argparse
from typing import Any, Dict, Union, List

import pytorch_lightning as pl
import torch

from esrgan.models import Generator


class PreTrainingESRGANModule(pl.LightningModule):
    """
    LightningModule for pre-training the ESRGAN Generator.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(PreTrainingESRGANModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = Generator(3, 3, 64, 23, 32)

        # metrics
        self.loss = torch.nn.L1Loss()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb):
        lr, hr, _ = batch["lr"], batch["hr"], batch["bicubic"]

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The L1 of the generated fake high-resolution image and real high-resolution image is calculated.
        loss = self.loss(sr, hr)

        self.log("train/pixel_level_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: Any) -> Dict[str, Union[int, float]]:
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
            "val/pixel_level_loss":  l1_loss_mean,
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

            sr = self(lr).clamp_(0, 1)
            self.logger.experiment.add_images('sr_images', sr, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net_G.parameters(), betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
           optimizer,
           max_lr=self.hparams.max_lr,
           total_steps=len(self.trainer.datamodule.train_dataloader()) * self.hparams.max_epochs,
           pct_start=self.hparams.pct_start
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument('--max_lr', default=1e-4, type=float)
        parser.add_argument('--pct_start', default=0.1, type=float)
        return parser
