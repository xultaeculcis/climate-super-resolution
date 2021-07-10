# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from torch import Tensor
from torch.optim import Adam

from sr.lightning_modules.pl_sr_module import SuperResolutionLightningModule


class GANLightningModule(SuperResolutionLightningModule):
    """
    LightningModule for training the GAN based models.
    """

    def __init__(self, **kwargs):
        super(GANLightningModule, self).__init__(**kwargs)

    def _real_fake(self, size: int) -> Tuple[Tensor, Tensor]:
        real_labels = torch.ones((size, 1), device=self.device)
        fake_labels = torch.zeros((size, 1), device=self.device)
        return real_labels, fake_labels

    def loss_g(
        self, hr: Tensor, sr: Tensor, real_labels: Tensor, fake_labels: Tensor
    ) -> Tuple[float, float, float, float]:
        score_real = self.net_D(hr)
        score_fake = self.net_D(sr)
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, fake_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

        perceptual_loss = self.perceptual_criterion(hr, sr)
        pixel_level_loss = self.pixel_level_criterion(sr, hr)

        loss_g = (
            self.hparams.pixel_level_loss_factor * pixel_level_loss
            + self.hparams.perceptual_loss_factor * perceptual_loss
            + self.hparams.adversarial_loss_factor * adversarial_loss
        )

        return perceptual_loss, adversarial_loss, pixel_level_loss, loss_g

    def loss_d(
        self, hr: Tensor, sr: Tensor, real_labels: Tensor, fake_labels: Tensor
    ) -> float:
        score_real = self.net_D(hr)
        score_fake = self.net_D(sr.detach())
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, fake_labels)
        loss_d = (adversarial_loss_fr + adversarial_loss_rf) / 2

        return loss_d

    def training_step(
        self, batch: Any, batch_idx: int, optimizer_idx: int
    ) -> Dict[str, Any]:
        lr, hr, sr_nearest, elevation = (
            batch["lr"],
            batch["hr"],
            batch["nearest"],
            batch["elevation"],
        )
        real_labels, fake_labels = self._real_fake(hr.size(0))

        sr = self(sr_nearest if self.hparams.generator == "srcnn" else lr, elevation)

        # train generator
        if optimizer_idx == 0:
            perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(
                hr, sr, real_labels, fake_labels
            )

            log_dict = {
                "train/perceptual_loss": perceptual_loss,
                "train/adversarial_loss": adversarial_loss,
                "train/pixel_level_loss": pixel_level_loss,
                "train/loss_G": loss_g,
            }
            self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False)

            return {
                "loss": loss_g,
                "log": log_dict,
            }

        # train discriminator
        if optimizer_idx == 1:
            loss_d = self.loss_d(hr, sr, real_labels, fake_labels)

            self.log(
                "train/loss_D", loss_d, prog_bar=True, on_step=True, on_epoch=False
            )

            return {
                "loss": loss_d,
                "log": {
                    "train/loss_D": loss_d,
                },
            }

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """

        lr, hr, sr_nearest, elevation = (
            batch["lr"],
            batch["hr"].half(),
            batch["nearest"],
            batch["elevation"],
        )
        real_labels, fake_labels = self._real_fake(hr.size(0))

        sr = self(sr_nearest if self.hparams.generator == "srcnn" else lr, elevation)

        perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(
            hr, sr, real_labels, fake_labels
        )

        return {
            "val/perceptual_loss": perceptual_loss,
            "val/adversarial_loss": adversarial_loss,
            "val/pixel_level_loss": pixel_level_loss,
            "val/loss_G": loss_g,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log validation losses at the epoch level."""

        loss_g_mean = torch.stack([output["val/loss_G"] for output in outputs]).mean()
        log_dict = {
            "hp_metric": loss_g_mean,
        }
        self.log_dict(log_dict)

    def configure_optimizers(
        self,
    ) -> Tuple[List[Adam], List[Dict[str, Union[str, Any]]]]:
        optimizerD = torch.optim.Adam(
            self.net_D.parameters(), weight_decay=self.hparams.weight_decay
        )
        optimizerG = torch.optim.Adam(
            self.net_G.parameters(), weight_decay=self.hparams.weight_decay
        )
        schedulerD = torch.optim.lr_scheduler.OneCycleLR(
            optimizerG,
            max_lr=self.hparams.lr,
            total_steps=len(self.trainer.datamodule.train_dataloader())
            * self.hparams.max_epochs,
            pct_start=self.hparams.pct_start,
        )
        schedulerG = torch.optim.lr_scheduler.OneCycleLR(
            optimizerD,
            max_lr=self.hparams.lr,
            total_steps=len(self.trainer.datamodule.train_dataloader())
            * self.hparams.max_epochs,
            pct_start=self.hparams.pct_start,
        )
        schedulerD = {"scheduler": schedulerD, "interval": "step"}
        schedulerG = {"scheduler": schedulerG, "interval": "step"}

        return [optimizerG, optimizerD], [schedulerG, schedulerD]
