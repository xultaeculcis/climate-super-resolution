# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

import climsr.consts as consts
from climsr.core.task import TaskSuperResolutionModule
from climsr.losses.perceptual import PerceptualLoss


class GANLightningModule(TaskSuperResolutionModule):
    """
    LightningModule for training the GAN based models.
    """

    def __init__(self, *args, **kwargs):
        super(GANLightningModule, self).__init__(*args, **kwargs)
        self.perceptual_criterion = PerceptualLoss()
        self.pixel_level_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()

    def _real_fake(self, size: int) -> Tuple[Tensor, Tensor]:
        real_labels = torch.ones((size, 1), device=self.device)
        fake_labels = torch.zeros((size, 1), device=self.device)
        return real_labels, fake_labels

    def loss_g(
        self, hr: Tensor, sr: Tensor, real_labels: Tensor, fake_labels: Tensor
    ) -> Tuple[Union[float, Tensor], Union[float, Tensor], Union[float, Tensor], Union[float, Tensor]]:
        score_real = self.discriminator(hr)
        score_fake = self.discriminator(sr)
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

    def loss_d(self, hr: Tensor, sr: Tensor, real_labels: Tensor, fake_labels: Tensor) -> Union[float, Tensor]:
        score_real = self.discriminator(hr)
        score_fake = self.discriminator(sr.detach())
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, fake_labels)
        loss_d = (adversarial_loss_fr + adversarial_loss_rf) / 2

        return loss_d

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int) -> Dict[str, Any]:
        hr: Tensor = batch[consts.batch_items.hr]
        real_labels, fake_labels = self._real_fake(hr.shape[0])

        hr, sr = self.common_step(batch)

        # train generator
        if optimizer_idx == 0:
            perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(hr, sr, real_labels, fake_labels)

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

            self.log("train/loss_D", loss_d, prog_bar=True, on_step=True, on_epoch=False)

            return {
                "loss": loss_d,
                "log": {
                    "train/loss_D": loss_d,
                },
            }

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[int, float, Tensor]]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """

        hr = batch[consts.batch_items.hr]
        real_labels, fake_labels = self._real_fake(hr.size(0))

        metric_dict = self.common_val_test_step(batch)

        perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(hr, metric_dict["sr"], real_labels, fake_labels)

        metric_dict.pop("sr", None)
        metric_dict.update(
            {
                "val/perceptual_loss": perceptual_loss,
                "val/adversarial_loss": adversarial_loss,
                "val/loss_G": loss_g,
            }
        )
        self.log_dict(metric_dict, on_step=False, on_epoch=True)

        return metric_dict
