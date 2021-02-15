import argparse
from typing import Any, List, Dict, Union, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from esrgan.loss import PerceptualLoss
from esrgan.models import Generator, Discriminator


class ClimateESRGANModule(pl.LightningModule):
    """
    LightningModule for pre-training the Climate ESRGAN.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(ClimateESRGANModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = Generator(3, 3, 64, 23, 32)

        if self.hparams.pretrained_gen_model:
            self.net_G.load_state_dict(torch.load(self.hparams.pretrained_gen_model), strict=True)

        self.net_D = Discriminator()

        self.perceptual_criterion = PerceptualLoss()
        self.pixel_level_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.net_G(x)

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

        loss_g = self.hparams.pixel_level_loss_factor * pixel_level_loss + \
                 self.hparams.perceptual_loss_factor * perceptual_loss + \
                 self.hparams.adversarial_loss_factor * adversarial_loss

        return perceptual_loss, adversarial_loss, pixel_level_loss, loss_g

    def loss_d(self, hr: Tensor, sr: Tensor, real_labels: Tensor, fake_labels: Tensor) -> float:
        score_real = self.net_D(hr)
        score_fake = self.net_D(sr.detach())
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, fake_labels)
        loss_d = (adversarial_loss_fr + adversarial_loss_rf) / 2

        return loss_d

    def training_step(self, batch: Any, batch_idx: Any, optimizer_idx: Any):
        lr, hr = batch["lr"], batch["hr"]

        real_labels = torch.ones((hr.size(0), 1), device=self.device)
        fake_labels = torch.zeros((hr.size(0), 1), device=self.device)

        sr = self(lr)

        # train generator
        if optimizer_idx == 0:
            perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(hr, sr, real_labels, fake_labels)

            log_dict = {
                "train/perceptual_loss": perceptual_loss,
                "train/adversarial_loss": adversarial_loss,
                "train/pixel_level_loss": pixel_level_loss,
                "train/loss_G": loss_g,
            }
            self.log_dict(log_dict, prog_bar=True)

            return {
                "loss": loss_g,
                "log": log_dict,
            }

        # train discriminator
        if optimizer_idx == 1:
            loss_d = self.loss_d(hr, sr, real_labels, fake_labels)

            self.log("train/loss_D", loss_d, prog_bar=True)

            return {
                "loss": loss_d,
                "log": {
                    "train/loss_D": loss_d,
                },
            }

    def training_epoch_end(self, outputs):
        """Compute and log training losses at the epoch level."""

        perceptual_loss_mean = torch.stack([output["log"]["train/perceptual_loss"] for output in outputs[1]]).mean()
        adversarial_loss_mean = torch.stack([output["log"]["train/adversarial_loss"] for output in outputs[1]]).mean()
        pixel_level_loss_mean = torch.stack([output["log"]["train/pixel_level_loss"] for output in outputs[1]]).mean()
        loss_g_mean = torch.stack([output["log"]["train/loss_G"] for output in outputs[1]]).mean()
        loss_d_mean = torch.stack([output["log"]["train/loss_D"] for output in outputs[0]]).mean()
        log_dict = {
            "train_epoch/loss_D": loss_d_mean,
            "train_epoch/perceptual_loss": perceptual_loss_mean,
            "train_epoch/adversarial_loss": adversarial_loss_mean,
            "train_epoch/pixel_level_loss": pixel_level_loss_mean,
            "train_epoch/loss_G": loss_g_mean,
        }
        self.log_dict(log_dict)

    def validation_step(self, batch: Any, batch_idx: Any) -> Dict[str, Union[int, float]]:
        lr, hr = batch["lr"], batch["hr"]
        real_labels = torch.ones((hr.size(0), 1), device=self.device)
        fake_labels = torch.zeros((hr.size(0), 1), device=self.device)

        sr = self(lr)

        perceptual_loss, adversarial_loss, pixel_level_loss, loss_g = self.loss_g(hr, sr, real_labels, fake_labels)

        return {
            "val/perceptual_loss": perceptual_loss,
            "val/adversarial_loss": adversarial_loss,
            "val/pixel_level_loss": pixel_level_loss,
            "val/loss_G": loss_g,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log validation losses at the epoch level."""

        vgg_loss_mean = torch.stack([output["val/perceptual_loss"] for output in outputs]).mean()
        adversarial_loss_mean = torch.stack([output["val/adversarial_loss"] for output in outputs]).mean()
        l1_loss_mean = torch.stack([output["val/pixel_level_loss"] for output in outputs]).mean()
        loss_g_mean = torch.stack([output["val/loss_G"] for output in outputs]).mean()
        log_dict = {
            "val/perceptual_loss": vgg_loss_mean,
            "val/adversarial_loss": adversarial_loss_mean,
            "val/pixel_level_loss": l1_loss_mean,
            "val/loss_G": loss_g_mean,
            "hp_metric": loss_g_mean,
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

    def configure_optimizers(self) -> Tuple[List[Adam], List[Dict[str, Union[str, Any]]]]:
        optimizerD = torch.optim.Adam(self.net_D.parameters())
        optimizerG = torch.optim.Adam(self.net_G.parameters())
        schedulerD = torch.optim.lr_scheduler.OneCycleLR(
            optimizerG,
            max_lr=self.hparams.lr,
            total_steps=len(self.trainer.datamodule.train_dataloader()) * self.hparams.max_epochs,
            pct_start=self.hparams.pct_start
        )
        schedulerG = torch.optim.lr_scheduler.OneCycleLR(
            optimizerD,
            max_lr=self.hparams.lr,
            total_steps=len(self.trainer.datamodule.train_dataloader()) * self.hparams.max_epochs,
            pct_start=self.hparams.pct_start
        )
        schedulerD = {'scheduler': schedulerD, 'interval': 'step'}
        schedulerG = {'scheduler': schedulerG, 'interval': 'step'}

        return [optimizerG, optimizerD], [schedulerG, schedulerD]

    @staticmethod
    def add_model_specific_args(parent) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-2, type=float)
        parser.add_argument('--pct_start', default=0.1, type=float)
        parser.add_argument('--pixel_level_loss_factor', default=1e-2, type=float)
        parser.add_argument('--perceptual_loss_factor', default=1.0, type=float)
        parser.add_argument('--adversarial_loss_factor', default=5e-3, type=float)

        return parser
