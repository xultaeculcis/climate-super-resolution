import argparse
from typing import Any, List, Dict, Union, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from loss import VGGLoss
from models import Generator, Discriminator


class ClimateRbfESRGANModule(pl.LightningModule):
    """
    LightningModule for pre-training the Climate RFB-ESRGAN.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(ClimateRbfESRGANModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = Generator(upscale_factor=self.hparams.scaling_factor)

        if self.hparams.train_fresh:
            state_dict = torch.load(self.hparams.pretrained_gen_model)["state_dict"]
            gen_state_dict = {}

            for key in state_dict.keys():
                if key.startswith("net_G"):
                    gen_state_dict[key.replace("net_G.", "")] = state_dict[key]

            self.net_G.load_state_dict(gen_state_dict)

        self.net_D = Discriminator()

        # metrics
        # We use vgg34 as our feature extraction method by default.
        self.vgg_criterion = VGGLoss()
        # Loss = 10 * mse_loss + vgg_loss + 0.005 * l1_loss
        self.pixel_level_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.net_G(x)

    def loss_g(self, lr, hr, real_label) -> Tuple[float, float, float, float]:
        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # According to the feature map, the root mean square error is regarded as the content loss.
        vgg_loss = self.vgg_criterion(sr, hr)

        # Train with fake high resolution image.
        hr_output = self.net_D(hr.detach())  # No train real fake image.
        sr_output = self.net_D(sr)  # Train fake image.

        # Adversarial loss (relativistic average GAN)
        adversarial_loss = self.adversarial_criterion(sr_output - torch.mean(hr_output), real_label)

        # Pixel level loss between two images.
        l1_loss = self.pixel_level_criterion(sr, hr)

        loss_g = 10 * l1_loss + vgg_loss + 0.005 * adversarial_loss

        return vgg_loss, adversarial_loss, l1_loss, loss_g

    def loss_d(self, hr: Tensor, sr: Tensor, real_label: Tensor, fake_label: Tensor) -> float:
        # Train with real high resolution image.
        hr_output = self.net_D(hr)  # Train real image.
        sr_output = self.net_D(sr.detach())  # No train fake image.

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_hr = self.adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
        loss_sr = self.adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
        loss_d = (loss_sr + loss_hr) / 2

        return loss_d

    def training_step(self, batch: Any, batch_idx: Any, optimizer_idx: Any):
        lr, hr, _ = batch
        real_label = torch.full((self.hparams.batch_size, 1), 1, dtype=lr.dtype, device=self.device)
        fake_label = torch.full((self.hparams.batch_size, 1), 0, dtype=lr.dtype, device=self.device)

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # train discriminator
        if optimizer_idx == 0:
            loss_d = self.loss_d(hr, sr, real_label, fake_label)

            self.log("train/loss_D", loss_d, prog_bar=True)

            return {
                "loss": loss_d,
                "log": {
                    "train/loss_D": loss_d,
                },
            }

        # train generator
        if optimizer_idx == 1:
            vgg_loss, adversarial_loss, l1_loss, loss_g = self.loss_g(lr, hr, real_label)

            log_dict = {
                "train/perceptual_loss": vgg_loss,
                "train/adversarial_loss":  adversarial_loss,
                "train/pixel_level_loss":  l1_loss,
                "train/loss_G": loss_g,
            }
            self.log_dict(log_dict, prog_bar=True)

            return {
                "loss": loss_g,
                "log": log_dict,
            }

    def training_epoch_end(self, outputs):
        """Compute and log training losses at the epoch level."""

        vgg_loss_mean = torch.stack([output["log"]["train/perceptual_loss"] for output in outputs[1]]).mean()
        adversarial_loss_mean = torch.stack([output["log"]["train/adversarial_loss"] for output in outputs[1]]).mean()
        l1_loss_mean = torch.stack([output["log"]["train/pixel_level_loss"] for output in outputs[1]]).mean()
        loss_g_mean = torch.stack([output["log"]["train/loss_G"] for output in outputs[1]]).mean()
        loss_d_mean = torch.stack([output["log"]["train/loss_D"] for output in outputs[0]]).mean()
        log_dict = {
            "train_epoch/loss_D": loss_d_mean,
            "train_epoch/perceptual_loss": vgg_loss_mean,
            "train_epoch/adversarial_loss": adversarial_loss_mean,
            "train_epoch/l1_loss": l1_loss_mean,
            "train_epoch/loss_G": loss_g_mean,
        }
        self.log_dict(log_dict)

    def validation_step(self, batch: Any, batch_idx: Any) -> Dict[str, Union[int, float]]:
        lr, hr, _ = batch
        real_label = torch.full((self.hparams.batch_size, 1), 1, dtype=lr.dtype, device=self.device)

        vgg_loss, adversarial_loss, l1_loss, loss_g = self.loss_g(lr, hr, real_label)

        return {
            "val/perceptual_loss": vgg_loss,
            "val/adversarial_loss":  adversarial_loss,
            "val/pixel_level_loss":  l1_loss,
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
            "val/adversarial_loss":  adversarial_loss_mean,
            "val/pixel_level_loss":  l1_loss_mean,
            "val/loss_G": loss_g_mean,
            "hp_metric": loss_g_mean,
        }
        self.log_dict(log_dict)

    def on_epoch_end(self) -> None:
        with torch.no_grad():
            lr, hr, sr_bicubic = next(iter(self.trainer.datamodule.val_dataloader()))
            lr = lr.to(self.device)

            self.logger.experiment.add_images('hr_images', hr, self.current_epoch)
            self.logger.experiment.add_images('lr_images', lr, self.current_epoch)
            self.logger.experiment.add_images('sr_bicubic', sr_bicubic, self.current_epoch)

            sr = self(lr)
            self.logger.experiment.add_images('sr_images', sr, self.current_epoch)

    def configure_optimizers(self) -> Tuple[List[Adam], List[StepLR]]:
        optimizerD = torch.optim.Adam(self.net_D.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        optimizerG = torch.optim.Adam(self.net_G.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=self.hparams.step_size, gamma=0.5)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=self.hparams.step_size, gamma=0.5)

        return [optimizerD, optimizerG], [schedulerD, schedulerG]

    @staticmethod
    def add_model_specific_args(parent) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--step_size", type=int, default=3)

        return parser
