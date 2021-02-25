# -*- coding: utf-8 -*-
import argparse
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from sr.losses.perceptual import PerceptualLoss
from sr.models.drln import DRLN
from sr.models.esrgan import ESRGANDiscriminator, ESRGANGenerator
from sr.models.rfb_esrgan import RFBESRGANDiscriminator, RFBESRGANGenerator
from sr.models.srcnn import SRCNN


class GANLightningModule(pl.LightningModule):
    """
    LightningModule for pre-training the ESRGAN.
    """

    def __init__(self, **kwargs):
        super(GANLightningModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G, self.net_D = self.build_models()

        if self.hparams.pretrained_gen_model:
            self.net_G.load_state_dict(
                torch.load(self.hparams.pretrained_gen_model), strict=True
            )

        self.perceptual_criterion = PerceptualLoss()
        self.pixel_level_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()

    def build_models(self) -> Tuple[nn.Module, nn.Module]:
        if self.hparams.generator == "esrgan":
            generator = ESRGANGenerator(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                nf=self.hparams.nf,
                nb=self.hparams.nb,
                gc=self.hparams.gc,
            )
            discriminator = ESRGANDiscriminator(
                in_channels=self.hparams.disc_in_channels
            )
        elif self.hparams.generator == "drln":
            generator = DRLN(scaling_factor=self.hparams.scale_factor)
            discriminator = ESRGANDiscriminator(
                in_channels=self.hparams.disc_in_channels
            )
        elif self.hparams.generator == "rfbesrgan":
            generator = RFBESRGANGenerator(
                upscale_factor=self.hparams.scale_factor,
                num_rrdb_blocks=self.hparams.num_rrdb_blocks,
                num_rrfdb_blocks=self.hparams.num_rrfdb_blocks,
            )
            discriminator = RFBESRGANDiscriminator(
                in_channels=self.hparams.disc_in_channels
            )
        elif self.hparams.generator == "srcnn":
            generator = SRCNN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
            )
            discriminator = ESRGANDiscriminator(
                in_channels=self.hparams.disc_in_channels
            )
        else:
            raise ValueError(
                f"Specified generator '{self.hparams.generator}' is not supported"
            )
        return generator, discriminator

    def forward(self, x: Tensor) -> Tensor:
        return self.net_G(x).clamp_(0, 1)

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
        lr, hr, sr_bicubic = batch["lr"], batch["hr"], batch["bicubic"]

        real_labels = torch.ones((hr.size(0), 1), device=self.device)
        fake_labels = torch.zeros((hr.size(0), 1), device=self.device)

        sr = self(sr_bicubic if self.hparams.generator == "srcnn" else lr)

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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log training losses at the epoch level."""

        perceptual_loss_mean = torch.stack(
            [output["log"]["train/perceptual_loss"] for output in outputs[1]]
        ).mean()
        adversarial_loss_mean = torch.stack(
            [output["log"]["train/adversarial_loss"] for output in outputs[1]]
        ).mean()
        pixel_level_loss_mean = torch.stack(
            [output["log"]["train/pixel_level_loss"] for output in outputs[1]]
        ).mean()
        loss_g_mean = torch.stack(
            [output["log"]["train/loss_G"] for output in outputs[1]]
        ).mean()
        loss_d_mean = torch.stack(
            [output["log"]["train/loss_D"] for output in outputs[0]]
        ).mean()
        log_dict = {
            "train_epoch/loss_D": loss_d_mean,
            "train_epoch/perceptual_loss": perceptual_loss_mean,
            "train_epoch/adversarial_loss": adversarial_loss_mean,
            "train_epoch/pixel_level_loss": pixel_level_loss_mean,
            "train_epoch/loss_G": loss_g_mean,
        }
        self.log_dict(log_dict)

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Dict[str, Union[int, float]]:
        lr, hr, sr_bicubic = batch["lr"], batch["hr"].half(), batch["bicubic"]

        real_labels = torch.ones((hr.size(0), 1), device=self.device)
        fake_labels = torch.zeros((hr.size(0), 1), device=self.device)

        sr = self(sr_bicubic if self.hparams.generator == "srcnn" else lr)

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

        vgg_loss_mean = torch.stack(
            [output["val/perceptual_loss"] for output in outputs]
        ).mean()
        adversarial_loss_mean = torch.stack(
            [output["val/adversarial_loss"] for output in outputs]
        ).mean()
        l1_loss_mean = torch.stack(
            [output["val/pixel_level_loss"] for output in outputs]
        ).mean()
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

            self.logger.experiment.add_images("hr_images", hr, self.global_step)
            self.logger.experiment.add_images("lr_images", lr, self.global_step)
            self.logger.experiment.add_images(
                "sr_bicubic", sr_bicubic, self.global_step
            )

            sr = self(lr)
            self.logger.experiment.add_images("sr_images", sr, self.global_step)

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

    @staticmethod
    def add_model_specific_args(parent) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            parents=[parent], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--max_lr",
            default=1e-4,
            type=float,
            help="The max learning rate for the 1Cycle LR Scheduler",
        )
        parser.add_argument(
            "--pct_start",
            default=0.05,
            type=Union[float, int],
            help="The percentage of the cycle (in number of steps) spent increasing the learning rate",
        )
        parser.add_argument(
            "--div_factor",
            default=2,
            type=float,
            help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
        )
        parser.add_argument(
            "--final_div_factor",
            default=1e2,
            type=float,
            help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
        )
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--pixel_level_loss_factor", default=1e-2, type=float)
        parser.add_argument("--perceptual_loss_factor", default=1.0, type=float)
        parser.add_argument("--adversarial_loss_factor", default=5e-3, type=float)
        parser.add_argument("--gen_in_channels", default=3, type=int)
        parser.add_argument("--gen_out_channels", default=3, type=int)
        parser.add_argument("--disc_in_channels", default=3, type=int)
        parser.add_argument("--nf", default=64, type=int)
        parser.add_argument("--nb", default=23, type=int)
        parser.add_argument("--gc", default=32, type=int)
        parser.add_argument("--num_rrdb_blocks", default=16, type=int)
        parser.add_argument("--num_rrfdb_blocks", default=8, type=int)

        return parser
