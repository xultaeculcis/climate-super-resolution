# -*- coding: utf-8 -*-
import argparse
import os
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.metrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    psnr,
    ssim,
)
from torch import Tensor
from torchvision.utils import save_image

from sr.losses.perceptual import PerceptualLoss
from sr.models.drln import DRLN
from sr.models.esrgan import ESRGANGenerator
from sr.models.rfb_esrgan import RFBESRGANGenerator
from sr.models.srcnn import SRCNN


class GeneratorPreTrainingLightningModule(pl.LightningModule):
    """
    LightningModule for pre-training the Generator Network.
    """

    def __init__(self, **kwargs):
        super(GeneratorPreTrainingLightningModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = self.build_model()

        # metrics
        self.loss = (
            torch.nn.L1Loss()
            if self.hparams.generator == "srcnn"
            else torch.nn.L1Loss()
        )
        self.perceptual_criterion = PerceptualLoss()

    def build_model(self) -> nn.Module:
        if self.hparams.generator == "esrgan":
            generator = ESRGANGenerator(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                nf=self.hparams.nf,
                nb=self.hparams.nb,
                gc=self.hparams.gc,
            )
        elif self.hparams.generator == "drln":
            generator = DRLN(scaling_factor=self.hparams.scale_factor)
        elif self.hparams.generator == "rfbesrgan":
            generator = RFBESRGANGenerator(
                upscale_factor=self.hparams.scale_factor,
                num_rrdb_blocks=self.hparams.num_rrdb_blocks,
                num_rrfdb_blocks=self.hparams.num_rrfdb_blocks,
            )
        elif self.hparams.generator == "srcnn":
            generator = SRCNN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
            )
        else:
            raise ValueError(
                f"Specified generator '{self.hparams.generator}' is not supported"
            )
        return generator

    def forward(self, x: Tensor) -> Tensor:
        return self.net_G(x).clamp(0, 1)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        lr, hr, sr_bicubic = batch["lr"], batch["hr"], batch["bicubic"]

        sr = self(sr_bicubic if self.hparams.generator == "srcnn" else lr)

        loss = self.loss(sr, hr)

        self.log("train/pixel_level_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Dict[str, Union[float, int]]:
        lr, hr, sr_bicubic = batch["lr"], batch["hr"].half(), batch["bicubic"]

        sr = self(sr_bicubic if self.hparams.generator == "srcnn" else lr)

        l1_loss = self.loss(sr, hr)
        psnr_score = psnr(sr, hr)
        ssim_score = ssim(sr, hr)
        mae = mean_absolute_error(sr, hr)
        mse = mean_squared_error(sr, hr)
        rmse = torch.sqrt(mse)
        perceptual_loss = self.perceptual_criterion(sr, hr)

        return {
            "val/pixel_level_loss": l1_loss,
            "val/psnr": psnr_score,
            "val/ssim": ssim_score,
            "val/mae": mae,
            "val/mse": mse,
            "val/rmse": rmse,
            "val/perceptual_loss": perceptual_loss,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log validation losses at the epoch level."""

        pixel_level_loss_mean = torch.stack(
            [output["val/pixel_level_loss"] for output in outputs]
        ).mean()
        psnr_score_mean = torch.stack([output["val/psnr"] for output in outputs]).mean()
        ssim_score_mean = torch.stack([output["val/ssim"] for output in outputs]).mean()
        mae_mean = torch.stack([output["val/mae"] for output in outputs]).mean()
        mse_mean = torch.stack([output["val/mse"] for output in outputs]).mean()
        rmse_mean = torch.stack([output["val/rmse"] for output in outputs]).mean()
        perceptual_loss_mean = torch.stack(
            [output["val/perceptual_loss"] for output in outputs]
        ).mean()
        log_dict = {
            "val/pixel_level_loss": pixel_level_loss_mean,
            "hp_metric": pixel_level_loss_mean,
            "val/psnr": psnr_score_mean,
            "val/ssim": ssim_score_mean,
            "val/mae": mae_mean,
            "val/mse": mse_mean,
            "val/rmse": rmse_mean,
            "val/perceptual_loss": perceptual_loss_mean,
        }
        self.log_dict(log_dict)

        with torch.no_grad():
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
            lr, hr, sr_bicubic = batch["lr"], batch["hr"], batch["bicubic"]

            self.logger.experiment.add_images("hr_images", hr, self.global_step)
            self.logger.experiment.add_images("lr_images", lr, self.global_step)
            self.logger.experiment.add_images(
                "sr_bicubic", sr_bicubic, self.global_step
            )

            lr = (
                sr_bicubic.to(self.device)
                if self.hparams.generator == "srcnn"
                else lr.to(self.device)
            )
            sr = self(lr)
            self.logger.experiment.add_images("sr_images", sr, self.global_step)

            if isinstance(self.logger, DummyLogger):
                return

            img_dir = os.path.join(self.logger.log_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            save_image(
                hr, os.path.join(img_dir, f"hr-{self.hparams.experiment_name}.png")
            )
            save_image(
                lr, os.path.join(img_dir, f"lr-{self.hparams.experiment_name}.png")
            )
            save_image(
                sr_bicubic,
                os.path.join(img_dir, f"sr-bicubic-{self.hparams.experiment_name}.png"),
            )
            save_image(
                sr,
                os.path.join(
                    img_dir,
                    f"sr-{self.hparams.experiment_name}-step={self.global_step}.png",
                ),
            )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, Any]]]]:
        optimizer = torch.optim.Adam(self.net_G.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.hparams.max_epochs,
            pct_start=self.hparams.pct_start,
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent: argparse.ArgumentParser):
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
        parser.add_argument("--gen_in_channels", default=3, type=int)
        parser.add_argument("--gen_out_channels", default=3, type=int)
        parser.add_argument("--nf", default=64, type=int)
        parser.add_argument("--nb", default=23, type=int)
        parser.add_argument("--gc", default=32, type=int)
        parser.add_argument("--num_rrdb_blocks", default=16, type=int)
        parser.add_argument("--num_rrfdb_blocks", default=8, type=int)
        return parser
