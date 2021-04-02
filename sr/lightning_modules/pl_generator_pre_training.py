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

    def build_model(self) -> nn.Module:
        if self.hparams.generator == "esrgan":
            generator = ESRGANGenerator(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                nf=self.hparams.nf,
                nb=self.hparams.nb,
                gc=self.hparams.gc,
                scale_factor=self.hparams.scale_factor,
            )
        elif self.hparams.generator == "drln":
            generator = DRLN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                scaling_factor=self.hparams.scale_factor,
            )
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

    def forward(self, x: Tensor, elevation: Tensor = None) -> Tensor:
        if self.hparams.generator == "srcnn":
            return self.net_G(x).clamp(0.0, 1.0)
        else:
            return self.net_G(x, elevation).clamp(0.0, 1.0)

    def common_step(self, batch: Any) -> Tuple[Tensor, Tensor]:
        """
        Runs common step during train/val/test pass.

        Args:
            batch (Any): A batch of data from a dataloader.

        Returns (Tuple[Tensor, Tensor]): A tuple with HR image and SR image.

        """
        lr, hr, sr_nearest, elev = (
            batch["lr"],
            batch["hr"],
            batch["nearest"],
            batch["elevation"],
        )
        x = torch.cat([sr_nearest, elev], dim=1)
        sr = self(x if self.hparams.generator == "srcnn" else lr, elev)

        return hr, sr

    def compute_metrics_common(
        self, hr: Tensor, sr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Common step to compute all of the metrics.

        Args:
            hr (Tensor): The ground truth HR image.
            sr (Tensor): The hallucinated SR image.

        Returns (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]): A tuple with metrics in following order:
            L1_Loss, PSNR, SSIM, MAE, MSE, RMSE.

        """
        loss = self.loss(sr, hr)
        psnr_score = psnr(sr, hr)
        ssim_score = ssim(sr, hr.half())
        mae = mean_absolute_error(sr, hr)
        mse = mean_squared_error(sr, hr)
        rmse = torch.sqrt(mse)

        return loss, psnr_score, ssim_score, mae, mse, rmse

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
        self.log("train/pixel_level_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Dict[str, Union[float, int, Tensor]]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """
        hr, sr = self.common_step(batch)
        l1_loss, psnr_score, ssim_score, mae, mse, rmse = self.compute_metrics_common(
            hr, sr
        )

        log_dict = {
            "val/pixel_level_loss": l1_loss,
            "val/psnr": psnr_score,
            "val/ssim": ssim_score,
            "val/mae": mae,
            "val/mse": mse,
            "val/rmse": rmse,
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return log_dict

    def test_step(self, batch: Any, batch_idx: int) -> Union[float, int, Tensor]:
        """
        Run test step.

        Args:
           batch (Any): A batch of data from test dataloader.
           batch_idx (int): The batch index.

        Returns (Union[float, int, Tensor]): A test loss for further processing.

        """
        hr, sr = self.common_step(batch)
        l1_loss, psnr_score, ssim_score, mae, mse, rmse = self.compute_metrics_common(
            hr, sr
        )

        log_dict = {
            "test/pixel_level_loss": l1_loss,
            "test/psnr": psnr_score,
            "test/ssim": ssim_score,
            "test/mae": mae,
            "test/mse": mse,
            "test/rmse": rmse,
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return l1_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log hp_metric at the epoch level."""

        pixel_level_loss_mean = torch.stack(
            [output["val/pixel_level_loss"] for output in outputs]
        ).mean()

        self.log("hp_metric", pixel_level_loss_mean)
        self.log_images()

    def log_images(self) -> None:
        """Log a single batch of images from the validation set to monitor image quality progress."""

        with torch.no_grad():
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
            lr, hr, sr_nearest, elev = (
                batch["lr"],
                batch["hr"],
                batch["nearest"],
                batch["elevation"],
            )

            self.logger.experiment.add_images("hr_images", hr, self.global_step)
            self.logger.experiment.add_images("lr_images", lr, self.global_step)
            self.logger.experiment.add_images(
                "sr_nearest", sr_nearest, self.global_step
            )

            lr = (
                sr_nearest.to(self.device)
                if self.hparams.generator == "srcnn"
                else lr.to(self.device)
            )

            x = torch.cat([sr_nearest, elev], dim=1).to(self.device)

            sr = self(x)
            self.logger.experiment.add_images("sr_images", sr, self.global_step)

            if isinstance(self.logger, DummyLogger):
                return

            img_dir = os.path.join(self.logger.log_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            save_image(
                hr,
                os.path.join(img_dir, f"hr-{self.hparams.experiment_name}.png"),
            )
            save_image(
                lr,
                os.path.join(img_dir, f"lr-{self.hparams.experiment_name}.png"),
            )
            save_image(
                sr_nearest,
                os.path.join(img_dir, f"sr-nearest-{self.hparams.experiment_name}.png"),
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
        parser.add_argument("--gen_in_channels", default=2, type=int)
        parser.add_argument("--gen_out_channels", default=1, type=int)
        parser.add_argument("--nf", default=64, type=int)
        parser.add_argument("--nb", default=23, type=int)
        parser.add_argument("--gc", default=32, type=int)
        parser.add_argument("--num_rrdb_blocks", default=16, type=int)
        parser.add_argument("--num_rrfdb_blocks", default=8, type=int)
        return parser
