# -*- coding: utf-8 -*-
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    psnr,
    ssim,
)
from torch import Tensor

from sr.data.utils import denormalize
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
            torch.nn.MSELoss()
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
            return self.net_G(x)
        else:
            return self.net_G(x, elevation)

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

        if self.hparams.use_elevation:
            x = torch.cat([sr_nearest, elev], dim=1)
        else:
            x = sr_nearest

        sr = self(x if self.hparams.generator == "srcnn" else lr, elev)

        return hr, sr

    def common_val_test_step(self, batch: Any) -> "MetricsResult":
        original = batch["original_data"]
        mask = batch["mask"].cpu().numpy()
        max_vals = batch["max"].cpu().numpy()
        min_vals = batch["min"].cpu().numpy()

        hr, sr = self.common_step(batch)

        metrics = self.compute_metrics_common(hr, sr)

        denormalized_mae = []
        denormalized_mse = []
        denormalized_rmse = []
        denormalized_r2 = []

        original = original.squeeze(1).cpu().numpy()
        sr = sr.squeeze(1).cpu().numpy()

        for i in range(sr.shape[0]):
            # to numpy
            i_original = original[i]
            i_sr = sr[i]

            # denormalize
            i_sr = denormalize(i_sr, min_vals[i], max_vals[i]).clip(
                min_vals[i], max_vals[i]
            )

            # ocean mask
            i_sr[mask[i]] = 0.0
            i_original[mask[i]] = 0.0

            # compute metrics
            diff = i_sr - i_original
            denormalized_mae.append(np.absolute(diff).mean())
            denormalized_mse.append((diff ** 2).mean())
            denormalized_rmse.append(np.sqrt(denormalized_mse[-1]))
            denormalized_r2.append(
                1
                - (
                    np.sum(diff ** 2)
                    / (np.sum((i_original - np.mean(i_original)) ** 2) + 1e-5)
                )
            )

        denormalized_mae = np.mean(denormalized_mae)
        denormalized_mse = np.mean(denormalized_mse)
        denormalized_rmse = np.mean(denormalized_rmse)
        denormalized_r2 = np.mean(denormalized_r2)

        return MetricsResult(
            denormalized_mae=denormalized_mae,
            denormalized_mse=denormalized_mse,
            denormalized_rmse=denormalized_rmse,
            denormalized_r2=denormalized_r2,
            pixel_level_loss=metrics.pixel_level_loss,
            mae=metrics.mae,
            mse=metrics.mse,
            psnr=metrics.psnr,
            rmse=metrics.rmse,
            ssim=metrics.ssim,
        )

    def compute_metrics_common(self, hr: Tensor, sr: Tensor) -> "MetricsSimple":
        """
        Common step to compute all of the metrics.

        Args:
            hr (Tensor): The ground truth HR image.
            sr (Tensor): The hallucinated SR image.

        Returns (MetricsSimple): A dataclass with metrics: L1_Loss, PSNR, SSIM, MAE, MSE, RMSE.

        """
        loss = self.loss(sr, hr)
        psnr_score = psnr(sr, hr)
        ssim_score = ssim(sr, hr.half() if self.hparams.precision == 16 else hr)
        mae = mean_absolute_error(sr, hr)
        mse = mean_squared_error(sr, hr)
        rmse = torch.sqrt(mse)

        return MetricsSimple(
            pixel_level_loss=loss,
            psnr=psnr_score,
            ssim=ssim_score,
            mae=mae,
            mse=mse,
            rmse=rmse,
        )

    def on_train_start(self):
        """Run additional steps when training starts."""
        self.logger.log_hyperparams(
            self.hparams, {"hp_metric": self.hparams.initial_hp_metric_val}
        )

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
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[float, int, Tensor]]:
        """
        Run validation step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """
        metrics = self.common_val_test_step(batch)

        log_dict = dict(
            list((f"val/{k}", v) for k, v in dataclasses.asdict(metrics).items())
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return log_dict

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log hp_metric at the epoch level."""
        pixel_level_loss_mean = torch.stack(
            [output["val/pixel_level_loss"] for output in outputs]
        ).mean()

        self.log("hp_metric", pixel_level_loss_mean)

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Union[float, int, Tensor]:
        """
        Run test step.

        Args:
            batch (Any): A batch of data from validation dataloader.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.

        Returns (Dict[str, Union[float, int, Tensor]]): A dictionary with outputs for further processing.

        """

        metrics = self.common_val_test_step(batch)

        log_dict = dict(
            list((f"test/{k}", v) for k, v in dataclasses.asdict(metrics).items())
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return metrics.pixel_level_loss

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
    def add_model_specific_args(
        parent: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
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
        parser.add_argument(
            "--initial_hp_metric_val",
            default=1e-2,
            type=float,
            help="The initial value for the `hp_metric`.",
        )
        parser.add_argument("--gen_in_channels", default=1, type=int)
        parser.add_argument("--gen_out_channels", default=1, type=int)
        parser.add_argument("--nf", default=64, type=int)
        parser.add_argument("--nb", default=23, type=int)
        parser.add_argument("--gc", default=32, type=int)
        parser.add_argument("--num_rrdb_blocks", default=16, type=int)
        parser.add_argument("--num_rrfdb_blocks", default=8, type=int)
        parser.add_argument("--use_elevation", default=False, type=bool)
        return parser


@dataclass
class MetricsResult:
    denormalized_mae: Union[np.ndarray, Tensor, float]
    denormalized_mse: Union[np.ndarray, Tensor, float]
    denormalized_rmse: Union[np.ndarray, Tensor, float]
    denormalized_r2: Union[np.ndarray, Tensor, float]
    pixel_level_loss: Union[np.ndarray, Tensor, float]
    mae: Union[np.ndarray, Tensor, float]
    mse: Union[np.ndarray, Tensor, float]
    psnr: Union[np.ndarray, Tensor, float]
    rmse: Union[np.ndarray, Tensor, float]
    ssim: Union[np.ndarray, Tensor, float]


@dataclass
class MetricsSimple:
    pixel_level_loss: Union[np.ndarray, Tensor, float]
    mae: Union[np.ndarray, Tensor, float]
    mse: Union[np.ndarray, Tensor, float]
    psnr: Union[np.ndarray, Tensor, float]
    rmse: Union[np.ndarray, Tensor, float]
    ssim: Union[np.ndarray, Tensor, float]
