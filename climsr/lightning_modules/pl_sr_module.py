# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass
from math import ceil
from typing import Any, Tuple, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    psnr,
    ssim,
)
from torch import Tensor

import climsr.consts as consts
from climsr.data import normalization
from climsr.pre_processing.variable_mappings import world_clim_to_cruts_mapping
from climsr.models.rcan import RCAN
from climsr.data.normalization import MinMaxScaler, StandardScaler
from climsr.models.drln import DRLN
from climsr.models.esrgan import ESRGANGenerator
from climsr.models.rfb_esrgan import RFBESRGANGenerator
from climsr.models.srcnn import SRCNN


class SuperResolutionLightningModule(pl.LightningModule):
    """
    A base LightningModule for Super-Resolution tasks.
    """

    def __init__(self, **kwargs):
        super(SuperResolutionLightningModule, self).__init__()

        # store parameters
        self.save_hyperparameters(
            # model specific
            "generator",
            "gen_in_channels",
            "gen_out_channels",
            "disc_in_channels",
            "nf",
            "nb",
            "gc",
            "scale_factor",
            "num_rrdb_blocks",
            "num_rrfdb_blocks",
            "n_resgroups",
            "n_resblocks",
            "n_feats",
            "reduction",
            # training specific
            "precision",
            "initial_hp_metric_val",
            "max_lr",
            "batch_size",
            "max_epochs",
            "accumulate_grad_batches",
            # optimizer & scheduler specific
            "pct_start",
            "div_factor",
            "final_div_factor",
            "weight_decay",
            # loss specific
            "pixel_level_loss_factor",
            "perceptual_loss_factor",
            "adversarial_loss_factor",
            # data specific
            "use_mask_as_3rd_channel",
            "use_global_min_max",
            "use_elevation",
            "normalization_method",
            "normalization_range",
            "world_clim_variable",
        )

        # networks
        self.net_G = self.build_model()

        # metrics
        self.loss = (
            torch.nn.MSELoss()
            if self.hparams.generator == consts.models.srcnn
            else torch.nn.L1Loss()
        )

        # standardization
        self.stats = consts.cruts.statistics[
            world_clim_to_cruts_mapping[self.hparams.world_clim_variable]
        ]
        self.scaler = (
            StandardScaler(
                self.stats[consts.stats.mean],
                self.stats[consts.stats.std],
            )
            if self.hparams.normalization_method == normalization.zscore
            else MinMaxScaler(feature_range=self.hparams.normalization_range)
        )

    def build_model(self) -> nn.Module:
        if self.hparams.generator == consts.models.esrgan:
            generator = ESRGANGenerator(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                nf=self.hparams.nf,
                nb=self.hparams.nb,
                gc=self.hparams.gc,
                scale_factor=self.hparams.scale_factor,
            )
        elif self.hparams.generator == consts.models.drln:
            generator = DRLN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                scaling_factor=self.hparams.scale_factor,
            )
        elif self.hparams.generator == consts.models.rfb_esrgan:
            generator = RFBESRGANGenerator(
                upscale_factor=self.hparams.scale_factor,
                num_rrdb_blocks=self.hparams.num_rrdb_blocks,
                num_rrfdb_blocks=self.hparams.num_rrfdb_blocks,
            )
        elif self.hparams.generator == consts.models.srcnn:
            generator = SRCNN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
            )
        elif self.hparams.generator == consts.models.rcan:
            generator = RCAN(
                in_channels=self.hparams.gen_in_channels,
                out_channels=self.hparams.gen_out_channels,
                n_resgroups=self.hparams.n_resgroups,
                n_resblocks=self.hparams.n_resblocks,
                n_feats=self.hparams.n_feats,
                reduction=self.hparams.reduction,
            )
        else:
            raise ValueError(
                f"Specified generator '{self.hparams.generator}' is not supported"
            )
        return generator

    def forward(
        self, x: Tensor, elevation: Tensor = None, mask: Tensor = None
    ) -> Tensor:
        if self.hparams.generator == consts.models.srcnn:
            return self.net_G(x)
        else:
            return self.net_G(x, elevation, mask)

    def common_step(self, batch: Any) -> Tuple[Tensor, Tensor]:
        """
        Runs common step during train/val/test pass.

        Args:
            batch (Any): A batch of data from a dataloader.

        Returns (Tuple[Tensor, Tensor]): A tuple with HR image and SR image.

        """
        lr, hr, elev, mask = (
            batch[consts.batch_items.lr],
            batch[consts.batch_items.hr],
            batch[consts.batch_items.elevation],
            batch[consts.batch_items.mask],
        )

        sr = self(lr, elev, mask)

        return hr, sr

    def common_val_test_step(self, batch: Any) -> "MetricsResult":
        """
        Rus common validation and test steps.

        Args:
            batch (Any): The batch of data.

        Returns (MetricsResult): The MetricsResult.

        """
        original = batch[consts.batch_items.original_data]
        mask = batch[consts.batch_items.mask_np].cpu().numpy()
        max_vals = batch[consts.batch_items.max].cpu().numpy()
        min_vals = batch[consts.batch_items.min].cpu().numpy()

        hr, sr = self.common_step(batch)

        metrics = self.compute_metrics_common(hr, sr)

        denormalized_mae = []
        denormalized_mse = []
        denormalized_rmse = []
        denormalized_r2 = []

        original = original.squeeze(1).cpu().numpy()
        squeezed_sr = sr.squeeze(1).cpu().numpy()

        for i in range(squeezed_sr.shape[0]):
            # to numpy
            i_original = original[i]
            i_sr = squeezed_sr[i]

            # denormalize/destandardize
            i_sr = (
                self.scaler.denormalize(i_sr)
                if self.hparams.normalization_method == normalization.zscore
                else self.scaler.denormalize(i_sr, min_vals[i], max_vals[i])
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
            sr=sr,
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

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        train_steps = ceil(batches / effective_accum) * self.trainer.max_epochs

        return train_steps

    @staticmethod
    def add_model_specific_args(
        parent: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            parents=[parent], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--max_lr",
            default=1e-5,
            type=float,
            help="The max learning rate for the 1Cycle LR Scheduler",
        )
        parser.add_argument(
            "--pct_start",
            default=0.03,
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
            default=3e-1,
            type=float,
            help="The initial value for the `hp_metric`.",
        )
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--pixel_level_loss_factor", default=1e-2, type=float)
        parser.add_argument("--perceptual_loss_factor", default=1.0, type=float)
        parser.add_argument("--adversarial_loss_factor", default=5e-3, type=float)
        parser.add_argument("--gen_in_channels", default=3, type=int)
        parser.add_argument("--gen_out_channels", default=1, type=int)
        parser.add_argument("--disc_in_channels", default=1, type=int)
        # esrgan specific
        parser.add_argument("--nf", default=64, type=int)
        parser.add_argument("--nb", default=11, type=int)
        parser.add_argument("--gc", default=16, type=int)
        # rfbesrgan specific
        parser.add_argument("--num_rrdb_blocks", default=16, type=int)
        parser.add_argument("--num_rrfdb_blocks", default=8, type=int)
        # rcan specific
        parser.add_argument("--n_resgroups", default=10, type=int)
        parser.add_argument("--n_resblocks", default=20, type=int)
        parser.add_argument("--reduction", default=16, type=int)
        parser.add_argument("--n_feats", default=64, type=int)
        # other
        parser.add_argument("--use_elevation", default=True, type=bool)
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
    sr: Optional[Union[np.ndarray, Tensor, float]]


@dataclass
class MetricsSimple:
    pixel_level_loss: Union[np.ndarray, Tensor, float]
    mae: Union[np.ndarray, Tensor, float]
    mse: Union[np.ndarray, Tensor, float]
    psnr: Union[np.ndarray, Tensor, float]
    rmse: Union[np.ndarray, Tensor, float]
    ssim: Union[np.ndarray, Tensor, float]
