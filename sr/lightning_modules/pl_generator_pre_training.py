# -*- coding: utf-8 -*-
import argparse
import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.metrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    psnr,
    ssim,
)
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

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
        self.log_images()

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

    def log_images(self) -> None:
        """Log a single batch of images from the validation set to monitor image quality progress."""
        if isinstance(self.logger, DummyLogger):
            return

        batch = next(iter(self.trainer.datamodule.val_dataloader()))

        with torch.no_grad():
            lr, hr, nearest, cubic, elev, mask = (
                batch["lr"],
                batch["hr"],
                batch["nearest"],
                batch["cubic"],
                batch["elevation"],
                batch["mask"],
            )

            lr = (
                nearest.to(self.device)
                if self.hparams.generator == "srcnn"
                else lr.to(self.device)
            )

            if self.hparams.use_elevation:
                x = torch.cat([nearest, elev], dim=1).to(self.device)
            else:
                x = lr

            sr = self(x)

            img_dir = os.path.join(self.logger.log_dir, "images")

            # run only on first epoch
            if self.current_epoch == 0:
                os.makedirs(img_dir, exist_ok=True)
                names = [
                    "hr_images",
                    "lr_images",
                    "elevation",
                    "nearest_interpolation",
                    "cubic_interpolation",
                ]
                tensors = [hr, lr, elev, nearest, cubic]
                self._log_images(img_dir, names, tensors)

            self._log_images(img_dir, ["sr_images"], [sr])
            self.save_fig(
                hr=hr,
                sr_nearest=nearest,
                sr_cubic=cubic,
                sr=sr,
                elev=elev,
                mask=mask,
                img_dir=img_dir,
            )

    def _log_images(self, img_dir, names, tensors):
        for name, tensor in zip(names, tensors):
            image_fp = os.path.join(
                img_dir,
                f"{name}-{self.hparams.experiment_name}-step={self.global_step}.png",
            )
            self._save_tensor_batch_as_image(image_fp, tensor)
            self._log_images_from_file(image_fp, name)

    def _log_images_from_file(self, fp: str, name: str) -> None:
        img = Image.open(fp).convert("RGB")
        tensor = ToTensor()(img).unsqueeze(0)  # make it NxCxHxW
        self.logger.experiment.add_images(
            name,
            tensor,
            self.global_step,
        )

    def _save_tensor_batch_as_image(
        self, out_path: str, images_tensor: Tensor, mask_tensor: Optional[Tensor] = None
    ) -> None:
        """Save a given Tensor into an image file.

        Args:
            out_path (str): The output filename.
            images_tensor (Tensor): The tensor with images.
            mask_tensor (Optional[Tensor]): The optional tensor with masks.
        """

        if mask_tensor is not None:
            assert mask_tensor.shape[0] == images_tensor.shape[0], (
                "Images tensor has to have the same number of elements as mask tensor, the shapes were: "
                f"{images_tensor.shape} (images) and {mask_tensor.shape} (masks)."
            )

        # ensure we only plot max of 88 images
        nitems = np.minimum(88, images_tensor.shape[0])
        nrows = 8
        ncols = nitems // nrows

        img_grid = (
            make_grid(images_tensor[:nitems], nrow=nrows)[0]
            .to("cpu", torch.float32)
            .numpy()
        )  # select only single channel since we deal with 2D data anyway

        if mask_tensor is not None:
            mask_grid = (
                self.batch_tensor_to_grid(mask_tensor[:nitems], nrow=nrows)
                .squeeze(0)
                .to("cpu")
                .numpy()
            )
            img_grid[mask_grid] = np.nan

        cmap = matplotlib.cm.jet.copy()
        cmap.set_bad("black", 1.0)
        plt.figure(figsize=(2 * nrows, 2 * ncols))
        plt.imshow(img_grid, cmap=cmap, aspect="auto")
        plt.axis("off")

        plt.savefig(out_path, bbox_inches="tight")

    def save_fig(
        self,
        hr: Tensor,
        sr_nearest: Tensor,
        sr_cubic: Tensor,
        sr: Tensor,
        elev: Tensor,
        mask: Tensor,
        img_dir: str,
        items: Optional[int] = 16,
    ) -> None:
        """
        Save batch data as plot.

        Args:
            hr (Tensor): The HR image tensor.
            sr_nearest (Tensor):The Nearest Interpolation image tensor.
            sr_cubic (Tensor):The Cubic Interpolation image tensor.
            sr (Tensor): The SR image tensor.
            elev (Tensor): The Elevation image tensor.
            mask (Tensor): The Land Mask image tensor.
            img_dir (str): The output dir.
            items (Optional[int]): Optional number of items from batch to save. 16 by default.

        """
        ncols = 5 if self.hparams.use_elevation else 4
        fig, axes = plt.subplots(
            nrows=items,
            ncols=ncols,
            figsize=(10, 2.2 * items),
            sharey=True,
            constrained_layout=True,
        )

        cmap = matplotlib.cm.jet.copy()
        cmap.set_bad("black", 1.0)

        cols = ["HR", "Elevation", "Nearest", "Cubic", "SR"]
        if not self.hparams.use_elevation:
            cols.remove("Elevation")

        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        nearest_arr = sr_nearest.squeeze(1).cpu().numpy()
        cubic_arr = sr_cubic.squeeze(1).cpu().numpy()
        hr_arr = hr.squeeze(1).cpu().numpy()
        elev_arr = elev.squeeze(1).cpu().numpy()
        sr_arr = sr.squeeze(1).cpu().numpy()

        for i in range(items):
            hr_arr[i][mask[i]] = np.nan
            elev_arr[i][mask[i]] = np.nan
            nearest_arr[i][mask[i]] = np.nan
            cubic_arr[i][mask[i]] = np.nan
            sr_arr[i][mask[i]] = np.nan

            axes[i][0].imshow(hr_arr[i], cmap=cmap, vmin=0, vmax=1)
            axes[i][0].set_xlabel("MAE/RMSE")

            if self.hparams.use_elevation:
                axes[i][1].imshow(elev_arr[i], cmap=cmap, vmin=0, vmax=1)

            hr_arr[i][mask[i]] = 0.0

            offset = 2 if self.hparams.use_elevation else 1
            for idx, arr in enumerate([nearest_arr, cubic_arr, sr_arr]):
                axes[i][offset + idx].imshow(arr[i], cmap=cmap, vmin=0, vmax=1)
                arr[i][mask[i]] = 0.0
                diff = hr - arr
                mae = np.absolute(diff).mean()
                rmse = np.sqrt((diff ** 2).mean())
                axes[i][offset + idx].set_xlabel(f"{rmse:.3f}/{mae:.3f}")

            for j in range(ncols):
                axes[i][j].xaxis.set_ticklabels([])
                axes[i][j].xaxis.set_ticklabels([])

        plt.xticks([])
        plt.yticks([])

        fig.suptitle(
            f"Validation batch, epoch={self.current_epoch}, step={self.global_step}",
            fontsize=16,
        )

        plt.savefig(
            os.path.join(
                img_dir,
                f"figure-{self.hparams.experiment_name}-epoch={self.current_epoch}-step={self.global_step}.png",
            )
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
    def batch_tensor_to_grid(
        tensor: Tensor, nrow: int = 8, padding: int = 2, pad_value: Any = False
    ):
        """
        Make the mini-batch of images into a grid

        Args:
            tensor (Tensor): The tensor.
            nrow (int): Number of rows
            padding (int): The padding.
            pad_value (Any): The padding value.

        Returns:

        """
        nmaps = tensor.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = int(tensor.size(1) + padding), int(tensor.size(2) + padding)
        grid = tensor.new_full(
            (1, height * ymaps + padding, width * xmaps + padding), pad_value
        )
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid.narrow(1, y * height + padding, height - padding).narrow(
                    2, x * width + padding, width - padding
                ).copy_(tensor[k])
                k = k + 1

        return grid

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
