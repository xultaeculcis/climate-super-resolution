# -*- coding: utf-8 -*-
import logging
import math
import os
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

import climsr.consts as consts
from climsr.core.task import TaskSuperResolutionModule
from climsr.data import normalization
from climsr.data.normalization import MinMaxScaler, StandardScaler

MAX_ITEMS = 88
MAX_MINI_BATCHES = 12

cmap_jet = matplotlib.cm.get_cmap("jet").copy()
cmap_gray = matplotlib.cm.get_cmap("gray").copy()
cmap_inferno = matplotlib.cm.get_cmap("inferno").copy()
cmap_jet.set_bad("black", 1.0)
cmap_gray.set_bad("black", 1.0)
cmap_inferno.set_bad("black", 1.0)


class LogImagesCallback(Callback):
    def __init__(
        self,
        generator: str,
        experiment_name: str,
        use_elevation: bool,
        world_clim_variable: str,
        normalization_method: Optional[str] = normalization.minmax,
        normalization_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ):
        super(LogImagesCallback, self).__init__()
        self.generator = generator
        self.experiment_name = experiment_name
        self.use_elevation = use_elevation
        self.standardize = normalization_method == normalization.zscore
        self.world_clim_variable = world_clim_variable
        self.normalization_range = normalization_range

    def on_validation_end(self, trainer: Trainer, pl_module: TaskSuperResolutionModule) -> None:
        """Log a single batch of images from the validation set to monitor image quality progress."""
        if isinstance(pl_module.logger, DummyLogger):
            return

        rank_zero_info("Saving generated images")

        if trainer.val_dataloaders:
            dl = trainer.val_dataloaders[0]
        else:
            logging.warning("The validation dataloader not attached to the trainer. No images will be logged or saved.")
            return

        batch = next(iter(dl))

        with torch.no_grad():
            lr, hr, original, nearest, cubic, elev, mask, mins, maxes = (
                batch[consts.batch_items.lr].to(pl_module.device),
                batch[consts.batch_items.hr].to(pl_module.device),
                batch[consts.batch_items.original_data].to(pl_module.device),
                batch[consts.batch_items.nearest].to(pl_module.device),
                batch[consts.batch_items.cubic],
                batch[consts.batch_items.elevation].to(pl_module.device),
                batch[consts.batch_items.mask].to(pl_module.device),
                batch[consts.batch_items.min],
                batch[consts.batch_items.max],
            )

            sr = pl_module(lr, elev, mask)
            error = torch.abs(sr - hr)

            img_dir = os.path.join(pl_module.logger.experiment[0].log_dir, "images")

            mask = ~mask.squeeze(1).bool()

            # run only on first epoch
            if pl_module.current_epoch == 0 and pl_module.global_step == 0:
                os.makedirs(img_dir, exist_ok=True)
                names = [
                    "hr_images",
                    consts.batch_items.elevation,
                    "nearest_interpolation",
                    "cubic_interpolation",
                ]
                tensors = [hr, elev, nearest, cubic]
                self._log_images(pl_module, img_dir, names, tensors, mask)

            self._log_images(pl_module, img_dir, ["sr_images", consts.batch_items.error], [sr, error], mask)
            self._save_fig(
                hr=hr,
                sr_nearest=nearest,
                sr_cubic=cubic,
                sr=sr,
                elev=elev,
                mask=mask,
                error=error,
                original=original,
                img_dir=img_dir,
                current_epoch=pl_module.current_epoch,
                global_step=pl_module.global_step,
                stats=pl_module.stats,
                mins=mins,
                maxes=maxes,
            )

    def _log_images(
        self,
        pl_module: TaskSuperResolutionModule,
        img_dir: str,
        names: List[str],
        tensors: List[Tensor],
        mask: Tensor,
    ):
        for name, tensor in zip(names, tensors):
            os.makedirs(img_dir, exist_ok=True)
            image_fp = os.path.join(
                img_dir,
                f"{name}-{self.experiment_name}-epoch={pl_module.current_epoch}-step={pl_module.global_step}.png",
            )

            if name == consts.batch_items.elevation:
                value_range = (
                    pl_module.stats.at[consts.world_clim.elev, consts.stats.normalized_min],
                    pl_module.stats.at[consts.world_clim.elev, consts.stats.normalized_max],
                )
            else:
                value_range = (
                    pl_module.stats.at[self.world_clim_variable, consts.stats.normalized_min],
                    pl_module.stats.at[self.world_clim_variable, consts.stats.normalized_max],
                )

            self._save_tensor_batch_as_image(
                out_path=image_fp,
                images_tensor=tensor,
                mask_tensor=mask,
                normalize=self.standardize,
                value_range=value_range if name != consts.batch_items.elevation else None,
            )
            self._log_images_from_file(pl_module, image_fp, name)

    def _save_tensor_batch_as_image(
        self,
        out_path: str,
        images_tensor: Tensor,
        mask_tensor: Optional[Tensor] = None,
        normalize: Optional[bool] = False,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Save a given Tensor into an image file.

        Args:
            out_path (str): The output filename.
            images_tensor (Tensor): The tensor with images.
            mask_tensor (Optional[Tensor]): The optional tensor with masks.
            normalize (Optional[bool]): If, True then the images will be normalized according to the value_range
                parameter, False otherwise. False by default.
            value_range (Optional[Tuple[float, float]]): The optional min, max value range for the image.
                None by default.
        """

        if mask_tensor is not None:
            assert mask_tensor.shape[0] == images_tensor.shape[0], (
                "Images tensor has to have the same number of elements as mask tensor, the shapes were: "
                f"{images_tensor.shape} (images) and {mask_tensor.shape} (masks)."
            )

        # ensure we only plot max of 88 images
        nitems = np.minimum(MAX_ITEMS, images_tensor.shape[0])
        nrows = 8
        ncols = nitems // nrows

        img_grid = (
            make_grid(
                images_tensor[:nitems],
                nrow=nrows,
                normalize=normalize,
                value_range=value_range,
            )[0]
            .to("cpu", torch.float32)
            .numpy()
        )  # select only single channel since we deal with 2D data anyway

        if mask_tensor is not None:
            mask_grid = self._batch_tensor_to_grid(mask_tensor[:nitems], nrow=nrows).squeeze(0).to("cpu").numpy()
            img_grid[mask_grid] = np.nan

        plt.figure(figsize=(2 * nrows, 2 * ncols))
        plt.imshow(
            img_grid, cmap=(cmap_inferno if os.path.basename(out_path).startswith(consts.cruts.elev) else cmap_jet), aspect="auto"
        )
        plt.axis("off")

        plt.savefig(out_path, bbox_inches="tight")

    def _save_fig(
        self,
        hr: Tensor,
        sr_nearest: Tensor,
        sr_cubic: Tensor,
        sr: Tensor,
        elev: Tensor,
        mask: Tensor,
        error: Tensor,
        original: Tensor,
        img_dir: str,
        current_epoch: int,
        global_step: int,
        stats: pd.DataFrame,
        mins: Tensor,
        maxes: Tensor,
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
            error (Tensor): The prediction error image tensor.
            original (Tensor): The original (non-normalized) image tensor.
            img_dir (str): The output dir.
            current_epoch (int): The number of current epoch.
            global_step (int): The number of the global step.
            mins (Tensor): A tensor with minimum pix value. Needed for de-normalization if min-max scaling was used.
            maxes (Tensor): A tensor with maximum pix value. Needed for de-normalization if min-max scaling was used.
            items (Optional[int]): Optional number of items from batch to save. 16 by default.

        """

        cols = [
            consts.plotting.hr,
            consts.plotting.elevation,
            consts.plotting.mask,
            consts.plotting.nearest_interpolation,
            consts.plotting.cubic_interpolation,
            consts.plotting.sr,
            consts.plotting.error,
        ]
        if not self.use_elevation:
            cols.remove(consts.plotting.elevation)

        ncols = len(cols)

        nearest = sr_nearest.squeeze(1).cpu().numpy()
        cubic = sr_cubic.squeeze(1).cpu().numpy()
        hr = hr.squeeze(1).cpu().numpy()
        original = original.squeeze(1).cpu().numpy()
        elev = elev.squeeze(1).cpu().numpy()
        sr = sr.squeeze(1).cpu().numpy()
        mask = mask.squeeze(1).cpu().numpy()
        error = error.squeeze(1).cpu().numpy()

        batches = np.min([hr.shape[0] // items, MAX_MINI_BATCHES])

        for mini_batch_idx in tqdm(range(batches), desc="Saving figures from mini-batches"):
            fig, axes = plt.subplots(
                nrows=items,
                ncols=ncols,
                figsize=(10, 2.2 * items),
                sharey="all",
                constrained_layout=True,
            )

            for ax, col in zip(axes[0], cols):
                ax.set_title(col)

            mini_batch_offset_start = mini_batch_idx * items
            mini_batch_offset_end = mini_batch_offset_start + items

            hr_arr = hr[mini_batch_offset_start:mini_batch_offset_end]
            original_arr = original[mini_batch_offset_start:mini_batch_offset_end]
            elev_arr = elev[mini_batch_offset_start:mini_batch_offset_end]
            nearest_arr = nearest[mini_batch_offset_start:mini_batch_offset_end]
            cubic_arr = cubic[mini_batch_offset_start:mini_batch_offset_end]
            sr_arr = sr[mini_batch_offset_start:mini_batch_offset_end]
            mask_arr = mask[mini_batch_offset_start:mini_batch_offset_end]
            error_arr = error[mini_batch_offset_start:mini_batch_offset_end]
            mins_arr = mins[mini_batch_offset_start:mini_batch_offset_end]
            maxes_arr = maxes[mini_batch_offset_start:mini_batch_offset_end]

            arrs = [nearest_arr, cubic_arr, sr_arr]

            if self.standardize:
                scaler = StandardScaler(
                    mean=stats.at[
                        consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.world_clim_variable], consts.stats.mean
                    ],
                    std=stats.at[
                        consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.world_clim_variable], consts.stats.std
                    ],
                    nan_substitution=stats.at[
                        consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.world_clim_variable],
                        consts.stats.normalized_min,
                    ],
                )
            else:
                scaler = MinMaxScaler(feature_range=self.normalization_range)

            hr_arr[mask_arr] = np.nan
            elev_arr[mask_arr] = np.nan
            nearest_arr[mask_arr] = np.nan
            cubic_arr[mask_arr] = np.nan
            sr_arr[mask_arr] = np.nan
            error_arr[mask_arr] = np.nan

            for i in range(items):
                axes[i][0].imshow(
                    hr_arr[i],
                    cmap=cmap_jet,
                    vmin=stats.at[consts.stats.normalized_min] if self.standardize else self.normalization_range[0],
                    vmax=stats[consts.stats.normalized_max] if self.standardize else self.normalization_range[1],
                )
                axes[i][0].set_xlabel("MAE/RMSE")

                if self.use_elevation:
                    axes[i][1].imshow(elev_arr[i], cmap=cmap_inferno)
                    axes[i][1].set_xlabel("-/-")

                axes[i][-1].imshow(error_arr[i], cmap=cmap_jet)
                axes[i][-1].set_xlabel("-/-")

                mask_col_idx = 2 if self.use_elevation else 1
                axes[i][mask_col_idx].imshow(~mask_arr[i], cmap=cmap_gray, vmin=0.0, vmax=1.0)
                axes[i][mask_col_idx].set_xlabel("-/-")

                offset = 3 if self.use_elevation else 2
                for idx, arr in enumerate(arrs):
                    axes[i][offset + idx].imshow(
                        arr[i],
                        cmap=cmap_jet,
                        vmin=(
                            stats.at[self.world_clim_variable, consts.stats.normalized_min]
                            if self.standardize
                            else self.normalization_range[0]
                        ),
                        vmax=(
                            stats.at[self.world_clim_variable, consts.stats.normalized_max]
                            if self.standardize
                            else self.normalization_range[1]
                        ),
                    )

                    denormalized_arr = (
                        scaler.denormalize(arr[i])
                        if self.standardize
                        else scaler.denormalize(arr[i], mins_arr[i].detach().cpu().item(), maxes_arr[i].detach().cpu().item())
                    )

                    original_arr[mask_arr] = 0.0
                    denormalized_arr[mask_arr[i]] = 0.0

                    diff = original_arr[i] - denormalized_arr
                    rmse_value = np.sqrt(np.nanmean(diff ** 2))
                    mae_value = np.nanmean(np.absolute(diff))

                    axes[i][offset + idx].set_xlabel(f"{mae_value:.3f}/{rmse_value:.3f}")

                for j in range(ncols):
                    axes[i][j].xaxis.set_ticklabels([])
                    axes[i][j].xaxis.set_ticklabels([])

            plt.xticks([])
            plt.yticks([])

            fig.suptitle(
                f"Validation batch, epoch={current_epoch}, step={global_step}",
                fontsize=16,
            )

            plt.savefig(
                os.path.join(
                    img_dir,
                    f"figure-{self.experiment_name}-epoch={current_epoch}-step={global_step}-{mini_batch_idx}.png",
                )
            )

    @staticmethod
    def _log_images_from_file(pl_module: LightningModule, fp: str, name: str) -> None:
        img = Image.open(fp).convert("RGB")
        tensor = ToTensor()(img).unsqueeze(0)  # make it NxCxHxW
        pl_module.logger.experiment[0].add_images(
            name,
            tensor,
            pl_module.global_step,
        )

    @staticmethod
    def _batch_tensor_to_grid(tensor: Tensor, nrow: int = 8, padding: int = 2, pad_value: Any = False) -> Tensor:
        """
        Make the mini-batch of images into a grid

        Args:
            tensor (Tensor): The tensor.
            nrow (int): Number of rows
            padding (int): The padding.
            pad_value (Any): The padding value.

        Returns (Tensor): The grid tensor.

        """
        nmaps = tensor.shape[0]
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
        grid = tensor.new_full((1, height * ymaps + padding, width * xmaps + padding), pad_value)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid.narrow(1, y * height + padding, height - padding).narrow(2, x * width + padding, width - padding).copy_(
                    tensor[k]
                )
                k = k + 1

        return grid
