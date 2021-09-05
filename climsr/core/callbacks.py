# -*- coding: utf-8 -*-
import logging
import math
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import DummyLogger
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import climsr.consts as consts
from climsr.consts.datasets_and_preprocessing import world_clim_to_cruts_mapping
from climsr.data import normalization

MAX_ITEMS = 88


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
        self.stats = consts.cruts.statistics[world_clim_to_cruts_mapping[self.world_clim_variable]]
        self.normalization_range = normalization_range

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log a single batch of images from the validation set to monitor image quality progress."""
        if isinstance(pl_module.logger, DummyLogger):
            return

        if trainer.val_dataloaders:
            dl = trainer.val_dataloaders[0]
        else:
            logging.warning("The validation dataloader not attached to the trainer. No images will be logged or saved.")
            return

        batch = next(iter(dl))

        with torch.no_grad():
            lr, hr, nearest, cubic, elev, mask, mask_tensor = (
                batch[consts.batch_items.lr].to(pl_module.device),
                batch[consts.batch_items.hr].to(pl_module.device),
                batch[consts.batch_items.nearest].to(pl_module.device),
                batch[consts.batch_items.cubic],
                batch[consts.batch_items.elevation].to(pl_module.device),
                batch[consts.batch_items.mask_np],
                batch[consts.batch_items.mask].to(pl_module.device),
            )

            sr = pl_module(lr, elev, mask_tensor)

            img_dir = os.path.join(pl_module.logger.log_dir, "images")

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

            self._log_images(pl_module, img_dir, ["sr_images"], [sr], mask)
            self._save_fig(
                hr=hr,
                sr_nearest=nearest,
                sr_cubic=cubic,
                sr=sr,
                elev=elev,
                mask=mask,
                img_dir=img_dir,
                current_epoch=pl_module.current_epoch,
                global_step=pl_module.global_step,
            )

    def _log_images(
        self,
        pl_module: LightningModule,
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
                    consts.cruts.statistics[consts.cruts.elev][consts.stats.normalized_min],
                    consts.cruts.statistics[consts.cruts.elev][consts.stats.normalized_max],
                )
            else:
                value_range = (
                    self.stats[consts.stats.normalized_min],
                    self.stats[consts.stats.normalized_max],
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

        cmap = deepcopy(matplotlib.cm.jet)
        cmap.set_bad("black", 1.0)
        plt.figure(figsize=(2 * nrows, 2 * ncols))
        plt.imshow(img_grid, cmap=cmap, aspect="auto")
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
        img_dir: str,
        current_epoch: int,
        global_step: int,
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
            current_epoch (int): The number of current epoch.
            global_step (int): The number of the global step.
            items (Optional[int]): Optional number of items from batch to save. 16 by default.

        """
        ncols = 5 if self.use_elevation else 4
        fig, axes = plt.subplots(
            nrows=items,
            ncols=ncols,
            figsize=(10, 2.2 * items),
            sharey=True,
            constrained_layout=True,
        )

        cmap = deepcopy(matplotlib.cm.jet)
        cmap.set_bad("black", 1.0)

        cols = [
            consts.plotting.hr,
            consts.plotting.elevation,
            consts.plotting.nearest_interpolation,
            consts.plotting.cubic_interpolation,
            consts.plotting.sr,
        ]
        if not self.use_elevation:
            cols.remove(consts.batch_items.elevation)

        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        nearest_arr = sr_nearest.squeeze(1).cpu().numpy()
        cubic_arr = sr_cubic.squeeze(1).cpu().numpy()
        hr_arr = hr.squeeze(1).cpu().numpy()
        elev_arr = elev.squeeze(1).cpu().numpy()
        sr_arr = sr.squeeze(1).cpu().numpy()

        hr_arr = hr_arr[:items]
        elev_arr = elev_arr[:items]
        nearest_arr = nearest_arr[:items]
        cubic_arr = cubic_arr[:items]
        sr_arr = sr_arr[:items]
        mask = mask[:items]

        hr_arr[mask] = 0.0
        elev_arr[mask] = 0.0
        nearest_arr[mask] = 0.0
        cubic_arr[mask] = 0.0
        sr_arr[mask] = 0.0

        arrs = [nearest_arr, cubic_arr, sr_arr]
        maes = []
        rmses = []
        for arr in arrs:
            diff = hr_arr - arr
            rmses.append(np.sqrt(np.mean(diff ** 2, axis=(1, 2))))
            maes.append(np.mean(np.absolute(diff), axis=(1, 2)))

        hr_arr[mask] = np.nan
        elev_arr[mask] = np.nan
        nearest_arr[mask] = np.nan
        cubic_arr[mask] = np.nan
        sr_arr[mask] = np.nan

        for i in range(items):
            axes[i][0].imshow(
                hr_arr[i],
                cmap=cmap,
                vmin=self.stats[consts.stats.normalized_min] if self.standardize else self.normalization_range[0],
                vmax=self.stats[consts.stats.normalized_max] if self.standardize else self.normalization_range[1],
            )
            axes[i][0].set_xlabel("MAE/RMSE")

            if self.use_elevation:
                axes[i][1].imshow(elev_arr[i], cmap=cmap)
                axes[i][1].set_xlabel("-/-")

            offset = 2 if self.use_elevation else 1
            for idx, arr in enumerate(arrs):
                axes[i][offset + idx].imshow(
                    arr[i],
                    cmap=cmap,
                    vmin=self.stats[consts.stats.normalized_min] if self.standardize else self.normalization_range[0],
                    vmax=self.stats[consts.stats.normalized_max] if self.standardize else self.normalization_range[1],
                )
                mae_value = maes[idx][i]
                rmse_value = rmses[idx][i]
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
                f"figure-{self.experiment_name}-epoch={current_epoch}-step={global_step}.png",
            )
        )

    @staticmethod
    def _log_images_from_file(pl_module: LightningModule, fp: str, name: str) -> None:
        img = Image.open(fp).convert("RGB")
        tensor = ToTensor()(img).unsqueeze(0)  # make it NxCxHxW
        pl_module.logger.experiment.add_images(
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
