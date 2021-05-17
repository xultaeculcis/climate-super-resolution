# -*- coding: utf-8 -*-
import logging
import math
import os
from typing import Optional, Any, List

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import DummyLogger
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid


class LogImagesCallback(Callback):
    def __init__(
        self,
        generator: str,
        experiment_name: str,
        use_elevation: bool,
    ):
        super(LogImagesCallback, self).__init__()
        self.generator = generator
        self.experiment_name = experiment_name
        self.use_elevation = use_elevation

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log a single batch of images from the validation set to monitor image quality progress."""
        if isinstance(pl_module.logger, DummyLogger):
            return

        if trainer.val_dataloaders:
            dl = trainer.val_dataloaders[0]
        else:
            logging.warning(
                "The validation dataloader not attached to the trainer. No images will be logged or saved."
            )
            return

        batch = next(iter(dl))

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
                nearest.to(pl_module.device)
                if pl_module.hparams.generator == "srcnn"
                else lr.to(pl_module.device)
            )

            if pl_module.hparams.use_elevation:
                x = torch.cat([nearest, elev], dim=1).to(pl_module.device)
            else:
                x = lr

            sr = pl_module(x)

            img_dir = os.path.join(pl_module.logger.log_dir, "images")

            # run only on first epoch
            if pl_module.current_epoch == 0 and pl_module.global_step == 0:
                os.makedirs(img_dir, exist_ok=True)
                names = [
                    "hr_images",
                    "lr_images",
                    "elevation",
                    "nearest_interpolation",
                    "cubic_interpolation",
                ]
                tensors = [hr, lr, elev, nearest, cubic]
                self._log_images(pl_module, img_dir, names, tensors)

            self._log_images(pl_module, img_dir, ["sr_images"], [sr])
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
    ):
        for name, tensor in zip(names, tensors):
            image_fp = os.path.join(
                img_dir,
                f"{name}-{self.experiment_name}-step={pl_module.global_step}.png",
            )
            self._save_tensor_batch_as_image(image_fp, tensor)
            self._log_images_from_file(pl_module, image_fp, name)

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
                self._batch_tensor_to_grid(mask_tensor[:nitems], nrow=nrows)
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

        cmap = matplotlib.cm.jet.copy()
        cmap.set_bad("black", 1.0)

        cols = ["HR", "Elevation", "Nearest", "Cubic", "SR"]
        if not self.use_elevation:
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

            if self.use_elevation:
                axes[i][1].imshow(elev_arr[i], cmap=cmap, vmin=0, vmax=1)

            hr_arr[i][mask[i]] = 0.0

            offset = 2 if self.use_elevation else 1
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
    def _batch_tensor_to_grid(
        tensor: Tensor, nrow: int = 8, padding: int = 2, pad_value: Any = False
    ) -> Tensor:
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