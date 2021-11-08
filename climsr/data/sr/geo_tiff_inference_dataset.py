# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms as transforms

import climsr.consts as consts
from climsr.data.normalization import MinMaxScaler, StandardScaler
from climsr.data.sr.climate_dataset_base import ClimateDatasetBase


class GeoTiffInferenceDataset(ClimateDatasetBase):
    def __init__(
        self,
        tiff_dir: str,
        tiff_df: pd.DataFrame,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        variable: str,
        hr_size: Optional[int] = 452,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: Dict[str, Dict[str, float]] = None,
        normalize_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        use_elevation: Optional[bool] = True,
        use_mask: Optional[bool] = True,
        use_global_min_max: Optional[bool] = True,
    ):
        super().__init__(
            generator_type=generator_type,
            variable=variable,
            scaling_factor=scaling_factor,
            normalize=normalize,
            standardize=standardize,
            standardize_stats=standardize_stats,
            normalize_range=normalize_range,
        )

        self.tiff_dir = tiff_dir
        self.tiffs = glob(f"{tiff_dir}/*.tif")
        self.tiff_df = tiff_df.set_index(consts.datasets_and_preprocessing.filename, drop=True)
        self.use_elevation = use_elevation
        self.use_mask = use_mask
        self.use_global_min_max = use_global_min_max
        self.elevation_file = elevation_file
        self.land_mask_file = land_mask_file
        self.hr_size = hr_size

        if self.standardize:
            self.scaler = StandardScaler(
                self.standardize_stats[self.variable][consts.stats.mean],
                self.standardize_stats[self.variable][consts.stats.std],
            )
            self.elevation_scaler = StandardScaler(
                standardize_stats[consts.cruts.elev][consts.stats.mean],
                standardize_stats[consts.cruts.elev][consts.stats.std],
            )
        else:
            self.scaler = MinMaxScaler(feature_range=normalize_range)
            self.elevation_scaler = MinMaxScaler(feature_range=normalize_range)

        self.to_tensor = transforms.ToTensor()

        self.land_mask_np = ~np.isnan(np.array(Image.open(land_mask_file), dtype=np.float32))
        self.land_mask_tensor = self.to_tensor(self.land_mask_np)

        elevation_arr = np.array(Image.open(elevation_file), dtype=np.float32)
        elevation_arr = np.where(self.land_mask_np, elevation_arr, np.nan)  # mask Antarctica
        elevation_arr = self.elevation_scaler.normalize(
            elevation_arr,
            missing_indicator=consts.world_clim.elevation_missing_indicator,
        )

        self.to_tensor = transforms.ToTensor()
        self.resize = A.Resize(
            height=self.hr_size // self.scaling_factor,
            width=self.hr_size // self.scaling_factor,
            interpolation=cv2.INTER_NEAREST,
            always_apply=True,
            p=1.0,
        )
        self.upscale = A.Resize(height=self.hr_size, width=self.hr_size, interpolation=cv2.INTER_NEAREST)

        self.elevation_data = self.to_tensor(elevation_arr)
        self.elevation_lr = self.to_tensor(self.resize(image=elevation_arr)["image"])
        self.mask_lr = self.to_tensor(self.resize(image=self.land_mask_np.astype(int))["image"].astype(bool))

    def _concat_if_needed(self, img_lr: Tensor, img_sr_nearest: Tensor) -> Tensor:
        """Concatenates elevation and/or mask data as 2nd and/or 3rd channel to LR raster data."""
        if self.generator_type == consts.models.srcnn:
            img_lr = img_sr_nearest

        if self.use_elevation:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_lr, self.elevation_data], dim=0)
            else:
                img_lr = torch.cat([img_lr, self.elevation_lr], dim=0)

        if self.use_mask:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_lr, self.land_mask_tensor], dim=0)
            else:
                mask_lr = self.to_tensor(self.resize(image=self.land_mask_np.astype(np.float32))["image"])
                img_lr = torch.cat([img_lr, mask_lr], dim=0)

        return img_lr

    def _common_to_tensor(self, img_lr: np.ndarray) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        img_sr_nearest = self.to_tensor(self.upscale(image=img_lr)["image"])

        img_lr = self.to_tensor(img_lr)

        img_lr = self._concat_if_needed(img_lr, img_sr_nearest)

        return (
            img_lr,
            self.elevation_data,
            self.elevation_lr,
            self.land_mask_np,
            self.land_mask_tensor,
            img_sr_nearest,
        )

    def _get_inference_sample(self, img_lr: np.ndarray) -> Dict[str, Union[Tensor, list]]:
        (
            img_lr,
            img_elev,
            img_elev_lr,
            mask,
            mask_tensor,
            img_sr_nearest,
        ) = self._common_to_tensor(img_lr)

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.elevation: img_elev,
            consts.batch_items.elevation_lr: img_elev_lr,
            consts.batch_items.nearest: img_sr_nearest,
            consts.batch_items.mask: mask_tensor,
            consts.batch_items.mask_np: mask,
        }

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, list]]:
        file_path = self.tiffs[index]
        file_name = os.path.basename(file_path)
        row = self.tiff_df.loc[file_name]
        min = row[consts.stats.min] if not self.use_global_min_max else row[consts.stats.global_min]
        max = row[consts.stats.max] if not self.use_global_min_max else row[consts.stats.global_max]

        # original, lr
        with Image.open(file_path) as img:
            original_image = np.flipud(np.array(img))
            img_lr = original_image.copy()

        # normalize/standardize
        if self.normalize:
            img_lr = self.scaler.normalize(img_lr, min, max)
        if self.standardize:
            img_lr = self.scaler.normalize(arr=img_lr)

        item = self._get_inference_sample(img_lr)
        item[consts.batch_items.filename] = file_name
        item[consts.batch_items.min] = min
        item[consts.batch_items.max] = max

        return item

    def __len__(self):
        return len(self.tiffs)
