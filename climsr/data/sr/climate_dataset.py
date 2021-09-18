# -*- coding: utf-8 -*-
import random
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


class ClimateDataset(ClimateDatasetBase):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_df: pd.DataFrame,
        generator_type: str,
        variable: str,
        hr_size: Optional[int] = 128,
        stage: Optional[str] = consts.stages.train,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: pd.DataFrame = None,
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

        self.df = df
        self.elevation_df = elevation_df
        self.hr_size = hr_size
        self.stage = stage
        self.use_elevation = use_elevation
        self.use_mask = use_mask
        self.use_global_min_max = use_global_min_max

        if self.standardize:
            self.scaler = StandardScaler(
                mean=self.standardize_stats.at[
                    consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.variable], consts.stats.mean
                ],
                std=self.standardize_stats.at[
                    consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.variable], consts.stats.std
                ],
                nan_substitution=self.standardize_stats.at[
                    consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.variable], consts.stats.normalized_min
                ],
            )
            self.elevation_scaler = StandardScaler(
                mean=standardize_stats.at[consts.cruts.elev, consts.stats.mean],
                std=standardize_stats.at[consts.cruts.elev, consts.stats.std],
                missing_indicator=consts.world_clim.elevation_missing_indicator,
                nan_substitution=self.standardize_stats.at[consts.cruts.elev, consts.stats.nan_sub],
            )
        else:
            self.scaler = MinMaxScaler(feature_range=self.normalize_range)
            self.elevation_scaler = MinMaxScaler(feature_range=self.normalize_range)

        self.to_tensor = transforms.ToTensor()
        self.resize = A.Resize(
            width=self.hr_size // self.scaling_factor,
            height=self.hr_size // self.scaling_factor,
            interpolation=cv2.INTER_NEAREST,
            always_apply=True,
            p=1.0,
        )
        self.upscale_nearest = A.Resize(height=self.hr_size, width=self.hr_size, interpolation=cv2.INTER_NEAREST)
        self.upscale_cubic = A.Resize(height=self.hr_size, width=self.hr_size, interpolation=cv2.INTER_CUBIC)

    def _concat_if_needed(
        self,
        img_lr: Tensor,
        img_sr_nearest: Tensor,
        img_elev: Tensor,
        img_elev_lr: Tensor,
        mask_tensor: Tensor,
        mask_np: np.ndarray,
    ) -> Tensor:
        """Concatenates elevation and/or mask data as 2nd and/or 3rd channel to LR raster data."""

        if self.use_elevation:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_sr_nearest, img_elev], dim=0)
            else:
                img_lr = torch.cat([img_lr, img_elev_lr], dim=0)

        if self.use_mask:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_lr, mask_tensor], dim=0)
            else:
                mask_lr = self.to_tensor(self.resize(image=mask_np.astype(np.float32))["image"])
                img_lr = torch.cat([img_lr, mask_lr], dim=0)

        return img_lr

    def _common_to_tensor(
        self,
        img_lr: np.ndarray,
        img_hr: np.ndarray,
        img_elev: np.ndarray,
        mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Performs common `to_tensor` transformation on raster data."""

        img_sr_nearest = self.to_tensor(self.upscale_nearest(image=img_lr)["image"])
        img_elev_lr = self.to_tensor(self.resize(image=img_elev)["image"])
        img_lr = self.to_tensor(img_lr)
        img_hr = self.to_tensor(img_hr.copy())
        img_elev = self.to_tensor(img_elev.copy())
        mask_tensor = self.to_tensor(mask.astype(np.float32))
        original_image = self.to_tensor(original_image.astype(np.float32)) if original_image is not None else None

        img_lr = self._concat_if_needed(img_lr, img_sr_nearest, img_elev, img_elev_lr, mask_tensor, mask)

        return img_lr, img_hr, img_elev, img_elev_lr, mask_tensor, img_sr_nearest, original_image

    def _get_training_sample(
        self,
        img_hr: np.ndarray,
        img_elev: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Union[Tensor, list]]:
        """Gets single training sample with applied transformations."""

        # Vertical flip
        if random.random() > 0.5:
            img_hr = np.flipud(img_hr)
            img_elev = np.flipud(img_elev)
            mask = np.flipud(mask)

        # Horizontal flip
        if random.random() > 0.5:
            img_hr = np.fliplr(img_hr)
            img_elev = np.fliplr(img_elev)
            mask = np.fliplr(mask)

        # Random 90 deg rotation
        if random.random() > 0.5:
            factor = random.randint(0, 3)
            img_hr = np.rot90(img_hr, factor)
            img_elev = np.rot90(img_elev, factor)
            mask = np.rot90(mask, factor)

        # lr
        img_lr = self.resize(image=img_hr)["image"]

        (
            img_lr,
            img_hr,
            img_elev,
            img_elev_lr,
            mask_tensor,
            img_sr_nearest,
            _,
        ) = self._common_to_tensor(img_lr, img_hr, img_elev, mask)

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.hr: img_hr,
            consts.batch_items.elevation: img_elev,
            consts.batch_items.mask: mask_tensor,
        }

    def _get_val_test_sample(
        self, img_hr: np.ndarray, img_elev: np.ndarray, mask: np.ndarray, original_image: np.ndarray, min: float, max: float
    ) -> Dict[str, Union[Tensor, list]]:
        img_lr = self.resize(image=img_hr)["image"]
        img_sr_cubic = self.to_tensor(self.upscale_cubic(image=img_lr)["image"])

        (
            img_lr,
            img_hr,
            img_elev,
            img_elev_lr,
            mask_tensor,
            img_sr_nearest,
            original_image,
        ) = self._common_to_tensor(img_lr, img_hr, img_elev, mask, original_image)

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.hr: img_hr,
            consts.batch_items.elevation: img_elev,
            consts.batch_items.elevation_lr: img_elev_lr,
            consts.batch_items.nearest: img_sr_nearest,
            consts.batch_items.cubic: img_sr_cubic,
            consts.batch_items.original_data: original_image,
            consts.batch_items.mask: mask_tensor,
            consts.batch_items.min: min,
            consts.batch_items.max: max,
        }

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, list]]:
        row = self.df.iloc[index]
        min = row[consts.stats.min] if not self.use_global_min_max else row[consts.stats.global_min]
        max = row[consts.stats.max] if not self.use_global_min_max else row[consts.stats.global_max]

        # original, hr
        original_image = np.array(Image.open(row[consts.datasets_and_preprocessing.tile_file_path]))
        img_hr = original_image.copy()

        # elevation
        elev_fp = self.elevation_df[
            (self.elevation_df[consts.datasets_and_preprocessing.x] == row[consts.datasets_and_preprocessing.x])
            & (self.elevation_df[consts.datasets_and_preprocessing.y] == row[consts.datasets_and_preprocessing.y])
            & (
                self.elevation_df[consts.datasets_and_preprocessing.resolution]
                == row[consts.datasets_and_preprocessing.resolution]
            )
        ][consts.datasets_and_preprocessing.tile_file_path]
        elev_fp = elev_fp.values[0]
        img_elev = np.array(Image.open(elev_fp))
        img_elev = img_elev.copy()

        # normalize/standardize
        if self.normalize:
            img_hr = self.scaler.normalize(img_hr, min, max)
            img_elev = self.elevation_scaler.normalize(
                img_elev,
                missing_indicator=consts.world_clim.elevation_missing_indicator,
            )
        if self.standardize:
            img_hr = self.scaler.normalize(arr=img_hr)
            img_elev = self.elevation_scaler.normalize(
                arr=img_elev,
            )

        # land mask
        mask = ~np.isnan(original_image)

        if self.stage == consts.stages.train:
            return self._get_training_sample(img_hr, img_elev, mask)

        return self._get_val_test_sample(img_hr, img_elev, mask, original_image, min, max)

    def __len__(self) -> int:
        return len(self.df)
