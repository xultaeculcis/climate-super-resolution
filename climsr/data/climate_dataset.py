# -*- coding: utf-8 -*-
from random import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import climsr.consts as consts
from climsr.data.normalization import MinMaxScaler, StandardScaler
from climsr.pre_processing.variable_mappings import world_clim_to_cruts_mapping
from data.climate_dataset_base import ClimateDatasetBase


class ClimateDataset(ClimateDatasetBase):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_df: pd.DataFrame,
        generator_type: str,
        variable: str,
        elevation_file: str,
        land_mask_file: str,
        hr_size: Optional[int] = 128,
        stage: Optional[str] = consts.stages.train,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: pd.DataFrame = None,
        normalize_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        use_elevation: Optional[bool] = True,
        use_mask_as_3rd_channel: Optional[bool] = True,
        use_global_min_max: Optional[bool] = True,
    ):
        super().__init__(
            elevation_file=elevation_file,
            land_mask_file=land_mask_file,
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
        self.use_mask_as_3rd_channel = use_mask_as_3rd_channel
        self.use_global_min_max = use_global_min_max

        if self.standardize:
            self.scaler = StandardScaler(
                mean=self.standardize_stats[world_clim_to_cruts_mapping[self.variable]][
                    consts.stats.mean
                ],
                std=self.standardize_stats[world_clim_to_cruts_mapping[self.variable]][
                    consts.stats.std
                ],
                nan_substitution=self.standardize_stats[
                    world_clim_to_cruts_mapping[self.variable]
                ][consts.stats.normalized_min],
            )
            self.elevation_scaler = StandardScaler(
                mean=standardize_stats[consts.cruts.elev][consts.stats.mean],
                std=standardize_stats[consts.cruts.elev][consts.stats.std],
                missing_indicator=consts.world_clim.elevation_missing_indicator,
                nan_substitution=self.standardize_stats[consts.cruts.elev][
                    consts.stats.nan_sub
                ],
            )
        else:
            self.scaler = MinMaxScaler(feature_range=self.normalize_range)
            self.elevation_scaler = MinMaxScaler(feature_range=self.normalize_range)

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(
            (self.hr_size // self.scaling_factor, self.hr_size // self.scaling_factor),
            InterpolationMode.NEAREST,
        )
        self.upscale_nearest = transforms.Resize(
            (self.hr_size, self.hr_size), interpolation=InterpolationMode.NEAREST
        )
        self.upscale_cubic = transforms.Resize(
            (self.hr_size, self.hr_size), interpolation=InterpolationMode.BICUBIC
        )

    def _concat_if_needed(
        self,
        img_lr: Tensor,
        img_sr_nearest: Tensor,
        img_elev: Tensor,
        img_elev_lr: Tensor,
        mask_tensor: Tensor,
        mask: np.ndarray,
    ) -> Tensor:
        """Concatenates elevation and/or mask data as 2nd and/or 3rd channel to LR raster data."""

        if self.use_elevation:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_sr_nearest, img_elev], dim=0)
            else:
                img_lr = torch.cat([img_lr, img_elev_lr], dim=0)

        if self.use_mask_as_3rd_channel:
            if self.generator_type == consts.models.srcnn:
                img_lr = torch.cat([img_lr, mask_tensor], dim=0)
            else:
                mask_lr = self.to_tensor(
                    self.resize(Image.fromarray(mask.astype(np.float32)))
                )
                img_lr = torch.cat([img_lr, mask_lr], dim=0)

        return img_lr

    def _common_to_tensor(
        self,
        img_lr: np.ndarray,
        img_hr: np.ndarray,
        img_elev: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray, Tensor, Tensor]:
        """Performs common `to_tensor` transformation on raster data."""

        img_sr_nearest = self.to_tensor(
            np.array(self.upscale_nearest(img_lr), dtype=np.float32)
        )
        img_elev_lr = self.to_tensor(np.array(self.resize(img_elev), dtype=np.float32))
        img_lr = self.to_tensor(np.array(img_lr, dtype=np.float32))
        img_hr = self.to_tensor(np.array(img_hr, dtype=np.float32))
        img_elev = self.to_tensor(np.array(img_elev, dtype=np.float32))
        mask_tensor = self.to_tensor(mask.astype(np.float32))

        img_lr = self._concat_if_needed(
            img_lr, img_sr_nearest, img_elev, img_elev_lr, mask_tensor, mask
        )

        return img_lr, img_hr, img_elev, img_elev_lr, mask, mask_tensor, img_sr_nearest

    def _get_training_sample(
        self, img_lr, img_hr, img_elev, mask
    ) -> Dict[str, Union[Tensor, list]]:
        """Gets single training sample with applied transformations."""

        if random() > 0.5:
            img_lr = TF.vflip(img_lr)
            img_hr = TF.vflip(img_hr)
            img_elev = TF.vflip(img_elev)

        if random() > 0.5:
            img_lr = TF.hflip(img_lr)
            img_hr = TF.hflip(img_hr)
            img_elev = TF.hflip(img_elev)

        (
            img_lr,
            img_hr,
            img_elev,
            _,
            _,
            mask_tensor,
            img_sr_nearest,
        ) = self._common_to_tensor(img_lr, img_hr, img_elev, mask)

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.hr: img_hr,
            consts.batch_items.elevation: img_elev,
            consts.batch_items.mask: mask_tensor,
        }

    def _get_val_test_sample(
        self, img_lr, img_hr, img_elev, mask, original_image, min, max
    ) -> Dict[str, Union[Tensor, list]]:
        img_sr_cubic = self.to_tensor(
            np.array(self.upscale_cubic(img_lr), dtype=np.float32)
        )

        (
            img_lr,
            img_hr,
            img_elev,
            img_elev_lr,
            mask,
            mask_tensor,
            img_sr_nearest,
        ) = self._common_to_tensor(img_lr, img_hr, img_elev, mask)

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.hr: img_hr,
            consts.batch_items.elevation: img_elev,
            consts.batch_items.elevation_lr: img_elev_lr,
            consts.batch_items.nearest: img_sr_nearest,
            consts.batch_items.cubic: img_sr_cubic,
            consts.batch_items.original_data: original_image,
            consts.batch_items.mask: mask_tensor,
            consts.batch_items.mask_np: mask,
            consts.batch_items.min: min,
            consts.batch_items.max: max,
        }

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        row = self.df.iloc[index]
        min = (
            row[consts.stats.min]
            if not self.use_global_min_max
            else row[consts.stats.global_min]
        )
        max = (
            row[consts.stats.max]
            if not self.use_global_min_max
            else row[consts.stats.global_max]
        )

        # original, hr
        with rio.open(row[consts.datasets_and_preprocessing.tile_file_path]) as ds:
            original_image = ds.read(1)
            img_hr = original_image.copy()

        # elevation
        elev_fp = self.elevation_df[
            (
                self.elevation_df[consts.datasets_and_preprocessing.x]
                == row[consts.datasets_and_preprocessing.x]
            )
            & (
                self.elevation_df[consts.datasets_and_preprocessing.y]
                == row[consts.datasets_and_preprocessing.y]
            )
        ][consts.datasets_and_preprocessing.file_path]
        elev_fp = elev_fp.values[0]
        with rio.open(elev_fp) as ds:
            img_elev = ds.read(1)

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

        img_hr = Image.fromarray(img_hr)

        # land mask
        mask = np.isnan(original_image)

        # lr
        img_lr = self.resize(img_hr)

        img_elev = Image.fromarray(img_elev)

        if self.stage == consts.stages.train:
            return self._get_training_sample(img_lr, img_hr, img_elev, mask)

        return self._get_val_test_sample(
            img_lr, img_hr, img_elev, mask, original_image, min, max
        )

    def __len__(self) -> int:
        return len(self.df)
