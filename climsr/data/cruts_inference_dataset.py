# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from torchvision import transforms as transforms
from torchvision.transforms import InterpolationMode

import climsr.consts as consts
from climsr.data import utils as utils
from climsr.data.normalization import MinMaxScaler, StandardScaler
from data.climate_dataset_base import ClimateDatasetBase


class CRUTSInferenceDataset(ClimateDatasetBase):
    def __init__(
        self,
        ds_path: str,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: pd.DataFrame = None,
        normalize_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ):
        super().__init__(
            elevation_file=elevation_file,
            land_mask_file=land_mask_file,
            generator_type=generator_type,
            variable=utils.get_variable_from_ds_fp(ds_path),
            scaling_factor=scaling_factor,
            normalize=normalize,
            standardize=standardize,
            standardize_stats=standardize_stats,
            normalize_range=normalize_range,
        )

        self.ds = xr.open_dataset(ds_path)

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

        self.v_flip = transforms.RandomVerticalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()

        self.upscale = transforms.Resize(
            (
                consts.cruts.cruts_original_shape[0] * self.scaling_factor,
                consts.cruts.cruts_original_shape[1] * self.scaling_factor,
            ),
            interpolation=InterpolationMode.NEAREST,
        )

        self.land_mask = np.array(Image.open(land_mask_file), dtype=np.float32)

        elevation_arr = np.array(Image.open(elevation_file), dtype=np.float32)
        mask = ~np.isnan(self.land_mask)
        elevation_arr = np.where(mask, elevation_arr, np.nan)  # mask Antarctica
        elevation_arr = self.elevation_scaler.normalize(elevation_arr)

        self.elevation_data = self.to_tensor(elevation_arr)

    def __getitem__(self, index):
        arr = self.ds[self.variable].isel(time=index)

        input_img = np.flipud(arr.values.astype(np.float32))

        min = np.nanmin(input_img)
        max = np.nanmax(input_img)

        if self.normalize:
            input_img = self.scaler.normalize(input_img)
        else:
            input_img = self.scaler.normalize(input_img)

        if self.generator_type == consts.models.srcnn:
            input_img = Image.fromarray(input_img)
            input_img = np.array(self.upscale(input_img), dtype=np.float32)

        img_lr = self.to_tensor(input_img)

        # extract date
        date_str = np.datetime_as_string(arr.time, unit="D")

        return {
            consts.batch_items.lr: img_lr,
            consts.batch_items.elevation: self.elevation_data,
            consts.batch_items.min: min,
            consts.batch_items.max: max,
            consts.batch_items.filename: f"cruts-{self.variable}-{date_str}.tif",
            consts.batch_items.normalized: self.normalize,
            consts.batch_items.standardized: self.standardize,
        }

    def __len__(self):
        return len(self.ds["time"])
