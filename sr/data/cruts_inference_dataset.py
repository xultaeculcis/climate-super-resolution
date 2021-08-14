# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import InterpolationMode

from sr.data import utils as utils
from sr.data.normalization import StandardScaler, MinMaxScaler
from sr.configs.cruts_config import CRUTSConfig


class CRUTSInferenceDataset(Dataset):
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
        if normalize == standardize:
            raise Exception(
                "Bad parameter combination: normalization and standardization! Choose one!"
            )

        self.ds = xr.open_dataset(ds_path)
        self.variable = utils.get_variable_from_ds_fp(ds_path)
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_stats = standardize_stats

        if self.standardize:
            self.scaler = StandardScaler(
                self.standardize_stats[self.variable]["mean"],
                self.standardize_stats[self.variable]["std"],
            )
            self.elevation_scaler = StandardScaler(
                standardize_stats[CRUTSConfig.elev]["mean"],
                standardize_stats[CRUTSConfig.elev]["std"],
            )
        else:
            self.scaler = MinMaxScaler(feature_range=normalize_range)
            self.elevation_scaler = MinMaxScaler(feature_range=normalize_range)

        self.v_flip = transforms.RandomVerticalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()

        self.upscale = transforms.Resize(
            (
                CRUTSConfig.cruts_original_shape[0] * self.scaling_factor,
                CRUTSConfig.cruts_original_shape[1] * self.scaling_factor,
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

        if self.generator_type == "srcnn":
            input_img = Image.fromarray(input_img)
            input_img = np.array(self.upscale(input_img), dtype=np.float32)

        img_lr = self.to_tensor(input_img)

        # extract date
        date_str = np.datetime_as_string(arr.time, unit="D")

        return {
            "lr": img_lr,
            "elevation": self.elevation_data,
            "min": min,
            "max": max,
            "filename": f"cruts-{self.variable}-{date_str}.tif",
            "normalized": self.normalize,
            "standardize": self.standardize,
        }

    def __len__(self):
        return len(self.ds["time"])
