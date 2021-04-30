# -*- coding: utf-8 -*-
from random import random
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import functional as TF

from sr.data.utils import get_variable_from_ds_fp, normalize
from sr.pre_processing.cruts_config import CRUTSConfig


class CRUTSInferenceDataset(Dataset):
    def __init__(
        self,
        ds_path: str,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        scaling_factor: Optional[int] = 4,
    ):
        self.ds = xr.open_dataset(ds_path)
        self.variable = get_variable_from_ds_fp(ds_path)
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type

        self.v_flip = transforms.RandomVerticalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()

        self.upscale = transforms.Resize(
            (
                CRUTSConfig.cruts_original_shape[0] * self.scaling_factor,
                CRUTSConfig.cruts_original_shape[1] * self.scaling_factor,
            ),
            interpolation=Image.NEAREST,
        )

        self.land_mask = np.array(Image.open(land_mask_file), dtype=np.float32)

        elevation_arr = np.array(Image.open(elevation_file), dtype=np.float32)
        mask = ~np.isnan(self.land_mask)
        elevation_arr = np.where(mask, elevation_arr, np.nan)  # mask Antarctica
        elevation_arr, _, _ = normalize(elevation_arr)

        self.elevation_data = self.to_tensor(elevation_arr)

    def __getitem__(self, index):
        arr = self.ds[self.variable].isel(time=index)

        input_img = np.flipud(arr.values.astype(np.float32))

        input_img, min, max = normalize(input_img)

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
        }

    def __len__(self):
        return len(self.ds["time"])


class ClimateDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_df: pd.DataFrame,
        generator_type: str,
        hr_size: Optional[int] = 128,
        stage: Optional[str] = "train",
        scaling_factor: Optional[int] = 4,
    ):
        self.df = df
        self.elevation_df = elevation_df
        self.hr_size = hr_size
        self.scaling_factor = scaling_factor
        self.stage = stage
        self.generator_type = generator_type

        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.hr_size // self.scaling_factor, self.hr_size // self.scaling_factor),
            Image.NEAREST,
        )

        self.upscale = transforms.Resize(
            (self.hr_size, self.hr_size), interpolation=Image.NEAREST
        )

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        row = self.df.iloc[index]
        min = row["min"]
        max = row["max"]

        # original, hr
        with rio.open(row["tile_file_path"]) as ds:
            original_image = ds.read(1)
            img_hr = original_image.copy()

        # normalize
        img_hr, min, max = normalize(img_hr, min, max)
        img_hr = Image.fromarray(img_hr)

        # land mask
        mask = np.isnan(original_image)

        # lr
        img_lr = self.resize(img_hr)
        img_lr = self.resize(img_hr)

        # elevation
        elev_fp = self.elevation_df[
            (self.elevation_df["x"] == row["x"]) & (self.elevation_df["y"] == row["y"])
        ]["file_path"].values[0]
        with rio.open(elev_fp) as ds:
            img_elev = ds.read(1)
        img_elev, elev_min, elev_max = normalize(img_elev, missing_indicator=-32768)
        img_elev = Image.fromarray(img_elev)

        # transforms
        if self.stage == "train":
            if random() > 0.5:
                img_lr = TF.vflip(img_lr)
                img_hr = TF.vflip(img_hr)
                img_elev = TF.vflip(img_elev)

            if random() > 0.5:
                img_lr = TF.hflip(img_lr)
                img_hr = TF.hflip(img_hr)
                img_elev = TF.hflip(img_elev)

        img_sr_nearest = self.to_tensor(
            np.array(self.upscale(img_lr), dtype=np.float32)
        )
        img_lr = self.to_tensor(np.array(img_lr, dtype=np.float32))
        img_hr = self.to_tensor(np.array(img_hr, dtype=np.float32))
        img_elev = self.to_tensor(np.array(img_elev, dtype=np.float32))

        return {
            "lr": img_lr,
            "hr": img_hr,
            "elevation": img_elev,
            "nearest": img_sr_nearest,
            "original_data": original_image,
            "mask": mask,
            "min": min,
            "max": max,
        }

    def __len__(self) -> int:
        return len(self.df)
