# -*- coding: utf-8 -*-
from random import random
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torchvision.transforms import functional as TF
import xarray as xr
from sr.pre_processing.cruts_config import CRUTSConfig
from sr.data.utils import plot_single_batch, normalize, get_variable_from_ds_fp


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
        img_hr = Image.open(row["file_path"])
        img_lr = self.resize(img_hr)
        img_sr_nearest = []
        elev_fp = self.elevation_df[
            (self.elevation_df["x"] == row["x"]) & (self.elevation_df["y"] == row["y"])
        ]["file_path"].values[0]
        img_elev = Image.open(elev_fp)

        if self.stage == "train":
            if random() > 0.5:
                img_lr = TF.vflip(img_lr)
                img_hr = TF.vflip(img_hr)
                img_elev = TF.vflip(img_elev)

            if random() > 0.5:
                img_lr = TF.hflip(img_lr)
                img_hr = TF.hflip(img_hr)
                img_elev = TF.hflip(img_elev)

        if self.generator_type == "srcnn" or self.stage != "train":
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
        }

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":
    ds = CRUTSInferenceDataset(
        ds_path="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.pre.dat.nc",
        elevation_file="/media/xultaeculcis/2TB/datasets/wc/pre-processed/elevation/resized/4x/wc2.1_2.5m_elev.tif",
        land_mask_file="/media/xultaeculcis/2TB/datasets/wc/pre-processed/prec/resized/4x/wc2.1_2.5m_prec_1961-01.tif",
        generator_type="srcnn",
        scaling_factor=4,
    )

    dl = DataLoader(dataset=ds, batch_size=1, pin_memory=True, num_workers=1)
    _ = plot_single_batch(loader=dl, keys=["lr", "elevation"])

    ds = ClimateDataset(
        df=pd.read_csv("../../datasets/prec/4x/train.csv"),
        elevation_df=pd.read_csv("../../datasets/elevation/4x/elevation.csv"),
        hr_size=128,
        stage="train",
        generator_type="srcnn",
        scaling_factor=4,
    )

    dl = DataLoader(dataset=ds, batch_size=32, pin_memory=True, num_workers=1)
    _ = plot_single_batch(loader=dl, keys=["lr", "hr", "elevation", "nearest"])
