# -*- coding: utf-8 -*-
from random import random
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import functional as TF, InterpolationMode

from sr.data.normalization import StandardScaler, MinMaxScaler
from sr.pre_processing.variable_mappings import world_clim_to_cruts_mapping
from sr.pre_processing.cruts_config import CRUTSConfig


class ClimateDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_df: pd.DataFrame,
        generator_type: str,
        variable: str,
        hr_size: Optional[int] = 128,
        stage: Optional[str] = "train",
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

        self.df = df
        self.elevation_df = elevation_df
        self.hr_size = hr_size
        self.scaling_factor = scaling_factor
        self.stage = stage
        self.generator_type = generator_type
        self.variable = variable
        self.normalize = normalize
        self.normalize_range = normalize_range
        self.standardize = standardize
        self.standardize_stats = standardize_stats

        if self.standardize:
            self.scaler = StandardScaler(
                mean=self.standardize_stats[world_clim_to_cruts_mapping[self.variable]][
                    "mean"
                ],
                std=self.standardize_stats[world_clim_to_cruts_mapping[self.variable]][
                    "std"
                ],
                nan_substitution=self.standardize_stats[
                    world_clim_to_cruts_mapping[self.variable]
                ]["normalized_min"],
            )
            self.elevation_scaler = StandardScaler(
                mean=standardize_stats[CRUTSConfig.elev]["mean"],
                std=standardize_stats[CRUTSConfig.elev]["std"],
                missing_indicator=-32768,
                nan_substitution=self.standardize_stats[CRUTSConfig.elev]["nan_sub"],
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

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        row = self.df.iloc[index]
        min = row["min"]
        max = row["max"]

        # original, hr
        with rio.open(row["tile_file_path"]) as ds:
            original_image = ds.read(1)
            img_hr = original_image.copy()

        # normalize/standardize
        if self.normalize:
            img_hr = self.scaler.normalize(img_hr, min, max)
        if self.standardize:
            img_hr = self.scaler.normalize(arr=img_hr)

        img_hr = Image.fromarray(img_hr)

        # land mask
        mask = np.isnan(original_image)

        # lr
        img_lr = self.resize(img_hr)

        # elevation
        elev_fp = self.elevation_df[
            (self.elevation_df["x"] == row["x"]) & (self.elevation_df["y"] == row["y"])
        ]["file_path"].values[0]
        with rio.open(elev_fp) as ds:
            img_elev = ds.read(1)

        if self.normalize:
            img_elev = self.elevation_scaler.normalize(
                img_elev, missing_indicator=-32768
            )
        if self.standardize:
            img_elev = self.elevation_scaler.normalize(
                arr=img_elev,
            )
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
            np.array(self.upscale_nearest(img_lr), dtype=np.float32)
        )

        img_sr_cubic = self.to_tensor(
            np.array(self.upscale_cubic(img_lr), dtype=np.float32)
        )

        img_elev_lr = self.to_tensor(np.array(self.resize(img_elev), dtype=np.float32))
        img_lr = self.to_tensor(np.array(img_lr, dtype=np.float32))
        img_hr = self.to_tensor(np.array(img_hr, dtype=np.float32))
        img_elev = self.to_tensor(np.array(img_elev, dtype=np.float32))

        return {
            "lr": img_lr,
            "hr": img_hr,
            "elevation": img_elev,
            "elevation_lr": img_elev_lr,
            "nearest": img_sr_nearest,
            "cubic": img_sr_cubic,
            "original_data": original_image,
            "mask": mask,
            "min": min,
            "max": max,
        }

    def __len__(self) -> int:
        return len(self.df)