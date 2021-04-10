# -*- coding: utf-8 -*-
from random import random
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import functional as TF

from sr.pre_processing.cruts_config import CRUTSConfig
from sr.pre_processing.world_clim_config import WorldClimConfig


class CRUTSInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_file: str,
        generator_type: str,
        normalization_mean: [float],
        normalization_std: [float],
        scaling_factor: Optional[int] = 4,
    ):
        self.df = df
        self.elevation_data = np.array(Image.open(elevation_file))
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type

        self.to_tensor = transforms.ToTensor()

        self.input_image_normalize = transforms.Normalize(
            normalization_mean, normalization_std
        )

        self.elevation_normalize = transforms.Normalize(
            WorldClimConfig.statistics[WorldClimConfig.elevation]["mean"],
            WorldClimConfig.statistics[WorldClimConfig.elevation]["std"],
        )

        self.upscale = transforms.Resize(
            (
                CRUTSConfig.cruts_original_shape[0] * self.scaling_factor,
                CRUTSConfig.cruts_original_shape[1] * self.scaling_factor,
            ),
            interpolation=Image.NEAREST,
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_lr = Image.open(row["file_path"])

        if self.generator_type == "srcnn":
            img_lr = self.upscale(img_lr)

        img_lr = self.input_image_normalize(self.to_tensor(np.array(img_lr)))
        img_elev = self.elevation_normalize(
            self.to_tensor(np.array(self.elevation_data))
        )

        return {
            "lr": img_lr,
            "elevation": img_elev,
        }

    def __len__(self):
        return len(self.df)


class ClimateDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        elevation_df: pd.DataFrame,
        generator_type: str,
        normalization_mean: [float],
        normalization_std: [float],
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

        self.input_image_normalize = transforms.Normalize(
            (normalization_mean,), (normalization_std,)
        )

        self.elevation_normalize = transforms.Normalize(
            (WorldClimConfig.statistics[WorldClimConfig.elevation]["mean"],),
            (WorldClimConfig.statistics[WorldClimConfig.elevation]["std"],),
        )

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
            arr = np.array(self.upscale(img_lr))
            t = self.to_tensor(arr)
            img_sr_nearest = torch.nan_to_num(self.input_image_normalize(t), nan=-7.0)

        img_lr = torch.nan_to_num(
            self.input_image_normalize(
                self.to_tensor(np.array(img_lr, dtype=np.float32))
            ),
            nan=-4.0,
        )
        img_hr = torch.nan_to_num(
            self.input_image_normalize(
                self.to_tensor(np.array(img_hr, dtype=np.float32))
            ),
            nan=-4.0,
        )
        img_elev = torch.nan_to_num(
            self.elevation_normalize(
                self.to_tensor(np.array(img_elev, dtype=np.float32))
            ),
            nan=-4.0,
        )

        return {
            "lr": img_lr,
            "hr": img_hr,
            "elevation": img_elev,
            "nearest": img_sr_nearest,
        }

    def __len__(self) -> int:
        return len(self.df)
