# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import InterpolationMode

from data import utils as utils
from data.normalization import StandardScaler, MinMaxScaler
from configs.cruts_config import CRUTSConfig


class GeoTiffInferenceDataset(Dataset):
    def __init__(
        self,
        tiff_folder_path: str,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: Dict[str, Dict[str, float]] = None,
        normalize_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        min_max_lookup_df: Optional[pd.DataFrame] = None,
    ):
        if normalize == standardize:
            raise Exception(
                "Bad parameter combination: normalization and standardization! Choose one!"
            )

        self.files = sorted(glob(os.path.join(tiff_folder_path, "*.tif")))
        self.variable = utils.get_variable_from_ds_fp(tiff_folder_path)
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_stats = standardize_stats
        self.min_max_lookup_df = min_max_lookup_df

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
        elevation_arr = self.elevation_scaler.normalize(
            elevation_arr, missing_indicator=-32768
        )

        resize = transforms.Resize(
            (
                elevation_arr.shape[0] // self.scaling_factor,
                elevation_arr.shape[1] // self.scaling_factor,
            ),
            InterpolationMode.NEAREST,
        )

        self.elevation_data = self.to_tensor(elevation_arr)
        self.elevation_lr = self.to_tensor(resize(Image.fromarray(elevation_arr)))

    def __getitem__(self, index):
        filename = self.files[index]
        arr = np.array(Image.open(filename))

        input_img = np.flipud(arr.astype(np.float32))

        basename = os.path.basename(filename)

        min = np.nanmin(input_img)
        max = np.nanmax(input_img)

        if self.normalize:
            input_img = self.scaler.normalize(arr=input_img, min=min, max=max)
        else:
            input_img = self.scaler.normalize(arr=input_img)

        if self.generator_type == "srcnn":
            input_img = Image.fromarray(input_img)
            input_img = np.array(self.upscale(input_img), dtype=np.float32)

        img_lr = self.to_tensor(input_img)

        return {
            "lr": img_lr,
            "elevation": self.elevation_data,
            "elevation_lr": self.elevation_lr,
            "min": min,
            "max": max,
            "mask": self.land_mask,
            "filename": basename,
            "normalized": self.normalize,
            "standardize": self.standardize,
        }

    def __len__(self):
        return len(self.files)
