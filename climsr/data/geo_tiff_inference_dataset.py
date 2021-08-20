# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import InterpolationMode

from climsr.data.normalization import StandardScaler, MinMaxScaler
from climsr.configs.cruts_config import CRUTSConfig


class GeoTiffInferenceDataset(Dataset):
    def __init__(
        self,
        tiff_dir: str,
        tiff_df: pd.DataFrame,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        variable: str,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: Dict[str, Dict[str, float]] = None,
        normalize_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        use_elevation: Optional[bool] = True,
        use_mask_as_3rd_channel: Optional[bool] = True,
        use_global_min_max: Optional[bool] = True,
    ):
        if normalize == standardize:
            raise Exception(
                "Bad parameter combination: normalization and standardization! Choose one!"
            )

        self.tiff_dir = tiff_dir
        self.tiffs = glob(f"{tiff_dir}/*.tif")
        self.tiff_df = tiff_df.set_index("filename", drop=True)
        self.variable = variable
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_stats = standardize_stats
        self.use_elevation = use_elevation
        self.use_mask_as_3rd_channel = use_mask_as_3rd_channel
        self.use_global_min_max = use_global_min_max

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

        self.land_mask_np = ~np.isnan(
            np.array(Image.open(land_mask_file), dtype=np.float32)
        )
        self.land_mask_tensor = self.to_tensor(self.land_mask_np)

        elevation_arr = np.array(Image.open(elevation_file), dtype=np.float32)
        elevation_arr = np.where(
            self.land_mask_np, elevation_arr, np.nan
        )  # mask Antarctica
        elevation_arr = self.elevation_scaler.normalize(
            elevation_arr, missing_indicator=-32768
        )

        self.resize = transforms.Resize(
            (
                elevation_arr.shape[0] // self.scaling_factor,
                elevation_arr.shape[1] // self.scaling_factor,
            ),
            InterpolationMode.NEAREST,
        )

        self.upscale = transforms.Resize(
            (
                elevation_arr.shape[0],
                elevation_arr.shape[1],
            ),
            interpolation=InterpolationMode.NEAREST,
        )

        self.elevation_data = self.to_tensor(elevation_arr)
        self.elevation_lr = self.to_tensor(self.resize(Image.fromarray(elevation_arr)))
        self.mask_lr = self.to_tensor(self.resize(Image.fromarray(self.land_mask_np)))

    def _concat_if_needed(self, img_lr, img_sr_nearest) -> Tensor:
        if self.use_elevation:
            if self.generator_type == "srcnn":
                img_lr = torch.cat([img_sr_nearest, self.elevation_data], dim=0)
            else:
                img_lr = torch.cat([img_lr, self.elevation_lr], dim=0)

        if self.use_mask_as_3rd_channel:
            if self.generator_type == "srcnn":
                img_lr = torch.cat([img_lr, self.land_mask_tensor], dim=0)
            else:
                img_lr = torch.cat([img_lr, self.mask_lr], dim=0)

        return img_lr

    def _common_to_tensor(
        self, img_lr
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        img_sr_nearest = self.to_tensor(
            np.array(self.upscale(img_lr), dtype=np.float32)
        )

        img_lr = self.to_tensor(np.array(img_lr, dtype=np.float32))

        img_lr = self._concat_if_needed(img_lr, img_sr_nearest)

        return (
            img_lr,
            self.elevation_data,
            self.elevation_lr,
            self.land_mask_np,
            self.land_mask_tensor,
            img_sr_nearest,
        )

    def _get_inference_sample(self, img_lr, min, max) -> Dict[str, Union[Tensor, list]]:
        (
            img_lr,
            img_elev,
            img_elev_lr,
            mask,
            mask_tensor,
            img_sr_nearest,
        ) = self._common_to_tensor(img_lr)

        return {
            "lr": img_lr,
            "elevation": img_elev,
            "elevation_lr": img_elev_lr,
            "nearest": img_sr_nearest,
            "mask": mask_tensor,
            "mask_np": mask,
            "min": min,
            "max": max,
        }

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        file_path = self.tiffs[index]
        file_name = os.path.basename(file_path)
        row = self.tiff_df.loc[file_name]
        min = row["min"] if not self.use_global_min_max else row["global_min"]
        max = row["max"] if not self.use_global_min_max else row["global_max"]

        # original, hr
        with rio.open(file_path) as ds:
            img_lr = np.flipud(ds.read(1))

        # normalize/standardize
        if self.normalize:
            img_lr = self.scaler.normalize(img_lr, min, max)
        if self.standardize:
            img_lr = self.scaler.normalize(arr=img_lr)

        img_lr = Image.fromarray(img_lr)

        item = self._get_inference_sample(img_lr, min, max)
        item["filename"] = file_name

        return item

    def __len__(self):
        return len(self.tiffs)
