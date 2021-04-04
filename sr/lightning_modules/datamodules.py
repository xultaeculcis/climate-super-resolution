# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from random import random
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pre_processing.world_clim_config import WorldClimConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"


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
        self.hr_size = hr_size
        self.scaling_factor = scaling_factor
        self.stage = stage
        self.generator_type = generator_type

        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.df = df
        self.elevation_df = elevation_df
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
        x = row["x"]
        y = row["y"]
        elev_fp = self.elevation_df[
            (self.elevation_df["x"] == x) & (self.elevation_df["y"] == y)
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
            img_sr_nearest = self.common_transforms(np.array(self.upscale(img_lr)))

        img_lr = self.common_transforms(np.array(img_lr))
        img_hr = self.common_transforms(np.array(img_hr))
        img_elev = self.common_transforms(np.array(img_elev))

        return {
            "lr": img_lr,
            "hr": img_hr,
            "elevation": img_elev,
            "nearest": img_sr_nearest,
        }

    def __len__(self) -> int:
        return len(self.df)


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        world_clim_variable: str,
        world_clim_multiplier: str,
        generator_type: str,
        scale_factor: Optional[int] = 4,
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 4,
        hr_size: Optional[int] = 128,
        seed: Optional[int] = 42,
    ):
        super(SuperResolutionDataModule, self).__init__()

        assert hr_size % scale_factor == 0

        self.data_path = data_path
        self.world_clim_variable = world_clim_variable
        self.world_clim_multiplier = world_clim_multiplier
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.seed = seed
        self.generator_type = generator_type

        train_df = pd.read_csv(
            os.path.join(
                data_path,
                self.world_clim_variable,
                self.world_clim_multiplier,
                "train.csv",
            )
        )
        val_df = pd.read_csv(
            os.path.join(
                data_path,
                self.world_clim_variable,
                self.world_clim_multiplier,
                "val.csv",
            )
        )
        test_df = pd.read_csv(
            os.path.join(
                data_path,
                self.world_clim_variable,
                self.world_clim_multiplier,
                "test.csv",
            )
        )
        elevation_df = pd.read_csv(
            os.path.join(
                data_path,
                WorldClimConfig.elevation,
                self.world_clim_multiplier,
                f"{WorldClimConfig.elevation}.csv",
            )
        )

        logging.info(
            f"Train/Validation/Test split sizes (HR): {len(train_df)}/{len(val_df)}/{len(test_df)}"
        )

        self.train_dataset = ClimateDataset(
            df=train_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage="train",
            generator_type=self.generator_type,
            scaling_factor=self.scale_factor,
        )
        self.val_dataset = ClimateDataset(
            df=val_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage="val",
            generator_type=self.generator_type,
            scaling_factor=self.scale_factor,
        )
        self.test_dataset = ClimateDataset(
            df=test_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage="test",
            generator_type=self.generator_type,
            scaling_factor=self.scale_factor,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds datamodule specific arguments.

        :param parent_parser: The parent parser.
        :returns: The parser.
        """
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default="../datasets/",
        )
        parser.add_argument(
            "--world_clim_variable",
            type=str,
            default="prec",
        )
        parser.add_argument(
            "--world_clim_multiplier",
            type=str,
            default="4x",
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--hr_size", type=int, default=128)
        parser.add_argument("--scale_factor", type=int, default=4)
        parser.add_argument("--seed", type=int, default=42)
        return parser


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    args = parser.parse_args()

    dm = SuperResolutionDataModule(
        data_path=os.path.join("..", args.data_path),
        world_clim_variable=args.world_clim_variable,
        world_clim_multiplier=args.world_clim_multiplier,
        generator_type="esrgan",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        scale_factor=args.scale_factor,
        seed=args.seed,
    )

    val_dl = dm.val_dataloader()

    def matplotlib_imshow(batch):
        # create grid of images
        img_grid = torchvision.utils.make_grid(batch, nrow=4, normalize=True, padding=0)
        # show images
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    for _, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
        lr = batch["lr"]
        hr = batch["hr"]
        sr_bicubic = batch["nearest"]
        elevation = batch["elevation"]

        expected_lr_shape = (32, 1, 32, 32)
        expected_hr_shape = (32, 1, 128, 128)
        assert lr.shape == expected_lr_shape, (
            f"Expected the LR batch to be in shape {expected_lr_shape}, "
            f"but found: {lr.shape}"
        )
        assert hr.shape == (32, 1, 128, 128), (
            f"Expected the LR batch to be in shape {expected_hr_shape}, "
            f"but found: {hr.shape}"
        )
        assert sr_bicubic.shape == (32, 1, 128, 128), (
            f"Expected the LR batch to be in shape {expected_hr_shape}, "
            f"but found: {sr_bicubic.shape}"
        )
        assert elevation.shape == (32, 1, 128, 128), (
            f"Expected the LR batch to be in shape {expected_hr_shape}, "
            f"but found: {elevation.shape}"
        )

        matplotlib_imshow(lr)
        matplotlib_imshow(hr)
        matplotlib_imshow(sr_bicubic)
        matplotlib_imshow(elevation)
        break
