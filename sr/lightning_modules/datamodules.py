# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision

from sr.data.datasets import ClimateDataset
from sr.pre_processing.world_clim_config import WorldClimConfig
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"


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
        normalize: Optional[bool] = False,
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
            normalization_mean=WorldClimConfig.statistics[self.world_clim_variable][
                "mean"
            ],
            normalization_std=WorldClimConfig.statistics[self.world_clim_variable][
                "std"
            ],
            generator_type=self.generator_type,
            scaling_factor=self.scale_factor,
        )
        self.val_dataset = ClimateDataset(
            df=val_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage="val",
            normalization_mean=WorldClimConfig.statistics[self.world_clim_variable][
                "mean"
            ],
            normalization_std=WorldClimConfig.statistics[self.world_clim_variable][
                "std"
            ],
            generator_type=self.generator_type,
            scaling_factor=self.scale_factor,
        )
        self.test_dataset = ClimateDataset(
            df=test_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage="test",
            normalization_mean=WorldClimConfig.statistics[self.world_clim_variable][
                "mean"
            ],
            normalization_std=WorldClimConfig.statistics[self.world_clim_variable][
                "std"
            ],
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
            default="tmin",
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
        generator_type="srcnn",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        scale_factor=args.scale_factor,
        seed=args.seed,
    )

    def matplotlib_imshow(batch):
        # create grid of images
        img_grid = torchvision.utils.make_grid(
            batch, nrow=8, normalize=False, padding=0
        )
        # show images
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def plot_single_batch(loader, stage):
        for _, batch in enumerate(loader):
            lr = batch["lr"]
            hr = batch["hr"]
            sr_nearest = batch["nearest"]
            elevation = batch["elevation"]

            matplotlib_imshow(lr)
            matplotlib_imshow(hr)
            matplotlib_imshow(elevation)

            if stage != "train":
                matplotlib_imshow(sr_nearest)

            break

    # plot_single_batch(dm.train_dataloader(), "train")
    plot_single_batch(dm.val_dataloader(), "val")
    # plot_single_batch(dm.test_dataloader(), "test")
