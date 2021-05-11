# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sr.data.datasets import ClimateDataset
from sr.pre_processing.world_clim_config import WorldClimConfig

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

        train_df, val_df, test_dfs, elevation_df = self.load_data()

        logging.info(
            f"'{self.world_clim_variable}' - Train/Validation/Test split sizes (HR): "
            f"{len(train_df)}/{len(val_df)}/{len(test_dfs)} - {[len(df) for df in test_dfs]}"
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
        self.test_datasets = [
            ClimateDataset(
                df=test_df,
                elevation_df=elevation_df,
                hr_size=self.hr_size,
                stage="test",
                generator_type=self.generator_type,
                scaling_factor=self.scale_factor,
            )
            for test_df in test_dfs
        ]

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

    def test_dataloader(self) -> List[DataLoader]:
        if self.world_clim_variable == "temp":
            return [
                DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                for test_dataset in self.test_datasets
            ]
        else:
            return [
                DataLoader(
                    self.test_datasets[0],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            ]

    def load_dataframe(self, var, filename) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(
                self.data_path,
                var,
                self.world_clim_multiplier,
                filename,
            )
        )

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
        elevation_df = self.load_dataframe(
            WorldClimConfig.elevation, f"{WorldClimConfig.elevation}.csv"
        )

        stats_df = pd.read_csv(os.path.join(self.data_path, "statistics.csv"))

        if self.world_clim_variable == "temp":
            train_dfs = []
            val_dfs = []
            test_dfs = []
            variables = [WorldClimConfig.tmin, WorldClimConfig.tmax]
            for var in variables:
                train_dfs.append(self.load_dataframe(var, "train.csv"))
                val_dfs.append(self.load_dataframe(var, "val.csv"))
                test_dfs.append(self.load_dataframe(var, "test.csv"))

            train_df = pd.concat(train_dfs)
            val_df = pd.concat(val_dfs)
        else:
            train_df = self.load_dataframe(self.world_clim_variable, "train.csv")
            val_df = self.load_dataframe(self.world_clim_variable, "val.csv")
            test_dfs = [self.load_dataframe(self.world_clim_variable, "test.csv")]

        train_df = pd.merge(
            train_df,
            stats_df,
            how="inner",
            on=["filename", "variable", "year", "month"],
        )

        val_df = pd.merge(
            val_df, stats_df, how="inner", on=["filename", "variable", "year", "month"]
        )

        output_test_dfs = []
        for test_df in test_dfs:
            test_df = pd.merge(
                test_df,
                stats_df,
                how="inner",
                on=["filename", "variable", "year", "month"],
            )
            output_test_dfs.append(test_df)

        return train_df, val_df, output_test_dfs, elevation_df

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
            default="tmax",
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
    import matplotlib
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = 64

    def plot_array(arr, figsize=None):
        plt.figure(figsize=figsize)
        plt.imshow(arr, cmap="jet")
        plt.show()

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

    # plot_single_batch(dm.train_dataloader(), keys=["lr", "hr", "elevation", "nearest"])
    # plot_single_batch(dm.val_dataloader(), keys=["lr", "hr", "elevation", "nearest"])
    # plot_single_batch(dm.test_dataloader(), keys=["lr", "hr", "elevation", "nearest"])

    dl = DataLoader(
        dataset=dm.val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1,
    )

    for _, batch in enumerate(dl):
        lr = batch["lr"]
        hr = batch["hr"]
        original = batch["original_data"]
        mask = batch["mask"]
        sr_nearest = batch["nearest"]
        elev = batch["elevation"]
        max_vals = batch["max"].cpu().numpy()
        min_vals = batch["min"].cpu().numpy()
        sr = hr

        mae = []
        mse = []
        rmse = []

        items = 16
        fig, axes = plt.subplots(
            nrows=items,
            ncols=4,
            figsize=(5, 1.5 * items),
            constrained_layout=True,
            sharey=True,
        )

        cmap = matplotlib.cm.jet.copy()
        cmap.set_bad("black", 1.0)

        cols = ["HR", "Nearest", "Elevation", "SR"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        nearest_arr = sr_nearest.squeeze(1).cpu().numpy()
        hr_arr = hr.squeeze(1).cpu().numpy()
        elev_arr = elev.squeeze(1).cpu().numpy()
        sr_arr = hr.squeeze(1).cpu().numpy()

        for i in range(items):
            hr_arr[i][mask[i]] = np.nan
            nearest_arr[i][mask[i]] = np.nan
            elev_arr[i][mask[i]] = np.nan
            sr_arr[i][mask[i]] = np.nan

            axes[i][0].imshow(hr_arr[i], cmap=cmap, vmin=0, vmax=1)
            axes[i][1].imshow(nearest_arr[i], cmap=cmap, vmin=0, vmax=1)
            axes[i][2].imshow(elev_arr[i], cmap=cmap, vmin=0, vmax=1)
            axes[i][3].imshow(sr_arr[i], cmap=cmap, vmin=0, vmax=1)

        fig.suptitle(f"Validation batch, epoch={0}", fontsize=16)
        plt.show()

        break
