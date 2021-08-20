# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple, Dict, Union

import inflection
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import climsr.consts as consts
from climsr.data import normalization
from climsr.pre_processing.variable_mappings import world_clim_to_cruts_mapping
from climsr.data.climate_dataset import ClimateDataset

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
        batch_size: Optional[int] = 128,
        num_workers: Optional[int] = 4,
        hr_size: Optional[int] = 128,
        seed: Optional[int] = 42,
        normalization_method: Optional[str] = normalization.minmax,
        normalization_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        pin_memory: Optional[bool] = True,
        use_elevation: Optional[bool] = True,
        use_mask_as_3rd_channel: Optional[bool] = True,
        use_global_min_max: Optional[bool] = True,
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
        self.normalization_method = normalization_method
        self.normalization_range = normalization_range
        self.pin_memory = pin_memory
        self.use_elevation = use_elevation
        self.use_mask_as_3rd_channel = use_mask_as_3rd_channel
        self.use_global_min_max = use_global_min_max

        train_df, val_df, test_dfs, elevation_df, standardize_stats = self.load_data()

        logging.info(
            f"'{self.world_clim_variable}' - Train/Validation/Test split sizes (HR): "
            f"{len(train_df)}/{len(val_df)}/{len(test_dfs)} - {[len(df) for df in test_dfs]}"
        )

        self.train_dataset = ClimateDataset(
            df=train_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage=consts.stages.train,
            generator_type=self.generator_type,
            variable=self.world_clim_variable,
            scaling_factor=self.scale_factor,
            normalize=self.normalization_method == normalization.minmax,
            standardize=self.normalization_method == normalization.zscore,
            standardize_stats=standardize_stats,
            normalize_range=self.normalization_range,
            use_elevation=self.use_elevation,
            use_mask_as_3rd_channel=self.use_mask_as_3rd_channel,
            use_global_min_max=self.use_global_min_max,
        )
        self.val_dataset = ClimateDataset(
            df=val_df,
            elevation_df=elevation_df,
            hr_size=self.hr_size,
            stage=consts.stages.val,
            generator_type=self.generator_type,
            variable=self.world_clim_variable,
            scaling_factor=self.scale_factor,
            normalize=self.normalization_method == normalization.minmax,
            standardize=self.normalization_method == normalization.zscore,
            standardize_stats=standardize_stats,
            normalize_range=self.normalization_range,
            use_elevation=self.use_elevation,
            use_mask_as_3rd_channel=self.use_mask_as_3rd_channel,
            use_global_min_max=self.use_global_min_max,
        )
        self.test_datasets = [
            ClimateDataset(
                df=test_df,
                elevation_df=elevation_df,
                hr_size=self.hr_size,
                stage=consts.stages.test,
                generator_type=self.generator_type,
                variable=self.world_clim_variable,
                scaling_factor=self.scale_factor,
                normalize=self.normalization_method == normalization.minmax,
                standardize=self.normalization_method == normalization.zscore,
                standardize_stats=standardize_stats,
                normalize_range=self.normalization_range,
                use_elevation=self.use_elevation,
                use_mask_as_3rd_channel=self.use_mask_as_3rd_channel,
                use_global_min_max=self.use_global_min_max,
            )
            for test_df in test_dfs
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> List[DataLoader]:
        if self.world_clim_variable == consts.world_clim.temp:
            return [
                DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
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
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        List[pd.DataFrame],
        pd.DataFrame,
        Union[Dict[str, float], None],
    ]:
        elevation_df = self.load_dataframe(
            consts.world_clim.elevation, f"{consts.world_clim.elevation}.csv"
        )

        stats_df = pd.read_csv(os.path.join(self.data_path, "statistics_min_max.csv"))

        if self.world_clim_variable == consts.world_clim.temp:
            train_dfs = []
            val_dfs = []
            test_dfs = []
            variables = [
                consts.world_clim.tmin,
                consts.world_clim.tmax,
                consts.world_clim.temp,
            ]
            for var in variables:
                train_dfs.append(
                    self.load_dataframe(
                        var, consts.datasets_and_preprocessing.train_csv
                    )
                )
                val_dfs.append(
                    self.load_dataframe(var, consts.datasets_and_preprocessing.val_csv)
                )
                test_dfs.append(
                    self.load_dataframe(var, consts.datasets_and_preprocessing.test_csv)
                )

            train_df = pd.concat(train_dfs)
            val_df = pd.concat(val_dfs)
        else:
            train_df = self.load_dataframe(
                self.world_clim_variable, consts.datasets_and_preprocessing.train_csv
            )
            val_df = self.load_dataframe(
                self.world_clim_variable, consts.datasets_and_preprocessing.val_csv
            )
            test_dfs = [
                self.load_dataframe(
                    self.world_clim_variable, consts.datasets_and_preprocessing.test_csv
                )
            ]

        train_df = pd.merge(
            train_df,
            stats_df,
            how="inner",
            on=[
                consts.datasets_and_preprocessing.filename,
                consts.datasets_and_preprocessing.variable,
                consts.datasets_and_preprocessing.year,
                consts.datasets_and_preprocessing.year,
            ],
        )

        val_df = pd.merge(
            val_df,
            stats_df,
            how="inner",
            on=[
                consts.datasets_and_preprocessing.filename,
                consts.datasets_and_preprocessing.variable,
                consts.datasets_and_preprocessing.year,
                consts.datasets_and_preprocessing.year,
            ],
        )

        output_test_dfs = []
        for test_df in test_dfs:
            test_df = pd.merge(
                test_df,
                stats_df,
                how="inner",
                on=[
                    consts.datasets_and_preprocessing.filename,
                    consts.datasets_and_preprocessing.variable,
                    consts.datasets_and_preprocessing.year,
                    consts.datasets_and_preprocessing.year,
                ],
            )
            output_test_dfs.append(test_df)

        return train_df, val_df, output_test_dfs, elevation_df, consts.cruts.statistics

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
            default="datasets/",
        )
        parser.add_argument(
            "--world_clim_variable",
            type=str,
            default=consts.world_clim.temp,
        )
        parser.add_argument(
            "--world_clim_multiplier",
            type=str,
            default="4x",
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--use_mask_as_3rd_channel", type=bool, default=True)
        parser.add_argument("--use_elevation", type=bool, default=True)
        parser.add_argument("--use_global_min_max", type=bool, default=True)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--hr_size", type=int, default=128)
        parser.add_argument("--scale_factor", type=int, default=4)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--normalization_method",
            type=str,
            default=normalization.minmax,
            choices=[normalization.minmax, normalization.zscore],
        )
        parser.add_argument(
            "--normalization_range",
            type=Tuple[float, float],
            default=(-1.0, 1.0),
        )
        return parser


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = 32
    args.num_workers = 1

    def plot_array(arr, figsize=None):
        plt.figure(figsize=figsize)
        plt.imshow(arr, cmap="jet")
        plt.show()

    dm = SuperResolutionDataModule(
        data_path=os.path.join("../..", args.data_path),
        world_clim_variable=args.world_clim_variable,
        world_clim_multiplier=args.world_clim_multiplier,
        generator_type=consts.models.rcan,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        scale_factor=args.scale_factor,
        seed=args.seed,
        normalization_method=args.normalization_method,
        normalization_range=args.normalization_range,
        pin_memory=args.pin_memory,
    )

    # plot_single_batch(dm.train_dataloader(), keys=[consts.batch_items.lr, consts.batch_items.hr, consts.batch_items.elevation, consts.batch_items.nearest])  # noqa E501
    # plot_single_batch(dm.val_dataloader(), keys=[consts.batch_items.lr, consts.batch_items.hr, consts.batch_items.elevation, consts.batch_items.nearest])  # noqa E501
    # plot_single_batch(dm.test_dataloader(), keys=[consts.batch_items.lr, consts.batch_items.hr, consts.batch_items.elevation, consts.batch_items.nearest])  # noqa E501

    dl = dm.val_dataloader()

    stats = consts.cruts.statistics[
        world_clim_to_cruts_mapping[args.world_clim_variable]
    ]
    stats_elev = consts.cruts.statistics[consts.cruts.elev]

    for _, batch in enumerate(dl):
        lr = batch[consts.batch_items.lr]
        hr = batch[consts.batch_items.hr]
        original = batch[consts.batch_items.original_data]
        mask = batch[consts.batch_items.mask]
        sr_nearest = batch[consts.batch_items.nearest]
        elev = batch[consts.batch_items.elevation]
        max_vals = batch[consts.batch_items.max].cpu().numpy()
        min_vals = batch[consts.batch_items.min].cpu().numpy()
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

        cols = [
            inflection.titleize(consts.batch_items.hr),
            inflection.titleize(consts.batch_items.nearest),
            inflection.titleize(consts.batch_items.elevation),
            consts.plotting.sr,
        ]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        nearest_arr = sr_nearest.squeeze(1).cpu().numpy()
        hr_arr = hr.squeeze(1).cpu().numpy()
        elev_arr = elev.squeeze(1).cpu().numpy()
        sr_arr = hr.squeeze(1).cpu().numpy()

        standardize = args.normalization_method == normalization.zscore

        for i in range(items):
            hr_arr[i][mask[i]] = np.nan
            nearest_arr[i][mask[i]] = np.nan
            elev_arr[i][mask[i]] = np.nan
            sr_arr[i][mask[i]] = np.nan

            axes[i][0].imshow(
                hr_arr[i],
                cmap=cmap,
                vmin=stats[consts.stats.normalized_min]
                if standardize
                else args.normalization_range[0],
                vmax=stats[consts.stats.normalized_max]
                if standardize
                else args.normalization_range[1],
            )
            axes[i][1].imshow(
                nearest_arr[i],
                cmap=cmap,
                vmin=stats[consts.stats.normalized_min]
                if standardize
                else args.normalization_range[0],
                vmax=stats[consts.stats.normalized_max]
                if standardize
                else args.normalization_range[1],
            )
            axes[i][2].imshow(
                elev_arr[i],
                cmap=cmap,
            )
            axes[i][3].imshow(
                sr_arr[i],
                cmap=cmap,
                vmin=stats[consts.stats.normalized_min]
                if standardize
                else args.normalization_range[0],
                vmax=stats[consts.stats.normalized_max]
                if standardize
                else args.normalization_range[1],
            )

        fig.suptitle(f"Validation batch, epoch={0}", fontsize=16)
        plt.show()

        break
