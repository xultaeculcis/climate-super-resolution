# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import climsr.consts as consts
from climsr.data import normalization
from climsr.data.climate_dataset import ClimateDataset
from climsr.core.config import SuperResolutionDataConfig

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"

default_cfg = SuperResolutionDataConfig()


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[SuperResolutionDataConfig] = default_cfg):
        super(SuperResolutionDataModule, self).__init__()

        assert cfg.hr_size % cfg.scale_factor == 0

        self.cfg = cfg
        # self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        train_df, val_df, test_dfs, elevation_df, standardize_stats = self.load_data()

        logging.info(
            f"'{self.cfg.world_clim_variable}' - Train/Validation/Test split sizes (HR): "
            f"{len(train_df)}/{len(val_df)}/{len(test_dfs)} - {[len(df) for df in test_dfs]}"
        )

        self.train_dataset = ClimateDataset(
            df=train_df,
            elevation_df=elevation_df,
            hr_size=self.cfg.hr_size,
            stage=consts.stages.train,
            generator_type=self.cfg.generator_type,
            variable=self.cfg.world_clim_variable,
            scaling_factor=self.cfg.scale_factor,
            normalize=self.cfg.normalization_method == normalization.minmax,
            standardize=self.cfg.normalization_method == normalization.zscore,
            standardize_stats=standardize_stats,
            normalize_range=self.cfg.normalization_range,
            use_elevation=self.cfg.use_elevation,
            use_mask_as_3rd_channel=self.cfg.use_mask_as_3rd_channel,
            use_global_min_max=self.cfg.use_global_min_max,
        )
        self.val_dataset = ClimateDataset(
            df=val_df,
            elevation_df=elevation_df,
            hr_size=self.cfg.hr_size,
            stage=consts.stages.val,
            generator_type=self.cfg.generator_type,
            variable=self.cfg.world_clim_variable,
            scaling_factor=self.cfg.scale_factor,
            normalize=self.cfg.normalization_method == normalization.minmax,
            standardize=self.cfg.normalization_method == normalization.zscore,
            standardize_stats=standardize_stats,
            normalize_range=self.cfg.normalization_range,
            use_elevation=self.cfg.use_elevation,
            use_mask_as_3rd_channel=self.cfg.use_mask_as_3rd_channel,
            use_global_min_max=self.cfg.use_global_min_max,
        )
        self.test_datasets = [
            ClimateDataset(
                df=test_df,
                elevation_df=elevation_df,
                hr_size=self.cfg.hr_size,
                stage=consts.stages.test,
                generator_type=self.cfg.generator_type,
                variable=self.cfg.world_clim_variable,
                scaling_factor=self.cfg.scale_factor,
                normalize=self.cfg.normalization_method == normalization.minmax,
                standardize=self.cfg.normalization_method == normalization.zscore,
                standardize_stats=standardize_stats,
                normalize_range=self.cfg.normalization_range,
                use_elevation=self.cfg.use_elevation,
                use_mask_as_3rd_channel=self.cfg.use_mask_as_3rd_channel,
                use_global_min_max=self.cfg.use_global_min_max,
            )
            for test_df in test_dfs
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> List[DataLoader]:
        if self.cfg.world_clim_variable == consts.world_clim.temp:
            return [
                DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                )
                for test_dataset in self.test_datasets
            ]
        else:
            return [
                DataLoader(
                    self.test_datasets[0],
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                )
            ]

    def load_dataframe(self, var, filename) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(
                self.cfg.data_path,
                var,
                self.cfg.world_clim_multiplier,
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

        stats_df = pd.read_csv(
            os.path.join(self.cfg.data_path, "statistics_min_max.csv")
        )

        if self.cfg.world_clim_variable == consts.world_clim.temp:
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
                self.cfg.world_clim_variable,
                consts.datasets_and_preprocessing.train_csv,
            )
            val_df = self.load_dataframe(
                self.cfg.world_clim_variable, consts.datasets_and_preprocessing.val_csv
            )
            test_dfs = [
                self.load_dataframe(
                    self.cfg.world_clim_variable,
                    consts.datasets_and_preprocessing.test_csv,
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
