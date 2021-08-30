# -*- coding: utf-8 -*-
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from core.data import DataModuleBase
from data.sr.climate_dataset import ClimateDataset

import climsr.consts as consts
from climsr.core.config import SuperResolutionDataConfig
from climsr.data import normalization

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"

default_cfg = SuperResolutionDataConfig()


class SuperResolutionDataModule(DataModuleBase):
    def __init__(self, cfg: Optional[SuperResolutionDataConfig] = default_cfg):
        super(SuperResolutionDataModule, self).__init__()

        assert cfg.hr_size % cfg.scale_factor == 0

        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        train_df, val_df, test_dfs, elevation_df, standardize_stats = self.load_data()

        logging.info(
            f"'{self.cfg.world_clim_variable}' - Train/Validation/Test split sizes (HR): "
            f"{len(train_df)}/{len(val_df)}/{len(test_dfs)} - {[len(df) for df in test_dfs]}"
        )

        self.ds["train"] = ClimateDataset(
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
        self.ds["val"] = ClimateDataset(
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
        self.ds["test"] = [
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], pd.DataFrame, Union[Dict[str, float], None]]:
        elevation_df = self.load_dataframe(consts.world_clim.elev, f"{consts.world_clim.elev}.csv")

        stats_df = pd.read_csv(os.path.join(self.cfg.data_path, "statistics_min_max.csv"))

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
                train_dfs.append(self.load_dataframe(var, consts.datasets_and_preprocessing.train_csv))
                val_dfs.append(self.load_dataframe(var, consts.datasets_and_preprocessing.val_csv))
                test_dfs.append(self.load_dataframe(var, consts.datasets_and_preprocessing.test_csv))

            train_df = pd.concat(train_dfs)
            val_df = pd.concat(val_dfs)
        else:
            train_df = self.load_dataframe(
                self.cfg.world_clim_variable,
                consts.datasets_and_preprocessing.train_csv,
            )
            val_df = self.load_dataframe(self.cfg.world_clim_variable, consts.datasets_and_preprocessing.val_csv)
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

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {
            "world_clim_variable": self.cfg.world_clim_variable,
            "normalization_method": self.cfg.normalization_method,
            "normalization_range": self.cfg.normalization_range,
            "generator_type": self.cfg.generator_type,
        }
