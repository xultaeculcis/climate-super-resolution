# -*- coding: utf-8 -*-
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from hydra.utils import to_absolute_path

import climsr.consts as consts
from climsr.core.config import SuperResolutionDataConfig
from climsr.core.data import DataModuleBase
from climsr.data import normalization
from climsr.data.sr.climate_dataset import ClimateDataset

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"

default_cfg = SuperResolutionDataConfig()


class SuperResolutionDataModule(DataModuleBase):
    def __init__(self, cfg: Optional[SuperResolutionDataConfig] = default_cfg):
        super(SuperResolutionDataModule, self).__init__()

        assert consts.world_clim.resolution_2_5m in cfg.resolutions, "2.5m resolution is required!"

        self.cfg = cfg
        self.ds = dict()
        self._setup()

    def _build_dataset(
        self, stage: str, df: pd.DataFrame, elevation_df: pd.DataFrame, standardize_stats_df: pd.DataFrame
    ) -> ClimateDataset:
        return ClimateDataset(
            df=df,
            elevation_df=elevation_df,
            stage=stage,
            generator_type=self.cfg.generator_type,
            variable=self.cfg.world_clim_variable,
            scaling_factor=self.cfg.scale_factor,
            normalize=self.cfg.normalization_method == normalization.minmax,
            standardize=self.cfg.normalization_method == normalization.zscore,
            standardize_stats=standardize_stats_df,
            normalize_range=self.cfg.normalization_range,
            use_elevation=self.cfg.use_elevation,
            use_mask=self.cfg.use_mask,
            use_global_min_max=self.cfg.use_global_min_max,
            europe_extent=self.cfg.europe_extent,
            transforms_cfg=self.cfg.transforms,
        )

    def _setup(self) -> None:
        train_df, val_df, test_dfs, elevation_df, standardize_stats = self._load_data()

        logging.info(
            f"'{self.cfg.world_clim_variable}' - Train/Validation/Test split sizes (HR): "
            f"{len(train_df)}/{len(val_df)}/{len(test_dfs)} - {[len(df) for df in test_dfs]}"
        )

        for stage, frames in zip(consts.stages.stages, [train_df, val_df, test_dfs]):
            if type(frames) is list:
                self.ds[stage] = [self._build_dataset(stage, df, elevation_df, standardize_stats) for df in frames]
            else:
                self.ds[stage] = self._build_dataset(stage, frames, elevation_df, standardize_stats)

    def _load_dataframe(self, var: str, filename: str) -> pd.DataFrame:
        def as_extent_filename_if_needed(fname: str) -> str:
            if self.cfg.europe_extent:
                fname, ext = os.path.splitext(fname)
                fname = fname + "_europe_extent"
                return f"{fname}{ext}"
            return fname

        return pd.read_feather(
            os.path.join(
                to_absolute_path(self.cfg.data_path),
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.feather_path,
                var,
                as_extent_filename_if_needed(filename),
            )
        )

    def _filter_df(self, df):
        if not self.cfg.use_extra_data:
            df = df[df[consts.datasets_and_preprocessing.year] <= 2020]
        df = df[df[consts.datasets_and_preprocessing.resolution].isin(self.cfg.resolutions)]
        return df

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        elevation_df = self._load_dataframe(consts.world_clim.elev, f"{consts.world_clim.elev}.feather")
        elevation_df = self._filter_df(elevation_df)

        stats_df = pd.read_feather(
            os.path.join(
                to_absolute_path(self.cfg.data_path),
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.feather_path,
                consts.datasets_and_preprocessing.min_max_stats_filename,
            )
        )
        stats_df = self._filter_df(stats_df)

        if self.cfg.world_clim_variable == consts.world_clim.temp:
            train_dfs = []
            val_dfs = []
            test_dfs = []
            for var in consts.world_clim.temperature_vars:
                train_dfs.append(self._filter_df(self._load_dataframe(var, consts.datasets_and_preprocessing.train_feather)))
                val_dfs.append(self._filter_df(self._load_dataframe(var, consts.datasets_and_preprocessing.val_feather)))
                test_dfs.append(self._filter_df(self._load_dataframe(var, consts.datasets_and_preprocessing.test_feather)))

            train_df = pd.concat(train_dfs)
            val_df = pd.concat(val_dfs)
        else:
            train_df = self._load_dataframe(
                self.cfg.world_clim_variable,
                consts.datasets_and_preprocessing.train_feather,
            )
            val_df = self._load_dataframe(self.cfg.world_clim_variable, consts.datasets_and_preprocessing.val_feather)
            test_dfs = [
                self._load_dataframe(
                    self.cfg.world_clim_variable,
                    consts.datasets_and_preprocessing.test_feather,
                )
            ]

        merge_columns = [
            consts.datasets_and_preprocessing.filename,
            consts.datasets_and_preprocessing.variable,
            consts.datasets_and_preprocessing.year,
            consts.datasets_and_preprocessing.month,
            consts.datasets_and_preprocessing.resolution,
        ]

        if self.cfg.europe_extent:
            stats_df = stats_df.drop(columns=consts.datasets_and_preprocessing.file_path, axis=1)

        train_df = pd.merge(
            train_df,
            stats_df,
            how="inner",
            on=merge_columns,
        )

        val_df = pd.merge(
            val_df,
            stats_df,
            how="inner",
            on=merge_columns,
        )

        output_test_dfs = []
        for test_df in test_dfs:
            test_df = pd.merge(
                test_df,
                stats_df,
                how="inner",
                on=merge_columns,
            )
            output_test_dfs.append(test_df)

        standardization_stats_df = pd.read_feather(
            os.path.join(
                to_absolute_path(self.cfg.data_path),
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.feather_path,
                consts.datasets_and_preprocessing.zscore_stats_filename,
            )
        )

        return train_df, val_df, output_test_dfs, elevation_df, standardization_stats_df

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {
            "data_path": to_absolute_path(self.cfg.data_path),
            "world_clim_variable": self.cfg.world_clim_variable,
            "normalization_method": self.cfg.normalization_method,
            "normalization_range": self.cfg.normalization_range,
            "generator_type": self.cfg.generator_type,
            "batch_size": self.cfg.batch_size,
            "use_elevation": self.cfg.use_elevation,
            "use_mask": self.cfg.use_mask,
            "use_global_min_max": self.cfg.use_global_min_max,
            "use_extra_data": self.cfg.use_extra_data,
            "resolutions": self.cfg.resolutions,
            "transforms": self.cfg.transforms,
            "seed": self.cfg.seed,
        }
