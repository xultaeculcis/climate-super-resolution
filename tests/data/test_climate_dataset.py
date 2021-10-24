# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from pytest_cases import parametrize_with_cases

from climsr import consts
from climsr.data.sr.climate_dataset import ClimateDataset

data_dir = os.path.join(str(Path(__file__).parent.parent.parent), "datasets/pre-processed/feather")

expected_shape_hr_3d = (3, 128, 128)
expected_shape_hr_2d = (2, 128, 128)
expected_shape_hr_1d = (1, 128, 128)
expected_shape_lr_3d = (3, 32, 32)
expected_shape_lr_2d = (2, 32, 32)
expected_shape_lr_1d = (1, 32, 32)


def case_rcan_no_elevation_and_no_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.rcan, False, False, expected_shape_lr_1d


def case_srcnn_no_elevation_and_no_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.srcnn, False, False, expected_shape_hr_1d


def case_rcan_with_elevation_and_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.rcan, True, True, expected_shape_lr_3d


def case_srcnn_with_elevation_and_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.srcnn, True, True, expected_shape_hr_3d


def case_rcan_with_elevation() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.rcan, True, False, expected_shape_lr_2d


def case_srcnn_with_elevation() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.srcnn, True, False, expected_shape_hr_2d


def case_rcan_with_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.rcan, False, True, expected_shape_lr_2d


def case_srcnn_with_mask() -> Tuple[str, bool, bool, Tuple[int, int, int]]:
    return consts.models.srcnn, False, True, expected_shape_hr_2d


@parametrize_with_cases("generator_name,use_elevation,use_mask,expected_shape", cases=".")
def test_should_return_proper_data(
    generator_name: str, use_elevation: bool, use_mask: bool, expected_shape: Tuple[int, int, int]
) -> None:
    # arrange
    tile_df = pd.read_feather(os.path.join(data_dir, "tmax/train.feather"))
    stats_df = pd.read_feather(os.path.join(data_dir, "statistics_min_max.feather"))
    zscore_stats_df = pd.read_feather(os.path.join(data_dir, "statistics_zscore.feather"))
    df = pd.merge(
        tile_df,
        stats_df,
        how="inner",
        on=[
            consts.datasets_and_preprocessing.filename,
            consts.datasets_and_preprocessing.variable,
            consts.datasets_and_preprocessing.year,
            consts.datasets_and_preprocessing.month,
            consts.datasets_and_preprocessing.resolution,
        ],
    )
    sut = ClimateDataset(
        df=df,
        elevation_df=pd.read_feather(os.path.join(data_dir, "elev/elev.feather")),
        generator_type=generator_name,
        variable=consts.world_clim.tmin,
        use_elevation=use_elevation,
        use_mask=use_mask,
        standardize_stats=zscore_stats_df,
    )

    # act
    out = sut[1]

    # assert
    assert out[consts.batch_items.lr].shape == expected_shape
    assert out[consts.batch_items.hr].shape == expected_shape_hr_1d
    assert out[consts.batch_items.elevation].shape == expected_shape_hr_1d
    assert out[consts.batch_items.mask].shape == expected_shape_hr_1d
