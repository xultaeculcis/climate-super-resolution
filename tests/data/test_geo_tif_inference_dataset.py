# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from pytest_cases import parametrize_with_cases

import climsr.consts as consts
from climsr.data.sr.geo_tiff_inference_dataset import GeoTiffInferenceDataset

europe_extent_expected_shape_hr_1d = (1, 412, 452)
europe_extent_expected_shape_hr_3d = (3, 412, 452)
europe_extent_expected_shape_lr_1d = (1, 103, 113)
europe_extent_expected_shape_lr_3d = (3, 103, 113)
var = "tmp"

root_dir = str(Path(__file__).parent.parent.parent)
data_dir = os.path.join(root_dir, "datasets/pre-processed")

df = pd.read_csv(os.path.join(data_dir, "csv/statistics_min_max.csv"))
df = df[(df[consts.datasets_and_preprocessing.dataset] == "cru-ts") & (df[consts.datasets_and_preprocessing.variable] == var)]
tiff_dir = os.path.join(data_dir, f"cruts/europe-extent/{var}")
elevation_file = os.path.join(data_dir, "cruts/europe-extent/elevation/wc2.1_2.5m_elev.tif")
land_mask_file = os.path.join(data_dir, "cruts/europe-extent/mask/wc2.1_2.5m_tmin_1961-01.tif")
variable = var
scaling_factor = 4
normalize = True
standardize = False
standardize_stats = None
normalize_range = (-1.0, 1.0)
use_elevation = True
use_global_min_max = True
use_mask_as_3rd_channel = True


def _get_dataset(gen):
    return GeoTiffInferenceDataset(
        tiff_dir=tiff_dir,
        tiff_df=df,
        elevation_file=elevation_file,
        land_mask_file=land_mask_file,
        generator_type=gen,
        variable=var,
        scaling_factor=scaling_factor,
        normalize=normalize,
        standardize=standardize,
        standardize_stats=standardize_stats,
        normalize_range=normalize_range,
        use_elevation=use_elevation,
        use_global_min_max=use_global_min_max,
        use_mask_as_3rd_channel=use_mask_as_3rd_channel,
    )


def case_rcan() -> Tuple[str, Tuple[int, int, int]]:
    return consts.models.rcan, europe_extent_expected_shape_lr_3d


def case_srcnn() -> Tuple[str, Tuple[int, int, int]]:
    return consts.models.srcnn, europe_extent_expected_shape_hr_3d


@parametrize_with_cases("generator_name,expected_shape", cases=".")
def test_should_return_expected_data_shapes_for_each_generator_type(generator_name: str, expected_shape: Tuple[int, int, int]):
    # arrange
    sut = _get_dataset(generator_name)

    # act
    out = sut[1]

    # assert
    assert out[consts.batch_items.lr].shape == expected_shape
    assert out[consts.batch_items.elevation].shape == europe_extent_expected_shape_hr_1d
    assert out[consts.batch_items.elevation_lr].shape == europe_extent_expected_shape_lr_1d
    assert out[consts.batch_items.nearest].shape == europe_extent_expected_shape_hr_1d
    assert out[consts.batch_items.mask].shape == europe_extent_expected_shape_hr_1d
    assert out[consts.batch_items.mask_np].shape == tuple(europe_extent_expected_shape_hr_1d[1:])
    assert out[consts.batch_items.min] is not None
    assert out[consts.batch_items.max] is not None
