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
data_dir = os.path.join(root_dir, "datasets")

df = pd.read_feather(
    os.path.join(data_dir, consts.datasets_and_preprocessing.preprocessing_output_path, "feather/statistics_min_max.feather")
)
df = df[(df[consts.datasets_and_preprocessing.dataset] == "cru-ts") & (df[consts.datasets_and_preprocessing.variable] == var)]
tiff_dir = os.path.join(data_dir, consts.datasets_and_preprocessing.preprocessing_output_path, f"cruts/europe-extent/{var}")
elevation_file = os.path.join(
    data_dir,
    consts.datasets_and_preprocessing.preprocessing_output_path,
    consts.datasets_and_preprocessing.world_clim_preprocessing_out_path,
    consts.cruts.europe_extent,
    consts.world_clim.elev,
    "wc2.1_5m_elev.tif",
)
land_mask_file = os.path.join(
    data_dir,
    consts.datasets_and_preprocessing.preprocessing_output_path,
    consts.datasets_and_preprocessing.world_clim_preprocessing_out_path,
    consts.cruts.europe_extent,
    consts.world_clim.tmin,
    "wc2.1_5m_tmin_01.tif",
)
variable = var
scaling_factor = 4
normalize = True
standardize = False
standardize_stats = pd.read_feather(
    os.path.join(data_dir, consts.datasets_and_preprocessing.preprocessing_output_path, "feather/statistics_zscore.feather")
)
normalize_range = (-1.0, 1.0)
use_elevation = True
use_global_min_max = True
use_mask = True


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
        use_mask=use_mask,
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
