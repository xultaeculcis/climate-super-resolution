# -*- coding: utf-8 -*-
import pandas as pd

from sr.data.geo_tiff_inference_dataset import GeoTiffInferenceDataset

europe_extent_expected_shape_hr = [412, 452]
europe_extent_expected_shape_lr = [103, 113]
var = "tmp"
df = pd.read_csv("./datasets/statistics_min_max.csv")
df = df[(df["dataset"] == "cru-ts") & (df["variable"] == var)]
tiff_dir = f"/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent/{var}"
tiff_df = df
elevation_file = "/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent/elevation/wc2.1_2.5m_elev.tif"
land_mask_file = "/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent/mask/wc2.1_2.5m_prec_1961-01.tif"
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
        tiff_df=tiff_df,
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


def test_should_return_proper_data_when_used_with_srcnn():
    # arrange
    sut = _get_dataset("srcnn")

    # act
    out = sut[1]

    # assert
    assert out["lr"].shape == (3, *europe_extent_expected_shape_hr)
    assert out["elevation"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["elevation_lr"].shape == (1, *europe_extent_expected_shape_lr)
    assert out["nearest"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["mask"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["mask_np"].shape == tuple(europe_extent_expected_shape_hr)
    assert out["min"] is not None
    assert out["max"] is not None


def test_should_return_proper_data_when_used_not_with_srcnn():
    # arrange
    sut = _get_dataset("rcan")

    # act
    out = sut[1]

    # assert
    assert out["lr"].shape == (3, *europe_extent_expected_shape_lr)
    assert out["elevation"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["elevation_lr"].shape == (1, *europe_extent_expected_shape_lr)
    assert out["nearest"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["mask"].shape == (1, *europe_extent_expected_shape_hr)
    assert out["mask_np"].shape == tuple(europe_extent_expected_shape_hr)
    assert out["min"] is not None
    assert out["max"] is not None
