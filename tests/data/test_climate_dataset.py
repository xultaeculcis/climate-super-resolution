# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd

from climsr import consts
from climsr.data.sr.climate_dataset import ClimateDataset

data_dir = os.path.join(str(Path(__file__).parent.parent.parent), "datasets/pre-processed/feather")


def test_should_return_proper_data():
    # arrange
    tile_df = pd.read_feather(os.path.join(data_dir, "tmax/train.feather"))
    stats_df = pd.read_feather(os.path.join(data_dir, "statistics_min_max.feather"))
    df = pd.merge(
        tile_df,
        stats_df,
        how="inner",
        on=[
            consts.datasets_and_preprocessing.filename,
            consts.datasets_and_preprocessing.variable,
            consts.datasets_and_preprocessing.year,
            consts.datasets_and_preprocessing.month,
        ],
    )
    sut = ClimateDataset(
        df=df,
        elevation_df=pd.read_feather(os.path.join(data_dir, "elev/elev.feather")),
        generator_type=consts.models.rcan,
        variable=consts.world_clim.tmin,
    )

    # act
    out = sut[1]

    # assert
    assert out[consts.batch_items.lr].shape == (3, 32, 32)
    assert out[consts.batch_items.hr].shape == (1, 128, 128)
    assert out[consts.batch_items.elevation].shape == (1, 128, 128)
    assert out[consts.batch_items.mask].shape == (1, 128, 128)
