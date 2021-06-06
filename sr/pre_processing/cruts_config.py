# -*- coding: utf-8 -*-
class CRUTSConfig:
    """Config class with default values for the CRU-TS dataset."""

    tmn = "tmn"
    tmx = "tmx"
    tmp = "tmp"
    pre = "pre"
    elev = "elevation"
    variables_cts = [
        tmn,
        tmx,
        tmp,
        pre,
    ]
    cts_variable_files = [f"cru_ts4.04.1901.2019.{var}.dat.nc" for var in variables_cts]
    full_res_dir = "full-res"
    tiles_dir = "tiles"
    sub_dirs_cts = [
        full_res_dir,
        tiles_dir,
    ]
    file_pattern = "cru_ts4.04.1901.2019.{0}.dat.nc"
    degree_per_pix = 0.5
    CRS = "EPSG:4326"

    statistics = {
        tmn: {
            "mean": 2.6400537490844727,
            "std": 17.716720581054688,
            "min": -63.70000076293945,
            "max": 33.60000228881836,
            "normalized_min": -3.7444864372429607,
            "normalized_max": 1.7474979004599809,
            "nan_sub": -7.0,
        },
        tmx: {
            "mean": 13.901198387145996,
            "std": 18.491439819335938,
            "min": -56.5,
            "max": 48.5,
            "normalized_min": -3.8072298293887132,
            "normalized_max": 1.871070197156785,
            "nan_sub": -7.0,
        },
        tmp: {
            "mean": 8.26048469543457,
            "std": 17.99039649963379,
            "min": -60.10000228881836,
            "max": 39.70000076293945,
            "normalized_min": -3.7998301424146868,
            "normalized_max": 1.7475711884634104,
            "nan_sub": -7.0,
        },
        pre: {
            "mean": 59.85238265991211,
            "std": 81.54020690917969,
            "min": 0.0,
            "max": 3332.0,
            "normalized_min": -0.7340228531226044,
            "normalized_max": 40.12924945640844,
            "nan_sub": -0.7340228531226044,
        },
        elev: {
            "mean": 1120.0989990234375,
            "std": 1154.4285888671875,
            "min": -415.0,
            "max": 7412.0,
            "normalized_min": -1.3297478947851713,
            "normalized_max": 5.45022956385552,
            "nan_sub": -0.9702626911032551,
        },
    }

    cruts_original_shape = (360, 720)
