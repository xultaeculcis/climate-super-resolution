# -*- coding: utf-8 -*-
class CRUTSConfig:
    """Config class with default values for the CRU-TS dataset."""

    tmn = "tmn"
    tmx = "tmx"
    tmp = "tmp"
    pre = "pre"
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
        },
        tmx: {
            "mean": 13.901198387145996,
            "std": 18.491439819335938,
        },
        tmp: {
            "mean": 8.26048469543457,
            "std": 17.99039649963379,
        },
        pre: {
            "mean": 59.85238265991211,
            "std": 81.54020690917969,
        },
    }

    cruts_original_shape = (360, 720)
