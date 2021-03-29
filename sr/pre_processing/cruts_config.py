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
