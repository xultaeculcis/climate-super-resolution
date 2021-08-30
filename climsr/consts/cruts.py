# -*- coding: utf-8 -*-
"""Consts with default values for the CRU-TS dataset."""

europe_extent = "europe-extent"
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
temperature_vars = [tmn, tmp, tmx]
cts_variable_files = [f"cru_ts4.05.1901.2020.{var}.dat.nc" for var in variables_cts]
full_res_dir = "full-res"
file_pattern = "cru_ts4.05.1901.2020.{0}.dat.nc"
degree_per_pix = 0.5
CRS = "EPSG:4326"
cruts_original_shape = (360, 720)
