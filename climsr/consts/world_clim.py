# -*- coding: utf-8 -*-
"""Consts with default values for the World Clim dataset."""
import numpy as np

elev = "elev"
tmin = "tmin"
tmax = "tmax"
temp = "temp"
tavg = "tavg"
prec = "prec"
variables_wc = [
    tmin,
    tmax,
    tavg,
    prec,
]
temperature_vars = [tmin, tavg, tmax]
pattern_wc = "*.tif"
resized_dir = "resized"
tiles_dir = "tiles"
CRS = "EPSG:4326"

elevation_missing_indicator = -32768.0
scenario_missing_indicator = np.float32(-3.4e38)
missing_indicators = [
    elevation_missing_indicator,
    scenario_missing_indicator,
]
target_missing_indicator = np.nan

gcm_BCC_CSM2_MR = "BCC-CSM2-MR"
gcm_CNRM_CM6_1 = "CNRM-CM6-1"
gcm_CNRM_ESM2_1 = "CNRM-ESM2-1"
gcm_CanESM5 = "CanESM5"
gcm_GFDL_ESM4 = "GFDL-ESM4"
gcm_IPSL_CM6A_LR = "IPSL-CM6A-LR"
gcm_MIROC_ES2L = "MIROC-ES2L"
gcm_MIROC6 = "MIROC6"
gcm_MRI_ESM2_0 = "MRI-ESM2-0"
GCMs = [
    gcm_BCC_CSM2_MR,
    gcm_CNRM_CM6_1,
    gcm_CNRM_ESM2_1,
    gcm_CanESM5,
    gcm_GFDL_ESM4,
    gcm_IPSL_CM6A_LR,
    gcm_MIROC_ES2L,
    gcm_MIROC6,
    gcm_MRI_ESM2_0,
]

scenario_ssp126 = "ssp126"
scenario_ssp245 = "ssp245"
scenario_ssp370 = "ssp370"
scenario_ssp585 = "ssp585"
scenarios = [
    scenario_ssp126,
    scenario_ssp245,
    scenario_ssp370,
    scenario_ssp585,
]

resolution_2_5m = "2.5m"
resolution_5m = "5m"
resolution_10m = "10m"
data_resolutions = [
    resolution_2_5m,
    resolution_5m,
    resolution_10m,
]

target_hr_resolution = (2880, 1440)
preprocessing_scaling_factor_2_5m = 1.0 / 3.0
preprocessing_scaling_factor_5m = 2.0 / 3.0
preprocessing_scaling_factor_10m = 4.0 / 3.0
