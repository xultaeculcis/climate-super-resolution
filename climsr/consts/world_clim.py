# -*- coding: utf-8 -*-
"""Consts with default values for the World Clim dataset."""

elevation = "elevation"
tmin = "tmin"
tmax = "tmax"
temp = "temp"
tavg = "tavg"
prec = "prec"
elev = "elev"
variables_wc = [
    tmin,
    tmax,
    temp,
    prec,
]
temperature_vars = [tmin, temp, tmax]
pattern_wc = "*.tif"
resized_dir = "resized"
tiles_dir = "tiles"
resolution_multipliers = [
    ("1x", 1 / 12),
    ("2x", 1 / 6),
    ("4x", 1 / 3),
]
CRS = "EPSG:4326"
elevation_missing_indicator = -32768.0

statistics = {
    tmin: {
        "mean": 2.6400537490844727,
        "std": 17.716720581054688,
    },
    tmax: {
        "mean": 13.901198387145996,
        "std": 18.491439819335938,
    },
    prec: {
        "mean": 59.85238265991211,
        "std": 81.54020690917969,
    },
    elevation: {
        "mean": 1120.0989990234375,
        "std": 1154.4285888671875,
    },
}

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
