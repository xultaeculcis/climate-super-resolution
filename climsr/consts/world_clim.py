# -*- coding: utf-8 -*-
"""Consts with default values for the World Clim dataset."""

elevation = "elevation"
tmin = "tmin"
tmax = "tmax"
temp = "temp"
prec = "prec"
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