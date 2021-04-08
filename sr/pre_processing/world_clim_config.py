# -*- coding: utf-8 -*-
class WorldClimConfig:
    """Config class with default values for the World Clim dataset."""

    elevation = "elevation"
    tmin = "tmin"
    tmax = "tmax"
    prec = "prec"
    variables_wc = [
        "tmin",
        "tmax",
        "prec",
    ]
    pattern_wc = "*.tif"
    resized_dir = "resized"
    tiles_dir = "tiles"
    resolution_multipliers = [
        ("1x", 1 / 12),
        ("2x", 1 / 6),
        ("4x", 1 / 3),
    ]
    CRS = "EPSG:4326"

    statistics = {
        elevation: {
            "mean": 1120.0989990234375,
            "std": 1154.4285888671875,
        },
    }
