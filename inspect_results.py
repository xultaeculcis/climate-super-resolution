# -*- coding: utf-8 -*-
import sys
import logging

import xarray as xr
import pandas as pd

from configs.cruts_config import CRUTSConfig
from result_inspection.consts import lats, lons, alts
from result_inspection.models import CompareStatsResults

logging.basicConfig(level=logging.INFO)
logging.info("Python %s on %s" % (sys.version, sys.platform))

ds_temp_nn = xr.open_dataset(
    "/media/xultaeculcis/2TB/datasets/cruts/inference-europe-extent-nc/esrgan.cru_ts4.04.nn.inference.1901.2019.tmp.dat.nc"  # noqa E501
)
ds_temp_cru = xr.open_dataset(
    "/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmp.dat.nc"
)
peaks = pd.read_csv("./datasets/mountain_peaks.csv")

if __name__ == "__main__":
    # data from Tomasz
    logging.info("Tomasz's data")
    may_only = pd.date_range(
        ds_temp_cru["time"][5].values, ds_temp_cru["time"][-1].values, freq="AS-MAY"
    )

    results = CompareStatsResults.compute(
        CRUTSConfig.tmp, may_only, lats, lons, alts, ds_temp_cru, ds_temp_nn
    )
    results.print_comparison_summary()
    results.line_plot()
    results.box_plot()

    # mountain peaks
    logging.info("Mountain peaks")
    results = CompareStatsResults.compute(
        CRUTSConfig.tmp,
        may_only,
        peaks["lat"].values,
        peaks["lon"].values,
        peaks["altitude"].values,
        ds_temp_cru,
        ds_temp_nn,
        peaks["mountain_peak_name"].values,
    )
    results.print_comparison_summary()
    results.line_plot()
    results.box_plot()

    # only 2 locations
    logging.info("Only 2 locations")
    results = CompareStatsResults.compute(
        CRUTSConfig.tmp,
        may_only,
        lats[-2:],
        lons[-2:],
        alts[-2:],
        ds_temp_cru,
        ds_temp_nn,
    )
    results.print_comparison_summary()
    results.line_plot()
    results.box_plot()
