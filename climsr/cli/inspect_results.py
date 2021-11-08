# -*- coding: utf-8 -*-
import logging

import hydra
import pandas as pd
import xarray as xr
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import climsr.consts as consts
from climsr.core.config import ResultInspectionConfig
from climsr.result_inspection.models import CompareStatsResults

logging.basicConfig(level=logging.INFO)


def run(ds_temp_nn: xr.Dataset, ds_temp_cru: xr.Dataset, peaks: pd.DataFrame) -> None:
    # data from Tomasz
    logging.info("Tomasz's data")
    may_only = pd.date_range(ds_temp_cru["time"][5].values, ds_temp_cru["time"][-1].values, freq="AS-MAY")

    results = CompareStatsResults.compute(
        consts.cruts.tmp,
        may_only,
        consts.result_inspection.lats,
        consts.result_inspection.lons,
        consts.result_inspection.alts,
        ds_temp_cru,
        ds_temp_nn,
    )
    results.print_comparison_summary()
    results.line_plot()
    results.box_plot()

    # mountain peaks
    logging.info("Mountain peaks")
    results = CompareStatsResults.compute(
        consts.cruts.tmp,
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
        consts.cruts.tmp,
        may_only,
        consts.result_inspection.lats[-2:],
        consts.result_inspection.lons[-2:],
        consts.result_inspection.alts[-2:],
        ds_temp_cru,
        ds_temp_nn,
    )
    results.print_comparison_summary()
    results.line_plot()
    results.box_plot()


def main(cfg: ResultInspectionConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    ds_temp_nn = xr.open_dataset(to_absolute_path(cfg.ds_temp_nn_path))
    ds_temp_cru = xr.open_dataset(to_absolute_path(cfg.ds_temp_cru_path))
    peaks = pd.read_feather(to_absolute_path(cfg.peaks_feather))
    run(ds_temp_nn, ds_temp_cru, peaks)


@hydra.main(config_path="../../conf", config_name="result_inspection")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg.get("result_inspection"))


if __name__ == "__main__":
    hydra_entry()
