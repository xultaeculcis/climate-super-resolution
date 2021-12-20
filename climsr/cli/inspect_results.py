# -*- coding: utf-8 -*-
import logging
import os
from typing import List, Optional, Union

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import climsr.consts as consts
from climsr.core.config import ResultInspectionConfig
from climsr.result_inspection.models import CompareStatsResults

logging.basicConfig(level=logging.INFO)


def _run_internal(
    prefix: str,
    var: str,
    time_range: pd.DatetimeIndex,
    lats: Union[List[float], np.ndarray],
    lons: Union[List[float], np.ndarray],
    alts: Union[List[float], np.ndarray],
    ds_cru: xr.Dataset,
    ds_nn: xr.Dataset,
    results_dir: str,
    names: Optional[Union[List[str], np.ndarray]] = None,
):
    logging.info(prefix)
    results = CompareStatsResults.compute(
        var=var,
        time_range=time_range,
        lats=lats,
        lons=lons,
        alts=alts,
        ds_cru=ds_cru,
        ds_nn=ds_nn,
        names=names,
    )
    results.print_comparison_summary()
    results.line_plot(save_path=os.path.join(results_dir, f"{prefix}_line_plot.png"))
    results.box_plot(save_path=os.path.join(results_dir, f"{prefix}_box_plot.png"))
    results.to_frame().to_csv(os.path.join(results_dir, f"{prefix}_results.csv"))


def run(ds_temp_nn: xr.Dataset, ds_temp_cru: xr.Dataset, peaks: pd.DataFrame, results_dir: str) -> None:
    # data from Tomasz
    results_dir = to_absolute_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    may_only = pd.date_range(ds_temp_cru["time"][5].values, ds_temp_cru["time"][-1].values, freq="AS-MAY")

    prefix = "tomasz_data"
    _run_internal(
        prefix=prefix,
        var=consts.cruts.tmp,
        time_range=may_only,
        lats=consts.result_inspection.lats,
        lons=consts.result_inspection.lons,
        alts=consts.result_inspection.alts,
        ds_cru=ds_temp_cru,
        ds_nn=ds_temp_nn,
        results_dir=results_dir,
    )

    # mountain peaks
    prefix = "mountain_peaks"
    logging.info(prefix)
    _run_internal(
        prefix=prefix,
        var=consts.cruts.tmp,
        time_range=may_only,
        lats=peaks["lat"].values,
        lons=peaks["lon"].values,
        alts=peaks["altitude"].values,
        ds_cru=ds_temp_cru,
        ds_nn=ds_temp_nn,
        names=peaks["mountain_peak_name"].values,
        results_dir=results_dir,
    )

    # only 2 locations
    prefix = "2_locations"
    _run_internal(
        prefix=prefix,
        var=consts.cruts.tmp,
        time_range=may_only,
        lats=consts.result_inspection.lats[-2:],
        lons=consts.result_inspection.lons[-2:],
        alts=consts.result_inspection.alts[-2:],
        ds_cru=ds_temp_cru,
        ds_nn=ds_temp_nn,
        results_dir=results_dir,
    )


def main(cfg: ResultInspectionConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    os.makedirs(to_absolute_path(cfg.results_dir), exist_ok=True)
    ds_temp_nn = xr.open_dataset(to_absolute_path(cfg.ds_temp_nn_path))
    ds_temp_cru = xr.open_dataset(to_absolute_path(cfg.ds_temp_cru_path))
    peaks = pd.read_feather(to_absolute_path(cfg.peaks_feather))
    run(ds_temp_nn, ds_temp_cru, peaks, cfg.results_dir)


@hydra.main(config_path="../../conf", config_name="result_inspection")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg.get("result_inspection"))


if __name__ == "__main__":
    hydra_entry()
