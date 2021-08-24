# -*- coding: utf-8 -*-
import os
from typing import List, Optional

import numpy as np
import requests
from sklearn.utils.extmath import cartesian

import climsr.consts as consts
from datasets import tqdm


def download_file(url: str, download_dir: Optional[str] = "../../datasets") -> str:
    os.makedirs(download_dir, exist_ok=True)
    fname = os.path.join(download_dir, url.split("/")[-1])

    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return fname


def get_cruts_data_download_urls() -> List[str]:
    return [
        f"https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.05/cruts.2103051243.v4.05/{var}/cru_ts4.05.1901.2020.{var}.dat.nc.gz"
        for var in consts.cruts.temperature_vars
    ]


def get_world_clim_historical_climate_data_download_urls() -> List[str]:
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tavg, consts.world_clim.tmax, consts.world_clim.prec]).astype(
        "<U12"
    )
    resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")

    product = cartesian([variables, resolutions])

    for var, res in product:
        world_clim_download_urls.append(f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_{res}_{var}.zip")

    return world_clim_download_urls


def get_world_clim_historical_weather_data_download_urls() -> List[str]:
    step = 10
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tmax, consts.world_clim.prec]).astype("<U12")
    resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")
    lowers = np.array([lower for lower in range(1960, 2019, step)])

    product = cartesian([variables, resolutions, lowers])

    for var, res, lower in product:
        upper = int(lower) + step - 1
        world_clim_download_urls.append(
            f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/hist/wc2.1_{res}_{var}_{lower}-{upper}.zip"
        )

    return world_clim_download_urls


def get_world_clim_future_climate_data_download_urls() -> List[str]:
    step = 20
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tmax, consts.world_clim.prec]).astype("<U12")
    resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")
    gcms = np.array(consts.world_clim.GCMs).astype("<U12")
    scenarios = np.array(consts.world_clim.scenarios).astype("<U12")
    lowers = np.array([lower for lower in range(2021, 2100, step)])

    product = cartesian([variables, resolutions, gcms, scenarios, lowers])
    for var, res, gcm, scenario, lower in product:
        upper = int(lower) + step - 1
        world_clim_download_urls.append(
            "https://biogeo.ucdavis.edu/data/worldclim/v2.1/fut/" f"{res}/wc2.1_{res}_{var}_{gcm}_{scenario}_{lower}-{upper}.zip"
        )

    return world_clim_download_urls
