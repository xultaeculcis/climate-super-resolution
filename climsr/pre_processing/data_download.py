# -*- coding: utf-8 -*-
import logging
import os
from pprint import pprint
from typing import Optional

import requests
from datasets import tqdm

import climsr.consts as consts

logging.basicConfig(level=logging.INFO)


cru_ts_download_url = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.05/cruts.2103051243.v4.05/tmn/cru_ts4.05.1901.2020.tmn.dat.nc.gz"  # noqa
world_clim_download_urls = []
for var in consts.world_clim.variables_wc:
    for i in range(1960, 2018, 10):
        upper = i + 9 if i != 2010 else 2018
        world_clim_download_urls.append(
            f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/hist/wc2.1_2.5m_{var}_{i}-{upper}.zip"
        )


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


if __name__ == "__main__":
    pprint(f"{cru_ts_download_url=}")  # noqa T003
    pprint(f"{world_clim_download_urls=}")  # noqa T003

    # download_file(cru_ts_download_url)
