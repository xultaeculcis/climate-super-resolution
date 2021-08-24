# -*- coding: utf-8 -*-
import gzip
import logging
import os
import zipfile
from typing import List, Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from parfive import Downloader, Results

from climsr.core.config import DataDownloadConfig
from climsr.preprocessing.data_download import (
    get_cruts_data_download_urls,
    get_world_clim_future_climate_data_download_urls,
    get_world_clim_historical_climate_data_download_urls,
    get_world_clim_historical_weather_data_download_urls,
)

logging.basicConfig(level=logging.INFO)


def handle_file_download(
    cru_ts_download_urls: List[str],
    world_clim_download_urls: List[str],
    parallel_downloads: Optional[int] = 8,
    download_path: Optional[str] = "../../datasets",
) -> Results:
    downloader = Downloader(max_conn=parallel_downloads)
    cruts_download_path = os.path.join(download_path, "cruts")
    world_clim_download_path = os.path.join(download_path, "world_clim")

    os.makedirs(cruts_download_path, exist_ok=True)
    os.makedirs(world_clim_download_path, exist_ok=True)

    paths = [cruts_download_path]
    paths.extend([world_clim_download_path for _ in world_clim_download_urls])

    all_urls = []
    all_urls.extend(cru_ts_download_urls)
    all_urls.extend(world_clim_download_urls)

    for url, download_path in zip(all_urls, paths):
        downloader.enqueue_file(url, download_path)

    return downloader.download()


def gunzip(source_filepath: str, dest_filepath: str, block_size: Optional[int] = 65536) -> None:
    with gzip.open(source_filepath, "rb") as s_file, open(dest_filepath, "wb") as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def unzip(source_filepath: str, dest_filepath: str) -> None:
    with zipfile.ZipFile(source_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_filepath)


def handle_file_extraction(download_results: Results) -> None:
    for idx, f_name in enumerate(download_results):
        logging.info(f"Extracting {f_name}. Progress {idx + 1:04d}/{len(download_results):04d}")
        if str(f_name).endswith(".zip"):
            unzip(f_name, os.path.basename(f_name))
        elif str(f_name).endswith(".gz"):
            gunzip(f_name, os.path.splitext(f_name)[0])
        else:
            raise ValueError(f"{f_name} compression type is unsupported! Supported: ZIP, GZ")


def run(
    cru_ts_download_urls: List[str],
    world_clim_download_urls: List[str],
    parallel_downloads: Optional[int] = 8,
    download_path: Optional[str] = "../../datasets",
) -> None:
    download_results = handle_file_download(
        cru_ts_download_urls=cru_ts_download_urls,
        world_clim_download_urls=world_clim_download_urls,
        parallel_downloads=parallel_downloads,
        download_path=download_path,
    )

    handle_file_extraction(download_results)


def main(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg.get("data_download")))

    cru_ts_download_urls = get_cruts_data_download_urls()
    world_clim_historical_climate_data_download_urls = get_world_clim_historical_climate_data_download_urls()
    world_clim_historical_weather_data_download_urls = get_world_clim_historical_weather_data_download_urls()
    world_clim_future_climate_data_download_urls = get_world_clim_future_climate_data_download_urls()

    download_config: DataDownloadConfig = cfg.get("data_download")

    world_clim_download_urls = []
    world_clim_download_urls.extend(world_clim_historical_weather_data_download_urls)
    world_clim_download_urls.extend(world_clim_historical_climate_data_download_urls)
    world_clim_download_urls.extend(world_clim_future_climate_data_download_urls)

    run(
        cru_ts_download_urls=cru_ts_download_urls,
        world_clim_download_urls=world_clim_download_urls[:1],
        parallel_downloads=download_config.parallel_downloads,
        download_path=to_absolute_path(download_config.download_path),
    )


@hydra.main(config_path="../../conf", config_name="data_download")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
