# -*- coding: utf-8 -*-
import logging
from typing import List, Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from climsr.core.config import DataDownloadConfig
from climsr.preprocessing.data_download import (
    get_cruts_data_download_urls,
    get_world_clim_future_climate_data_download_urls,
    get_world_clim_historical_climate_data_download_urls,
    get_world_clim_historical_weather_data_download_urls,
    handle_file_download,
)

logging.basicConfig(level=logging.INFO)


def run(
    cru_ts_download_urls: List[str],
    world_clim_download_urls: List[str],
    download_path: Optional[str] = "../../datasets/download",
) -> None:
    handle_file_download(
        cru_ts_download_urls=cru_ts_download_urls,
        world_clim_download_urls=world_clim_download_urls,
        download_path=download_path,
    )


def main(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg.get("data_download")))

    cru_ts_download_urls = get_cruts_data_download_urls()
    world_clim_historical_climate_data_download_urls = get_world_clim_historical_climate_data_download_urls()
    world_clim_historical_weather_data_download_urls = get_world_clim_historical_weather_data_download_urls()
    world_clim_future_climate_data_download_urls = get_world_clim_future_climate_data_download_urls()

    download_config: DataDownloadConfig = cfg.get("data_download")

    world_clim_download_urls = []
    world_clim_download_urls.extend(world_clim_historical_climate_data_download_urls)
    world_clim_download_urls.extend(world_clim_historical_weather_data_download_urls)
    world_clim_download_urls.extend(world_clim_future_climate_data_download_urls)

    run(
        cru_ts_download_urls=cru_ts_download_urls,
        world_clim_download_urls=world_clim_download_urls,
        download_path=to_absolute_path(download_config.download_path),
    )


@hydra.main(config_path="../../conf", config_name="data_download")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
