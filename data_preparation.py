# -*- coding: utf-8 -*-
import logging

import hydra
from omegaconf import DictConfig

from climsr.cli.data_download import main as data_download_main
from climsr.cli.preprocess import main as preprocessing_main

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./conf", config_name="data_preparation")
def hydra_entry_data_preparation(cfg: DictConfig) -> None:
    if cfg.get("run_download"):
        data_download_main(cfg)
    if cfg.get("run_preprocessing"):
        preprocessing_main(cfg.get("preprocessing"))


if __name__ == "__main__":
    hydra_entry_data_preparation()
