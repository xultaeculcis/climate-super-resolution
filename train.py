# -*- coding: utf-8 -*-
import logging

import hydra
from omegaconf import DictConfig

from climsr.cli.train import main

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
