# -*- coding: utf-8 -*-
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def my_app(cfg: DictConfig):
    logging.info(f"{OmegaConf.to_yaml(cfg)}")


if __name__ == "__main__":
    my_app()
