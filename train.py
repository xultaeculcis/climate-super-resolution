# -*- coding: utf-8 -*-
import logging
from typing import Optional, Union

import hydra
from omegaconf import DictConfig
from torch import Tensor

from climsr.cli.train import main

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> Optional[Union[float, Tensor]]:
    return main(cfg)


if __name__ == "__main__":
    hydra_entry()
