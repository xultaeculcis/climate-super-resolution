# -*- coding: utf-8 -*-
import hydra
from omegaconf import DictConfig

from climsr.cli.inference import main


@hydra.main(config_path="./conf", config_name="inference")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
