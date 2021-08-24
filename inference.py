# -*- coding: utf-8 -*-
import logging

import hydra
from omegaconf import DictConfig

from climsr.cli.inference import main as inference_main
from climsr.cli.inspect_results import main as result_inspection_main

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./conf", config_name="inference")
def hydra_entry_inference(cfg: DictConfig) -> None:
    inference_main(cfg)


@hydra.main(config_path="./conf", config_name="result_inspection")
def hydra_entry_result_inspection(cfg: DictConfig) -> None:
    result_inspection_main(cfg.get("result_inspection"))


if __name__ == "__main__":
    hydra_entry_inference()
    hydra_entry_result_inspection()
