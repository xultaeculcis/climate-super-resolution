# -*- coding: utf-8 -*-
import hydra
from cli.inspect_results import main
from omegaconf import DictConfig


@hydra.main(config_path="./conf", config_name="result_inspection")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg.get("result_inspection"))


if __name__ == "__main__":
    hydra_entry()
