# -*- coding: utf-8 -*-
import logging

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import climsr.consts as consts
import climsr.inference.inference as inference
from climsr.core import utils
from climsr.core.config import InferenceConfig

logging.basicConfig(level=logging.INFO)


def run(cfg: InferenceConfig) -> None:
    variables = [cfg.cruts_variable] if cfg.cruts_variable else consts.cruts.variables_cts

    utils.set_gpu_power_limit_if_needed()

    # Run inference
    if cfg.run_inference:
        logging.info("Running inference")
        inference.run_inference(cfg, variables)

    # Run tiff file transformation to net-cdf datasets.
    if cfg.to_netcdf:
        logging.info("Building NET CDF datasets from extent tiff files.")
        inference.transform_tiff_files_to_net_cdf(
            to_absolute_path(cfg.extent_out_path_sr),
            to_absolute_path(cfg.extent_out_path_sr_nc),
            variables,
            prefix=cfg.generator_type,
        )

    logging.info("Done")


def main(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg.get("inference"))


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg.get("inference"))


if __name__ == "__main__":
    hydra_entry()
