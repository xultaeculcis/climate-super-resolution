# -*- coding: utf-8 -*-
import logging
import os

import hydra
from dask.diagnostics import progress
from distributed import Client
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import climsr.preprocessing.preprocessing as preprocessing
from climsr import consts
from climsr.core.config import PreProcessingConfig

pbar = progress.ProgressBar()
pbar.register()
logging.basicConfig(level=logging.INFO)


def main(cfg: PreProcessingConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    client = Client(n_workers=cfg.n_workers, threads_per_worker=cfg.threads_per_worker)
    try:
        preprocessing.ensure_sub_dirs_exist_cts(
            os.path.join(
                to_absolute_path(cfg.output_path),
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.cruts_preprocessing_out_path,
            )
        )
        preprocessing.ensure_sub_dirs_exist_wc(
            os.path.join(
                to_absolute_path(cfg.output_path),
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.world_clim_preprocessing_out_path,
            )
        )
        preprocessing.run_cruts_to_tiff(cfg)
        preprocessing.run_world_clim_resize(cfg)
        preprocessing.run_tavg_rasters_generation(cfg)
        preprocessing.run_world_clim_tiling(cfg)
        preprocessing.run_statistics_computation(cfg)
        preprocessing.run_train_val_test_split(cfg)
        preprocessing.run_extent_extraction(cfg)
        logging.info("DONE")
    finally:
        client.close()


@hydra.main(config_path="../../conf", config_name="preprocessing")
def hydra_entry(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    main(cfg.get("preprocessing"))


if __name__ == "__main__":
    hydra_entry()
