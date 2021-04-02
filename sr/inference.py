# -*- coding: utf-8 -*-
import argparse
import os
from glob import glob
from tqdm import tqdm
import logging

import pytorch_lightning as pl

from pre_processing.cruts_config import CRUTSConfig
from utils import prepare_pl_module


def parse_args() -> argparse.Namespace:
    """
    Parses the arguments.

    Returns (argparse.Namespace): The `argparse.Namespace`.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/full-res",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/inference",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="../model_weights/gen-pre-training-srcnn-prec-4x-epoch=14-step=0-hp_metric=0.00132.ckpt",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="inference",
    )
    parser.add_argument(
        "--cruts_variable",
        type=str,
        default=CRUTSConfig.pre,
    )

    return parser.parse_args()


def run_inference(
    model: pl.LightningModule, data_dir: str, cruts_variable: str, out_dir: str
) -> None:
    """
    Runs the inference on CRU-TS dataset.

    Args:
        model (pl.LightningModule): The LightningModule.
        data_dir (str): The path to the data.
        cruts_variable (str): The name of the CRU-TS variable
        out_dir (str): The output path.

    Returns:

    """
    logging.info(f"Running Inference for '{cruts_variable}'")
    tiffs = glob(os.path.join(data_dir, cruts_variable, "*.tif"))
    logging.info(
        f"Found {len(tiffs)} files under {data_dir} for variable '{cruts_variable}'"
    )

    for _ in tqdm(tiffs):
        pass


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Running with following config: {vars(args)}")

    out_path = os.path.join(args.out_path, args.cruts_variable)
    os.makedirs(out_path, exist_ok=True)
    net = prepare_pl_module(args)
    run_inference(
        model=net,
        data_dir=args.data_dir,
        cruts_variable=args.cruts_variable,
        out_dir=out_path,
    )
