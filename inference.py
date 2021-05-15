# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import pytorch_lightning as pl
import rasterio as rio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr.data.datasets import CRUTSInferenceDataset
from sr.data.utils import denormalize
from sr.lightning_modules.utils import prepare_pl_module
from sr.pre_processing.cruts_config import CRUTSConfig


def parse_args() -> argparse.Namespace:
    """
    Parses the arguments.

    Returns (argparse.Namespace): The `argparse.Namespace`.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        "--ds_path",
        type=str,
        # default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmn.dat.nc",
        # default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmp.dat.nc",
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmx.dat.nc",
        # default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.pre.dat.nc",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/full-res",
    )
    parser.add_argument(
        "--elevation_file",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed/elevation/resized/4x/wc2.1_2.5m_elev.tif",
    )
    parser.add_argument(
        "--land_mask_file",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed/prec/resized/4x/wc2.1_2.5m_prec_1961-01.tif",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/inference",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="../model_weights/gen-pre-training-srcnn-temp-4x-epoch=29-step=165419-hp_metric=0.00083.ckpt",
        # default="../model_weights/gen-pre-training-srcnn-prec-4x-epoch=29-step=82709-hp_metric=0.00007.ckpt",
    )
    parser.add_argument("--experiment_name", type=str, default="inference")
    parser.add_argument("--cruts_variable", type=str, default=CRUTSConfig.tmx)
    parser.add_argument("--scaling_factor", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)

    return parser.parse_args()


def run_inference_on_full_images(
    model: pl.LightningModule,
    ds_path: str,
    elevation_file: str,
    land_mask_file: str,
    out_dir: str,
) -> None:
    # ensure gpu
    model = model.cuda()

    # load mask
    with rio.open(land_mask_file) as mask_src:
        mask_data = mask_src.read()
        profile = mask_src.profile

    mask = np.isnan(mask_data).squeeze(0)

    # prepare dataset
    ds = CRUTSInferenceDataset(
        ds_path=ds_path,
        elevation_file=elevation_file,
        land_mask_file=land_mask_file,
        generator_type="srcnn",
        scaling_factor=4,
    )

    # prepare dataloader
    dl = DataLoader(dataset=ds, batch_size=1, pin_memory=True, num_workers=1)

    # run inference
    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        lr = batch["lr"].cuda()
        elev = batch["elevation"].cuda()
        min = batch["min"].numpy()
        max = batch["max"].numpy()
        filename = batch["filename"]

        outputs = model(torch.cat([lr, elev], dim=1))
        outputs = outputs.cpu().numpy()

        for idx, output in enumerate(outputs):
            arr = output.squeeze(0)
            arr = denormalize(arr, min[idx], max[idx]).clip(min[idx], max[idx])
            arr[mask] = np.nan

            with rio.open(
                os.path.join(out_dir, filename[idx]), "w", **profile
            ) as dataset:
                dataset.write(arr, 1)


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Running with following config: {vars(args)}")

    out_path = os.path.join(args.out_path, args.cruts_variable)
    os.makedirs(out_path, exist_ok=True)
    net = prepare_pl_module(args)
    net.eval()
    with torch.no_grad():
        run_inference_on_full_images(
            model=net,
            ds_path=args.ds_path,
            land_mask_file=args.land_mask_file,
            elevation_file=args.elevation_file,
            out_dir=out_path,
        )
