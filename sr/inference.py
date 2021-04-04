# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import rasterio as rio
import torch
from PIL import Image
from pre_processing.clim_scaler import ClimScaler
from pre_processing.cruts_config import CRUTSConfig
from rasterio.transform import from_origin
from torchvision import transforms
from tqdm import tqdm
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
        "--elevation_file",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed/elevation/resized/4x/wc2.1_2.5m_elev.tif",
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
    parser.add_argument(
        "--scaling_factor",
        type=int,
        default=4,
    )

    return parser.parse_args()


def run_inference(
    model: pl.LightningModule,
    data_dir: str,
    cruts_variable: str,
    elevation_file: str,
    out_dir: str,
    scaling_factor: Optional[int] = 4,
) -> None:
    """
    Runs the inference on CRU-TS dataset.

    Args:
        model (pl.LightningModule): The LightningModule.
        data_dir (str): The path to the data.
        cruts_variable (str): The name of the CRU-TS variable.
        elevation_file (str): The path to the elevation file.
        out_dir (str): The output path.
        scaling_factor (Optional[int]): The scaling factor.

    Returns:

    """
    logging.info(f"Running Inference for '{cruts_variable}'")
    tiffs = sorted(glob(os.path.join(data_dir, cruts_variable, "*.tif")))
    logging.info(
        f"Found {len(tiffs)} files under {data_dir} for variable '{cruts_variable}'"
    )

    # get and normalize elevation data
    elev_scaler = ClimScaler()
    elevation_data = np.array(Image.open(elevation_file))
    elevation_data = elev_scaler.fit_transform_single(elevation_data)

    to_tensor = transforms.ToTensor()
    upscale = transforms.Resize((1440, 2880), interpolation=Image.NEAREST)

    for fp in tqdm(tiffs):
        # load file
        img = Image.open(fp)
        arr = np.array(img)

        # normalize
        scaler = ClimScaler()
        arr = scaler.fit_transform_single(arr)

        # to tensor
        t_lr = to_tensor(np.array(upscale(Image.fromarray(arr)))).unsqueeze_(0)
        t_elev = to_tensor(elevation_data).unsqueeze_(0)
        t = torch.cat(tensors=[t_lr, t_elev], dim=1)

        # run inference with model
        out = model(t).clamp(0.0, 1.0)

        # back to array
        arr = out.squeeze_(0).squeeze_(0).numpy()

        # denormalize
        arr = scaler.inverse_transform(arr)

        # get filename
        filename = os.path.basename(os.path.splitext(fp)[0])
        filename = os.path.join(out_dir, f"{filename}-inference.tif")

        # prepare transform using prior knowledge
        transform = from_origin(
            west=-180.0,
            north=90.0,
            xsize=CRUTSConfig.degree_per_pix / scaling_factor,
            ysize=CRUTSConfig.degree_per_pix / scaling_factor,
        )

        # create COG with enhanced data
        with rio.open(
            filename,
            "w",
            driver="COG",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=str(arr.dtype),
            crs=CRUTSConfig.CRS,
            transform=transform,
        ) as new_dataset:
            new_dataset.write(arr, 1)

        break


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Running with following config: {vars(args)}")

    out_path = os.path.join(args.out_path, args.cruts_variable)
    os.makedirs(out_path, exist_ok=True)
    net = prepare_pl_module(args)
    net.eval()
    with torch.no_grad():
        run_inference(
            model=net,
            data_dir=args.data_dir,
            cruts_variable=args.cruts_variable,
            elevation_file=args.elevation_file,
            out_dir=out_path,
            scaling_factor=args.scaling_factor,
        )
