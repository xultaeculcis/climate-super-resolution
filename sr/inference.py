# -*- coding: utf-8 -*-
import argparse
import logging
import os
import tempfile
from glob import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
from PIL import Image
from pre_processing.clim_scaler import ClimScaler
from pre_processing.cruts_config import CRUTSConfig
from pre_processing.preprocessing import get_tiles
from pre_processing.world_clim_config import WorldClimConfig
from rasterio.merge import merge
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
    tile_shape: Optional[Tuple[int, int]] = (32, 32),
    stride: Optional[int] = 16,
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
        tile_shape (Optional[Tuple[int, int]]): Optional tile shape. 32x32 by default.
        stride (Optional[int]): Optional stride. 16 by default.

    """
    logging.info(f"Running Inference for '{cruts_variable}'")
    tiffs = sorted(glob(os.path.join(data_dir, cruts_variable, "*.tif")))
    logging.info(
        f"Found {len(tiffs)} files under {data_dir} for variable '{cruts_variable}'"
    )

    # get and normalize elevation data
    elevation_df = pd.read_csv(
        os.path.join(
            "../datasets",
            WorldClimConfig.elevation,
            f"{scaling_factor}x",
            f"{WorldClimConfig.elevation}.csv",
        )
    )

    to_tensor = transforms.ToTensor()
    upscale = transforms.Resize(
        size=(tile_shape[0] * scaling_factor, tile_shape[1] * scaling_factor),
        interpolation=Image.NEAREST,
    )

    # for each CRU-TS raster file
    for idx, fp in enumerate(tiffs):
        logging.info(f"Processing raster {idx}/{len(tiffs)}: {fp}")

        # crate two temp dirs
        # one for temp raster tiles
        # one for temp prediction tiles
        with tempfile.TemporaryDirectory() as tmp_tiles_dir:
            with tempfile.TemporaryDirectory() as tmp_tile_predictions_dir:
                scaler = ClimScaler()
                # open the raster file and generate overlapping tiles from it
                with rio.open(fp) as in_dataset:
                    logging.info("Generating temporary raster tiles")
                    tile_width, tile_height = tile_shape
                    fname = os.path.basename(os.path.splitext(fp)[0]) + ".{}.{}.tif"

                    data = in_dataset.read()
                    meta = in_dataset.meta.copy()

                    scaler.fit_single(data)

                    for window, transform in get_tiles(
                        in_dataset, tile_width, tile_height, stride
                    ):
                        meta["transform"] = transform
                        meta["dtype"] = np.float32
                        meta["width"], meta["height"] = window.width, window.height
                        out_fp = os.path.join(
                            tmp_tiles_dir,
                            fname.format(int(window.col_off), int(window.row_off)),
                        )

                        subset = in_dataset.read(window=window)

                        with rio.open(out_fp, "w", **meta) as out_dataset:
                            out_dataset.write(subset)

                # get tmp tiles
                tiles = sorted(glob(os.path.join(tmp_tiles_dir, "*.tif")))

                logging.info(f"Running predictions on {len(tiles)} raster tiles")

                for tile in tqdm(tiles):
                    # load file
                    img = Image.open(tile)
                    arr = np.array(img)

                    # get west and north of the tile
                    splitted = tile.split(".")
                    x = int(splitted[-3]) * scaling_factor
                    y = int(splitted[-2]) * scaling_factor

                    # get corresponding elevation tile
                    elev_fp = elevation_df[
                        (elevation_df["x"] == x) & (elevation_df["y"] == y)
                    ]["file_path"].values[0]

                    elevation_data = np.array(Image.open(elev_fp))

                    # normalize
                    arr = scaler.transform_single(arr)

                    # to tensor
                    t_lr = to_tensor(
                        np.array(upscale(Image.fromarray(arr)))
                    ).unsqueeze_(0)
                    t_elev = to_tensor(elevation_data).unsqueeze_(0)
                    t = torch.cat(tensors=[t_lr, t_elev], dim=1)

                    # run inference with model
                    out = model(t).clamp(0.0, 1.0)

                    # back to array
                    arr = out.squeeze_(0).squeeze_(0).numpy()

                    # denormalize
                    # arr = scaler.inverse_transform(arr)

                    # get filename
                    filename = os.path.basename(os.path.splitext(tile)[0])
                    filename = os.path.join(
                        tmp_tile_predictions_dir, f"{filename}-inference.tif"
                    )

                    # prepare transform using prior knowledge
                    transform = from_origin(
                        west=-180.0 + (x / 8),
                        north=90.0 + (-y / 8),
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

                logging.info("Merging generated prediction tiles into single mosaic")
                # get tmp predictions
                tiles = glob(os.path.join(tmp_tile_predictions_dir, "*.tif"))

                src_files_to_mosaic = []
                for tile in tiles:
                    src = rio.open(tile)
                    src_files_to_mosaic.append(src)

                mosaic, out_trans = merge(src_files_to_mosaic)

                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "COG",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "crs": "EPSG:4326",
                    }
                )

                filename = os.path.join(
                    out_dir,
                    f"{os.path.basename(os.path.splitext(fp)[0])}-inference.tif",
                )
                with rio.open(filename, "w", **out_meta) as dest:
                    dest.write(mosaic)

        break


def run_inference_v2(
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

    """
    logging.info(f"Running Inference for '{cruts_variable}'")
    tiffs = glob(os.path.join(data_dir, cruts_variable, "*.tif"))
    logging.info(
        f"Found {len(tiffs)} files under {data_dir} for variable '{cruts_variable}'"
    )

    # get and normalize elevation data
    elev_scaler = ClimScaler()
    elevation_data = np.array(Image.open(elevation_file))
    elevation_data_scaled = elev_scaler.fit_transform_single(elevation_data)

    # load mask
    with rio.open(
        "/media/xultaeculcis/2TB/datasets/wc/pre-processed/prec/resized/4x/wc2.1_2.5m_prec_1961-01.tif"
    ) as mask_src:
        mask_data = mask_src.read()

    mask = np.isnan(mask_data).squeeze(0)

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
        t_elev = to_tensor(elevation_data_scaled).unsqueeze_(0)
        t = torch.cat(tensors=[t_lr, t_elev], dim=1)

        # run inference with model
        out = model(t).clamp(0.0, 1.0)

        # back to array
        arr = out.squeeze_(0).squeeze_(0).numpy()

        # denormalize
        arr = scaler.inverse_transform(arr)
        arr[mask] = np.nan

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


def run_inference_single_path(
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

    """
    logging.info(f"Running Inference for '{cruts_variable}'")
    tiffs = glob(os.path.join(data_dir, cruts_variable, "*.tif"))
    logging.info(
        f"Found {len(tiffs)} files under {data_dir} for variable '{cruts_variable}'"
    )

    # get and normalize elevation data
    elev_scaler = ClimScaler()
    elevation_data = np.array(Image.open(elevation_file))
    elevation_data_scaled = elev_scaler.fit_transform_single(elevation_data)

    # load mask
    with rio.open(
        "/media/xultaeculcis/2TB/datasets/wc/pre-processed/prec/resized/4x/wc2.1_2.5m_prec_1961-01.tif"
    ) as mask_src:
        mask_data = mask_src.read()

    mask = np.isnan(mask_data).squeeze(0)

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
        t_elev = to_tensor(elevation_data_scaled).unsqueeze_(0)
        t = torch.cat(tensors=[t_lr, t_elev], dim=1)

        # run inference with model
        out = model(t).clamp(0.0, 1.0)

        # back to array
        arr = out.squeeze_(0).squeeze_(0).numpy()

        # denormalize
        arr = scaler.inverse_transform(arr)
        arr[mask] = np.nan

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
        run_inference_v2(
            model=net,
            data_dir=args.data_dir,
            cruts_variable=args.cruts_variable,
            elevation_file=args.elevation_file,
            out_dir=out_path,
            scaling_factor=args.scaling_factor,
        )
