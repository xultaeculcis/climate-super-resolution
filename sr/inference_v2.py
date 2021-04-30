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
from affine import Affine
from pre_processing.preprocessing import get_tiles
from pre_processing.world_clim_config import WorldClimConfig
from rasterio.enums import Resampling
from rasterio.merge import merge
from torchvision.transforms import transforms
from tqdm import tqdm

from sr.data.utils import denormalize, normalize
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
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.pre.dat.nc",
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
        default="../model_weights/gen-pre-training-srcnn-prec-4x-epoch=14-step=41354-hp_metric=0.00152.ckpt",
    )
    parser.add_argument("--experiment_name", type=str, default="inference")
    parser.add_argument("--cruts_variable", type=str, default=CRUTSConfig.pre)
    parser.add_argument("--scaling_factor", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)

    return parser.parse_args()


def run_inference_on_tiles(
    model: pl.LightningModule,
    data_dir: str,
    cruts_variable: str,
    land_mask_file: str,
    elevation_file: str,
    out_dir: str,
    scaling_factor: Optional[int] = 4,
    tile_shape: Optional[Tuple[int, int]] = (128, 128),
    stride: Optional[int] = None,
) -> None:
    """
    Runs the inference on CRU-TS dataset.

    Args:
        model (pl.LightningModule): The LightningModule.
        data_dir (str): The path to the data.
        cruts_variable (str): The name of the CRU-TS variable.
        land_mask_file (str): The land mask file path.
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

    # ensure GPU
    model = model.cuda()

    # get elevation data
    elevation_df = pd.read_csv(
        os.path.join(
            "../datasets",
            WorldClimConfig.elevation,
            f"{scaling_factor}x",
            f"{WorldClimConfig.elevation}.csv",
        )
    )

    to_tensor = transforms.ToTensor()

    # load mask
    with rio.open(land_mask_file) as mask_src:
        mask_data = mask_src.read()
        profile = mask_src.profile

    mask = np.isnan(mask_data).squeeze(0)

    # for each CRU-TS raster file
    for idx, fp in enumerate(tiffs[720:721]):
        logging.info(f"Processing raster {idx}/{len(tiffs)}: {fp}")

        # crate three temp dirs
        # one for temp normalized rasters
        # one for temp raster tiles
        # one for temp prediction tiles
        tmp_normalized_rasters_dir = tempfile.TemporaryDirectory()
        tmp_normalized_rasters_dir_path = tmp_normalized_rasters_dir.name
        tmp_tiles_dir = tempfile.TemporaryDirectory()
        tmp_tiles_dir_path = tmp_tiles_dir.name
        tmp_tile_predictions_dir = tempfile.TemporaryDirectory()
        tmp_tile_predictions_dir_path = tmp_tile_predictions_dir.name

        # open the raster file and normalize it
        fp, input_max, input_min = resize_and_normalize_input_raster(
            fp, scaling_factor, tmp_normalized_rasters_dir_path
        )

        # generate temp raster tiles
        generate_temp_raster_tiles(fp, tmp_tiles_dir_path, tile_shape, stride)

        # get tmp tiles
        tiles = sorted(glob(os.path.join(tmp_tiles_dir_path, "*.tif")))

        logging.info(f"Running predictions on {len(tiles)} raster tiles")

        for tile in tqdm(tiles):
            run_inference_on_single_tile(
                elevation_df,
                input_max,
                input_min,
                model,
                tile,
                tmp_tile_predictions_dir_path,
                to_tensor,
            )

        merge_tiles(fp, mask, out_dir, profile, tmp_tile_predictions_dir_path)

        # cleanup
        tmp_tile_predictions_dir.cleanup()
        tmp_tiles_dir.cleanup()
        tmp_normalized_rasters_dir.cleanup()

        break


def merge_tiles(fp, mask, out_dir, profile, tmp_tile_predictions_dir):
    logging.info("Merging generated prediction tiles into single mosaic")
    # get tmp predictions
    tiles = glob(os.path.join(tmp_tile_predictions_dir, "*.tif"))

    # merge tiles
    src_files_to_mosaic = []
    for tile in tiles:
        src = rio.open(tile)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    mosaic = mosaic.squeeze(0)

    # ocean mask
    mosaic[mask] = np.nan
    filename = os.path.join(
        out_dir,
        f"{os.path.basename(os.path.splitext(fp)[0])}-inference.tif",
    )

    with rio.open(filename, "w", **profile) as dest:
        dest.write(mosaic, 1)


def run_inference_on_single_tile(
    elevation_df,
    input_max,
    input_min,
    model,
    tile,
    tmp_tile_predictions_dir,
    to_tensor,
):
    # load file
    with rio.open(tile) as dataset:
        arr = np.flipud(dataset.read(1))
        transform = dataset.transform

    # get west and north of the tile
    splitted = tile.split(".")
    x = int(splitted[-3])
    y = int(splitted[-2])

    # get corresponding elevation tile
    elev_fp = elevation_df[(elevation_df["x"] == x) & (elevation_df["y"] == y)][
        "file_path"
    ].values[0]

    with rio.open(elev_fp) as dataset:
        elevation_data = np.flipud(dataset.read(1))

    # to tensor
    t_lr = to_tensor(arr.copy()).unsqueeze(0)
    t_elev = to_tensor(elevation_data.copy()).unsqueeze(0)
    t = torch.cat(tensors=[t_lr, t_elev], dim=1).cuda()

    # run inference with model
    out = model(t)

    # back to array
    arr = out.squeeze_(0).squeeze_(0).cpu().numpy()

    # denormalize
    arr = np.flipud(denormalize(arr, input_min, input_max).clip(input_min, input_max))

    # get filename
    filename = os.path.basename(os.path.splitext(tile)[0])
    filename = os.path.join(tmp_tile_predictions_dir, f"{filename}-inference.tif")

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


def resize_and_normalize_input_raster(fp, scaling_factor, tmp_normalized_rasters_dir):
    logging.info("Normalizing the TIFF file")
    with rio.open(fp) as raster:
        # resample data to target shape
        t = raster.transform
        transform = Affine(
            t.a / scaling_factor, t.b, t.c, t.d, t.e / -scaling_factor, -t.f
        )
        height = int(raster.height * scaling_factor)
        width = int(raster.width * scaling_factor)

        profile = raster.profile
        profile.update(transform=transform, height=height, width=width)

        data = raster.read(
            1,
            out_shape=(raster.count, height, width),
            resampling=Resampling.nearest,
        )

        # normalize
        data, input_min, input_max = normalize(data)
        data = np.flipud(data)
        fp = os.path.join(tmp_normalized_rasters_dir, os.path.basename(fp))

        # save normalized data
        with rio.open(fp, "w", **profile) as tmp_raster:
            tmp_raster.write(data, 1)

        return fp, input_max, input_min


def generate_temp_raster_tiles(fp, tmp_tiles_dir, tile_shape, stride):
    logging.info("Generating temporary raster tiles")

    # open the raster file and generate overlapping tiles from it
    with rio.open(fp) as in_dataset:
        tile_width, tile_height = tile_shape
        fname = os.path.basename(os.path.splitext(fp)[0]) + ".{}.{}.tif"
        meta = in_dataset.meta.copy()

        for window, transform in get_tiles(in_dataset, tile_width, tile_height, stride):
            meta["transform"] = transform
            meta["dtype"] = np.float32
            meta["width"], meta["height"] = window.width, window.height
            out_fp = os.path.join(
                tmp_tiles_dir,
                fname.format(int(window.col_off), int(window.row_off)),
            )

            subset = in_dataset.read(window=window).astype(np.float32)

            with rio.open(out_fp, "w", **meta) as out_dataset:
                out_dataset.write(subset)


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Running with following config: {vars(args)}")

    out_path = os.path.join(args.out_path, args.cruts_variable)
    os.makedirs(out_path, exist_ok=True)
    net = prepare_pl_module(args)
    net.eval()
    with torch.no_grad():
        run_inference_on_tiles(
            model=net,
            data_dir=args.data_dir,
            cruts_variable=args.cruts_variable,
            land_mask_file=args.land_mask_file,
            elevation_file=args.elevation_file,
            out_dir=out_path,
        )
