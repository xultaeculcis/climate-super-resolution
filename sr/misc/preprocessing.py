# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from itertools import product
from typing import Any, List, Optional, Tuple

import dask.bag
import datacube.utils.geometry as dcug
import numpy as np
import rasterio as rio
import xarray
from dask.diagnostics import progress
from datacube.utils.cog import write_cog
from distributed import Client
from misc.clim_scaler import ClimScaler
from misc.cruts_config import CRUTSConfig
from misc.world_clim_config import WorldClimConfig
from rasterio import Affine, windows
from rasterio.enums import Resampling

pbar = progress.ProgressBar()
pbar.register()

# create logger
logger = logging.getLogger("pre-processing")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def parse_args() -> argparse.Namespace:
    """
    Parse arguments.

    Returns (argparse.Namespace): A namespace with parsed arguments.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        "--data_dir_cruts",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/",
    )
    parser.add_argument(
        "--data_dir_world_clim",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/weather/",
    )
    parser.add_argument(
        "--out_dir_cruts",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/",
    )
    parser.add_argument(
        "--out_dir_world_clim",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed/",
    )
    parser.add_argument(
        "--world_clim_elevation_fp",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/elevation/wc2.1_2.5m_elev.tif",
    )
    parser.add_argument("--run_cruts", type=bool, default=False)
    parser.add_argument("--run_cruts_tiling", type=bool, default=False)
    parser.add_argument("--run_world_clim_resize", type=bool, default=False)
    parser.add_argument("--run_world_clim_tiling", type=bool, default=False)
    parser.add_argument("--run_world_clim_elevation_resize", type=bool, default=False)
    parser.add_argument("--patch_size", type=Tuple[int, int], default=(128, 128))
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--threads_per_worker", type=int, default=1)
    parser.add_argument(
        "--resolutions",
        type=List[Tuple[int, int]],
        default=[
            (720, 360),
            (1080, 720),
            (2160, 1080),
        ],
    )
    return parser.parse_args()


def ensure_sub_dirs_exist_cts(out_dir: str) -> None:
    """
    Ensures, that output dir structure exists for CRU-TS data.

    Args:
        out_dir (str): Output dir.

    """
    logger.info("Creating sub-dirs for CRU-TS")
    for dir_name in CRUTSConfig.sub_dirs_cts:
        for var in CRUTSConfig.variables_cts:
            sub_dir_name = os.path.join(out_dir, dir_name, var)
            logger.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)


def cruts_as_cog(variable: str, data_dir: str, out_dir: str) -> None:
    """
    Creates a Cloud Optimized Geo-Tiff file for each time step in the CRU-TS dataset.

    Args:
        variable (str): The variable name.
        data_dir (str): Data dir.
        out_dir (str): Where to save the Geo-Tiffs.

    """
    fp = CRUTSConfig.file_pattern.format(variable)
    file_path = os.path.join(data_dir, fp)
    out_path = os.path.join(out_dir, CRUTSConfig.full_res_dir, variable)
    ds = xarray.open_dataset(file_path)
    for i in range(ds.dims["time"]):
        # get frame at time index i
        arr = ds[variable].isel(time=i)

        # make it geo
        arr = dcug.assign_crs(arr, "EPSG:4326")

        # extract date
        date_str = np.datetime_as_string(arr.time, unit="D")

        # Write as Cloud Optimized GeoTIFF
        write_cog(
            geo_im=arr,
            fname=os.path.join(out_path, f"cruts-{variable}-{date_str}.tif"),
            overwrite=True,
        )


def ensure_sub_dirs_exist_wc(out_dir: str) -> None:
    """
    Ensures, that output dir structure exists for World Clim data.

    Args:
        out_dir (str): Output dir.

    """
    logger.info("Creating sub-dirs for WorldClim")

    variables = WorldClimConfig.variables_wc
    variables.append(WorldClimConfig.elevation)
    for var in variables:
        for rm in WorldClimConfig.resolution_multipliers:
            sub_dir_name = os.path.join(
                out_dir, var, WorldClimConfig.resized_dir, rm[0]
            )
            logger.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)

            sub_dir_name = os.path.join(out_dir, var, WorldClimConfig.tiles_dir, rm[0])
            logger.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)


def resize_raster(
    file_path: str,
    variable: str,
    scaling_factor: float,
    resolution_multiplier: str,
    out_dir: str,
) -> None:
    """
    Resizes specified World Clim Geo-Tiff file with specified scaling factor.

    Args:
        file_path (str): The path to the WorldClim .tif file.
        variable (str): The mame of the variable.
        scaling_factor (float): The scaling factor.
        resolution_multiplier (str): Resolution multiplier in correspondence with CRU-TS.
            eg. if resized resolution is the same as cru-ts then it should be "1x",
            if its twice as big, then it should be "2x" etc.
        out_dir (str): Where to save the resized Geo-Tiff files.

    """

    with rio.open(file_path) as raster:
        # resample data to target shape
        t = raster.transform
        transform = Affine(
            t.a / scaling_factor, t.b, t.c, t.d, t.e / scaling_factor, t.f
        )
        height = int(raster.height * scaling_factor)
        width = int(raster.width * scaling_factor)

        profile = raster.profile
        profile.update(transform=transform, driver="COG", height=height, width=width)

        data = raster.read(
            out_shape=(raster.count, height, width),
            resampling=Resampling.nearest,
        )

        fname = os.path.join(
            out_dir,
            variable,
            WorldClimConfig.resized_dir,
            resolution_multiplier,
            os.path.basename(file_path),
        )

        with rio.open(fname, "w", **profile) as dataset:
            dataset.write(data)


def get_tiles(
    ds: Any, width: Optional[int] = 128, height: Optional[int] = 128
) -> Tuple[rio.windows.Window, rio.Affine]:
    """
    Using `rasterio` generate windows and transforms.

    Args:
        ds (Any): The input dataset.
        width (Optional[int]): The window width. Default: 128.
        height (Optional[int]): The window height. Default: 128.

    Returns (Tuple[rio.windows.Window, rio.Affine]): A tuple with the window and the transform.

    """
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = np.array(list(product(range(0, ncols, width), range(0, nrows, height))))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    for col_off, row_off in offsets:
        leftover_w = ncols - col_off
        leftover_h = nrows - row_off

        if leftover_w < width:
            col_off = ncols - width

        if leftover_h < height:
            row_off = nrows - height

        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def make_patches(
    file_path: str, out_path: str, tile_shape: Tuple[int, int] = (128, 128)
) -> None:
    """
    Create tiles of specified size for ML purposes from specified geo-tiff file.

    Args:
        file_path (str): The path to the geo-tiff raster file.
        out_path (str): The output path.
        tile_shape (Tuple[int, int]): The shape of the image patches.

    """
    with rio.open(file_path) as in_dataset:
        tile_width, tile_height = tile_shape
        fname = os.path.basename(os.path.splitext(file_path)[0]) + ".{}.{}.tif"

        data = in_dataset.read()
        meta = in_dataset.meta.copy()

        scaler = ClimScaler()
        scaler.fit_single(data)

        for window, transform in get_tiles(in_dataset, tile_width, tile_height):
            meta["transform"] = transform
            meta["dtype"] = np.float32
            meta["width"], meta["height"] = window.width, window.height
            out_fp = os.path.join(
                out_path, fname.format(int(window.col_off), int(window.row_off))
            )
            with rio.open(out_fp, "w", **meta) as out_dataset:
                subset = in_dataset.read(window=window)

                # ignore tiles with less than more than 85% nan values
                if np.count_nonzero(np.isnan(subset)) / np.prod(subset.shape) > 0.85:
                    continue

                scaled = scaler.transform_single(subset)
                out_dataset.write(scaled)


if __name__ == "__main__":
    args = parse_args()
    client = Client(
        n_workers=args.n_workers, threads_per_worker=args.threads_per_worker
    )

    try:
        if args.run_cruts:
            ensure_sub_dirs_exist_cts(args.out_dir_cruts)

            logger.info("Running CRU-TS pre-processing - Geo Tiff generation")

            results = (
                dask.bag.from_sequence(CRUTSConfig.variables_cts)
                .map(cruts_as_cog, args.data_dir_cruts, args.out_dir_cruts)
                .compute()
            )

        if args.run_world_clim_resize:
            ensure_sub_dirs_exist_wc(args.out_dir_world_clim)

            for var in WorldClimConfig.variables_wc:
                files = sorted(
                    glob(
                        os.path.join(
                            args.data_dir_world_clim,
                            var,
                            "**",
                            WorldClimConfig.pattern_wc,
                        ),
                        recursive=True,
                    )
                )
                for multiplier, scale in WorldClimConfig.resolution_multipliers:
                    logger.info(
                        "Running WorldClim pre-processing for variable: "
                        f"{var}, scale: {scale}, multiplier: {multiplier}. Total files to process: {len(files)}"
                    )
                    dask.bag.from_sequence(files, npartitions=1000).map(
                        resize_raster, var, scale, multiplier, args.out_dir_world_clim
                    ).compute()

        if args.run_world_clim_elevation_resize:
            ensure_sub_dirs_exist_wc(args.out_dir_world_clim)
            for multiplier, scale in WorldClimConfig.resolution_multipliers:
                logger.info(
                    "Running WorldClim pre-processing for variable: "
                    f"elevation, scale: {scale}, multiplier: {multiplier}"
                )
                resize_raster(
                    args.world_clim_elevation_fp,
                    WorldClimConfig.elevation,
                    scale,
                    multiplier,
                    args.out_dir_world_clim,
                )

        if args.run_cruts_tiling:
            for var in CRUTSConfig.variables_cts:
                files = sorted(
                    glob(
                        os.path.join(
                            args.out_dir_cruts, CRUTSConfig.full_res_dir, var, "*.tif"
                        )
                    )
                )
                logger.info(
                    f"CRU-TS - Running tile generation for {len(files)} {var} files"
                )
                dask.bag.from_sequence(files).map(
                    make_patches,
                    os.path.join(args.out_dir_cruts, CRUTSConfig.tiles_dir, var),
                    args.patch_size,
                ).compute()

        if args.run_world_clim_tiling:
            variables = WorldClimConfig.variables_wc + [WorldClimConfig.elevation]
            for var in variables:
                for multiplier, scale in WorldClimConfig.resolution_multipliers:
                    files = sorted(
                        glob(
                            os.path.join(
                                args.out_dir_world_clim,
                                var,
                                WorldClimConfig.resized_dir,
                                multiplier,
                                "*.tif",
                            )
                        )
                    )
                    logger.info(
                        f"WorldClim - Running tile generation. Total files: {len(files)}, "
                        f"variable: {var}, scale: {scale:.4f}, multiplier: {multiplier}"
                    )
                    dask.bag.from_sequence(files).map(
                        make_patches,
                        os.path.join(
                            args.out_dir_world_clim,
                            var,
                            WorldClimConfig.tiles_dir,
                            multiplier,
                        ),
                        args.patch_size,
                    ).compute()

        logger.info("DONE")
    finally:
        client.close()
