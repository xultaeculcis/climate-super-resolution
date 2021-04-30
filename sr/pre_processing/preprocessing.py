# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from itertools import product
from typing import Any, Optional, Tuple

import dask.bag
import datacube.utils.geometry as dcug
import numpy as np
import pandas as pd
import rasterio as rio
import xarray
from dask.diagnostics import progress
from datacube.utils.cog import write_cog
from distributed import Client
from rasterio import Affine, windows
from rasterio.enums import Resampling
from tqdm import tqdm

from sr.pre_processing.cruts_config import CRUTSConfig
from sr.pre_processing.world_clim_config import WorldClimConfig

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
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed",
    )
    parser.add_argument(
        "--out_dir_world_clim",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed",
    )
    parser.add_argument(
        "--world_clim_elevation_fp",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/elevation/wc2.1_2.5m_elev.tif",
    )
    parser.add_argument(
        "--dataframe_output_path",
        type=str,
        default="../../datasets/",
    )
    parser.add_argument("--run_cruts_to_cog", type=bool, default=False)
    parser.add_argument("--run_statistics_computation", type=bool, default=True)
    parser.add_argument("--run_world_clim_resize", type=bool, default=False)
    parser.add_argument("--run_world_clim_tiling", type=bool, default=True)
    parser.add_argument("--run_world_clim_elevation_resize", type=bool, default=False)
    parser.add_argument("--patch_size", type=Tuple[int, int], default=(128, 128))
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--normalize_patches", type=bool, default=False)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--res_mult_inx", type=int, default=2)
    parser.add_argument("--threads_per_worker", type=int, default=1)
    parser.add_argument("--train_years", type=Tuple[int, int], default=(1961, 1999))
    parser.add_argument("--val_years", type=Tuple[int, int], default=(2000, 2004))
    parser.add_argument("--test_years", type=Tuple[int, int], default=(2005, 2019))

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


def cruts_as_cog(
    variable: str, data_dir: str, out_dir: str, dataframe_output_path: str
) -> None:
    """
    Creates a Cloud Optimized Geo-Tiff file for each time step in the CRU-TS dataset.

    Args:
        variable (str): The variable name.
        data_dir (str): Data dir.
        out_dir (str): Where to save the Geo-Tiffs.
        dataframe_output_path (str): Where to save the dataframe with results.

    """
    fp = CRUTSConfig.file_pattern.format(variable)
    file_path = os.path.join(data_dir, fp)
    out_path = os.path.join(out_dir, CRUTSConfig.full_res_dir, variable)
    ds = xarray.open_dataset(file_path)
    file_paths = []
    dataframe_output_path = os.path.join(dataframe_output_path, "cruts_inference")
    os.makedirs(dataframe_output_path, exist_ok=True)

    for i in range(ds.dims["time"]):
        # get frame at time index i
        arr = ds[variable].isel(time=i)

        # make it geo
        arr = dcug.assign_crs(arr, "EPSG:4326")

        # extract date
        date_str = np.datetime_as_string(arr.time, unit="D")

        fname = os.path.join(out_path, f"cruts-{variable}-{date_str}.tif")
        file_paths.append(fname)

        # Write as Cloud Optimized GeoTIFF
        write_cog(
            geo_im=arr,
            fname=fname,
            overwrite=True,
        )

    pd.DataFrame(file_paths, columns=["file_path"]).to_csv(
        os.path.join(dataframe_output_path, f"{variable}.csv"), index=False, header=True
    )


def ensure_sub_dirs_exist_wc(out_dir: str) -> None:
    """
    Ensures, that output dir structure exists for World Clim data.

    Args:
        out_dir (str): Output dir.

    """
    logger.info("Creating sub-dirs for WorldClim")

    variables = WorldClimConfig.variables_wc + [WorldClimConfig.elevation]
    for var in variables:
        for multiplier, _ in WorldClimConfig.resolution_multipliers:
            sub_dir_name = os.path.join(
                out_dir, var, WorldClimConfig.resized_dir, multiplier
            )
            logger.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)

            sub_dir_name = os.path.join(
                out_dir, var, WorldClimConfig.tiles_dir, multiplier
            )
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
    ds: Any,
    width: Optional[int] = 128,
    height: Optional[int] = 128,
    stride: Optional[int] = 64,
) -> Tuple[rio.windows.Window, rio.Affine]:
    """
    Using `rasterio` generate windows and transforms.

    Args:
        ds (Any): The input dataset.
        width (Optional[int]): The window width. Default: 128.
        height (Optional[int]): The window height. Default: 128.
        stride (Optional[int]): The stride of the tiles. Default: 64.

    Returns (Tuple[rio.windows.Window, rio.Affine]): A tuple with the window and the transform.

    """
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = np.array(
        list(
            product(
                range(0, ncols, stride if stride else width),
                range(0, nrows, stride if stride else height),
            )
        )
    )
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
    file_path: str,
    out_path: str,
    tile_shape: Optional[Tuple[int, int]] = (128, 128),
    stride: Optional[int] = 64,
    normalize: Optional[bool] = True,
) -> None:
    """
    Create tiles of specified size for ML purposes from specified geo-tiff file.

    Args:
        file_path (str): The path to the geo-tiff raster file.
        out_path (str): The output path.
        tile_shape (Optional[Tuple[int, int]]): The shape of the image patches.
        stride (Optional[int]): The stride of the tiles. Default: 64.
        normalize (Optional[bool]): Whether to normalize the patches to [0,1] range. True by default.

    """
    with rio.open(file_path) as in_dataset:
        tile_width, tile_height = tile_shape
        fname = os.path.basename(os.path.splitext(file_path)[0]) + ".{}.{}.tif"

        if normalize:
            data = in_dataset.read().astype(np.float32)
            data[data == -32768.0] = np.nan
            min = np.nanmin(data)
            max = np.nanmax(data)

        meta = in_dataset.meta.copy()

        for window, transform in get_tiles(in_dataset, tile_width, tile_height, stride):
            meta["transform"] = transform
            meta["dtype"] = np.float32
            meta["width"], meta["height"] = window.width, window.height

            out_fp = os.path.join(
                out_path, fname.format(int(window.col_off), int(window.row_off))
            )

            subset = in_dataset.read(window=window).astype(np.float32)

            # ignore tiles with more than 85% nan values
            # unless it's the elevation file
            if (
                np.count_nonzero(np.isnan(subset)) / np.prod(subset.shape) > 0.85
                and "elev" not in file_path
            ):
                continue

            with rio.open(out_fp, "w", **meta) as out_dataset:
                if normalize:
                    subset[subset == -32768.0] = np.nan
                    subset = (subset + (-min)) / (max - min + 1e-5)
                    subset[np.isnan(subset)] = 0.0

                out_dataset.write(subset)


def run_cruts_to_cog(args: argparse.Namespace) -> None:
    """
    Runs CRU-TS transformation from Net CDF to Cloud Optimized Geo-Tiffs.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_cruts_to_cog:
        logger.info("Running CRU-TS pre-processing - Geo Tiff generation")

        dask.bag.from_sequence(CRUTSConfig.variables_cts).map(
            cruts_as_cog,
            args.data_dir_cruts,
            args.out_dir_cruts,
            args.dataframe_output_path,
        ).compute()


def run_statistics_computation(args: argparse.Namespace) -> None:
    """
    Runs CRU-TS and World Clim statistics computation.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_statistics_computation:
        logger.info("Running statistics computation")

        def compute_stats(fp):
            with rio.open(fp) as ds:
                arr = ds.read(1)
                min = np.nanmin(arr)
                max = np.nanmax(arr)
                return min, max

        results = []

        for var in CRUTSConfig.variables_cts:
            logger.info(f"Computing stats for CRU-TS - '{var}'")
            for fp in tqdm(
                sorted(
                    glob(
                        os.path.join(
                            args.out_dir_cruts, CRUTSConfig.full_res_dir, var, "*.tif"
                        )
                    )
                )
            ):
                year_from_filename = int(os.path.basename(fp).split("-")[-3])
                month_from_filename = int(os.path.basename(fp).split("-")[-2])
                results.append(
                    (
                        "cru-ts",
                        fp,
                        os.path.basename(fp),
                        var,
                        year_from_filename,
                        month_from_filename,
                        *compute_stats(fp),
                    )
                )

        for var in WorldClimConfig.variables_wc + [WorldClimConfig.elevation]:
            logger.info(f"Computing stats for World Clim - '{var}'")
            for fp in tqdm(
                sorted(
                    glob(
                        os.path.join(
                            args.out_dir_world_clim,
                            var,
                            WorldClimConfig.resized_dir,
                            WorldClimConfig.resolution_multipliers[args.res_mult_inx][
                                0
                            ],
                            "*.tif",
                        )
                    )
                )
            ):
                year_from_filename = (
                    int(os.path.basename(fp).split("-")[0].split("_")[-1])
                    if var != WorldClimConfig.elevation
                    else -1
                )
                month_from_filename = (
                    int(os.path.basename(fp).split("-")[1].split(".")[0])
                    if var != WorldClimConfig.elevation
                    else -1
                )
                results.append(
                    (
                        "world-clim",
                        fp,
                        os.path.basename(fp),
                        var,
                        year_from_filename,
                        month_from_filename,
                        *compute_stats(fp),
                    )
                )

        output_file = os.path.join(args.dataframe_output_path, "statistics.csv")
        columns = [
            "dataset",
            "file_path",
            "filename",
            "variable",
            "year",
            "month",
            "min",
            "max",
        ]
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(output_file, header=True, index=False)


def run_world_clim_resize(args: argparse.Namespace) -> None:
    """
    Runs WorldClim resize operation.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_world_clim_resize:
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
            multiplier, scale = WorldClimConfig.resolution_multipliers[
                args.res_mult_inx
            ]
            logger.info(
                "Running WorldClim pre-processing for variable: "
                f"{var}, scale: {scale:.4f}, multiplier: {multiplier}. Total files to process: {len(files)}"
            )
            dask.bag.from_sequence(files, npartitions=1000).map(
                resize_raster, var, scale, multiplier, args.out_dir_world_clim
            ).compute()


def run_world_clim_elevation_resize(args: argparse.Namespace) -> None:
    """
    Runs WorldClim elevation resize operation.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_world_clim_elevation_resize:
        multiplier, scale = WorldClimConfig.resolution_multipliers[args.res_mult_inx]
        logger.info(
            "Running WorldClim pre-processing for variable: "
            f"elevation, scale: {scale:.4f}, multiplier: {multiplier}"
        )
        resize_raster(
            args.world_clim_elevation_fp,
            WorldClimConfig.elevation,
            scale,
            multiplier,
            args.out_dir_world_clim,
        )


def run_world_clim_tiling(args: argparse.Namespace) -> None:
    """
    Runs WorldClim tiling operation.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_world_clim_tiling:
        variables = WorldClimConfig.variables_wc + [WorldClimConfig.elevation]
        for var in variables:
            multiplier, scale = WorldClimConfig.resolution_multipliers[
                args.res_mult_inx
            ]
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
                args.patch_stride,
                args.normalize_patches,
            ).compute()


def run_train_val_test_split(args: argparse.Namespace) -> None:
    """
    Runs split into train, validation and test datasets based on provided configuration.

    Args:
        args (argparse.Namespace): The arguments.

    """
    variables = WorldClimConfig.variables_wc + [WorldClimConfig.elevation]

    for var in variables:
        multiplier, scale = WorldClimConfig.resolution_multipliers[args.res_mult_inx]
        if var != WorldClimConfig.elevation:
            logger.info(
                f"Generating Train/Validation/Test splits for variable: {var}, "
                f"multiplier: {multiplier}, scale:{scale:.4f}"
            )

        files = sorted(
            glob(
                os.path.join(
                    args.out_dir_world_clim,
                    var,
                    WorldClimConfig.tiles_dir,
                    multiplier,
                    "*.tif",
                )
            )
        )

        train_images = []
        val_images = []
        test_images = []
        elevation_images = []

        train_years_lower_bound, train_years_upper_bound = args.train_years
        val_years_lower_bound, val_years_upper_bound = args.val_years
        test_years_lower_bound, test_years_upper_bound = args.test_years

        os.makedirs(
            os.path.join(args.dataframe_output_path, var, multiplier), exist_ok=True
        )

        for file_path in files:
            filename = os.path.basename(file_path)
            x = int(file_path.split(".")[-3])
            y = int(file_path.split(".")[-2])
            original_filename = os.path.basename(file_path).replace(f".{x}.{y}.", ".")
            year_from_filename = (
                int(filename.split("-")[0].split("_")[-1])
                if var != WorldClimConfig.elevation
                else -1
            )
            month_from_filename = (
                int(filename.split("-")[1].split(".")[0])
                if var != WorldClimConfig.elevation
                else -1
            )

            if train_years_lower_bound <= year_from_filename <= train_years_upper_bound:
                train_images.append(
                    (
                        file_path,
                        original_filename,
                        var,
                        multiplier,
                        year_from_filename,
                        month_from_filename,
                        x,
                        y,
                    )
                )

            elif (
                (val_years_lower_bound <= year_from_filename <= val_years_upper_bound)
                and x % args.patch_size[1] == 0
                and y % args.patch_size[0] == 0
            ):
                val_images.append(
                    (
                        file_path,
                        original_filename,
                        var,
                        multiplier,
                        year_from_filename,
                        month_from_filename,
                        x,
                        y,
                    )
                )

            elif (
                (test_years_lower_bound <= year_from_filename <= test_years_upper_bound)
                and x % args.patch_size[1] == 0
                and y % args.patch_size[0] == 0
            ):
                test_images.append(
                    (
                        file_path,
                        original_filename,
                        var,
                        multiplier,
                        year_from_filename,
                        month_from_filename,
                        x,
                        y,
                    )
                )

            elif WorldClimConfig.elevation in file_path:
                elevation_images.append((file_path, var, multiplier, x, y))

        for stage, images in zip(
            ["train", "val", "test", WorldClimConfig.elevation],
            [train_images, val_images, test_images, elevation_images],
        ):
            if images:
                columns = (
                    ["file_path", "variable", "multiplier", "x", "y"]
                    if stage == WorldClimConfig.elevation
                    else [
                        "tile_file_path",
                        "filename",
                        "variable",
                        "multiplier",
                        "year",
                        "month",
                        "x",
                        "y",
                    ]
                )
                df = pd.DataFrame(
                    images,
                    columns=columns,
                )
                df.to_csv(
                    os.path.join(
                        args.dataframe_output_path, var, multiplier, f"{stage}.csv"
                    ),
                    index=False,
                    header=True,
                )

        if var != WorldClimConfig.elevation:
            logger.info(
                f"Generated Train ({len(train_images)}) / "
                f"Validation ({len(val_images)}) / "
                f"Test ({len(test_images)}) splits "
                f"for variable: {var}, multiplier: {multiplier}, scale:{scale:.4f}"
            )
        else:
            logger.info(
                f"({len(elevation_images)}) images "
                f"for variable: {var}, multiplier: {multiplier}, scale:{scale:.4f}"
            )


if __name__ == "__main__":
    arguments = parse_args()

    client = Client(
        n_workers=arguments.n_workers, threads_per_worker=arguments.threads_per_worker
    )

    try:
        ensure_sub_dirs_exist_cts(arguments.out_dir_cruts)
        ensure_sub_dirs_exist_wc(arguments.out_dir_world_clim)

        run_cruts_to_cog(arguments)
        run_statistics_computation(arguments)
        run_world_clim_resize(arguments)
        run_world_clim_elevation_resize(arguments)
        run_world_clim_tiling(arguments)
        run_train_val_test_split(arguments)
        logger.info("DONE")
    finally:
        client.close()
