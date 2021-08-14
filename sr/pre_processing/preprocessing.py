# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from itertools import product
from typing import Any, Optional, Tuple, List, Dict

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
from rasterio.mask import mask
from tqdm import tqdm

from sr.configs.cruts_config import CRUTSConfig
from sr.configs.world_clim_config import WorldClimConfig

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

# consts
europe_bbox_lr = ((-16.0, 84.5), (40.5, 33.0))
europe_bbox_hr = ((-16.0, 84.5), (40.5, 33.0))
left_upper_lr = [-16.0, 84.5]
left_lower_lr = [-16.0, 33.0]
right_upper_lr = [40.5, 84.5]
right_lower_lr = [40.5, 33.0]

left_upper_hr = [-16.0, 84.5]
left_lower_hr = [-16.0, 33.0]
right_upper_hr = [40.5, 84.5]
right_lower_hr = [40.5, 33.0]

lr_polygon = [
    [
        left_upper_lr,
        right_upper_lr,
        right_lower_lr,
        left_lower_lr,
        left_upper_lr,
    ]
]
hr_polygon = [
    [
        left_upper_hr,
        right_upper_hr,
        right_lower_hr,
        left_lower_hr,
        left_upper_hr,
    ]
]

var_to_variable = {
    CRUTSConfig.pre: "Precipitation",
    CRUTSConfig.tmn: "Minimum Temperature",
    CRUTSConfig.tmp: "Average Temperature",
    CRUTSConfig.tmx: "Maximum Temperature",
}

lr_bbox = [
    {
        "coordinates": lr_polygon,
        "type": "Polygon",
    }
]

hr_bbox = [
    {
        "coordinates": hr_polygon,
        "type": "Polygon",
    }
]


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
    parser.add_argument("--run_cruts_to_cog", type=bool, default=False)
    parser.add_argument("--run_temp_rasters_generation", type=bool, default=False)
    parser.add_argument("--run_statistics_computation", type=bool, default=False)
    parser.add_argument("--run_world_clim_resize", type=bool, default=False)
    parser.add_argument("--run_world_clim_tiling", type=bool, default=False)
    parser.add_argument("--run_world_clim_elevation_resize", type=bool, default=False)
    parser.add_argument("--run_train_val_test_split", type=bool, default=True)
    parser.add_argument("--run_extent_extraction", type=bool, default=False)
    parser.add_argument("--run_z_score_stats_computation", type=bool, default=False)
    parser.add_argument("--run_min_max_stats_computation", type=bool, default=True)
    parser.add_argument("--patch_size", type=Tuple[int, int], default=(128, 128))
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--normalize_patches", type=bool, default=False)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--res_mult_inx", type=int, default=2)
    parser.add_argument("--threads_per_worker", type=int, default=1)
    parser.add_argument(
        "--train_years", type=Tuple[int, int], default=(1961, 2004)
    )  # 1961, 1999
    parser.add_argument(
        "--val_years", type=Tuple[int, int], default=(2005, 2017)
    )  # 2000, 2004
    parser.add_argument(
        "--test_years", type=Tuple[int, int], default=(2018, 2019)
    )  # 2005, 2019

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


def compute_stats_for_zscore(args: argparse.Namespace) -> None:
    """
    Computes dataset statistics for z-score standardization.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if not args.run_z_score_stats_computation:
        return

    logger.info("Running statistical computation for z-score")

    def compute_stats(var_name, arr):
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        min = np.nanmin(arr)
        max = np.nanmax(arr)
        normalized_min = (min - mean) / (std + 1e-5)
        normalized_max = (max - mean) / (std + 1e-5)
        results.append((var_name, mean, std, min, max, normalized_min, normalized_max))

    results = []
    for var in tqdm(CRUTSConfig.variables_cts + [WorldClimConfig.elevation]):
        if var == WorldClimConfig.elevation:
            elevation = rio.open(args.world_clim_elevation_fp).read().astype(np.float32)
            elevation[elevation == -32768.0] = np.nan
            compute_stats(var, elevation)
        else:
            ds = xarray.open_dataset(
                os.path.join(args.data_dir_cruts, CRUTSConfig.file_pattern.format(var))
            )
            compute_stats(var, ds[var].values)

    output_file = os.path.join(args.dataframe_output_path, "statistics_zscore.csv")
    df = pd.DataFrame(
        results,
        columns=[
            "variable",
            "mean",
            "std",
            "min",
            "max",
            "normalized_min",
            "normalized_max",
        ],
    )
    df.to_csv(output_file, header=True, index=False)


def compute_stats_for_min_max_normalization(args: argparse.Namespace) -> None:
    """
    Computes dataset statistics for min max normalization.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if not args.run_min_max_stats_computation:
        return

    logger.info("Running statistical computation for min-max normalization")

    def compute_stats(fp):
        with rio.open(fp) as ds:
            arr = ds.read(1)
            arr[arr == -32768.0] = 0.0  # handle elevation masked values
            min = np.nanmin(arr)
            max = np.nanmax(arr)
            return min, max

    results = []

    def _stats_for_cruts(fp):
        year_from_filename = int(os.path.basename(fp).split("-")[-3])
        month_from_filename = int(os.path.basename(fp).split("-")[-2])
        return (
            "cru-ts",
            fp,
            os.path.basename(fp),
            var,
            year_from_filename,
            month_from_filename,
            *compute_stats(fp),
        )

    for var in CRUTSConfig.variables_cts:
        logger.info(f"Computing stats for CRU-TS - '{var}'")
        results.extend(
            dask.bag.from_sequence(
                sorted(
                    glob(
                        os.path.join(
                            args.out_dir_cruts, CRUTSConfig.full_res_dir, var, "*.tif"
                        )
                    )
                )
            )
            .map(_stats_for_cruts)
            .compute()
        )

    def _stats_for_wc(fp):
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
        return (
            "world-clim",
            fp,
            os.path.basename(fp),
            var,
            year_from_filename,
            month_from_filename,
            *compute_stats(fp),
        )

    for var in WorldClimConfig.variables_wc + [WorldClimConfig.elevation]:
        logger.info(f"Computing stats for World Clim - '{var}'")
        results.extend(
            dask.bag.from_sequence(
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
            )
            .map(_stats_for_wc)
            .compute()
        )

    output_file = os.path.join(args.dataframe_output_path, "statistics_min_max.csv")
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

    # compute global min & max
    df["global_min"] = np.nan
    df["global_max"] = np.nan

    grouped_min_series = df.groupby("variable").min()["min"]
    grouped_max_series = df.groupby("variable").max()["max"]
    global_min_max_lookup = pd.DataFrame(
        {
            "global_min": grouped_min_series,
            "global_max": grouped_max_series,
        }
    ).to_dict(orient="index")

    cruts_global_min = 0.0
    cruts_global_max = 0.0
    wc_global_min = 0.0
    wc_global_max = 0.0

    for key, val in global_min_max_lookup.items():
        if key in CRUTSConfig.temperature_vars:
            cruts_global_min = np.minimum(cruts_global_min, val["global_min"])
            cruts_global_max = np.maximum(cruts_global_max, val["global_max"])
        if key in WorldClimConfig.temperature_vars:
            wc_global_min = np.minimum(wc_global_min, val["global_min"])
            wc_global_max = np.maximum(wc_global_max, val["global_max"])

    for key, val in global_min_max_lookup.items():
        if key in CRUTSConfig.temperature_vars:
            val["global_min"] = cruts_global_min
            val["global_max"] = cruts_global_max
        if key in WorldClimConfig.temperature_vars:
            val["global_min"] = wc_global_min
            val["global_max"] = wc_global_max

    global_min_max_lookup[WorldClimConfig.elevation]["global_min"] = 0.0

    for idx, row in df.iterrows():
        var = row["variable"]
        df.loc[idx, "global_min"] = global_min_max_lookup[var]["global_min"]
        df.loc[idx, "global_max"] = global_min_max_lookup[var]["global_max"]

    df.to_csv(output_file, header=True, index=False)


def run_statistics_computation(args: argparse.Namespace) -> None:
    """
    Runs CRU-TS and World Clim statistics computation.

    Args:
        args (argparse.Namespace): The arguments.

    """
    if args.run_statistics_computation:
        logger.info("Running statistics computation")

        compute_stats_for_zscore(args)
        compute_stats_for_min_max_normalization(args)


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
    if args.run_train_val_test_split:
        variables = WorldClimConfig.variables_wc + [WorldClimConfig.elevation]

        for var in variables:
            multiplier, scale = WorldClimConfig.resolution_multipliers[
                args.res_mult_inx
            ]
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
                original_filename = os.path.basename(file_path).replace(
                    f".{x}.{y}.", "."
                )
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

                if (
                    train_years_lower_bound
                    <= year_from_filename
                    <= train_years_upper_bound
                ):
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
                    (
                        val_years_lower_bound
                        <= year_from_filename
                        <= val_years_upper_bound
                    )
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
                    (
                        test_years_lower_bound
                        <= year_from_filename
                        <= test_years_upper_bound
                    )
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


def extract_extent_single(
    fp: str,
    bbox: List[Dict[str, Any]],
    variable: str,
    extent_out_path: str,
) -> None:
    """
    Extracts polygon extent form a single Geo-Tiff file.

    Args:
        fp (str): The Geo-Tiff file path.
        bbox (List[Dict[str, Any]]): The bounding box acceptable by `rasterio.mask()` method.
        variable (str): The name of the variable.
        extent_out_path (str): The out file path.

    """
    filename = os.path.basename(fp)

    with rio.open(fp) as ds:
        crop, transform = mask(ds, bbox, crop=True)
        meta = ds.meta

    meta.update(
        {
            "driver": "GTiff",
            "height": crop.shape[1],
            "width": crop.shape[2],
            "transform": transform,
        }
    )

    with rio.open(
        os.path.join(extent_out_path, variable, filename), "w", **meta
    ) as dest:
        dest.write(crop)


def extract_extent(
    src_dir: str,
    extent_out_path: str,
    cruts_variables: List[str],
    bbox: List[Dict[str, Any]],
) -> None:
    """
    Extracts the polygon extent from the CRU-TS variable Geo-Tiff files.

    Args:
        src_dir (str): The source Geo-Tiff directory.
        extent_out_path (str): The extent out path.
        cruts_variables (List[str]): The CRU-TS variables.
        bbox (List[Dict[str, Any]]): The bounding box of the extent.

    """

    for var in cruts_variables:
        logger.info(f"Extracting Europe polygon extents for variable '{var}'")
        files = sorted(glob(os.path.join(src_dir, var, "*.tif")))
        os.makedirs(os.path.join(extent_out_path, var), exist_ok=True)
        dask.bag.from_sequence(files).map(
            extract_extent_single,
            bbox=bbox,
            variable=var,
            extent_out_path=extent_out_path,
        ).compute()
        logger.info(f"Done for variable '{var}'")


def run_cruts_extent_extraction(args: argparse.Namespace) -> None:
    """
    Run Europe extent extraction for Geo-Tiff files.

    Args:
        args (argparse.Namespace): The arguments.

    """

    if args.run_extent_extraction:
        # handle extent dirs
        extent_dir = os.path.join(args.out_dir_cruts, CRUTSConfig.europe_extent)
        os.makedirs(os.path.join(extent_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(extent_dir, CRUTSConfig.elev), exist_ok=True)

        logger.info("Extracting polygon extents for Europe.")
        extract_extent(
            os.path.join(args.out_dir_cruts, CRUTSConfig.full_res_dir),
            extent_dir,
            CRUTSConfig.variables_cts,
            lr_bbox,
        )
        logger.info("Extracting polygon extents for Europe for land mask file.")
        extract_extent_single(
            args.land_mask_file,
            hr_bbox,
            "mask",
            extent_dir,
        )
        logger.info("Extracting polygon extents for Europe for elevation file.")
        extract_extent_single(
            args.elevation_file,
            hr_bbox,
            CRUTSConfig.elev,
            extent_dir,
        )


def generate_temp_raster(tmin_raster_fname: str) -> None:
    """
    Generates temp raster from tmin and tmax rasters.
    Args:
        tmin_raster_fname (str): The filename of the tmin raster.
    """
    output_file_name = tmin_raster_fname.replace("/tmin/", "/temp/").replace(
        "_tmin_", "_temp_"
    )
    tmax_raster_fname = tmin_raster_fname.replace("/tmin/", "/tmax/").replace(
        "_tmin_", "_tmax_"
    )

    with rio.open(tmin_raster_fname) as tmin_raster:
        with rio.open(tmax_raster_fname) as tmax_raster:
            with rio.open(
                output_file_name, **tmin_raster.profile, mode="w"
            ) as temp_raster:
                tmin_values = tmin_raster.read()
                tmax_values = tmax_raster.read()
                temp_values = (tmax_values + tmin_values) / 2.0
                temp_raster.write(temp_values)


def run_temp_rasters_generation(args: argparse.Namespace) -> None:
    """
    Runs temp raster generation from tmin and tmax rasters.
    Args:
        args (argparse.Namespace): The arguments.
    """
    if args.run_temp_rasters_generation:
        logger.info("Running temp raster generation")
        multiplier, scale = WorldClimConfig.resolution_multipliers[args.res_mult_inx]
        os.makedirs(
            os.path.join(
                args.out_dir_world_clim,
                WorldClimConfig.temp,
                WorldClimConfig.resized_dir,
                multiplier,
            ),
            exist_ok=True,
        )

        tmin_files = sorted(
            glob(
                os.path.join(
                    args.out_dir_world_clim,
                    WorldClimConfig.tmin,
                    WorldClimConfig.resized_dir,
                    multiplier,
                    "*.tif",
                )
            )
        )

        dask.bag.from_sequence(tmin_files).map(generate_temp_raster).compute()

        logger.info("Done with temp raster generation")


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
        run_temp_rasters_generation(arguments)
        run_world_clim_elevation_resize(arguments)
        run_world_clim_tiling(arguments)
        run_train_val_test_split(arguments)
        run_cruts_extent_extraction(arguments)
        logger.info("DONE")
    finally:
        client.close()
