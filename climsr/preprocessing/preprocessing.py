# -*- coding: utf-8 -*-
import logging
import os
from glob import glob
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import dask.bag
import datacube.utils.geometry as dcug
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
import rasterio.windows
import xarray
from datacube.utils.cog import write_cog
from tqdm import tqdm

import climsr.consts as consts
from climsr.core.config import PreProcessingConfig


def ensure_sub_dirs_exist_cts(out_dir: str) -> None:
    """
    Ensures, that output dir structure exists for CRU-TS data.

    Args:
        out_dir (str): Output dir.

    """
    logging.info("Creating sub-dirs for CRU-TS")
    for dir_name in consts.cruts.sub_dirs_cts:
        for var in consts.cruts.variables_cts:
            sub_dir_name = os.path.join(out_dir, dir_name, var)
            logging.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)


def cruts_as_cog(variable: str, data_dir: str, out_dir: str, dataframe_output_path: str) -> None:
    """
    Creates a Cloud Optimized Geo-Tiff file for each time step in the CRU-TS dataset.

    Args:
        variable (str): The variable name.
        data_dir (str): Data dir.
        out_dir (str): Where to save the Geo-Tiffs.
        dataframe_output_path (str): Where to save the dataframe with results.

    """
    fp = consts.cruts.file_pattern.format(variable)
    file_path = os.path.join(data_dir, fp)
    out_path = os.path.join(out_dir, consts.cruts.full_res_dir, variable)
    ds = xarray.open_dataset(file_path)
    file_paths = []
    dataframe_output_path = os.path.join(dataframe_output_path, "cruts_inference")
    os.makedirs(dataframe_output_path, exist_ok=True)

    for i in range(ds.dims["time"]):
        # todo https://pratiman-91.github.io/2020/08/01/NetCDF-to-GeoTIFF-using-Python.html

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

    pd.DataFrame(file_paths, columns=[consts.datasets_and_preprocessing.file_path]).to_csv(
        os.path.join(dataframe_output_path, f"{variable}.csv"), index=False, header=True
    )


def ensure_sub_dirs_exist_wc(out_dir: str) -> None:
    """
    Ensures, that output dir structure exists for World Clim data.

    Args:
        out_dir (str): Output dir.

    """
    logging.info("Creating sub-dirs for WorldClim")

    variables = consts.world_clim.variables_wc + [consts.world_clim.elevation]
    for var in variables:
        for multiplier, _ in consts.world_clim.resolution_multipliers:
            sub_dir_name = os.path.join(out_dir, var, consts.world_clim.resized_dir, multiplier)
            logging.info(f"Creating sub-dir: '{sub_dir_name}'")
            os.makedirs(sub_dir_name, exist_ok=True)

            sub_dir_name = os.path.join(out_dir, var, consts.world_clim.tiles_dir, multiplier)
            logging.info(f"Creating sub-dir: '{sub_dir_name}'")
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
        transform = rio.Affine(t.a / scaling_factor, t.b, t.c, t.d, t.e / scaling_factor, t.f)
        height = int(raster.height * scaling_factor)
        width = int(raster.width * scaling_factor)

        profile = raster.profile
        profile.update(transform=transform, driver="COG", height=height, width=width)

        data = raster.read(
            out_shape=(raster.count, height, width),
            resampling=rio.enums.Resampling.nearest,
        )

        fname = os.path.join(
            out_dir,
            variable,
            consts.world_clim.resized_dir,
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
    big_window = rio.windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    for col_off, row_off in offsets:
        leftover_w = ncols - col_off
        leftover_h = nrows - row_off

        if leftover_w < width:
            col_off = ncols - width

        if leftover_h < height:
            row_off = nrows - height

        window = rio.windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = rio.windows.transform(window, ds.transform)

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
            data[data == consts.world_clim.elevation_missing_indicator] = np.nan
            min = np.nanmin(data)
            max = np.nanmax(data)

        meta = in_dataset.meta.copy()

        for window, transform in get_tiles(in_dataset, tile_width, tile_height, stride):
            meta["transform"] = transform
            meta["dtype"] = np.float32
            meta["width"], meta["height"] = window.width, window.height

            out_fp = os.path.join(out_path, fname.format(int(window.col_off), int(window.row_off)))

            subset = in_dataset.read(window=window).astype(np.float32)

            # ignore tiles with more than 85% nan values
            # unless it's the elevation file
            if np.count_nonzero(np.isnan(subset)) / np.prod(subset.shape) > 0.85 and "elev" not in file_path:
                continue

            with rio.open(out_fp, "w", **meta) as out_dataset:
                if normalize:
                    subset[subset == consts.world_clim.elevation_missing_indicator] = np.nan
                    subset = (subset + (-min)) / (max - min + 1e-5)
                    subset[np.isnan(subset)] = 0.0

                out_dataset.write(subset)


def run_cruts_to_cog(cfg: PreProcessingConfig) -> None:
    """
    Runs CRU-TS transformation from Net CDF to Cloud Optimized Geo-Tiffs.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_cruts_to_cog:
        logging.info("Running CRU-TS pre-processing - Geo Tiff generation")

        dask.bag.from_sequence(consts.cruts.variables_cts).map(
            cruts_as_cog,
            cfg.data_dir_cruts,
            cfg.out_dir_cruts,
            cfg.dataframe_output_path,
        ).compute()


def compute_stats_for_zscore(cfg: PreProcessingConfig) -> None:
    """
    Computes dataset statistics for z-score standardization.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if not cfg.run_z_score_stats_computation:
        return

    logging.info("Running statistical computation for z-score")

    def compute_stats(var_name, arr):
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        min = np.nanmin(arr)
        max = np.nanmax(arr)
        normalized_min = (min - mean) / (std + 1e-5)
        normalized_max = (max - mean) / (std + 1e-5)
        results.append((var_name, mean, std, min, max, normalized_min, normalized_max))

    results = []
    for var in tqdm(consts.cruts.variables_cts + [consts.world_clim.elevation]):
        if var == consts.world_clim.elevation:
            elevation = rio.open(cfg.world_clim_elevation_fp).read().astype(np.float32)
            elevation[elevation == consts.world_clim.elevation_missing_indicator] = np.nan
            compute_stats(var, elevation)
        else:
            ds = xarray.open_dataset(os.path.join(cfg.data_dir_cruts, consts.cruts.file_pattern.format(var)))
            compute_stats(var, ds[var].values)

    output_file = os.path.join(cfg.dataframe_output_path, "statistics_zscore.csv")
    df = pd.DataFrame(
        results,
        columns=[
            consts.datasets_and_preprocessing.variable,
            consts.stats.mean,
            consts.stats.std,
            consts.stats.min,
            consts.stats.max,
            consts.stats.normalized_min,
            consts.stats.normalized_max,
        ],
    )
    df.to_csv(output_file, header=True, index=False)


def compute_stats_for_min_max_normalization(cfg: PreProcessingConfig) -> None:
    """
    Computes dataset statistics for min max normalization.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if not cfg.run_min_max_stats_computation:
        return

    logging.info("Running statistical computation for min-max normalization")

    def compute_stats(fp):
        with rio.open(fp) as ds:
            arr = ds.read(1)
            arr[arr == consts.world_clim.elevation_missing_indicator] = 0.0  # handle elevation masked values
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

    for var in consts.cruts.variables_cts:
        logging.info(f"Computing stats for CRU-TS - '{var}'")
        results.extend(
            dask.bag.from_sequence(sorted(glob(os.path.join(cfg.out_dir_cruts, consts.cruts.full_res_dir, var, "*.tif"))))
            .map(_stats_for_cruts)
            .compute()
        )

    def _stats_for_wc(fp):
        year_from_filename = int(os.path.basename(fp).split("-")[0].split("_")[-1]) if var != consts.world_clim.elevation else -1
        month_from_filename = int(os.path.basename(fp).split("-")[1].split(".")[0]) if var != consts.world_clim.elevation else -1
        return (
            "world-clim",
            fp,
            os.path.basename(fp),
            var,
            year_from_filename,
            month_from_filename,
            *compute_stats(fp),
        )

    for var in consts.world_clim.variables_wc + [consts.world_clim.elevation]:
        logging.info(f"Computing stats for World Clim - '{var}'")
        results.extend(
            dask.bag.from_sequence(
                sorted(
                    glob(
                        os.path.join(
                            cfg.out_dir_world_clim,
                            var,
                            consts.world_clim.resized_dir,
                            consts.world_clim.resolution_multipliers[cfg.res_mult_inx][0],
                            "*.tif",
                        )
                    )
                )
            )
            .map(_stats_for_wc)
            .compute()
        )

    output_file = os.path.join(cfg.dataframe_output_path, "statistics_min_max.csv")
    columns = [
        consts.datasets_and_preprocessing.dataset,
        consts.datasets_and_preprocessing.file_path,
        consts.datasets_and_preprocessing.filename,
        consts.datasets_and_preprocessing.variable,
        consts.datasets_and_preprocessing.year,
        consts.datasets_and_preprocessing.year,
        consts.stats.min,
        consts.stats.max,
    ]

    df = pd.DataFrame(results, columns=columns)

    # compute global min & max
    df[consts.stats.global_min] = np.nan
    df[consts.stats.global_max] = np.nan

    grouped_min_series = df.groupby(consts.datasets_and_preprocessing.variable).min()[consts.stats.min]
    grouped_max_series = df.groupby(consts.datasets_and_preprocessing.variable).max()[consts.stats.max]
    global_min_max_lookup = pd.DataFrame(
        {
            consts.stats.global_min: grouped_min_series,
            consts.stats.global_max: grouped_max_series,
        }
    ).to_dict(orient="index")

    cruts_global_min = 0.0
    cruts_global_max = 0.0
    wc_global_min = 0.0
    wc_global_max = 0.0

    for key, val in global_min_max_lookup.items():
        if key in consts.cruts.temperature_vars:
            cruts_global_min = np.minimum(cruts_global_min, val[consts.stats.global_min])
            cruts_global_max = np.maximum(cruts_global_max, val[consts.stats.global_max])
        if key in consts.world_clim.temperature_vars:
            wc_global_min = np.minimum(wc_global_min, val[consts.stats.global_min])
            wc_global_max = np.maximum(wc_global_max, val[consts.stats.global_max])

    for key, val in global_min_max_lookup.items():
        if key in consts.cruts.temperature_vars:
            val[consts.stats.global_min] = cruts_global_min
            val[consts.stats.global_max] = cruts_global_max
        if key in consts.world_clim.temperature_vars:
            val[consts.stats.global_min] = wc_global_min
            val[consts.stats.global_max] = wc_global_max

    global_min_max_lookup[consts.world_clim.elevation][consts.stats.global_min] = 0.0

    for idx, row in df.iterrows():
        var = row[consts.datasets_and_preprocessing.variable]
        df.loc[idx, consts.stats.global_min] = global_min_max_lookup[var][consts.stats.global_min]
        df.loc[idx, consts.stats.global_max] = global_min_max_lookup[var][consts.stats.global_max]

    df.to_csv(output_file, header=True, index=False)


def run_statistics_computation(cfg: PreProcessingConfig) -> None:
    """
    Runs CRU-TS and World Clim statistics computation.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_statistics_computation:
        logging.info("Running statistics computation")

        compute_stats_for_zscore(cfg)
        compute_stats_for_min_max_normalization(cfg)


def run_world_clim_resize(cfg: PreProcessingConfig) -> None:
    """
    Runs WorldClim resize operation.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_world_clim_resize:
        for var in consts.world_clim.variables_wc:
            files = sorted(
                glob(
                    os.path.join(
                        cfg.data_dir_world_clim,
                        var,
                        "**",
                        consts.world_clim.pattern_wc,
                    ),
                    recursive=True,
                )
            )
            multiplier, scale = consts.world_clim.resolution_multipliers[cfg.res_mult_inx]
            logging.info(
                "Running WorldClim pre-processing for variable: "
                f"{var}, scale: {scale:.4f}, multiplier: {multiplier}. Total files to process: {len(files)}"
            )
            dask.bag.from_sequence(files, npartitions=1000).map(
                resize_raster, var, scale, multiplier, cfg.out_dir_world_clim
            ).compute()


def run_world_clim_elevation_resize(cfg: PreProcessingConfig) -> None:
    """
    Runs WorldClim elevation resize operation.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_world_clim_elevation_resize:
        multiplier, scale = consts.world_clim.resolution_multipliers[cfg.res_mult_inx]
        logging.info("Running WorldClim pre-processing for variable: " f"elevation, scale: {scale:.4f}, multiplier: {multiplier}")
        resize_raster(
            cfg.world_clim_elevation_fp,
            consts.world_clim.elevation,
            scale,
            multiplier,
            cfg.out_dir_world_clim,
        )


def run_world_clim_tiling(cfg: PreProcessingConfig) -> None:
    """
    Runs WorldClim tiling operation.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_world_clim_tiling:
        variables = consts.world_clim.variables_wc + [consts.world_clim.elevation]
        for var in variables:
            multiplier, scale = consts.world_clim.resolution_multipliers[cfg.res_mult_inx]
            files = sorted(
                glob(
                    os.path.join(
                        cfg.out_dir_world_clim,
                        var,
                        consts.world_clim.resized_dir,
                        multiplier,
                        "*.tif",
                    )
                )
            )
            logging.info(
                f"WorldClim - Running tile generation. Total files: {len(files)}, "
                f"variable: {var}, scale: {scale:.4f}, multiplier: {multiplier}"
            )
            dask.bag.from_sequence(files).map(
                make_patches,
                os.path.join(
                    cfg.out_dir_world_clim,
                    var,
                    consts.world_clim.tiles_dir,
                    multiplier,
                ),
                cfg.patch_size,
                cfg.patch_stride,
                cfg.normalize_patches,
            ).compute()


def run_train_val_test_split(cfg: PreProcessingConfig) -> None:
    """
    Runs split into train, validation and test datasets based on provided configuration.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """
    if cfg.run_train_val_test_split:
        variables = consts.world_clim.variables_wc + [consts.world_clim.elevation]

        for var in variables:
            multiplier, scale = consts.world_clim.resolution_multipliers[cfg.res_mult_inx]
            if var != consts.world_clim.elevation:
                logging.info(
                    f"Generating Train/Validation/Test splits for variable: {var}, "
                    f"multiplier: {multiplier}, scale:{scale:.4f}"
                )

            files = sorted(
                glob(
                    os.path.join(
                        cfg.out_dir_world_clim,
                        var,
                        consts.world_clim.tiles_dir,
                        multiplier,
                        "*.tif",
                    )
                )
            )

            train_images = []
            val_images = []
            test_images = []
            elevation_images = []

            train_years_lower_bound, train_years_upper_bound = cfg.train_years
            val_years_lower_bound, val_years_upper_bound = cfg.val_years
            test_years_lower_bound, test_years_upper_bound = cfg.test_years

            os.makedirs(os.path.join(cfg.dataframe_output_path, var, multiplier), exist_ok=True)

            for file_path in files:
                filename = os.path.basename(file_path)
                x = int(file_path.split(".")[-3])
                y = int(file_path.split(".")[-2])
                original_filename = os.path.basename(file_path).replace(f".{x}.{y}.", ".")
                year_from_filename = int(filename.split("-")[0].split("_")[-1]) if var != consts.world_clim.elevation else -1
                month_from_filename = int(filename.split("-")[1].split(".")[0]) if var != consts.world_clim.elevation else -1

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
                    and x % cfg.patch_size[1] == 0
                    and y % cfg.patch_size[0] == 0
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
                    and x % cfg.patch_size[1] == 0
                    and y % cfg.patch_size[0] == 0
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

                elif consts.world_clim.elevation in file_path:
                    elevation_images.append((file_path, var, multiplier, x, y))

            for stage, images in zip(
                [
                    consts.stages.train,
                    consts.stages.val,
                    consts.stages.test,
                    consts.world_clim.elevation,
                ],
                [train_images, val_images, test_images, elevation_images],
            ):
                if images:
                    columns = (
                        [
                            consts.datasets_and_preprocessing.file_path,
                            consts.datasets_and_preprocessing.variable,
                            consts.datasets_and_preprocessing.multiplier,
                            consts.datasets_and_preprocessing.x,
                            consts.datasets_and_preprocessing.y,
                        ]
                        if stage == consts.world_clim.elevation
                        else [
                            consts.datasets_and_preprocessing.tile_file_path,
                            consts.datasets_and_preprocessing.filename,
                            consts.datasets_and_preprocessing.variable,
                            consts.datasets_and_preprocessing.multiplier,
                            consts.datasets_and_preprocessing.year,
                            consts.datasets_and_preprocessing.year,
                            consts.datasets_and_preprocessing.x,
                            consts.datasets_and_preprocessing.y,
                        ]
                    )
                    df = pd.DataFrame(
                        images,
                        columns=columns,
                    )
                    df.to_csv(
                        os.path.join(cfg.dataframe_output_path, var, multiplier, f"{stage}.csv"),
                        index=False,
                        header=True,
                    )

            if var != consts.world_clim.elevation:
                logging.info(
                    f"Generated Train ({len(train_images)}) / "
                    f"Validation ({len(val_images)}) / "
                    f"Test ({len(test_images)}) splits "
                    f"for variable: {var}, multiplier: {multiplier}, scale:{scale:.4f}"
                )
            else:
                logging.info(
                    f"({len(elevation_images)}) images " f"for variable: {var}, multiplier: {multiplier}, scale:{scale:.4f}"
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
        crop, transform = rio.mask.mask(ds, bbox, crop=True)
        meta = ds.meta

    meta.update(
        {
            "driver": "GTiff",
            "height": crop.shape[1],
            "width": crop.shape[2],
            "transform": transform,
        }
    )

    with rio.open(os.path.join(extent_out_path, variable, filename), "w", **meta) as dest:
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
        logging.info(f"Extracting Europe polygon extents for variable '{var}'")
        files = sorted(glob(os.path.join(src_dir, var, "*.tif")))
        os.makedirs(os.path.join(extent_out_path, var), exist_ok=True)
        dask.bag.from_sequence(files).map(
            extract_extent_single,
            bbox=bbox,
            variable=var,
            extent_out_path=extent_out_path,
        ).compute()
        logging.info(f"Done for variable '{var}'")


def run_cruts_extent_extraction(cfg: PreProcessingConfig) -> None:
    """
    Run Europe extent extraction for Geo-Tiff files.

    Args:
        cfg (PreProcessingConfig): The arguments.

    """

    if cfg.run_extent_extraction:
        # handle extent dirs
        extent_dir = os.path.join(cfg.out_dir_cruts, consts.cruts.europe_extent)
        os.makedirs(os.path.join(extent_dir, consts.batch_items.mask), exist_ok=True)
        os.makedirs(os.path.join(extent_dir, consts.cruts.elev), exist_ok=True)

        logging.info("Extracting polygon extents for Europe.")
        extract_extent(
            os.path.join(cfg.out_dir_cruts, consts.cruts.full_res_dir),
            extent_dir,
            consts.cruts.variables_cts,
            consts.datasets_and_preprocessing.lr_bbox,
        )
        logging.info("Extracting polygon extents for Europe for land mask file.")
        extract_extent_single(
            cfg.land_mask_file,
            consts.datasets_and_preprocessing.hr_bbox,
            consts.batch_items.mask,
            extent_dir,
        )
        logging.info("Extracting polygon extents for Europe for elevation file.")
        extract_extent_single(
            cfg.elevation_file,
            consts.datasets_and_preprocessing.hr_bbox,
            consts.cruts.elev,
            extent_dir,
        )


def generate_temp_raster(tmin_raster_fname: str) -> None:
    """
    Generates temp raster from tmin and tmax rasters.
    Args:
        tmin_raster_fname (str): The filename of the tmin raster.
    """
    output_file_name = tmin_raster_fname.replace("/tmin/", "/temp/").replace("_tmin_", "_temp_")
    tmax_raster_fname = tmin_raster_fname.replace("/tmin/", "/tmax/").replace("_tmin_", "_tmax_")

    with rio.open(tmin_raster_fname) as tmin_raster:
        with rio.open(tmax_raster_fname) as tmax_raster:
            with rio.open(output_file_name, **tmin_raster.profile, mode="w") as temp_raster:
                tmin_values = tmin_raster.read()
                tmax_values = tmax_raster.read()
                temp_values = (tmax_values + tmin_values) / 2.0
                temp_raster.write(temp_values)


def run_temp_rasters_generation(cfg: PreProcessingConfig) -> None:
    """
    Runs temp raster generation from tmin and tmax rasters.
    Args:
        cfg (PreProcessingConfig): The arguments.
    """
    if cfg.run_temp_rasters_generation:
        logging.info("Running temp raster generation")
        multiplier, scale = consts.world_clim.resolution_multipliers[cfg.res_mult_inx]
        os.makedirs(
            os.path.join(
                cfg.out_dir_world_clim,
                consts.world_clim.temp,
                consts.world_clim.resized_dir,
                multiplier,
            ),
            exist_ok=True,
        )

        tmin_files = sorted(
            glob(
                os.path.join(
                    cfg.out_dir_world_clim,
                    consts.world_clim.tmin,
                    consts.world_clim.resized_dir,
                    multiplier,
                    "*.tif",
                )
            )
        )

        dask.bag.from_sequence(tmin_files).map(generate_temp_raster).compute()

        logging.info("Done with temp raster generation")
