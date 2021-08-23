# -*- coding: utf-8 -*-
import logging
import os
from glob import glob
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
import xarray as xr
from core.config import InferenceConfig
from core.task import TaskSuperResolutionModule
from data.sr.cruts_inference_dataset import CRUTSInferenceDataset
from data.sr.geo_tiff_inference_dataset import GeoTiffInferenceDataset
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import climsr.consts as consts
from climsr.data.normalization import MinMaxScaler


def inference_on_full_images(
    model: pl.LightningModule,
    ds: Dataset,
    land_mask_file: str,
    out_dir: str,
    normalization_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> None:
    """
    Runs inference pipeline on full sized images.

    Args:
        model (pl.LightningModule): The model.
        ds (Dataset): The dataset.
        land_mask_file (str): The HR land mask file.
        out_dir (str): The output SR Geo-Tiff dir.
        normalization_range (Optional[Tuple[float, float]]): Optional normalization range. `(-1, 1)` by default.

    """

    # ensure gpu
    model = model.cuda()

    # load mask
    with rio.open(land_mask_file) as mask_src:
        profile = mask_src.profile
        plt.imshow(mask_src.read().squeeze(0), cmap="jet")
        plt.show()

    # prepare dataloader
    dl = DataLoader(dataset=ds, batch_size=1, pin_memory=True, num_workers=1)

    scaler = MinMaxScaler(feature_range=normalization_range)

    # run inference
    for i, batch in tqdm(enumerate(dl), total=len(dl)):
        lr = batch[consts.batch_items.lr].cuda()
        elev = batch[consts.batch_items.elevation].cuda()
        mask = batch[consts.batch_items.mask].cuda()
        mask_np = batch[consts.batch_items.mask_np].squeeze(0).squeeze(0).numpy()
        min = batch[consts.stats.min].numpy()
        max = batch[consts.batch_items.max].numpy()
        filename = batch[consts.batch_items.filename]

        outputs = model(lr, elev, mask)
        outputs = outputs.cpu().numpy()

        for idx, output in enumerate(outputs):
            arr = output.squeeze(0)
            arr = scaler.denormalize(arr, min[idx], max[idx]).clip(min[idx], max[idx])
            arr[~mask_np] = np.nan

            with rio.open(os.path.join(out_dir, filename[idx]), "w", **profile) as dataset:
                dataset.write(arr, 1)

        if i == 0:
            plt.imshow(lr.cpu().squeeze(0).permute(1, 2, 0).numpy())
            plt.show()
            plt.imshow(mask_np, cmap="jet")
            plt.show()
            plt.imshow(mask.cpu().squeeze(0).squeeze(0).numpy(), cmap="jet")
            plt.show()
            plt.imshow(elev.cpu().squeeze(0).squeeze(0).numpy(), cmap="jet")
            plt.show()
            plt.imshow(arr, cmap="jet")
            plt.show()


def run_inference(cfg: Union[InferenceConfig, DictConfig], cruts_variables: List[str]) -> None:
    """
    Runs the inference on specified variables.

    Args:
        cfg (InferenceConfig): The cfg.
        cruts_variables (List[str]): The CRU-TS variables.

    """

    for var in cruts_variables:
        out_path = os.path.join(cfg.inference_out_path, var)
        os.makedirs(out_path, exist_ok=True)

        if var in [consts.cruts.tmn, consts.cruts.tmx] and cfg.temp_only:
            logging.info(f"TEMP_ONLY detected - 'temp' model will be used instead of '{var}' model.")
            model_file = cfg[f"pretrained_model_{consts.cruts.tmp}"]
        else:
            model_file = cfg[f"pretrained_model_{var}"]

        cfg.pretrained_model = model_file
        net = TaskSuperResolutionModule.load_from_checkpoint(model_file)
        net.eval()

        logging.info(f"Running inference for variable: {var}")
        logging.info(f"Running inference with model: {model_file}")

        min_max_lookup = pd.read_csv(cfg.min_max_lookup)
        min_max_lookup = min_max_lookup[
            (min_max_lookup[consts.datasets_and_preprocessing.dataset] == "cru-ts")
            & (min_max_lookup[consts.datasets_and_preprocessing.variable] == var)
        ]

        # prepare dataset
        dataset = (
            CRUTSInferenceDataset(
                ds_path=cfg.ds_path,
                elevation_file=cfg.elevation_file,
                land_mask_file=cfg.land_mask_file,
                generator_type=cfg.generator_type,
                scaling_factor=4,
                normalize=cfg.normalize,
                standardize=not cfg.normalize,
                normalize_range=cfg.normalization_range,
            )
            if cfg.use_netcdf_datasets
            else GeoTiffInferenceDataset(
                tiff_dir=os.path.join(cfg.extent_out_path_lr, var),
                tiff_df=min_max_lookup,
                variable=var,
                elevation_file=cfg.elevation_file,
                land_mask_file=cfg.land_mask_file,
                generator_type=cfg.generator_type,
                scaling_factor=4,
                normalize=cfg.normalize,
                standardize=not cfg.normalize,
                normalize_range=cfg.normalization_range,
                standardize_stats=None,
                use_elevation=cfg.use_elevation,
                use_mask_as_3rd_channel=cfg.use_mask_as_3rd_channel,
                use_global_min_max=cfg.use_global_min_max,
            )
        )

        with torch.no_grad():
            inference_on_full_images(
                model=net,
                ds=dataset,
                land_mask_file=cfg.land_mask_file,
                out_dir=out_path,
                normalization_range=cfg.normalization_range,
            )

        logging.info(f"Inference for variable {var} finished. Removing network.")

        del net


def transform_tiff_files_to_net_cdf(
    tiff_dir: str,
    nc_out_path: str,
    cruts_variables: str,
    prefix: Optional[str] = "inference",
) -> None:
    """
    Transforms generated Geo-Tiff files into Net-CDF datasets.

    Args:
        tiff_dir (str): The directory with tiff files to convert.
        nc_out_path (str): The out directory where to place the Net-CDF Datasets.
        cruts_variables (str): The name of the CRU-TS variable.
        prefix (Optional[str]): The optional prefix of the saved file. 'inference' by default.

    """

    os.makedirs(nc_out_path, exist_ok=True)

    for var in cruts_variables:
        logging.info(f"Building NET-CDF Dataset from Geo-TIFF files for variable {var}")
        fps = sorted(glob(os.path.join(tiff_dir, var, "*.tif")))
        timestamps = []
        lat = None
        lon = None
        arrs = []

        logging.info("Loading data arrays...")
        for fp in tqdm(fps):
            filename = os.path.basename(fp)

            splitted = filename.replace(".tif", "").split("-")
            timestamp = "-".join(splitted[-3:])
            timestamps.append(timestamp)

            da = xr.open_rasterio(fp).rename(var)
            if lat is None:
                lat = da.y.data
            if lon is None:
                lon = da.x.data
            arr = da.data
            arrs.append(arr)

        logging.info(f"Concatenating {len(arrs)} arrays together...")
        var_data = np.concatenate(arrs, axis=0)
        time = pd.to_datetime(timestamps)
        ds = xr.Dataset(
            {
                var: (("time", "lat", "lon"), var_data),
            },
            {"time": time, "lon": lon, "lat": lat},
            {
                "Conventions": "CF-1.4",
                "title": f"CRU TS4.04 {consts.datasets_and_preprocessing.var_to_variable[var]}",
                "source": "Neural-Downscaling approach.",
                "extent": "Europe. Based on ETRS89.",
            },
        )

        logging.info("Saving the dataset...")
        ds.to_netcdf(
            os.path.join(
                nc_out_path,
                f"{prefix}.cru_ts4.04.nn.inference.1901.2019.{var}.dat.nc",
            )
        )
        logging.info(f"Done for {var}")
