# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
import xarray as xr
from data.sr.cruts_inference_dataset import CRUTSInferenceDataset
from data.sr.geo_tiff_inference_dataset import GeoTiffInferenceDataset
from matplotlib import pyplot as plt
from pre_processing.preprocessing import extract_extent, hr_bbox, var_to_variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import climsr.consts as consts
from climsr.core.utils import prepare_pl_module
from climsr.data.normalization import MinMaxScaler


def parse_args(arg_str: Optional[str] = None) -> argparse.Namespace:
    """
    Parses the arguments.

    Returns (argparse.Namespace): The `argparse.Namespace`.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        f"--ds_path_{consts.cruts.tmn}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmn.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{consts.cruts.tmp}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmp.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{consts.cruts.tmx}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmx.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{consts.cruts.pre}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.pre.dat.nc",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/full-res",
    )
    parser.add_argument(
        "--tiff_dir",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent",
    )
    parser.add_argument(
        "--elevation_file",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent/elevation/wc2.1_2.5m_elev.tif",
    )
    parser.add_argument(
        "--land_mask_file",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent/mask/wc2.1_2.5m_prec_1961-01.tif",
    )
    parser.add_argument(
        "--inference_out_path",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/inference",
    )
    parser.add_argument(
        "--original_full_res_cruts_data_path",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/full-res",
    )
    parser.add_argument(
        "--extent_out_path_lr",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/pre-processed/europe-extent",
    )
    parser.add_argument(
        "--extent_out_path_sr",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/inference-europe-extent",
    )
    parser.add_argument(
        "--extent_out_path_sr_nc",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/inference-europe-extent-nc",
    )
    parser.add_argument(
        "--min_max_lookup",
        type=str,
        default="./datasets/statistics_min_max.csv",
    )
    parser.add_argument(
        f"--pretrained_model_{consts.cruts.tmn}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-tmin-4x-epoch=29-step=82709-hp_metric=0.00165.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-tmin-4x-epoch=29-step=82709-hp_metric=0.00571.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-tmin-4x-epoch=29-step=20699-hp_metric=0.00064.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-rcan-tmin-4x-epoch=29-step=110279-hp_metric=0.00397.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{consts.cruts.tmp}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-temp-4x-epoch=29-step=165419-hp_metric=0.00083.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-temp-4x-epoch=24-step=137849-hp_metric=0.00516.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-temp-4x-epoch=29-step=41369-hp_metric=0.00056.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-rcan-temp-4x-epoch=29-step=220559-hp_metric=0.00317.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-rcan-tmax-4x-epoch=29-step=110279-hp_metric=0.00417.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=64/gen-pre-training-esrgan-temp-4x-epoch=29-step=165419-hp_metric=0.00608.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=64/gan-training-esrgan-temp-4x-epoch=18-step=104765-hp_metric=0.50164.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-100epoch--rcan-temp-4x-epoch=99-step=155599-hp_metric=0.00261.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{consts.cruts.tmx}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-tmax-4x-epoch=29-step=82709-hp_metric=0.00142.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-tmax-4x-epoch=18-step=52382-hp_metric=0.00468.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-tmax-4x-epoch=29-step=20699-hp_metric=0.00059.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-rcan-tmax-4x-epoch=29-step=110279-hp_metric=0.00417.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{consts.cruts.pre}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-prec-4x-epoch=29-step=82709-hp_metric=0.00007.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-prec-4x-epoch=21-step=60653-hp_metric=0.00017.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=256/normalize/gen-pre-training-srcnn-prec-4x-epoch=29-step=20699-hp_metric=0.00005.ckpt",  # noqa E501
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=consts.training.experiment_name_gan_training,
    )
    parser.add_argument("--use_netcdf_datasets", type=bool, default=False)
    parser.add_argument("--temp_only", type=bool, default=True)
    parser.add_argument("--use_elevation", type=bool, default=True)
    parser.add_argument("--use_mask_as_3rd_channel", type=bool, default=True)
    parser.add_argument("--use_global_min_max", type=bool, default=True)
    parser.add_argument("--run_inference", type=bool, default=True)
    parser.add_argument("--extract_polygon_extent", type=bool, default=True)
    parser.add_argument("--to_netcdf", type=bool, default=True)
    parser.add_argument("--cruts_variable", type=str, default=consts.cruts.tmp)
    parser.add_argument("--generator_type", type=str, default=consts.models.rcan)
    parser.add_argument("--scaling_factor", type=int, default=4)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--normalization_range", type=Tuple[float, float], default=(-1.0, 1.0))
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)

    return parser.parse_args(arg_str)


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


def run_inference(arguments: argparse.Namespace, cruts_variables: List[str]) -> None:
    """
    Runs the inference on specified variables.

    Args:
        arguments (argparse.Namespace): The arguments.
        cruts_variables (List[str]): The CRU-TS variables.

    """

    for var in cruts_variables:
        out_path = os.path.join(arguments.inference_out_path, var)
        os.makedirs(out_path, exist_ok=True)

        if var in [consts.cruts.tmn, consts.cruts.tmx] and arguments.temp_only:
            logging.info(f"TEMP_ONLY detected - 'temp' model will be used instead of '{var}' model.")
            model_file = param_dict[f"pretrained_model_{consts.cruts.tmp}"]
        else:
            model_file = param_dict[f"pretrained_model_{var}"]

        arguments.pretrained_model = model_file
        net = prepare_pl_module(arguments)
        net.eval()

        logging.info(f"Running inference for variable: {var}")
        logging.info(f"Running inference with model: {model_file}")

        min_max_lookup = pd.read_csv(arguments.min_max_lookup)
        min_max_lookup = min_max_lookup[
            (min_max_lookup[consts.datasets_and_preprocessing.dataset] == "cru-ts")
            & (min_max_lookup[consts.datasets_and_preprocessing.variable] == var)
        ]

        # prepare dataset
        dataset = (
            CRUTSInferenceDataset(
                ds_path=arguments.ds_path,
                elevation_file=arguments.elevation_file,
                land_mask_file=arguments.land_mask_file,
                generator_type=arguments.generator_type,
                scaling_factor=4,
                normalize=arguments.normalize,
                standardize=not arguments.normalize,
                normalize_range=arguments.normalization_range,
            )
            if args.use_netcdf_datasets
            else GeoTiffInferenceDataset(
                tiff_dir=os.path.join(arguments.extent_out_path_lr, var),
                tiff_df=min_max_lookup,
                variable=var,
                elevation_file=arguments.elevation_file,
                land_mask_file=arguments.land_mask_file,
                generator_type=arguments.generator_type,
                scaling_factor=4,
                normalize=arguments.normalize,
                standardize=not arguments.normalize,
                normalize_range=arguments.normalization_range,
                standardize_stats=None,
                use_elevation=arguments.use_elevation,
                use_mask_as_3rd_channel=arguments.use_mask_as_3rd_channel,
                use_global_min_max=arguments.use_global_min_max,
            )
        )

        with torch.no_grad():
            inference_on_full_images(
                model=net,
                ds=dataset,
                land_mask_file=arguments.land_mask_file,
                out_dir=out_path,
                normalization_range=arguments.normalization_range,
            )

        logging.info(f"Inference for variable {var} finished. Removing network.")

        del net


def transform_tiff_files_to_net_cdf(tiff_dir: str, nc_out_path: str, cruts_variables: str, generator: str) -> None:
    """
    Transforms generated Geo-Tiff files into Net-CDF datasets.

    Args:
        tiff_dir (str): The directory with tiff files to convert.
        nc_out_path (str): The out directory where to place the Net-CDF Datasets.
        cruts_variables (str): The name of the CRU-TS variable.
        generator (str): The name of the generator network that was used.

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
                "title": f"CRU TS4.04 {var_to_variable[var]}",
                "source": "Neural-Downscaling approach.",
                "extent": "Europe. Based on ETRS89.",
            },
        )

        logging.info("Saving the dataset...")
        ds.to_netcdf(
            os.path.join(
                nc_out_path,
                f"{generator}.cru_ts4.04.nn.inference.1901.2019.{var}.dat.nc",
            )
        )
        logging.info(f"Done for {var}")


if __name__ == "__main__":
    args = parse_args()
    param_dict = vars(args)
    logging.info("Running with following config: ")
    for k, v in param_dict.items():
        logging.info(f"Param: '{k}': {v}")

    variables = [args.cruts_variable] if args.cruts_variable else consts.cruts.variables_cts

    # Run inference
    if args.run_inference:
        logging.info("Running inference")
        run_inference(args, variables)

    # Run Europe extent extraction
    if args.extract_polygon_extent:
        logging.info("Extracting SR polygon extents for Europe.")
        extract_extent(args.inference_out_path, args.extent_out_path_sr, variables, hr_bbox)

    # Run tiff file transformation to net-cdf datasets.
    if args.to_netcdf:
        logging.info("Building NET CDF datasets from extent tiff files.")
        transform_tiff_files_to_net_cdf(
            args.extent_out_path_sr,
            args.extent_out_path_sr_nc,
            variables,
            args.generator_type,
        )

    logging.info("Done")
