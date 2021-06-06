# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
import xarray as xr
from rasterio.mask import mask
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr.data.datasets import CRUTSInferenceDataset
from sr.data.normalization import MinMaxScaler
from sr.lightning_modules.utils import prepare_pl_module
from sr.pre_processing.cruts_config import CRUTSConfig


europe_bbox_lr = ((-16, 84), (40.25, 33))
europe_bbox_hr = ((-16, 84.25), (40.25, 32.875))
left_upper_lr = [-16, 84.375]
left_lower_lr = [-16, 33.125]
right_upper_lr = [40.25, 84.375]
right_lower_lr = [40.25, 33.125]

left_upper_hr = [-16, 84.375]
left_lower_hr = [-16, 33]
right_upper_hr = [40.25, 84.375]
right_lower_hr = [40.25, 33]

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


def parse_args() -> argparse.Namespace:
    """
    Parses the arguments.

    Returns (argparse.Namespace): The `argparse.Namespace`.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        f"--ds_path_{CRUTSConfig.tmn}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmn.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{CRUTSConfig.tmp}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmp.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{CRUTSConfig.tmx}",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmx.dat.nc",
    )
    parser.add_argument(
        f"--ds_path_{CRUTSConfig.pre}",
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
        f"--pretrained_model_{CRUTSConfig.tmn}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-tmin-4x-epoch=29-step=82709-hp_metric=0.00165.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-tmin-4x-epoch=29-step=82709-hp_metric=0.00571.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-tmin-4x-epoch=29-step=20699-hp_metric=0.00064.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{CRUTSConfig.tmp}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-temp-4x-epoch=29-step=165419-hp_metric=0.00083.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-temp-4x-epoch=24-step=137849-hp_metric=0.00516.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-temp-4x-epoch=29-step=41369-hp_metric=0.00056.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{CRUTSConfig.tmx}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-tmax-4x-epoch=29-step=82709-hp_metric=0.00142.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-tmax-4x-epoch=18-step=52382-hp_metric=0.00468.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=256/normalize-v2/gen-pre-training-srcnn-tmax-4x-epoch=29-step=20699-hp_metric=0.00059.ckpt",  # noqa E501
    )
    parser.add_argument(
        f"--pretrained_model_{CRUTSConfig.pre}",
        type=str,
        # default="./model_weights/with_elevation/gen-pre-training-srcnn-prec-4x-epoch=29-step=82709-hp_metric=0.00007.ckpt",  # noqa E501
        # default="./model_weights/no_elevation/gen-pre-training-srcnn-prec-4x-epoch=21-step=60653-hp_metric=0.00017.ckpt",  # noqa E501
        default="./model_weights/use_elevation=True-batch_size=256/normalize/gen-pre-training-srcnn-prec-4x-epoch=29-step=20699-hp_metric=0.00005.ckpt",  # noqa E501
    )
    parser.add_argument("--experiment_name", type=str, default="inference")
    parser.add_argument("--temp_only", type=bool, default=False)
    parser.add_argument("--use_elevation", type=bool, default=True)
    parser.add_argument("--run_inference", type=bool, default=True)
    parser.add_argument("--extract_polygon_extent", type=bool, default=True)
    parser.add_argument("--with_lr_extent", type=bool, default=False)
    parser.add_argument("--to_netcdf", type=bool, default=True)
    parser.add_argument("--cruts_variable", type=str, default=CRUTSConfig.tmn)
    # parser.add_argument(
    #     "--cruts_variable", type=str, default=CRUTSConfig.tmp
    # )  # single variable
    parser.add_argument("--scaling_factor", type=int, default=4)
    parser.add_argument(
        "--normalization_range", type=Tuple[float, float], default=(-1.0, 1.0)
    )
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)

    return parser.parse_args()


def inference_on_full_images(
    model: pl.LightningModule,
    ds_path: str,
    elevation_file: str,
    land_mask_file: str,
    out_dir: str,
    use_elevation: bool = False,
    normalization_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
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
        normalize_range=normalization_range,
    )

    # prepare dataloader
    dl = DataLoader(dataset=ds, batch_size=1, pin_memory=True, num_workers=1)

    scaler = MinMaxScaler(feature_range=normalization_range)

    # run inference
    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        lr = batch["lr"].cuda()
        elev = batch["elevation"].cuda()
        min = batch["min"].numpy()
        max = batch["max"].numpy()
        filename = batch["filename"]

        x = torch.cat([lr, elev], dim=1) if use_elevation else lr
        outputs = model(x)
        outputs = outputs.cpu().numpy()

        for idx, output in enumerate(outputs):
            arr = output.squeeze(0)
            arr = scaler.denormalize(arr, min[idx], max[idx]).clip(min[idx], max[idx])
            arr[mask] = np.nan

            with rio.open(
                os.path.join(out_dir, filename[idx]), "w", **profile
            ) as dataset:
                dataset.write(arr, 1)


def extract_extent(src_dir, extent_out_path, cruts_variables, polygon):
    bbox = [
        {
            "coordinates": polygon,
            "type": "Polygon",
        }
    ]

    for var in cruts_variables:
        logging.info(f"Extracting Europe polygon extents for variable '{var}'")
        files = sorted(glob(os.path.join(src_dir, var, "*.tif")))
        os.makedirs(os.path.join(extent_out_path, var), exist_ok=True)
        for fp in tqdm(files):
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
                os.path.join(extent_out_path, var, filename), "w", **meta
            ) as dest:
                dest.write(crop)

        logging.info(f"Done for variable '{var}'")


def run_inference(arguments, cruts_variables):
    for var in cruts_variables:
        out_path = os.path.join(arguments.inference_out_path, var)
        os.makedirs(out_path, exist_ok=True)

        if var in [CRUTSConfig.tmn, CRUTSConfig.tmx] and arguments.temp_only:
            logging.info(
                f"TEMP_ONLY detected - 'temp' model will be used instead of '{var}' model."
            )
            model_file = param_dict[f"pretrained_model_{CRUTSConfig.tmp}"]
        else:
            model_file = param_dict[f"pretrained_model_{var}"]

        arguments.pretrained_model = model_file
        net = prepare_pl_module(arguments)
        net.eval()

        logging.info(f"Running inference for variable: {var}")
        logging.info(f"Running inference with model: {model_file}")

        with torch.no_grad():
            inference_on_full_images(
                model=net,
                ds_path=param_dict[f"ds_path_{var}"],
                land_mask_file=arguments.land_mask_file,
                elevation_file=arguments.elevation_file,
                out_dir=out_path,
                use_elevation=arguments.use_elevation,
                normalization_range=arguments.normalization_range,
            )

        logging.info(f"Inference for variable {var} finished. Removing network.")

        del net


def transform_tiff_files_to_net_cdf(tiff_dir, nc_out_path, cruts_variables):
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
            os.path.join(nc_out_path, f"cru_ts4.04.nn.inference.1901.2019.{var}.dat.nc")
        )
        logging.info(f"Done for {var}")


if __name__ == "__main__":
    args = parse_args()
    param_dict = vars(args)
    logging.info("Running with following config: ")
    for k, v in param_dict.items():
        logging.info(f"Param: '{k}': {v}")

    variables = (
        [args.cruts_variable] if args.cruts_variable else CRUTSConfig.variables_cts
    )

    # Run inference
    if args.run_inference:
        logging.info("Running inference")
        run_inference(args, variables)

    # Run Europe extent extraction
    if args.extract_polygon_extent:
        logging.info("Extracting SR polygon extents for Europe.")
        extract_extent(
            args.inference_out_path, args.extent_out_path_sr, variables, hr_polygon
        )

        if args.with_lr_extent:
            logging.info("Extracting LR polygon extents for Europe.")
            extract_extent(
                args.original_full_res_cruts_data_path,
                args.extent_out_path_lr,
                variables,
                lr_polygon,
            )

    # Run tiff file transformation to net-cdf datasets.
    if args.to_netcdf:
        logging.info("Building NET CDF datasets from extent tiff files.")
        transform_tiff_files_to_net_cdf(
            args.extent_out_path_sr, args.extent_out_path_sr_nc, variables
        )

    logging.info("Done")
