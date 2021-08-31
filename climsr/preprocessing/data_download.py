# -*- coding: utf-8 -*-
import gzip
import logging
import os
import shutil
import traceback
import zipfile
from glob import glob
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
from requests import Response
from sklearn.utils.extmath import cartesian
from tqdm import tqdm

import climsr.consts as consts


def download_file(url: str, download_dir: Optional[str] = "../../datasets/download") -> Tuple[Union[str, None], Union[str, None]]:
    os.makedirs(download_dir, exist_ok=True)
    fname = os.path.join(download_dir, url.split("/")[-1])

    if os.path.exists(fname):
        logging.info(f"File {fname} already exists. Skipping download...")
        return fname, None

    resp: Response = requests.get(url, stream=True)

    # This is done due to World Clim missing some of the files for some of the scenarios...
    # Handle 404 as a valid response and notify the calling function by the error reason
    # But raise exception on any other kind of error
    if resp.status_code == 404:
        return None, resp.reason
    else:
        resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)

    return fname, None


def get_cruts_data_download_urls() -> List[str]:
    return [
        f"https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.05/cruts.2103051243.v4.05/{var}/cru_ts4.05.1901.2020.{var}.dat.nc.gz"
        for var in consts.cruts.temperature_vars
    ]


def get_world_clim_historical_climate_data_download_urls() -> List[str]:
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tavg, consts.world_clim.tmax, consts.world_clim.elev]).astype(
        "<U12"
    )
    resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")

    product = cartesian([variables, resolutions])

    for var, res in product:
        world_clim_download_urls.append(f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_{res}_{var}.zip")

    return world_clim_download_urls


def get_world_clim_historical_weather_data_download_urls() -> List[str]:
    step = 10
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tmax]).astype("<U12")
    lowers = np.array([lower for lower in range(1960, 2019, step)])

    product = cartesian([variables, lowers])

    for var, lower in product:
        upper = int(lower) + step - 1

        if upper == 2019:
            upper = 2018

        world_clim_download_urls.append(
            f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/hist/wc2.1_2.5m_{var}_{lower}-{upper}.zip"
        )

    return world_clim_download_urls


def get_world_clim_future_climate_data_download_urls() -> List[str]:
    step = 20
    world_clim_download_urls = []

    # To `ndarray` and as U12 since `cartesian` tends to assign U4 as dtype.
    variables = np.array([consts.world_clim.tmin, consts.world_clim.tmax]).astype("<U12")
    resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")
    gcms = np.array(consts.world_clim.GCMs).astype("<U12")
    scenarios = np.array(consts.world_clim.scenarios).astype("<U12")
    lowers = np.array([lower for lower in range(2021, 2100, step)])

    product = cartesian([variables, resolutions, gcms, scenarios, lowers])
    for var, res, gcm, scenario, lower in product:
        upper = int(lower) + step - 1
        world_clim_download_urls.append(
            "https://biogeo.ucdavis.edu/data/worldclim/v2.1/fut/" f"{res}/wc2.1_{res}_{var}_{gcm}_{scenario}_{lower}-{upper}.zip"
        )

    return world_clim_download_urls


def handle_file_download(
    cru_ts_download_urls: List[str],
    world_clim_download_urls: List[str],
    download_path: Optional[str] = "../../datasets/download",
) -> None:
    cruts_download_path = os.path.join(
        download_path, consts.datasets_and_preprocessing.cruts_download_dir, consts.datasets_and_preprocessing.archives
    )
    world_clim_download_path = os.path.join(
        download_path, consts.datasets_and_preprocessing.world_clim_download_dir, consts.datasets_and_preprocessing.archives
    )

    os.makedirs(cruts_download_path, exist_ok=True)
    os.makedirs(world_clim_download_path, exist_ok=True)

    paths = [cruts_download_path for _ in cru_ts_download_urls]
    paths.extend([world_clim_download_path for _ in world_clim_download_urls])

    replace_underscore_flags = [False for _ in cru_ts_download_urls]
    replace_underscore_flags.extend([True for _ in world_clim_download_urls])

    all_urls = []
    all_urls.extend(cru_ts_download_urls)
    all_urls.extend(world_clim_download_urls)

    for idx, (url, download_path, replace_underscore_flag) in enumerate(zip(all_urls, paths, replace_underscore_flags)):
        logging.info(f"PROGRESS: {idx + 1}/{len(all_urls)}")
        try_file_download_and_extraction(url, download_path, replace_underscore_flag)

    fix_paths_for_world_clim(world_clim_download_path)


def try_file_download_and_extraction(url: str, download_path: str, replace_underscore_flag: Optional[bool] = False) -> None:
    retry = 0
    MAX_RETRY_COUNT = 3
    while True and retry < MAX_RETRY_COUNT:
        if retry > 0:
            logging.warning(
                f"Attempting to re-download the file {url} due to issues with the file integrity. Attempt #{retry + 1}"
            )

        f_name, error = download_file(url, download_path)

        # break on failed download
        if f_name is None:
            logging.info(f"File {url} could not be downloaded because of the following error: {error}")
            break

        try:
            handle_file_extraction(f_name, replace_underscore_flag)
            break
        except Exception as ex:
            logging.error(f"File: {url} could not be extracted due to following error: {ex}.")
            logging.error(f"{traceback.format_exc()}")
            os.remove(f_name)

        retry = retry + 1

    if retry == 3:
        logging.error(f"Maximum number of retries for file {url} has been reached. Re-download the file manually")


def gunzip(source_filepath: str, dest_filepath: str, block_size: Optional[int] = 65536) -> None:
    with gzip.open(source_filepath, "rb") as s_file, open(dest_filepath, "wb") as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def unzip(source_filepath: str, dest_filepath: str) -> None:
    os.makedirs(dest_filepath, exist_ok=True)
    with zipfile.ZipFile(source_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_filepath)


def handle_file_extraction(f_name: str, replace_underscore: Optional[bool] = False) -> None:
    logging.info(f"Extracting {f_name}")
    extraction_path = os.path.splitext(f_name)[0].replace(
        consts.datasets_and_preprocessing.archives, consts.datasets_and_preprocessing.extracted
    )

    if replace_underscore:
        extraction_path = extraction_path.replace("_", os.sep)

    if os.path.exists(extraction_path):
        logging.info(f"File {f_name} was already extracted... Skipping extraction...")
        return

    try:
        if str(f_name).endswith(".zip"):
            unzip(f_name, extraction_path)
        elif str(f_name).endswith(".gz"):
            gunzip(f_name, extraction_path)
        else:
            raise ValueError(f"{f_name} compression type is unsupported! Supported: ZIP, GZ")
    except Exception:
        if os.path.isfile(extraction_path):
            os.remove(extraction_path)
        else:
            os.rmdir(extraction_path)
        raise


def fix_paths_for_world_clim(world_clim_download_path: str = "downloads/world-clim") -> None:
    world_clim_extraction_path = os.path.join(world_clim_download_path, consts.datasets_and_preprocessing.extracted, "wc2.1")
    logging.info(f"Fixing folder structure for files in World Clim extraction directory: {world_clim_extraction_path}")
    pattern = os.path.join(world_clim_extraction_path, "**/*.tif")
    files = glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} files in total. Processing...")

    def build_lookup() -> List[str]:
        resolutions = np.array(consts.world_clim.data_resolutions).astype("<U12")
        gcms = np.array(consts.world_clim.GCMs).astype("<U12")
        scenarios = np.array(consts.world_clim.scenarios).astype("<U12")
        prod = cartesian([resolutions, gcms, scenarios])
        lookup_strings: List[str] = []
        for res, gcm, scenario in prod:
            lookup_strings.append(f"share/spatial03/worldclim/cmip6/7_fut/{res}/{gcm}/{scenario}/")
        return lookup_strings

    def move_if_match_in_lookup(fp: str, lookup: List[str]) -> None:
        for lookup_str in lookup:
            if lookup_str not in fp:
                continue
            # match found, move file to new destination
            destination = fp.replace(lookup_str, "")
            shutil.move(fp, destination)
            break

    str_lookup_array = build_lookup()

    for file_path in tqdm(files):
        move_if_match_in_lookup(file_path, str_lookup_array)

    logging.info("Cleaning up after file movement... Removing empty directories...")
    for directory in tqdm(glob(os.path.join(world_clim_extraction_path, "**/share"), recursive=True)):
        shutil.rmtree(directory)
