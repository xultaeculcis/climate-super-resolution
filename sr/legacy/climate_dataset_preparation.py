# -*- coding: utf-8 -*-
import logging
import os
from glob import glob
from typing import Dict, List, Optional, Set, Tuple

import dask
import dask.bag
import numpy as np
from dask.distributed import Client
from misc.clim_scaler import ClimScaler
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def make_patches(
    X: np.ndarray,
    idx: str,
    target_path: str,
    stage: Optional[str] = None,
    hr_image_size: Optional[int] = 128,
    lr_image_size: Optional[int] = 32,
    stride_hr: Optional[int] = 4,
    stride_lr: Optional[int] = 1,
    scaling_factor: Optional[int] = 4,
    indices_to_skip: Optional[Set[str]] = None,
) -> int:
    """
    Makes patches of specified size out of the source image.

    Args:
        X (np.ndarray): The image as a numpy array.
        idx (str): The index or name of the image.
        target_path (str): The target directory where to save the patches.
        stage (Optional[str]): The name of the stage, one of: "train", "val", "test". Optional, default: None.
        hr_image_size (Optional[int]): The size of the HR image. Optional, default: 128.
        lr_image_size (Optional[int]): The size of the LR image. Optional, default: 32.
        stride_hr (Optional[int]): The stride to use for the HR image. Optional, default: 4.
        stride_lr (Optional[int]): The stride to use for the LR image. Optional, default: 1.
        scaling_factor (Optional[int]): The scaling factor. Optional, default: 4.

    Returns (int): The last index of generated image patches.

    """

    def generate(image: np.ndarray, image_size: int, stride: int, is_lr: bool) -> int:
        h, w = image.shape
        num_row = h // stride
        num_col = w // stride
        image_index = 0

        for i in range(num_row):
            if (i + 1) * image_size > h:
                break
            for j in range(num_col):
                if (j + 1) * image_size > w:
                    break

                if (
                    indices_to_skip is not None
                    and str(image_index).rjust(5, "0") in indices_to_skip
                ):
                    image_index = image_index + 1
                    continue

                image_to_save = image[
                    i * image_size : (i + 1) * image_size,
                    j * image_size : (j + 1) * image_size,
                ]

                path_to_img = (
                    os.path.join(
                        target_path,
                        stage,
                        "lr" if is_lr else "hr",
                        f'{idx}_{str(image_index).rjust(5, "0")}.tiff',
                    )
                    if stage
                    else os.path.join(
                        target_path,
                        "lr" if is_lr else "hr",
                        f'{idx}_{str(image_index).rjust(5, "0")}.tiff',
                    )
                )

                Image.fromarray(image_to_save).save(path_to_img)

                image_index = image_index + 1
        return image_index

    # HR
    max_index = generate(X, hr_image_size, stride_hr, False)
    if (
        not stage
    ):  # only hr images if no stage was provided - this handles elevation layer
        return max_index

    # LR
    width = int(X.shape[1] / scaling_factor)
    height = int(X.shape[0] / scaling_factor)
    img = Image.fromarray(X)
    img = img.resize_raster((width, height), resample=Image.BICUBIC)
    img = np.array(img)
    return generate(img, lr_image_size, stride_lr, True)


def get_files(
    data_dir: str, variable: str, resolution: str, extension: Optional[str] = ".tif"
) -> List[str]:
    """
    Gets the files of specified types.

    Args:
        data_dir (str): The source data directory.
        variable (str): The list of variables for which to get the files.
            One of: "prec", "tmin", "tmax", "elevation".
        resolution (List[str]): The list of resolutions for which to get the files. One of: "30s", "2.5m", "5m", "10m".
        extension (Optional[str]): The file extension. Optional, default: ".tif".

    Returns (List[str]): A list of the `file paths`.

    """

    pattern = os.path.join(data_dir, variable, "**", f"*_{resolution}*{extension}")
    return sorted(glob(pattern, recursive=True))


def prepara_data(
    fpath: str,
    output_dir: str,
    stage: Optional[str] = None,
    indices_to_skip: Optional[Set[str]] = None,
) -> int:
    """
    Normalizes values in the image, then prepares HR & LR image patches from specified file and saves them
    in the specified `output_dir`.

    Args:
        fpath (str): The path to the file.
        output_dir (str): The output dir.
        stage (Optional[str]): The name of the stage, one of: "train", "val", "test". Optional, default: None.
        indices_to_skip (Optional[Set[str]]): An optional set of indices to skip when generating image patches.

    Returns (int): The last index of generated image patches.

    """
    scaler = ClimScaler()
    arr = np.array(Image.open(fpath))
    scaled = scaler.fit_transform_single(arr)
    return make_patches(
        scaled,
        os.path.basename(os.path.splitext(fpath)[0]),
        output_dir,
        stage,
        indices_to_skip=indices_to_skip,
    )


def stage_from_year(year: int, stage_years: Dict[str, Tuple[int, int]]) -> str:
    """
    Helper function. Maps year to correct stage using supplied config dictionary.
    Args:
        year (int): The year.
        stage_years (Dict[str, Tuple[int, int]]): Mapping dictionary.

    Returns (str): The name of the stage.

    """
    for key, value in stage_years.items():
        lower, upper = value
        if lower <= year <= upper:
            return key

    raise ValueError("Cannot map year to stage based on provided mapping")


def remove_based_on_index_set(fpath: str, indices_to_remove: Set[str]) -> None:
    """
    Removes specified file if index in it's name indicates, that it should be removed.

    Args:
        fpath (str): The file path.
        indices_to_remove (Set[str]): A set of indices to remove.

    """
    idx = os.path.basename(os.path.splitext(fpath)[0]).split("_")[-1]
    if idx in indices_to_remove:
        os.remove(fpath)


def should_be_removed(fpath) -> Optional[str]:
    arr = np.array(Image.open(fpath))
    if np.count_nonzero(arr) / np.prod(arr.shape) < 0.3:
        idx = os.path.basename(os.path.splitext(fpath)[0]).split("_")[-1]
        return idx
    return None


def find_indices_to_skip(
    var: str,
    resolution: str,
    out_dir: str,
    stage_years_mapping: Dict[str, Tuple[int, int]],
) -> Set[str]:
    """
    Finds which image patch indices to skip based on a single file from the variable.

    Args:
        var (str): The variable name.
        resolution (str): The resolution.
        out_dir (str): The out dir.
        stage_years_mapping (Dict[str, Tuple[int, int]]): The mapping.

    Returns (Set[str]): A set of image indices to skip during image patch generation.

    """
    indices_to_skip = set()
    logging.info(
        f"Running pre-processing for a single '{var}' file just to find a set of image patch indices to skip."
    )
    f_paths = get_files(os.path.join(root_data_directory, "weather"), var, resolution)
    for fp in f_paths[:1]:
        file_name = os.path.basename(os.path.splitext(fp)[0])
        year = int(file_name.split("-")[0].split("_")[-1])
        stage = stage_from_year(year, stage_years_mapping)
        prepara_data(fp, os.path.join(out_dir, var, resolution), stage, None)

        logging.info(
            "Checking which files do not meet the min. of 30% non zero values."
        )
        single_file = os.path.basename(os.path.splitext(fp)[0])
        files = glob(
            os.path.join(
                out_dir,
                var,
                resolution,
                stage,
                "hr",
                f"*{single_file}*.tiff",
            )
        )
        logging.info(f"Found {len(files)} files to check. Scheduling checks using DASK")
        results = (
            dask.bag.from_sequence(files, npartitions=1000)
            .map(should_be_removed)
            .compute()
        )
        logging.info("Generating a set of indices to skip from Dask results.")
        for result in tqdm(results):
            if result:
                indices_to_skip.add(result)

        logging.info(f"Found {len(indices_to_skip)} file indices to skip")
        return indices_to_skip


def cleanup(out_dir: str) -> None:
    """
    Performs a cleanup operation on specified target directory.

    Args:
        out_dir (str): The target directory to remove all ".tiff" files from.

    Returns:

    """

    logging.info(
        f"Cleaning up all previously generated files under: {output_directory}"
    )
    files = glob(os.path.join(out_dir, "**", "*.tiff"), recursive=True)
    logging.info(f"Found {len(files)} to remove. Scheduling removal using DASK.")
    dask.bag.from_sequence(files, npartitions=1000).map(os.remove).compute()


def schedule_image_patch_generation(
    client: Client,
    variables: List[str],
    resolution: str,
    stage_years_mapping: Dict[str, Tuple[int, int]],
    data_dir: str,
    out_dir: str,
    indices_to_skip: Optional[Set[str]] = None,
) -> None:
    """
    Schedules image patch generation to run on DASK using specified client connection.

    Args:
        client (Client): The client.
        variables (List[str]): List of variables.
        resolution (str): The dataset resolution.
        stage_years_mapping (Dict[str, Tuple[int, int]]): The stage to years mapping.
        data_dir (str): The root data directory.
        out_dir (str): The output directory.
        indices_to_skip (Optional[Set[str]]): Optional set of indices to skip during image patch generation.

    """

    futures = []
    for var in variables:
        logging.info(f"Running pre-processing for '{var}'")
        dask_params = []
        f_paths = get_files(os.path.join(data_dir, "weather"), var, resolution)
        logging.info(f"Found {len(f_paths)} files for '{var}'")
        logging.info(f"Scheduling '{var}' files to be processed using DASK")
        for fp in tqdm(f_paths):
            file_name = os.path.basename(os.path.splitext(fp)[0])
            year = int(file_name.split("-")[0].split("_")[-1])
            stage = stage_from_year(year, stage_years_mapping)
            dask_params.append((fp, stage))

            for item in dask_params:
                fp, stage = item
                futures.append(
                    client.submit(
                        prepara_data,
                        fp,
                        os.path.join(out_dir, var, resolution),
                        stage,
                        indices_to_skip,
                    )
                )

        logging.info(f"Done for '{var}'")
    client.gather(futures)


def ensure_target_dirs_exist(
    stages: List[str],
    variables: List[str],
    img_sub_folders: List[str],
    out_dir: str,
    resolution: str,
) -> None:
    """
    Ensures that output paths exist.

    Args:
        stages (List[str]): The training stages.
        variables (List[str]): The dataset variables.
        img_sub_folders (List[str]): The image sub-folders.
        out_dir (str): The output directory.
        resolution (str): The dataset resolution.

    """

    logging.info("Creating target directories")
    for stage in stages:
        for var in variables:
            for subfolder in img_sub_folders:
                os.makedirs(
                    os.path.join(out_dir, var, resolution, stage, subfolder),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(out_dir, "elevation", resolution, "hr"),
                    exist_ok=True,
                )


if __name__ == "__main__":
    root_data_directory = "/media/xultaeculcis/2TB/datasets/sr/fine-tuning/wc/"
    output_directory = os.path.join(root_data_directory, "pre-processed")
    train_stages = ["train", "val", "test"]
    image_sub_folders = ["hr", "lr"]
    dataset_variables = [
        "pre",
        "tmin",
        "tmax",
    ]
    dataset_resolution = "2.5m"
    n_workers = 8
    threads_per_worker = 1

    stage_years_mapping = {
        "train": (1961, 1995),
        "val": (1996, 2004),
        "test": (2005, 2019),
    }

    ensure_target_dirs_exist(
        train_stages,
        dataset_variables,
        image_sub_folders,
        output_directory,
        dataset_resolution,
    )

    indices_to_skip = find_indices_to_skip(
        dataset_variables[0], dataset_resolution, output_directory, stage_years_mapping
    )

    logging.info(
        f"{'='*10} - Dataset pre-processing and image patches generation begins - {'='*10}"
    )
    # Once the client is created, you'll be able to view the progress on: http://localhost:8787/status
    c = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    try:
        # cleanup first
        cleanup(output_directory)

        schedule_image_patch_generation(
            c,
            dataset_variables,
            dataset_resolution,
            stage_years_mapping,
            root_data_directory,
            output_directory,
            indices_to_skip,
        )

        # handle elevation layer separately
        var = "elevation"
        logging.info(f"Running pre-processing for '{var}'")
        elev_files = get_files(root_data_directory, var, dataset_resolution)
        for fp in tqdm(elev_files):  # there should be only one file per resolution
            prepara_data(
                fp,
                os.path.join(output_directory, var, dataset_resolution),
                indices_to_skip=indices_to_skip,
            )

    finally:
        c.close()
