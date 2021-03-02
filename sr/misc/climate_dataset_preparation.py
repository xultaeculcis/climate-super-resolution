# -*- coding: utf-8 -*-
import logging
import os
from glob import glob
from typing import List, Optional, Tuple, Dict

import dask
import dask.bag
import numpy as np
from dask.distributed import Client
from PIL import Image

logging.basicConfig(level=logging.INFO)


class WorldClimScaler:
    """
    The custom Min-Max scaler for the WorldClim dataset.

    There are a couple of ways you can use this class. Similarly to `scikit-learn`'s `MinMaxScaler`
    you can fit on a single `numpy` array and then do a transformation on other array.
    Or you can run `fit_transform`, which will do both.

    There are methods which work on a collection of files as well - will use `Dask` to distribute the workload.
    """

    def __init__(
        self,
    ):
        self.feature_range = (0.0, 1.0)
        self.lower_bound, self.upper_bound = self.feature_range
        self.nan_replacement = 0.0

    def fit(self, fpaths: List[str]) -> None:
        """
        Compute the minimum and maximum to be used for later scaling.

        Args:
            fpaths (List[str]): The list of paths to .tif files.

        """

        self.fpaths = fpaths

        def find_min_max(file_path: str) -> Tuple[float, float]:
            """
            Find min and max value in the single file.

            Args:
                file_path (str): A path to a single file.

            Returns (Tuple[float, float]): Tuple with min and max value.

            """
            img = Image.open(file_path)
            arr = np.array(img, dtype=np.float)
            arr[arr == -32768] = np.nan
            xmin = np.nanmin(arr)
            xmax = np.nanmax(arr)

            return xmin, xmax

        c = Client(n_workers=8, threads_per_worker=1)

        results = (
            dask.bag.from_sequence(self.fpaths, npartitions=len(self.fpaths))
            .map(find_min_max)
            .compute()
        )

        c.close()

        current_min = 9999
        current_max = -9999

        for xmin, xmax in results:
            if current_min > xmin:
                current_min = xmin
            if current_max < xmax:
                current_max = xmax

        self.org_data_min_ = current_min
        self.org_data_max_ = current_max
        self.data_min_ = current_min - 1
        self.data_max_ = current_max + 1
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.upper_bound - self.lower_bound) / self.data_range_
        self.min_ = self.lower_bound - self.data_min_ * self.scale_

    def fit_single(self, X: np.ndarray) -> None:
        """
        Compute the minimum and maximum to be used for later scaling.

        Args:
            X (np.ndarray): The numpy array with data

        """

        arr = X.astype(float)
        arr[arr == -32768] = np.nan
        xmin = np.nanmin(arr)
        xmax = np.nanmax(arr)

        self.org_data_min_ = xmin
        self.org_data_max_ = xmax
        self.data_min_ = xmin - 1
        self.data_max_ = xmax + 1
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.upper_bound - self.lower_bound) / self.data_range_
        self.min_ = self.lower_bound - self.data_min_ * self.scale_

    def transform_single(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X according to feature_range.

        Args:
            X (np.ndarray): The array.

        Returns (np.ndarray): The scaled array.

        """

        X = X.astype(float)
        X[X == -32768.0] = np.nan
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = X_std * (self.upper_bound - self.lower_bound) + self.lower_bound
        X_scaled[np.isnan(X_scaled)] = self.nan_replacement
        return X_scaled

    def fit_transform_single(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): The numpy array with data.

        Returns (np.ndarray): The transformed data.

        """
        self.fit_single(X)
        return self.transform_single(X)

    def transform(self, fpaths: List[str], out_dir: str) -> None:
        """
        Scale features of files according to feature_range.

        Args:
            fpaths (List[str]): The file paths.
            out_dir (str): The output directory.

        """
        os.makedirs(out_dir, exist_ok=True)

        def transform_(
            fpath: str,
            out_dir: str,
            min_val: float,
            max_val: float,
            lower: Optional[float] = 0.0,
            upper: Optional[float] = 1.0,
            nan_replacement: Optional[float] = 0.0,
        ) -> None:
            """
            Scale features of single file according to feature_range.

            Args:
                fpath (str): The file path.
                out_dir (str): The output directory.
                min_val (float): The min val.
                max_val (float): The max val.
                lower (Optional[float]): The lower bound of feature range.
                upper (Optional[float]): The upper bound of feature range.
                nan_replacement (Optional[float]): The value to use in order to replace NaNs.

            """

            im_name = os.path.basename(os.path.splitext(fpath)[0]) + ".tiff"
            if os.path.exists(im_name):
                return
            X = np.array(Image.open(fpath), dtype=np.float)
            X[X == -32768.0] = np.nan
            X_std = (X - min_val) / (max_val - min_val)
            X_scaled = X_std * (upper - lower) + lower
            X_scaled[np.isnan(X_scaled)] = nan_replacement

            im = Image.fromarray(X_scaled)
            im.save(os.path.join(out_dir, im_name))

        c = Client(n_workers=8, threads_per_worker=1)

        _ = (
            dask.bag.from_sequence(fpaths, npartitions=len(fpaths))
            .map(
                transform_,
                out_dir=out_dir,
                min_val=self.data_min_,
                max_val=self.data_max_,
                lower=self.lower_bound,
                upper=self.upper_bound,
                nan_replacement=self.nan_replacement,
            )
            .compute()
        )

        c.close()

    def fit_transform(self, fpaths: List[str], out_dir: str) -> None:
        """
        Fit to data, then transform it.

        Args:
            fpaths (List[str]): The file paths.
            out_dir (str): The output directory.

        """
        self.fit(fpaths)
        self.transform(fpaths, out_dir)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling of X according to feature_range.

        Args:
            X (np.ndarray): The numpy array with data.

        """
        X = X.copy()
        X[X == 0.0] = np.nan
        return X * (self.data_max_ - self.data_min_) + self.data_min_


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
    indices_to_skip = []

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

                if image_index in indices_to_skip:
                    image_index = image_index + 1
                    continue

                image_to_save = image[
                    i * image_size : (i + 1) * image_size,
                    j * image_size : (j + 1) * image_size,
                ]

                if (
                    not is_lr
                    and np.count_nonzero(image_to_save) / image_to_save.size < 0.3
                ):
                    indices_to_skip.append(image_index)
                    image_index = image_index + 1
                    continue

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
    img = img.resize((width, height), resample=Image.BICUBIC)
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
    logging.info(pattern)
    return sorted(glob(pattern, recursive=True))


def prepara_data(fpath: str, output_dir: str, stage: Optional[str] = None) -> int:
    """
    Normalizes values in the image, then prepares HR & LR image patches from specified file and saves them
    in the specified `output_dir`.

    Args:
        fpath (str): The path to the file.
        output_dir (str): The output dir.
        stage (Optional[str]): The name of the stage, one of: "train", "val", "test". Optional, default: None.

    Returns (int): The last index of generated image patches.

    """
    scaler = WorldClimScaler()
    arr = np.array(Image.open(fpath))
    scaled = scaler.fit_transform_single(arr)
    return make_patches(
        scaled, os.path.basename(os.path.splitext(fpath)[0]), output_dir, stage
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


if __name__ == "__main__":
    root_data_directory = "/media/xultaeculcis/2TB/datasets/sr/fine-tuning/wc/"
    output_directory = os.path.join(root_data_directory, "pre-processed")
    train_stages = ["train", "val", "test"]
    image_subfolders = ["hr", "lr"]
    dataset_variables = [
        "pre",
        "tmin",
        "tmax",
    ]
    dataset_resolutions = ["2.5m"]

    stage_years_mapping = {
        "train": (1961, 1995),
        "val": (1996, 2004),
        "test": (2005, 2019),
    }

    for stage in train_stages:
        for var in dataset_variables:
            for res in dataset_resolutions:
                for subfolder in image_subfolders:
                    os.makedirs(
                        os.path.join(output_directory, var, res, stage, subfolder),
                        exist_ok=True,
                    )
                    os.makedirs(
                        os.path.join(output_directory, "elevation", res, "hr"),
                        exist_ok=True,
                    )

    c = Client(n_workers=8, threads_per_worker=1)
    try:
        futures = []

        for var in dataset_variables:
            for res in dataset_resolutions:
                dask_params = []
                f_paths = get_files(root_data_directory, "weather", var, res)
                for fp in f_paths[:16]:
                    file_name = os.path.basename(os.path.splitext(fp)[0])
                    year = int(file_name.split("-")[0].split("_")[-1])
                    logging.info(year)
                    stage = stage_from_year(year, stage_years_mapping)
                    dask_params.append((fp, stage))

                    for item in dask_params:
                        fp, stage = item
                        futures.append(
                            c.submit(
                                prepara_data,
                                fp,
                                os.path.join(output_directory, var, res),
                                stage,
                            )
                        )

                break

        c.gather(futures)
    finally:
        c.close()

    # handle elevation layer separately
    var = "elevation"
    for res in dataset_resolutions:
        elev_files = get_files(root_data_directory, var, res)
        for fp in elev_files:  # there should be only one file per resolution
            prepara_data(fp, os.path.join(output_directory, var, res))
