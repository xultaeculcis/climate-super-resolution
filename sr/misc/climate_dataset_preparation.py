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


def get_files(
    data_dir: str, variables: List[str], resolutions: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    results = {}
    for var in variables:
        results[var] = {}
        for res in resolutions:
            pattern = os.path.join(data_dir, var, "**", f"*_{res}*.tif")
            logging.info(pattern)
            results[var][res] = sorted(glob(pattern, recursive=True))

    return results


def prepara_dataset(
    data_dir: str, output_dir: str, variables: List[str], resolutions: List[str]
) -> None:
    fpaths = get_files(data_dir, variables, resolutions)

    for var in variables:
        scaler = WorldClimScaler()
        scaler.fit(fpaths=fpaths[var]["2.5m"])
        scaler.transform(
            fpaths=fpaths[var]["2.5m"], out_dir=os.path.join(output_dir, var)
        )


if __name__ == "__main__":
    data_dir = "/media/xultaeculcis/2TB/datasets/sr/fine-tuning/wc/"
    output_dir = os.path.join(data_dir, "scaled")

    variables = [
        os.path.join("weather", "pre"),
        os.path.join("weather", "tmin"),
        os.path.join("weather", "tmax"),
        "elevation",
    ]
    resolutions = ["2.5m"]

    prepara_dataset(data_dir, output_dir, variables, resolutions)
