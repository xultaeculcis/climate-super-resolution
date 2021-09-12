# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

minmax = "minmax"
zscore = "zscore"


class Scaler:
    def _normalize(self, *args, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        pass

    def _denormalize(self, *args, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        pass

    def normalize(self, *args, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        return self._normalize(*args, **kwargs)

    def denormalize(self, *args, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        return self._denormalize(*args, **kwargs)


class MinMaxScaler(Scaler):
    def __init__(
        self,
        eps: Optional[float] = 1e-8,
        feature_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        nan_substitution: Optional[float] = 0.0,
    ):
        self.eps = eps
        self.feature_range = feature_range
        self.nan_substitution = nan_substitution
        self.a, self.b = self.feature_range

    def _normalize(
        self,
        arr: np.ndarray,
        min: Optional[float] = None,
        max: Optional[float] = None,
        missing_indicator: Optional[float] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        out_arr = arr.copy()
        if missing_indicator:
            out_arr[arr == missing_indicator] = np.nan

        if min is None or max is None:
            max = np.nanmax(out_arr)
            min = np.nanmin(out_arr)

        data_range = max - min
        scale = (self.b - self.a) / (data_range + self.eps)
        min_ = self.a - min * scale

        out_arr = out_arr * scale
        out_arr += min_

        out_arr[np.isnan(out_arr)] = self.nan_substitution

        return out_arr.astype(np.float32)

    def _denormalize(
        self,
        arr: np.ndarray,
        min: float,
        max: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        data_range = max - min
        scale = (self.b - self.a) / (data_range + self.eps)
        min_ = self.a - min * scale

        out_arr = arr - min_
        out_arr /= scale

        return out_arr


class StandardScaler(Scaler):
    def __init__(
        self,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        nan_sub: Optional[float] = None,
        eps: Optional[float] = 1e-8,
        missing_indicator: Optional[float] = None,
        nan_substitution: Optional[float] = None,
    ):
        self.mean = mean
        self.std = std
        self.nan_sub = nan_sub
        self.eps = eps
        self.missing_indicator = missing_indicator
        self.nan_substitution = nan_substitution

    def _normalize(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.missing_indicator:
            arr[arr == self.missing_indicator] = np.nan

        out_arr = (arr - self.mean) / (self.std + self.eps)

        if self.nan_substitution:
            out_arr[np.isnan(out_arr)] = self.nan_substitution

        return out_arr.astype(np.float32)

    def _denormalize(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (arr * self.std) + self.mean
