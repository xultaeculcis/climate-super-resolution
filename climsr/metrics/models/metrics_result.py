# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from torch import Tensor


@dataclass
class MetricsResult:
    denormalized_mae: Union[np.ndarray, Tensor, float]
    denormalized_mse: Union[np.ndarray, Tensor, float]
    denormalized_rmse: Union[np.ndarray, Tensor, float]
    denormalized_r2: Union[np.ndarray, Tensor, float]
    pixel_level_loss: Union[np.ndarray, Tensor, float]
    mae: Union[np.ndarray, Tensor, float]
    mse: Union[np.ndarray, Tensor, float]
    psnr: Union[np.ndarray, Tensor, float]
    rmse: Union[np.ndarray, Tensor, float]
    ssim: Union[np.ndarray, Tensor, float]
    sr: Optional[Union[np.ndarray, Tensor, float]]
