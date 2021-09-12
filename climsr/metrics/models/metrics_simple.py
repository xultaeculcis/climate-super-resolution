# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Union

import numpy as np
from torch import Tensor


@dataclass
class MetricsSimple:
    pixel_level_loss: Union[np.ndarray, Tensor, float]
    mae: Union[np.ndarray, Tensor, float]
    mse: Union[np.ndarray, Tensor, float]
    psnr: Union[np.ndarray, Tensor, float]
    rmse: Union[np.ndarray, Tensor, float]
    ssim: Union[np.ndarray, Tensor, float]
