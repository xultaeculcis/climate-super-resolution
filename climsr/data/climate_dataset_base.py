# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class ClimateDatasetBase(Dataset):
    def __init__(
        self,
        elevation_file: str,
        land_mask_file: str,
        generator_type: str,
        variable: Optional[str] = None,
        scaling_factor: Optional[int] = 4,
        normalize: Optional[bool] = True,
        standardize: Optional[bool] = False,
        standardize_stats: Dict[str, Dict[str, float]] = None,
        normalize_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
    ):
        if normalize == standardize:
            raise Exception("Bad parameter combination: normalization and standardization! Choose one!")

        self.elevation_file = elevation_file
        self.land_mask_file = land_mask_file
        self.scaling_factor = scaling_factor
        self.generator_type = generator_type
        self.variable = variable
        self.normalize = normalize
        self.normalize_range = normalize_range
        self.standardize = standardize
        self.standardize_stats = standardize_stats

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
