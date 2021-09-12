# -*- coding: utf-8 -*-
import torch
from torchmetrics import Metric


class RegressionAccuracy(Metric):
    def __init__(self, eps: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.eps = eps

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(torch.abs(preds - target) <= self.eps)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
