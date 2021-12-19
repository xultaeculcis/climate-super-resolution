# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torchmetrics import (
    PSNR,
    SSIM,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
    SymmetricMeanAbsolutePercentageError,
)

import climsr.consts as consts
from climsr.core.config import DiscriminatorConfig, GeneratorConfig, OptimizerConfig, SchedulerConfig
from climsr.core.instantiator import HydraInstantiator, Instantiator
from climsr.data import normalization
from climsr.data.normalization import MinMaxScaler, Scaler, StandardScaler
from climsr.metrics.regression_accuracy import RegressionAccuracy

default_instantiator = HydraInstantiator()


class LitSuperResolutionModule(pl.LightningModule):
    """
    Base class for SR. Provides a few helper functions primarily for optimization.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: Optional[torch.nn.Module] = None,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        # some optimizers/schedulers need parameters only known dynamically
        # allow users to override the getter to instantiate them lazily
        self._optimizers = optimizers
        self._schedulers = schedulers

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, torch.optim.lr_scheduler._LRScheduler]]]]:
        """Prepare optimizers and schedulers"""
        schedulers = []
        for scheduler in self._schedulers:
            schedulers.append({"scheduler": scheduler, "interval": "step"})
        return self._optimizers, schedulers

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and -1 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float) and num_warmup_steps < 1.0:
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)

    def configure_metrics(self, stage: Optional[str] = None) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass

    def on_epoch_start(self) -> None:
        self.configure_metrics()


class TaskSuperResolutionModule(LitSuperResolutionModule):
    """
    Base class for task specific SR modules
    """

    def __init__(
        self,
        generator: GeneratorConfig,
        optimizers: Dict[str, OptimizerConfig],
        schedulers: Dict[str, SchedulerConfig],
        discriminator: Optional[DiscriminatorConfig] = None,
        instantiator: Optional[Instantiator] = default_instantiator,
        **kwargs,
    ):
        self.instantiator = instantiator
        self.optimizer_cfgs = optimizers
        self.scheduler_cfgs = schedulers
        super().__init__(generator=instantiator.instantiate(generator), discriminator=instantiator.instantiate(discriminator))

        # store parameters
        self.save_hyperparameters(
            "generator",
            "discriminator",
            "optimizers",
            "schedulers",
            *kwargs.keys(),
        )

        # metrics placeholder
        self.metrics = {}

        # loss
        self.loss = MSELoss() if self.hparams.generator_type == consts.models.srcnn else L1Loss()

        # normalization
        self.stats, self.scaler = self._configure_scaler()

    def _configure_scaler(self) -> Tuple[pd.DataFrame, Scaler]:
        stats = pd.read_feather(
            os.path.join(
                self.hparams.data_path,
                consts.datasets_and_preprocessing.preprocessing_output_path,
                consts.datasets_and_preprocessing.feather_path,
                consts.datasets_and_preprocessing.zscore_stats_filename,
            )
        ).set_index(consts.datasets_and_preprocessing.variable, drop=True)

        scaler = (
            StandardScaler(
                mean=stats.at[
                    consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.hparams.world_clim_variable],
                    consts.stats.mean,
                ],
                std=stats.at[
                    consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.hparams.world_clim_variable],
                    consts.stats.std,
                ],
            )
            if self.hparams.normalization_method == normalization.zscore
            else MinMaxScaler(feature_range=self.hparams.normalization_range)
        )

        return stats, scaler

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, torch.optim.lr_scheduler._LRScheduler]]]]:
        if self.instantiator is None:
            raise MisconfigurationException(
                "To train you must provide an instantiator to instantiate the optimizer and scheduler "
                "or override `configure_optimizers` in the `LightningModule`."
            )

        # compute_warmup needs the datamodule to be available when `self.num_training_steps`
        # is called that is why this is done here and not in the __init__
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=getattr(self.scheduler_cfgs[consts.training.generator_scheduler_key], "num_training_steps", -1),
            num_warmup_steps=getattr(self.scheduler_cfgs[consts.training.generator_scheduler_key], "num_warmup_steps", None),
        )
        for key in self.scheduler_cfgs.keys():
            if self.scheduler_cfgs[key] is not None:
                self.scheduler_cfgs[key].num_training_steps = num_training_steps
                self.scheduler_cfgs[key].num_warmup_steps = num_warmup_steps

        rank_zero_info(
            "Inferring number of training steps, set to "
            f"{self.scheduler_cfgs[consts.training.generator_scheduler_key].num_training_steps}"
        )
        rank_zero_info(
            "Inferring number of warmup steps from ratio, set to "
            f"{self.scheduler_cfgs[consts.training.generator_scheduler_key].num_warmup_steps}"
        )

        self._optimizers = []
        self._schedulers = []

        generator_optimizer = self.instantiator.optimizer(
            self.generator, self.optimizer_cfgs[consts.training.generator_optimizer_key]
        )
        self._optimizers.append(generator_optimizer)
        self._schedulers.append(
            self.instantiator.scheduler(self.scheduler_cfgs[consts.training.generator_scheduler_key], generator_optimizer)
        )

        if self.optimizer_cfgs[consts.training.discriminator_optimizer_key] is not None:
            discriminator_optimizer = self.instantiator.optimizer(
                self.discriminator,
                self.optimizer_cfgs[consts.training.discriminator_optimizer_key],
            )
            self._optimizers.append(discriminator_optimizer)
            self._schedulers.append(
                self.instantiator.scheduler(
                    self.scheduler_cfgs[consts.training.discriminator_scheduler_key],
                    discriminator_optimizer,
                )
            )

        return super().configure_optimizers()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.instantiator:
            checkpoint["instantiator"] = self.instantiator

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.instantiator = checkpoint.get("instantiator")

    def forward(self, x: Tensor, elevation: Tensor = None, mask: Tensor = None) -> Tensor:
        if self.hparams.generator_type == consts.models.srcnn:
            return self.generator(x)
        else:
            return self.generator(x, elevation, mask)

    def common_step(self, batch: Any) -> Tuple[Tensor, Tensor]:
        """
        Runs common step during train/val/test pass.

        Args:
            batch (Any): A batch of data from a dataloader.

        Returns (Tuple[Tensor, Tensor]): A tuple with HR image and SR image.

        """
        lr, hr, elev, mask = (
            batch[consts.batch_items.lr],
            batch[consts.batch_items.hr],
            batch[consts.batch_items.elevation],
            batch[consts.batch_items.mask],
        )

        sr = self(lr, elev, mask)

        return hr, sr

    def common_val_test_step(self, batch: Any, prefix: Optional[str] = consts.stages.val) -> Dict[str, Tensor]:
        """
        Runs common validation and test steps.

        Args:
            batch (Any): The batch of data.
            prefix (Optional[str]): The prefix for metrics grouping.

        Returns (Tensor): The loss.

        """
        original = batch[consts.batch_items.original_data]
        mask = batch[consts.batch_items.mask]
        max_vals = batch[consts.batch_items.max]
        min_vals = batch[consts.batch_items.min]

        hr, sr = self.common_step(batch)
        sr_copy = sr.detach().clone()

        # denormalize/destandardize
        denormalized_sr = (
            self.scaler.denormalize(sr)
            if self.hparams.normalization_method == normalization.zscore
            else self.scaler.denormalize(sr, min_vals, max_vals)
        )

        sr[(~mask.bool())] = 0.0
        hr[(~mask.bool())] = 0.0
        denormalized_sr[(~mask.bool())] = 0.0
        original[(~mask.bool())] = 0.0

        normal_loss = self.loss(sr, hr)
        loss = self.loss(sr, hr)
        metric_dict = self.compute_metrics(sr, hr, denormalized_sr, original, mode=prefix)
        metric_dict[f"{prefix}/normalized_loss"] = normal_loss
        metric_dict[f"{prefix}/loss"] = loss
        metric_dict["sr"] = sr_copy

        return metric_dict

    def configure_metrics(self, stage: Optional[str] = None) -> None:
        self.acc_at_0_1 = RegressionAccuracy(eps=0.1)
        self.acc_at_0_25 = RegressionAccuracy(eps=0.25)
        self.acc_at_0_5 = RegressionAccuracy(eps=0.5)
        self.acc_at_0_75 = RegressionAccuracy(eps=0.75)
        self.acc_at_1 = RegressionAccuracy(eps=1.0)
        self.acc_at_1_25 = RegressionAccuracy(eps=1.25)
        self.acc_at_1_5 = RegressionAccuracy(eps=1.5)
        self.acc_at_2 = RegressionAccuracy(eps=2.0)
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.rmse = MeanSquaredError(squared=False)
        self.mape = MeanAbsolutePercentageError()
        self.smape = SymmetricMeanAbsolutePercentageError()
        self.r2 = R2Score()
        self.metrics = {
            "acc@0.1": self.acc_at_0_1,
            "acc@0.25": self.acc_at_0_25,
            "acc@0.5": self.acc_at_0_5,
            "acc@0.75": self.acc_at_0_75,
            "acc@1": self.acc_at_1,
            "acc@01.25": self.acc_at_1_25,
            "acc@1.5": self.acc_at_1_5,
            "acc@2": self.acc_at_2,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "r2": self.r2,
        }

        # ensure the same device as main model
        for k, metric in self.metrics.items():
            self.metrics[k] = metric.to(self.device)

    def compute_metrics(
        self,
        normalized_sr: Tensor,
        normalized_hr: Tensor,
        denormalized_sr: Tensor,
        denormalized_hr: Tensor,
        mode: Optional[str] = consts.stages.val,
    ) -> Dict[str, torch.Tensor]:
        """
        Common step to compute all of the metrics.

        Args:
            normalized_sr (Tensor): The hallucinated SR normalized image.
            normalized_hr (Tensor): The ground truth HR normalized image.
            denormalized_sr (Tensor): The hallucinated SR denormalized image.
            denormalized_hr (Tensor): The ground truth HR denormalized image.
            mode (Optional[str]): The optional mode. "val" by default.

        Returns (Dict[str, torch.Tensor]): A dictionary with metrics.

        """
        normalized_hr = normalized_hr.to(normalized_sr.dtype)
        denormalized_hr = denormalized_hr.to(denormalized_sr.dtype)

        results = dict()
        for k, metric in self.metrics.items():
            if k in ["ssim", "mape"]:
                hr = normalized_hr
                sr = normalized_sr
            elif k == "r2":
                hr = torch.flatten(denormalized_hr)
                sr = torch.flatten(denormalized_sr)
            else:
                hr = denormalized_hr
                sr = denormalized_sr

            results[f"{mode}/{k}"] = metric(sr, hr)

        return results

    def on_train_start(self):
        """Run additional steps when training starts."""
        for logger in self.logger:
            if type(logger) == TensorBoardLogger:
                self.logger[0].log_hyperparams(self.hparams, {"hp_metric": self.hparams.initial_hp_metric_val})

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Compute and log hp_metric at the epoch level."""
        hp_metric = torch.stack([output[f"{consts.stages.val}/rmse"] for output in outputs]).mean()
        self.log("hp_metric", hp_metric)
