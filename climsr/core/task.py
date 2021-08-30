# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torchmetrics.functional import mean_absolute_error, mean_squared_error, psnr, ssim

import climsr.consts as consts
from climsr.core.config import DiscriminatorConfig, GeneratorConfig, OptimizerConfig, SchedulerConfig
from climsr.core.instantiator import Instantiator
from climsr.data import normalization
from climsr.data.normalization import MinMaxScaler, StandardScaler
from climsr.task.metrics import MetricsResult, MetricsSimple


class LitSuperResolutionModule(pl.LightningModule):
    """
    Base class for SR.
    Provides a few helper functions primarily for optimization.
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
        self.optimizers = optimizers
        self.schedulers = schedulers

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, torch.optim.lr_scheduler._LRScheduler]]]]:
        """Prepare optimizers and schedulers"""
        schedulers = []
        for idx, scheduler in enumerate(self.schedulers):
            schedulers.append({"scheduler": scheduler, "interval": "step", "name": f"scheduler-{idx}"})
        return self.optimizers, schedulers

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

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass


class TaskSuperResolutionModule(LitSuperResolutionModule):
    """
    Base class for task specific SR modules
    """

    def __init__(
        self,
        generator: GeneratorConfig,
        optimizers: List[OptimizerConfig],
        schedulers: List[SchedulerConfig],
        discriminator: Optional[DiscriminatorConfig] = None,
        instantiator: Optional[Instantiator] = None,
        **kwargs,
    ):
        self.instantiator = instantiator
        self.optimizer_cfgs = optimizers
        self.scheduler_cfgs = schedulers
        super().__init__(generator=instantiator.instantiate(generator), discriminator=instantiator.instantiate(discriminator))

        # store parameters
        self.save_hyperparameters()

        # loss
        self.loss = torch.nn.MSELoss() if self.hparams.generator == consts.models.srcnn else torch.nn.L1Loss()

        # standardization
        self.stats = consts.cruts.statistics[
            consts.datasets_and_preprocessing.world_clim_to_cruts_mapping[self.hparams.world_clim_variable]
        ]
        self.scaler = (
            StandardScaler(
                self.stats[consts.stats.mean],
                self.stats[consts.stats.std],
            )
            if self.hparams.normalization_method == normalization.zscore
            else MinMaxScaler(feature_range=self.hparams.normalization_range)
        )

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
            num_training_steps=self.scheduler_cfgs[0].num_training_steps,
            num_warmup_steps=getattr(self.scheduler_cfg, "num_warmup_steps", None),
        )
        for scheduler_cfg in self.scheduler_cfgs:
            scheduler_cfg.num_training_steps = num_training_steps
            scheduler_cfg.num_warmup_steps = num_warmup_steps

        rank_zero_info(f"Inferring number of training steps, set to {self.scheduler_cfg.num_training_steps}")
        rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.scheduler_cfg.num_warmup_steps}")

        self.optimizers = []
        self.schedulers = []

        generator_optimizer = self.instantiator.optimizer(self.generator, self.optimizer_cfgs[0])
        self.optimizers.append(generator_optimizer)
        self.schedulers.append(self.instantiator.scheduler(self.scheduler_cfgs[0], generator_optimizer))

        if len(self.optimizer_cfgs) == 2:
            discriminator_optimizer = self.instantiator.optimizer(self.generator, self.optimizer_cfgs[1])
            self.optimizers.append(discriminator_optimizer)
            self.schedulers.append(self.instantiator.scheduler(self.scheduler_cfgs[1], discriminator_optimizer))

        return super().configure_optimizers()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.instantiator:
            checkpoint["instantiator"] = self.instantiator

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.instantiator = checkpoint.get("instantiator")

    def forward(self, x: Tensor, elevation: Tensor = None, mask: Tensor = None) -> Tensor:
        if self.hparams.generator == consts.models.srcnn:
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

    def common_val_test_step(self, batch: Any) -> "MetricsResult":
        """
        Rus common validation and test steps.

        Args:
            batch (Any): The batch of data.

        Returns (MetricsResult): The MetricsResult.

        """
        original = batch[consts.batch_items.original_data]
        mask = batch[consts.batch_items.mask_np].cpu().numpy()
        max_vals = batch[consts.batch_items.max].cpu().numpy()
        min_vals = batch[consts.batch_items.min].cpu().numpy()

        hr, sr = self.common_step(batch)

        metrics = self.compute_metrics_common(hr, sr)

        denormalized_mae = []
        denormalized_mse = []
        denormalized_rmse = []
        denormalized_r2 = []

        original = original.squeeze(1).cpu().numpy()
        squeezed_sr = sr.squeeze(1).cpu().numpy()

        for i in range(squeezed_sr.shape[0]):
            # to numpy
            i_original = original[i]
            i_sr = squeezed_sr[i]

            # denormalize/destandardize
            i_sr = (
                self.scaler.denormalize(i_sr)
                if self.hparams.normalization_method == normalization.zscore
                else self.scaler.denormalize(i_sr, min_vals[i], max_vals[i])
            )

            # ocean mask
            i_sr[mask[i]] = 0.0
            i_original[mask[i]] = 0.0

            # compute metrics
            diff = i_sr - i_original
            denormalized_mae.append(np.absolute(diff).mean())
            denormalized_mse.append((diff ** 2).mean())
            denormalized_rmse.append(np.sqrt(denormalized_mse[-1]))
            denormalized_r2.append(1 - (np.sum(diff ** 2) / (np.sum((i_original - np.mean(i_original)) ** 2) + 1e-8)))

        denormalized_mae = np.mean(denormalized_mae)
        denormalized_mse = np.mean(denormalized_mse)
        denormalized_rmse = np.mean(denormalized_rmse)
        denormalized_r2 = np.mean(denormalized_r2)

        return MetricsResult(
            denormalized_mae=denormalized_mae,
            denormalized_mse=denormalized_mse,
            denormalized_rmse=denormalized_rmse,
            denormalized_r2=denormalized_r2,
            pixel_level_loss=metrics.pixel_level_loss,
            mae=metrics.mae,
            mse=metrics.mse,
            psnr=metrics.psnr,
            rmse=metrics.rmse,
            ssim=metrics.ssim,
            sr=sr,
        )

    def compute_metrics_common(self, hr: Tensor, sr: Tensor) -> "MetricsSimple":
        """
        Common step to compute all of the metrics.

        Args:
            hr (Tensor): The ground truth HR image.
            sr (Tensor): The hallucinated SR image.

        Returns (MetricsSimple): A dataclass with metrics: L1_Loss, PSNR, SSIM, MAE, MSE, RMSE.

        """
        loss = self.loss(sr, hr)
        psnr_score = psnr(sr, hr)
        ssim_score = ssim(sr, hr.half() if self.hparams.precision == 16 else hr)
        mae = mean_absolute_error(sr, hr)
        mse = mean_squared_error(sr, hr)
        rmse = torch.sqrt(mse)

        return MetricsSimple(
            pixel_level_loss=loss,
            psnr=psnr_score,
            ssim=ssim_score,
            mae=mae,
            mse=mse,
            rmse=rmse,
        )

    def on_train_start(self):
        """Run additional steps when training starts."""
        self.logger.log_hyperparams(self.hparams, {"hp_metric": self.hparams.initial_hp_metric_val})
