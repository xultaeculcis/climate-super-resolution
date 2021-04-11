# -*- coding: utf-8 -*-
import argparse
import os
from typing import Tuple

import pytorch_lightning as pl

from lightning_modules.pl_gan import GANLightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from sr.lightning_modules.datamodules import SuperResolutionDataModule
from sr.lightning_modules.pl_generator_pre_training import (
    GeneratorPreTrainingLightningModule,
)


def prepare_pl_module(args: argparse.Namespace) -> pl.LightningModule:
    """
    Prepares the Lightning Module.

    :param args: The arguments.
    :return: The Lightning Module.
    """
    if args.pretrained_model:
        net = (
            GANLightningModule.load_from_checkpoint(
                checkpoint_path=args.pretrained_model,
            )
            if args.experiment_name == "gan-training"
            else GeneratorPreTrainingLightningModule.load_from_checkpoint(
                checkpoint_path=args.pretrained_model,
            )
        )
    else:
        net = (
            GANLightningModule(**vars(args))
            if args.experiment_name == "gan-training"
            else GeneratorPreTrainingLightningModule(**vars(args))
        )

    return net


def prepare_pl_trainer(args: argparse.Namespace) -> pl.Trainer:
    """
    Prepares the Pytorch Lightning Trainer.

    :param args: The arguments.
    :return: The Pytorch Lightning Trainer.
    """
    experiment_name = f"{args.experiment_name}-{args.generator}-{args.world_clim_variable}-{args.world_clim_multiplier}"
    tb_logger = pl_loggers.TensorBoardLogger(
        args.log_dir, name=experiment_name, default_hp_metric=True
    )
    monitor_metric = args.checkpoint_monitor_metric
    mode = "min"
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stopping_patience,
        verbose=False,
        mode=mode,
    )
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        verbose=False,
        mode=mode,
        filepath=os.path.join(
            args.save_model_path,
            f"{experiment_name}-{{epoch}}-{{step}}-{{{monitor_metric}:.5f}}",
        ),
        save_top_k=args.save_top_k,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, early_stop_callback]
    checkpoint_callback = model_checkpoint
    pl_trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback,
    )
    return pl_trainer


def prepare_pl_datamodule(args: argparse.Namespace) -> pl.LightningDataModule:
    """
    Prepares the Tabular Lightning Data Module.

    :param args: The arguments.
    :return: The Tabular Lightning Data Module.
    """
    data_module = SuperResolutionDataModule(
        data_path=args.data_path,
        world_clim_variable=args.world_clim_variable,
        world_clim_multiplier=args.world_clim_multiplier,
        generator_type=args.generator,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        seed=args.seed,
    )
    return data_module


def prepare_training(
    args: argparse.Namespace,
) -> Tuple[pl.LightningModule, pl.LightningDataModule, pl.Trainer]:
    """
    Prepares everything for training. `DataModule` is prepared by setting up the train/val/test sets for specified fold.
    Creates new `PreTrainingESRGANModule` Lightning Module together with `pl.Trainer`.

    :param args: The arguments.
    :returns: A tuple with model and the trainer.
    """
    pl.seed_everything(args.seed)
    data_module = prepare_pl_datamodule(args)
    lightning_module = prepare_pl_module(args)
    pl_trainer = prepare_pl_trainer(args)

    return lightning_module, data_module, pl_trainer
