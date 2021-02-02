import argparse
import os
import warnings
import logging
from collections import OrderedDict
from datetime import datetime
from pprint import pprint
from typing import Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from datamodules import SuperResolutionDataModule
from models import Generator
from pl_pre_training_generator import PreTrainingClimateSRGanModule

np.set_printoptions(precision=3)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings('ignore')


def parse_args(arguments: argparse.Namespace = None) -> argparse.Namespace:
    """
    Parses the program arguments.

    :param arguments: The argparse Namespace. Optional.
    :return: The Namespace with parsed parameters.
    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    parser = PreTrainingClimateSRGanModule.add_model_specific_args(parser)

    # training config args
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fast_dev_run', type=bool, default=True)
    parser.add_argument('--print_config', type=bool, default=True)
    parser.add_argument('--experiment_name', type=str, default="gen-pre-training")
    parser.add_argument('--log_dir', type=str, default="../logs")
    parser.add_argument('--save_model_path', type=str, default="../model_weights")
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--checkpoint_monitor_metric', type=str, default="hp_metric")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=5)
    parser.add_argument('--flush_logs_every_n_steps', type=int, default=10)

    # override args
    parser.add_argument('--scaling_factor', type=int, default=4)

    return parser.parse_args(arguments)


def prepare_pl_module(args: argparse.Namespace) -> PreTrainingClimateSRGanModule:
    """
    Prepares the Ambulance Network Lightning Module.

    :param args: The arguments.
    :return: The Ambulance Network Lightning Module.
    """
    net = PreTrainingClimateSRGanModule(
        **vars(args)
    )
    return net


def prepare_pl_trainer(args: argparse.Namespace) -> pl.Trainer:
    """
    Prepares the Pytorch Lightning Trainer.

    :param args: The arguments.
    :return: The Pytorch Lightning Trainer.
    """
    experiment_name = f"{args.experiment_name}-{int(datetime.utcnow().timestamp())}"
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=experiment_name, default_hp_metric=True)
    monitor_metric = args.checkpoint_monitor_metric
    mode = "min"
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stopping_patience,
        verbose=False,
        mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        verbose=False,
        mode=mode,
        filepath=os.path.join(
            args.save_model_path, f"{experiment_name}-{{epoch:02d}}-{{{monitor_metric}:.2f}}"
        ),
        save_top_k=args.save_top_k
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


def prepare_pl_datamodule(args: argparse.Namespace) -> SuperResolutionDataModule:
    """
    Prepares the Tabular Lightning Data Module.

    :param args: The arguments.
    :return: The Tabular Lightning Data Module.
    """
    data_module = SuperResolutionDataModule(
        data_path=args.data_path,
        scaling_factor=args.scaling_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        seed=args.seed,
    )
    return data_module


def prepare_training(
        args: argparse.Namespace
) -> Tuple[PreTrainingClimateSRGanModule, SuperResolutionDataModule, pl.Trainer]:
    """
    Prepares everything for training. `DataModule` is prepared by setting up the train/val/test sets for specified fold.
    Creates new `PreTrainingClimateSRGanModule` Lightning Module together with `pl.Trainer`.

    :param args: The arguments.
    :returns: A tuple with model and the trainer.
    """
    pl.seed_everything(args.seed)
    data_module = prepare_pl_datamodule(args)
    lightning_module = prepare_pl_module(args)
    pl_trainer = prepare_pl_trainer(args)

    return lightning_module, data_module, pl_trainer


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.train_both_networks:
        arguments.experiment_name = "pre-train-both"

    if arguments.print_config:
        print(f"Running with following configuration:")
        pprint(vars(arguments))

    net, dm, trainer = prepare_training(arguments)
    trainer.fit(model=net, datamodule=dm)
