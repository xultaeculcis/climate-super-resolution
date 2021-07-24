# -*- coding: utf-8 -*-
import argparse
import logging
import warnings
from pprint import pprint
from typing import Union

import numpy as np
import pytorch_lightning as pl

from sr.lightning_modules.pl_sr_module import SuperResolutionLightningModule
from sr.lightning_modules.utils import prepare_training
from sr.lightning_modules.datamodules import SuperResolutionDataModule

np.set_printoptions(precision=3)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


def parse_args(arguments: argparse.Namespace = None) -> argparse.Namespace:
    """
    Parses the program arguments.

    :param arguments: The argparse Namespace. Optional.
    :return: The Namespace with parsed parameters.
    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    parser = SuperResolutionLightningModule.add_model_specific_args(parser)

    # training config args
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--val_check_interval", type=Union[int, float], default=1.0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--benchmark", type=bool, default=True)
    parser.add_argument("--lr_find_only", type=bool, default=False)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--print_config", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_model_path", type=str, default="./model_weights")
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--checkpoint_monitor_metric", type=str, default="hp_metric")
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--save_top_k", type=int, default=None)
    parser.add_argument("--flush_logs_every_n_steps", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--generator", type=str, default="rcan")
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default="./model_weights/use_elevation=True-batch_size=48/gen-pre-training-rcan-temp-4x-epoch=14-step=110279-hp_metric=0.00405.ckpt",  # noqa E501
    # )
    parser.add_argument("--compare", type=bool, default=False)

    # args for training from pre-trained model
    parser.add_argument(
        "--pretrained_model",
        type=str,
        # default="./model_weights/use_elevation=True-batch_size=64/gen-pre-training-esrgan-temp-4x-epoch=29-step=165419-hp_metric=0.00608.ckpt",  # noqa E501
        # default="./model_weights/use_elevation=True-batch_size=64/gan-training-esrgan-temp-4x-epoch=18-step=104765-hp_metric=0.50164.ckpt",  # noqa E501
        default=None,  # noqa E501
        help="A path to pre-trained model checkpoint. Required for fine tuning.",
    )
    parsed_arguments = parser.parse_args()
    parsed_arguments.experiment_name = arguments.experiment_name

    return parsed_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    # parser.add_argument("--experiment_name", type=str, default="gan-training")
    parser.add_argument("--experiment_name", type=str, default="gen-pre-training")

    arguments = parser.parse_args()
    arguments = parse_args(arguments)
    pl.seed_everything(seed=arguments.seed)

    if arguments.print_config:
        print("Running with following configuration:")  # noqa T001
        pprint(vars(arguments))  # noqa T003

    net, dm, trainer = prepare_training(arguments)

    if arguments.lr_find_only:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model=net, datamodule=dm, max_lr=8e-4)

        # Plot lr find results
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logging.info(f"LR Finder suggestion: {new_lr}")
    else:
        trainer.fit(model=net, datamodule=dm)
        if not arguments.fast_dev_run:
            trainer.test()
