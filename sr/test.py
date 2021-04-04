# -*- coding: utf-8 -*-
import argparse
import logging
import warnings
from pprint import pprint
from typing import Union

import numpy as np
import pytorch_lightning as pl
from utils import prepare_training

from sr.lightning_modules.datamodules import SuperResolutionDataModule
from sr.lightning_modules.pl_gan import GANLightningModule

np.set_printoptions(precision=3)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """
    Parses the program arguments.

    :return: The Namespace with parsed parameters.
    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    parser = GANLightningModule.add_model_specific_args(parser)

    # training config args
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--val_check_interval", type=Union[int, float], default=1.0)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_find_only", type=bool, default=False)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--print_config", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--experiment_name", type=str, default="gen-pre-training")
    parser.add_argument("--save_model_path", type=str, default="../model_weights")
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--checkpoint_monitor_metric", type=str, default="hp_metric")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=5)
    parser.add_argument("--flush_logs_every_n_steps", type=int, default=10)
    parser.add_argument("--generator", type=str, default="srcnn")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # args for training from pre-trained model
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="../model_weights/gen-pre-training-srcnn-tmax-4x-epoch=14-step=0-hp_metric=0.00402.ckpt",
        help="A path to pre-trained model checkpoint. Required for fine tuning.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    if arguments.print_config:
        print("Running with following configuration:")  # noqa T001
        pprint(vars(arguments))  # noqa T003

    pl.seed_everything(seed=arguments.seed)

    net, dm, trainer = prepare_training(arguments)

    trainer.test(model=net, datamodule=dm)
