# -*- coding: utf-8 -*-
import datetime as dt
import logging
import os
import warnings
from functools import partial, wraps
from typing import List, Sequence

import pytorch_lightning as pl
import pytorch_lightning.loggers
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"


def set_gpu_power_limit_if_needed():
    """Helper function, that sets GPU power limit if RTX 3090 is used"""
    stream = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv")
    gpu_list = stream.read()
    if "NVIDIA GeForce RTX 3090" in gpu_list:
        os.system("sudo nvidia-smi -pm 1")
        os.system("sudo nvidia-smi -pl 300")


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "training",
        "trainer",
        "task",
        "datamodule",
        "callbacks",
        "generator",
        "discriminator",
        "logger",
        "optimizers",
        "schedulers",
        "profiler",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def log_step(
    func=None,
    *,
    time_taken=True,
):
    """
    Decorates a function to add automated logging statements
    :param func: callable, function to log, defaults to None
    :param time_taken: bool, log the time it took to run a function, defaults to True
    :returns: the result of the function
    """

    if func is None:
        return partial(
            log_step,
            time_taken=time_taken,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()

        result = func(*args, **kwargs)

        optional_strings = [
            f"time={dt.datetime.now() - tic}" if time_taken else None,
        ]

        combined = " ".join([s for s in optional_strings if s])

        logging.info(
            f"[{func.__name__}]" + combined,
        )
        return result

    return wrapper


def finish(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
