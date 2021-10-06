# -*- coding: utf-8 -*-
import datetime as dt
import logging
import os
import warnings
from functools import partial, wraps


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
