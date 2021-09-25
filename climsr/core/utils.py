# -*- coding: utf-8 -*-
import os
import warnings


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
