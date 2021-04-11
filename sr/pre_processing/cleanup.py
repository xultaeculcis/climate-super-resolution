# -*- coding: utf-8 -*-
import argparse
import os
from glob import glob
import logging

import dask.bag
from distributed import Client

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    Parse arguments.

    Returns (argparse.Namespace): A namespace with parsed arguments.

    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument(
        "--dir",
        type=str,
        default="/media/xultaeculcis/2TB/datasets/wc/pre-processed-v2",
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--threads_per_worker", type=int, default=16)

    return parser.parse_args()


def remove(fp):
    if os.path.isdir(fp):
        return
    else:
        os.remove(fp)


if __name__ == "__main__":
    arguments = parse_args()

    client = Client(
        n_workers=arguments.n_workers, threads_per_worker=arguments.threads_per_worker
    )

    files = glob(os.path.join(arguments.dir, "**/*"), recursive=True)
    logging.info(f"Deleting {len(files)} files from {arguments.dir}")

    try:
        dask.bag.from_sequence(files).map(remove).compute()
    finally:
        client.close()
