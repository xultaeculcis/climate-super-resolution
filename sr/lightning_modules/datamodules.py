# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from glob import glob
from random import random
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class ImageDataset(Dataset):
    def __init__(
        self,
        hr_image_list: List[str],
        lr_image_list: List[str],
        generator_type: str,
        hr_size: Optional[int] = 128,
        stage: Optional[str] = "train",
    ):
        self.hr_size = hr_size
        self.stage = stage
        self.generator_type = generator_type

        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.hr_image_list = hr_image_list
        self.lr_image_list = lr_image_list

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        img_lr = Image.open(self.lr_image_list[index]).convert(mode="RGB")
        img_hr = Image.open(self.hr_image_list[index]).convert(mode="RGB")
        img_sr_bicubic = []

        if self.stage == "train":
            if random() > 0.5:
                img_lr = TF.vflip(img_lr)
                img_hr = TF.vflip(img_hr)

            if random() > 0.5:
                img_lr = TF.hflip(img_lr)
                img_hr = TF.hflip(img_hr)
        if self.generator_type == "srcnn" or self.stage != "train":
            upscale = transforms.Resize((self.hr_size, self.hr_size))
            img_sr_bicubic = self.common_transforms(upscale(img_lr))

        img_lr = self.common_transforms(img_lr)
        img_hr = self.common_transforms(img_hr)

        return {"lr": img_lr, "hr": img_hr, "bicubic": img_sr_bicubic}

    def __len__(self) -> int:
        return len(self.hr_image_list)


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        generator_type: str,
        scale_factor: Optional[int] = 4,
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 4,
        hr_size: Optional[int] = 256,
        seed: Optional[int] = 42,
    ):
        super(SuperResolutionDataModule, self).__init__()

        assert hr_size % scale_factor == 0

        self.data_path = data_path
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.seed = seed
        self.generator_type = generator_type

        train_lr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "train", "lr")
        )
        train_hr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "train", "hr")
        )
        val_lr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "val", "lr")
        )
        val_hr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "val", "hr")
        )
        test_lr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "test", "lr")
        )
        test_hr_images = SuperResolutionDataModule.search_for_images(
            os.path.join(data_path, "test", "hr")
        )

        total = (
            len(train_lr_images)
            + len(train_hr_images)
            + len(val_lr_images)
            + len(val_hr_images)
            + len(test_lr_images)
            + len(test_hr_images)
        )
        logging.info(f"Total of {total} found under the {data_path}")

        assert len(train_hr_images) == len(
            train_lr_images
        ), f"Train HR size ({len(train_hr_images)}) does not match LR size ({len(train_lr_images)})"
        assert len(val_hr_images) == len(
            val_lr_images
        ), f"Validation HR size ({len(val_hr_images)}) does not match LR size ({len(val_lr_images)})"
        assert len(test_hr_images) == len(
            test_lr_images
        ), f"Test HR size ({len(test_hr_images)}) does not match LR size ({len(test_lr_images)})"

        logging.info(
            f"Train/Validation/Test split sizes (HR): {len(train_hr_images)}/{len(val_hr_images)}/{len(test_hr_images)}"
        )
        logging.info(
            f"Train/Validation/Test split sizes (LR): {len(train_lr_images)}/{len(val_lr_images)}/{len(test_lr_images)}"
        )

        self.train_dataset = ImageDataset(
            hr_image_list=train_hr_images,
            lr_image_list=train_lr_images,
            hr_size=self.hr_size,
            stage="train",
            generator_type=self.generator_type,
        )
        self.val_dataset = ImageDataset(
            hr_image_list=val_hr_images,
            lr_image_list=val_lr_images,
            hr_size=self.hr_size,
            stage="val",
            generator_type=self.generator_type,
        )
        self.test_dataset = ImageDataset(
            hr_image_list=test_hr_images,
            lr_image_list=test_lr_images,
            hr_size=self.hr_size,
            stage="test",
            generator_type=self.generator_type,
        )

    @staticmethod
    def search_for_images(data_path: str):
        logging.info(f"Searching for images under: '{data_path}'")
        images_lookup = os.path.join(data_path, "images.txt")

        if os.path.exists(images_lookup):
            logging.info("Using lookup file.")
            with open(images_lookup, "r") as f:
                images = f.readlines()
                return [i.replace("\n", "") for i in images]

        logging.info("Lookup file not found... Searching for images...")
        glob_images = [
            glob(p, recursive=True)
            for p in [
                os.path.join(data_path, "**", ext)
                for ext in [
                    ".jpeg",
                    "*.jpg",
                    "*.png",
                    ".bmp",
                    ".JPEG",
                    ".JPG",
                    ".PNG",
                    ".BMP",
                ]
            ]
        ]
        images = []
        for img_list in glob_images:
            images.extend(img_list)

        logging.info(
            f"Found {len(images)} images under '{data_path}'. Saving lookup file."
        )

        with open(images_lookup, "w") as f:
            f.writelines(sorted([f"{i}\n" for i in images]))

        return sorted(images)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds datamodule specific arguments.

        :param parent_parser: The parent parser.
        :returns: The parser.
        """
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default="/media/xultaeculcis/2TB/datasets/sr/pre-training/classic/",
        )
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--hr_size", type=int, default=128)
        parser.add_argument("--scale_factor", type=int, default=4)
        parser.add_argument("--seed", type=int, default=42)
        return parser


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    args = parser.parse_args()

    dm = SuperResolutionDataModule(
        data_path=args.data_path,
        generator_type="esrgan",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_size=args.hr_size,
        scale_factor=args.scale_factor,
        seed=args.seed,
    )

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    def matplotlib_imshow(batch):
        # create grid of images
        img_grid = torchvision.utils.make_grid(batch, nrow=4, normalize=True, padding=0)
        # show images
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    img_grid = None

    for _, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        lr = batch["lr"]
        hr = batch["hr"]
        matplotlib_imshow(lr)
        matplotlib_imshow(hr)
        break

    for _, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
        lr = batch["lr"]
        hr = batch["hr"]
        sr_bicubic = batch["bicubic"]
        matplotlib_imshow(lr)
        matplotlib_imshow(hr)
        matplotlib_imshow(sr_bicubic)
        break
