import os
import random
from argparse import ArgumentParser
from glob import glob
import logging
from typing import Union, Optional, List, Tuple

import numpy as np

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from tqdm import tqdm

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

logging.basicConfig(level=logging.INFO)


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(
            self,
            image_list: List[str],
            hr_shape: Optional[Tuple[int, int]] = (224, 224),
            scaling_factor: Optional[int] = 4,
    ):
        hr_height, hr_width = hr_shape
        self.hr_shape = hr_shape
        self.common_transforms = transforms.Compose(
            [
                transforms.RandomCrop(hr_shape),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scaling_factor, hr_height // scaling_factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.image_list = image_list

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = self.common_transforms(img)
        img_lr = self.lr_transform(img)
        img_hr = self.lr_transform(img)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.image_list)


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root_data_path: str,
            scaling_factor: Optional[int] = 4,
            batch_size: Optional[int] = 32,
            num_workers: Optional[int] = 8,
            hr_shape: Optional[Tuple[int, int]] = (224, 224),
            seed: Optional[int] = 42,
    ):
        super(SuperResolutionDataModule, self).__init__()
        self.root_data_path = root_data_path
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_shape = hr_shape
        self.seed = seed

        glob_images = [
            glob(p, recursive=True) for p in [os.path.join(root_data_path, "**", ext) for ext in ["*.jpg", "*.png"]]
        ]
        images = []
        for img_list in glob_images:
            images.extend(img_list)

        logging.info(f"Total of {len(images)} found under the {root_data_path}")

        train_images = []
        val_images = []
        for img_path in images:
            if "val_images" in img_path or "valid_HR" in img_path:
                val_images.append(img_path)
            else:
                train_images.append(img_path)

        logging.info(f"Train/Validation split sizes: {len(train_images)}/{len(val_images)}")

        self.train_dataset = ImageDataset(
            image_list=train_images,
            hr_shape=self.hr_shape,
            scaling_factor=self.scaling_factor
        )
        self.val_dataset = ImageDataset(
            image_list=val_images,
            hr_shape=self.hr_shape,
            scaling_factor=self.scaling_factor
        )

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

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds datamodule specific arguments.

        :param parent_parser: The parent parser.
        :returns: The parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument('--root_data_path', type=str, default="../datasets/original")
        parser.add_argument('--scaling_factor', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--hr_shape', type=Tuple[int, int], default=(224, 224))
        parser.add_argument('--seed', type=int, default=42)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = SuperResolutionDataModule.add_data_specific_args(parser)
    args = parser.parse_args()

    dm = SuperResolutionDataModule(
        root_data_path=args.root_data_path,
        scaling_factor=args.scaling_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_shape=args.hr_shape,
        seed=args.seed,
    )
    for idx, batch in tqdm(enumerate(dm.train_dataloader())):
        pass
