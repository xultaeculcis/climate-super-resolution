import os
from argparse import ArgumentParser
from glob import glob
import logging
from typing import Optional, List, Tuple

import numpy as np

import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(
            self,
            image_list: List[str],
            hr_size: Optional[int] = 128,
            scaling_factor: Optional[int] = 4,
            stage: Optional[str] = "train"
    ):
        self.hr_size = hr_size
        self.stage = stage
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(hr_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_size // scaling_factor, hr_size // scaling_factor), Image.BICUBIC),
            ]
        )
        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.CenterCrop(hr_size)
            ]
        )

        self.image_list = image_list

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img = Image.open(self.image_list[index]).convert(mode="RGB")
        if self.stage == "train":
            img = self.train_transforms(img)
        else:
            img = self.val_transforms(img)

        img_lr = self.common_transforms(self.lr_transform(img))
        img_hr = self.common_transforms(img)

        return img_lr, img_hr

    def __len__(self) -> int:
        return len(self.image_list)


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            scaling_factor: Optional[int] = 4,
            batch_size: Optional[int] = 32,
            num_workers: Optional[int] = 4,
            hr_size: Optional[int] = 256,
            seed: Optional[int] = 42,
    ):
        super(SuperResolutionDataModule, self).__init__()

        assert hr_size % scaling_factor == 0

        self.data_path = data_path
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.seed = seed

        glob_images = [
            glob(p, recursive=True) for p in [
                os.path.join(data_path, "**", ext) for ext in [
                    ".jpeg", "*.jpg", "*.png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP"
                ]
            ]
        ]
        images = []
        for img_list in glob_images:
            images.extend(img_list)

        logging.info(f"Total of {len(images)} found under the {data_path}")

        train_images = []
        val_images = []
        test_images = []
        for img_path in images:
            if "/val/" in img_path:
                val_images.append(img_path)
            elif "/test/" in img_path:
                test_images.append(img_path)
            else:
                train_images.append(img_path)

        logging.info(f"Train/Validation split sizes: {len(train_images)}/{len(val_images)}")

        self.train_dataset = ImageDataset(
            image_list=train_images,
            hr_size=self.hr_size,
            scaling_factor=self.scaling_factor,
            stage="train"
        )
        self.val_dataset = ImageDataset(
            image_list=val_images,
            hr_size=self.hr_size,
            scaling_factor=self.scaling_factor,
            stage="val"
        )
        self.test_dataset = ImageDataset(
            image_list=test_images,
            hr_size=self.hr_size,
            scaling_factor=self.scaling_factor,
            stage="test"
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
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument(
            '--data_path', type=str, default="/media/xultaeculcis/2TB/datasets/sr/original/pre-training/")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--hr_size', type=int, default=128)
        parser.add_argument('--scale_factor', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        return parser


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_images = sorted(
        glob("/media/xultaeculcis/2TB/datasets/sr/original/pre-training/val/road_testing/**/*.png", recursive=True))
    test_dataloader = DataLoader(ImageDataset(test_images), batch_size=4, num_workers=1, shuffle=False)


    def matplotlib_imshow(batch):
        # create grid of images
        img_grid = torchvision.utils.make_grid(batch, nrow=2, normalize=True)
        # show images
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    print(len(test_images))

    img_grid = None
    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        lr, hr = batch
        matplotlib_imshow(lr)
        matplotlib_imshow(hr)
        break
