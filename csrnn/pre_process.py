import os
from PIL import Image
from glob import glob
from dask.distributed import Client
from dask.bag import from_sequence
import time
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
MIN_WIDTH = 512
MIN_HEIGHT = 512


def convert_image(img_path, subdirs, output_path):
    """Converts the image and saves the converted image in train/val dirs"""
    stage = subdirs[0] if "train_images" in img_path else subdirs[1]
    outfile = os.path.join(output_path, stage, os.path.basename(img_path))[:-3] + "jpg"
    im = Image.open(img_path)
    out = im.convert("RGB")
    out.save(outfile, "JPEG", quality=95)


def ensure_min_dimensions(img_path):
    """Removes the image if it does not have min required dimensions (HxW)"""
    try:
        img = Image.open(img_path)
        if img.width < MIN_WIDTH:
            os.remove(img_path)
        if img.height < MIN_HEIGHT:
            os.remove(img_path)
    except Exception:
        pass


def pre_process_xview():
    """
    Processes XView dataset (tif -> jpg)
    """
    subdirs = ["train", "val"]
    images = glob("../../xview/datasets/**/*.tif", recursive=True)
    output_path = "../datasets/original/xview/"

    # TODO make sure to remove corrupted images afterwards
    for d in subdirs:
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

    c = Client()
    from_sequence(images).map(convert_image, subdirs, output_path).compute()
    c.close()


def size_check(base_dir: str):
    """Checks the dimensions of images in the specified dir"""
    image_lists = [
        glob(f"{base_dir}/**/{ext}", recursive=True) for ext in ["*.jpg", "*.png"]
    ]

    images = []
    for image_list in image_lists:
        images.extend(image_list)

    logging.info(f"Found {len(images)} images to process in {base_dir}")
    for img in tqdm(images):
        ensure_min_dimensions(img)
    logging.info(f"Done for {base_dir}")


if __name__ == '__main__':
    datasets_path = "/home/xultaeculcis/repos/super-resolution/datasets/original/pre-training"
    # pre_process_xview()
    for directory in [os.path.join(datasets_path, folder) for folder in ["corals", "coco"]]:
        size_check(directory)
