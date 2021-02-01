import glob
import os
import logging

from PIL import Image
from dask import bag
from dask.diagnostics import ProgressBar
from distributed import Client

logging.basicConfig(level=logging.INFO)

pbar = ProgressBar()
pbar.register()


def get_tifs(data_dir):
    pattern = os.path.join(data_dir, "**/*.tif")
    imgs = glob.glob(pattern, recursive=True)
    return imgs


def to_jpg(img_path, data_dir):
    out_path = os.path.join(data_dir, "xview_train" if "/train_images/" in img_path else "xview_val")

    os.makedirs(out_path, exist_ok=True)

    img_name = os.path.basename(img_path).replace(".tif", ".jpg")

    out_filename = os.path.join(out_path, img_name)

    img = Image.open(img_path).convert("RGB")
    img.save(out_filename, "jpeg")


if __name__ == '__main__':
    data_dir = "/media/xultaeculcis/2TB/datasets/sr/original/archives/xview/"
    tifs = get_tifs(data_dir)

    logging.info(f"Total of {len(tifs)} images found under {data_dir}")

    c = Client()

    bag.from_sequence(tifs).map(to_jpg, data_dir).compute()

    c.close()

    logging.info("DONE.")
