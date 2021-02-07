import os
from collections import Counter
from glob import glob

from PIL import Image
from dask import bag
from distributed import Client
import logging

logging.basicConfig(level=logging.INFO)


def validate_image(img_path, min_size=128):
    def remove_image(image_path):
        os.remove(image_path)

    try:
        img = Image.open(img_path).convert("RGB")
        h, w = img.size

        if h < min_size or w < min_size:
            remove_image(img_path)
            return "Invalid size"
        return "Ok"
    except Exception:
        remove_image(img_path)
        return "Erred"


def get_images(data_path):
    glob_images = [
        glob(p, recursive=True) for p in [
            os.path.join(data_path, "**", ext) for ext in [
                ".jpeg", "*.jpg", "*.png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP",
            ]
        ]
    ]
    images = []
    for img_list in glob_images:
        images.extend(img_list)

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

    return train_images, val_images, test_images


if __name__ == '__main__':
    c = Client(n_workers=8, threads_per_worker=1)

    train, val, test = get_images("/media/xultaeculcis/2TB/datasets/sr/original/pre-training/")

    logging.info(f"Train: {len(train)}")
    logging.info(f"Val: {len(val)}")
    logging.info(f"Test: {len(test)}")

    results = bag.from_sequence(train, npartitions=1000).map(validate_image, min_size=128).compute()
    logging.info(f"Train: {Counter(results)}")

    results = bag.from_sequence(val, npartitions=1000).map(validate_image, min_size=128).compute()
    logging.info(f"Val: {Counter(results)}")

    # results = bag.from_sequence(test, npartitions=1000).map(validate_image, min_size=128).compute()
    # logging.info(f"Test: {Counter(results)}")


    c.close()

    logging.info("DONE.")
