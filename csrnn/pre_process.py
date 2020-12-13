import os
from PIL import Image
from glob import glob
from dask.distributed import Client
from dask.bag import from_sequence


def dasked_convert(img_path, subdirs, output_path):
    stage = subdirs[0] if "train_images" in img_path else subdirs[1]
    outfile = os.path.join(output_path, stage, os.path.basename(img_path))[:-3] + "jpg"
    im = Image.open(img_path)
    out = im.convert("RGB")
    out.save(outfile, "JPEG", quality=95)


if __name__ == '__main__':
    subdirs = ["train", "val"]
    images = glob("../../xview/datasets/**/*.tif", recursive=True)
    output_path = "../datasets/original/xview/"

    # TODO make sure to remove corrupted images afterwards
    for d in subdirs:
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

    c = Client()
    from_sequence(images).map(dasked_convert, subdirs, output_path).compute()
    c.close()
