# -*- coding: utf-8 -*-
import torchvision
from matplotlib import pyplot as plt

import climsr.consts as consts


def matplotlib_imshow(batch, title=None, nrow=8, normalize=False, padding=2, truncate=88):
    # create grid of images
    img_grid = torchvision.utils.make_grid(batch[:truncate], nrow=nrow, normalize=normalize, padding=padding)

    # img_grid has 3 channels - they have the same values
    # as they were created using 1 channel data
    # we can select only one of those channels
    npimg = img_grid.numpy()[0, :, :]

    im_ratio = npimg.shape[0] / npimg.shape[1]

    # show images
    c = plt.imshow(npimg, cmap="jet")
    plt.colorbar(c, fraction=0.047 * im_ratio)
    plt.title(title, fontweight="bold")
    plt.show()


def plot_single_batch(loader, keys):
    for _, batch in enumerate(loader):
        for key in keys:
            images = batch[key]

            matplotlib_imshow(images, key)

        return batch


def get_variable_from_ds_fp(fp):
    if f".{consts.cruts.pre}." in fp:
        return consts.cruts.pre

    if f".{consts.cruts.tmn}." in fp:
        return consts.cruts.tmn

    if f".{consts.cruts.tmp}." in fp:
        return consts.cruts.tmp

    if f".{consts.cruts.tmx}." in fp:
        return consts.cruts.tmx


def plot_array(arr):
    plt.figure(figsize=(20, 10))
    plt.imshow(arr, cmap="jet")
    plt.show()
