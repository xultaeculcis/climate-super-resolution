# -*- coding: utf-8 -*-
from core.config import SuperResolutionDataConfig
from data.super_resolution_data_module import SuperResolutionDataModule

import climsr.consts as consts

args = SuperResolutionDataConfig()

dm = SuperResolutionDataModule(args)


def common_asserts(dl, stage=consts.stages.test):
    for _, batch in enumerate(dl):
        lr = batch[consts.batch_items.lr]
        hr = batch[consts.batch_items.hr]
        elevation = batch[consts.batch_items.elevation]

        expected_lr_shape = (
            args.batch_size,
            3,
            args.hr_size // args.scale_factor,
            args.hr_size // args.scale_factor,
        )
        expected_hr_shape = (args.batch_size, 1, args.hr_size, args.hr_size)

        assert lr.shape == expected_lr_shape, (
            f"Expected the LR batch to be in shape {expected_lr_shape}, " f"but found: {lr.shape}"
        )
        assert hr.shape == (args.batch_size, 1, 128, 128), (
            f"Expected the HR batch to be in shape {expected_hr_shape}, " f"but found: {hr.shape}"
        )
        assert elevation.shape == (args.batch_size, 1, 128, 128), (
            f"Expected the Elev batch to be in shape {expected_hr_shape}, " f"but found: {elevation.shape}"
        )

        if stage != consts.stages.train:
            sr_nearest = batch[consts.batch_items.nearest]

            assert sr_nearest.shape == (args.batch_size, 1, 128, 128), (
                f"Expected the SR batch to be in shape {expected_hr_shape}, " f"but found: {sr_nearest.shape}"
            )

        break


def test_train_dl():
    train_dl = dm.train_dataloader()
    common_asserts(train_dl, consts.stages.train)


def test_val_dl():
    val_dl = dm.val_dataloader()
    common_asserts(val_dl)


def test_test_dl():
    test_dls = dm.test_dataloader()
    for test_dl in test_dls:
        common_asserts(test_dl)
