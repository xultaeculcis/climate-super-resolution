# -*- coding: utf-8 -*-
from sr.pre_processing.world_clim_config import WorldClimConfig
from sr.lightning_modules.datamodules import SuperResolutionDataModule


class Args:
    batch_size = 32
    data_path = "./datasets"
    world_clim_variable = WorldClimConfig.tmin
    world_clim_multiplier = "4x"
    num_workers = 4
    hr_size = 128
    scale_factor = 4
    seed = 42


args = Args()

dm = SuperResolutionDataModule(
    data_path=args.data_path,
    world_clim_variable=args.world_clim_variable,
    world_clim_multiplier=args.world_clim_multiplier,
    generator_type="esrgan",
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    hr_size=args.hr_size,
    scale_factor=args.scale_factor,
    seed=args.seed,
)


def common_asserts(dl, stage="test"):
    for _, batch in enumerate(dl):
        lr = batch["lr"]
        hr = batch["hr"]
        sr_nearest = batch["nearest"]
        elevation = batch["elevation"]

        expected_lr_shape = (
            args.batch_size,
            1,
            args.hr_size // args.scale_factor,
            args.hr_size // args.scale_factor,
        )
        expected_hr_shape = (args.batch_size, 1, args.hr_size, args.hr_size)

        assert lr.shape == expected_lr_shape, (
            f"Expected the LR batch to be in shape {expected_lr_shape}, "
            f"but found: {lr.shape}"
        )
        assert hr.shape == (args.batch_size, 1, 128, 128), (
            f"Expected the HR batch to be in shape {expected_hr_shape}, "
            f"but found: {hr.shape}"
        )
        assert elevation.shape == (args.batch_size, 1, 128, 128), (
            f"Expected the Elev batch to be in shape {expected_hr_shape}, "
            f"but found: {elevation.shape}"
        )

        if stage != "train":
            assert sr_nearest.shape == (args.batch_size, 1, 128, 128), (
                f"Expected the SR batch to be in shape {expected_hr_shape}, "
                f"but found: {sr_nearest.shape}"
            )

        break


def test_train_dl():
    train_dl = dm.train_dataloader()
    common_asserts(train_dl, "train")


def test_val_dl():
    val_dl = dm.val_dataloader()
    common_asserts(val_dl)


def test_test_dl():
    test_dl = dm.test_dataloader()
    common_asserts(test_dl)