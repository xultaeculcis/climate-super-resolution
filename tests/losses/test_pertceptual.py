# -*- coding: utf-8 -*-
import torch

from climsr.losses.perceptual import PerceptualLoss

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def test_should_return_zero_when_sr_and_hr_images_are_the_same():
    # Arrange
    sut = PerceptualLoss().to(device)
    hr = torch.rand(64, 1, 128, 128, dtype=torch.float32, device=device)
    sr = hr.detach().clone().to(device)

    # Act
    loss = sut.forward(sr, hr)

    # Assert
    assert loss == 0.0


def test_should_return_nonzero_when_sr_and_hr_images_are_different():
    # Arrange
    sut = PerceptualLoss().to(device)
    hr = torch.rand(64, 1, 128, 128, dtype=torch.float32, device=device)
    sr = torch.rand(64, 1, 128, 128, dtype=torch.float32, device=device)

    # Act
    loss = sut.forward(sr, hr)

    # Assert
    assert loss != 0.0
