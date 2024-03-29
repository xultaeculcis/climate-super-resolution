# -*- coding: utf-8 -*-
import torch

from climsr.models.rcan import RCAN


def test_should_return_tensor_with_correct_shape_after_forward():
    # arrange
    lr_batch_size = (32, 3, 32, 32)
    hr_elev_batch_size = (32, 1, 128, 128)
    expected = (32, 1, 128, 128)
    model = RCAN(
        in_channels=3,
        out_channels=1,
        scaling_factor=4,
    ).cuda()
    elev = torch.rand(hr_elev_batch_size).cuda()
    mask = torch.rand(hr_elev_batch_size).cuda()
    x = torch.rand(lr_batch_size).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x, elev, mask)

    # assert
    assert out.shape == expected
