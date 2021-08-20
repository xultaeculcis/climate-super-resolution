# -*- coding: utf-8 -*-
import torch

from climsr.models.drln import DRLN


def test_should_return_tensor_with_correct_shape_after_forward():
    # arrange
    lr_batch = (32, 3, 32, 32)
    expected = (32, 3, 128, 128)
    model = DRLN(in_channels=3, out_channels=3, scaling_factor=4).cuda()
    x = torch.rand(lr_batch).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x)

    # assert
    assert out.shape == expected
