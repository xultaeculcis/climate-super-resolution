# -*- coding: utf-8 -*-
import torch

from sr.models.esrgan import ESRGANGenerator


def test_should_return_tensor_with_correct_shape_after_forward():
    # arrange
    lr_batch = (32, 3, 32, 32)
    expected = (32, 3, 128, 128)
    model = ESRGANGenerator(3, 3)
    x = torch.rand(lr_batch)

    # act
    out = model.forward(x)

    # assert
    assert out.shape == expected
