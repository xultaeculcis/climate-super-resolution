# -*- coding: utf-8 -*-
import torch

from climsr.models.esrgan import ESRGANGenerator


def test_should_return_tensor_with_correct_shape_after_forward():
    # arrange
    hr_elev_batch_size = (32, 1, 128, 128)
    lr_batch = (32, 2, 32, 32)
    expected = (32, 1, 128, 128)
    model = ESRGANGenerator(2, 1).cuda()
    x = torch.rand(lr_batch).cuda()
    elev = torch.rand(hr_elev_batch_size).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x, elev)

    # assert
    assert out.shape == expected
