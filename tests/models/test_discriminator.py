# -*- coding: utf-8 -*-
import torch

from climsr.models.discriminator import Discriminator
from climsr.models.rfb_esrgan import RFBESRGANDiscriminator


def test_should_return_tensor_with_correct_shape_after_forward_esrgan():
    input_shape = (64, 1, 128, 128)
    expected = (64, 1)
    model = Discriminator(in_channels=1).cuda()

    x = torch.rand(input_shape).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x)

    assert out.shape == expected


def test_should_return_tensor_with_correct_shape_after_forward_rbf_esrgan():
    input_shape = (64, 1, 452, 452)
    expected = (64, 1)
    model = RFBESRGANDiscriminator(in_channels=1).cuda()

    x = torch.rand(input_shape).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x)

    assert out.shape == expected
