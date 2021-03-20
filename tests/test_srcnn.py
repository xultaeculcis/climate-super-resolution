# -*- coding: utf-8 -*-
import torch

from sr.models.srcnn import SRCNN


def test_should_return_tensor_with_correct_shape_after_forward():
    expected = (32, 3, 32, 32)
    model = SRCNN(3, 3).cuda()

    x = torch.rand(expected).cuda()

    # act
    with torch.no_grad():
        out = model.forward(x)

    assert out.shape == expected
