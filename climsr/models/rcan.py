# -*- coding: utf-8 -*-
import logging
import math
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from climsr.models.srcnn import SRCNN


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias
    )


class Upsampler(nn.Sequential):
    def __init__(
        self,
        conv: nn.Module,
        scale: int,
        n_feat: int,
        bn: bool = False,
        act: Union[nn.Module, bool] = False,
        bias: bool = True,
    ):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    """Channel Attention (CA) Layer"""

    def __init__(self, channel: int, reduction: int = 16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB)"""

    def __init__(
        self,
        conv: nn.Module,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool = True,
        bn: bool = False,
        res_scale: int = 1,
    ):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    """Residual Group (RG)"""

    def __init__(
        self,
        conv: nn.Module,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        n_resblocks: int,
    ):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    """Residual Channel Attention Network (RCAN)"""

    def __init__(
        self,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        reduction: int = 16,
        scaling_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 1,
        conv: nn.Module = default_conv,
    ):
        super(RCAN, self).__init__()

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.kernel_size = 3
        self.reduction = reduction
        self.scaling_factor = scaling_factor

        # define head module
        modules_head = [conv(in_channels, n_feats, self.kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, self.kernel_size, reduction, n_resblocks=n_resblocks
            )
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, self.kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scaling_factor, n_feats, act=False),
            conv(n_feats, out_channels, self.kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.srcnn = SRCNN(in_channels=3, out_channels=out_channels)

    def forward(self, x: Tensor, elev: Tensor, mask: Tensor) -> Tensor:
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        x = self.srcnn(torch.cat([x, elev, mask], 1))

        return x

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") >= 0:
                        logging.info("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
