# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sr.models.srcnn import SRCNN


class ESRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_conv_block=4):
        super(ESRGANDiscriminator, self).__init__()

        block = []

        for _ in range(num_conv_block):
            block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(out_channels),
            ]
            in_channels = out_channels

            block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, 2),
                nn.LeakyReLU(),
            ]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3),
        ]

        self.feature_extraction = nn.Sequential(*block)

        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classification = nn.Sequential(nn.Linear(8192, 100), nn.Linear(100, 1))

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()

        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class ResidualInResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ESRGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
        scale_factor: int = 4,
    ):
        super(ESRGANGenerator, self).__init__()

        self.scale_factor = scale_factor

        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(
            *[ResidualInResidualDenseBlock(nf=nf, gc=gc) for _ in range(nb)]
        )
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        if self.scale_factor == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.srcnn = SRCNN(in_channels=2, out_channels=out_channels)

    def forward(self, x: Tensor, elev: Tensor):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )

        if self.scale_factor == 4:
            fea = self.lrelu(
                self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out = self.srcnn(torch.cat([out, elev], 1))

        return out
