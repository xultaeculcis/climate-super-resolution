# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Discriminator", "Generator", "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock",
    "ResidualOfReceptiveFieldDenseBlock", "ResidualDenseBlock", "ResidualInResidualDenseBlock"
]


class Discriminator(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # input is 3 x 216 x 216
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (64) x 108 x 108
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 54 x 54
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 27 x 27
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 14 x 14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))

        self.fc = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor, num_rrdb_blocks=2, num_rrfdb_blocks=2):
        r""" This is an esrgan model defined by the author himself.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
            num_rrdb_blocks (int): How many RRDB structures make up Trunk-A? (Default: 2).
            num_rrfdb_blocks (int): How many RRDB structures make up Trunk-RFB? (Default: 2).

        Notes:
            Use `num_rrdb_blocks` is 2  and `num_rrfdb_blocks` is 2 for RTX 2080.
            Use `num_rrdb_blocks` is 4  and `num_rrfdb_blocks` is 2 for RTX 2080Ti.
            Use `num_rrdb_blocks` is 8  and `num_rrfdb_blocks` is 4 for TITAN RTX.
            Use `num_rrdb_blocks` is 16 and `num_rrfdb_blocks` is 8 for Tesla A100.
        """
        super(Generator, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 16 ResidualInResidualDenseBlock layer
        residual_residual_dense_blocks = []
        for _ in range(num_rrdb_blocks):
            residual_residual_dense_blocks += [ResidualInResidualDenseBlock(64, 32, 0.2)]
        self.Trunk_A = nn.Sequential(*residual_residual_dense_blocks)

        # 8 ResidualInResidualDenseBlock layer
        residual_residual_fields_dense_blocks = []
        for _ in range(num_rrfdb_blocks):
            residual_residual_fields_dense_blocks += [ResidualOfReceptiveFieldDenseBlock(64, 32, 0.2)]
        self.Trunk_RFB = nn.Sequential(*residual_residual_fields_dense_blocks)

        # Second conv layer post residual field blocks
        self.RFB = ReceptiveFieldBlock(64, 64, non_linearity=False)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                ReceptiveFieldBlock(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                ReceptiveFieldBlock(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next layer after upper sampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        out1 = self.conv1(input)
        out = self.Trunk_A(out1)
        out2 = self.Trunk_RFB(out)
        out = torch.add(out1, out2)
        out = self.RFB(out)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class ReceptiveFieldBlock(nn.Module):
    r"""This structure is similar to the main building blocks in the GoogLeNet model.
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    """

    def __init__(self, in_channels, out_channels, scale_ratio=0.2, non_linearity=True):
        super(ReceptiveFieldBlock, self).__init__()
        channels = in_channels // 4
        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, (channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((channels // 4) * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) if non_linearity else None

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat((branch1, branch2, branch3, branch4), 1)
        out = self.conv1x1(out)

        out = out.mul(self.scale_ratio) + shortcut
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out


class ReceptiveFieldDenseBlock(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.RFB1 = ReceptiveFieldBlock(in_channels, growth_channels, scale_ratio)
        self.RFB2 = ReceptiveFieldBlock(in_channels + 1 * growth_channels, growth_channels, scale_ratio)
        self.RFB3 = ReceptiveFieldBlock(in_channels + 2 * growth_channels, growth_channels, scale_ratio)
        self.RFB4 = ReceptiveFieldBlock(in_channels + 3 * growth_channels, growth_channels, scale_ratio)
        self.RFB5 = ReceptiveFieldBlock(in_channels + 4 * growth_channels, in_channels, scale_ratio,
                                        non_linearity=False)

        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        rfb1 = self.RFB1(input)
        rfb2 = self.RFB2(torch.cat((input, rfb1), 1))
        rfb3 = self.RFB3(torch.cat((input, rfb1, rfb2), 1))
        rfb4 = self.RFB4(torch.cat((input, rfb1, rfb2, rfb3), 1))
        rfb5 = self.RFB5(torch.cat((input, rfb1, rfb2, rfb3, rfb4), 1))

        return rfb5.mul(self.scale_ratio) + input


class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, in_channels=64, growth_channels=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB2 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB3 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        out = self.RFDB1(input)
        out = self.RFDB2(out)
        out = self.RFDB3(out)

        return out.mul(self.scale_ratio) + input


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1, bias=False)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input
