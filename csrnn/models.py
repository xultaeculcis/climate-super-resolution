"""
ESRGAN (Enhanced SRGAN)

@InProceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
"""
from math import log2

import torch.nn as nn
import torch
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class SRGANBlock(nn.Module):
    """
    Building block of SRGAN.
    """

    def __init__(self, dim):
        super(SRGANBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.net(x)


class SRResNet(nn.Module):
    """
    PyTorch Module for SRGAN, https://arxiv.org/pdf/1609.04802.
    """

    def __init__(self, scale_factor=4, ngf=64, n_blocks=16):
        super(SRResNet, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, ngf, kernel_size=9),
            nn.PReLU()
        )
        self.body = nn.Sequential(
            *[SRGANBlock(ngf) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
            nn.BatchNorm2d(ngf)
        )
        self.tail = nn.Sequential(
            UpscaleBlock(scale_factor, ngf, act='prelu'),
            nn.ReflectionPad2d(4),
            nn.Conv2d(ngf, 3, kernel_size=9),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return (x + 1) / 2


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(self, scale_factor, dim, act=None):
        assert scale_factor in [2, 3, 4, 8]

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * r * r, kernel_size=3),
                nn.PixelShuffle(r),
            ]

            if act == 'relu':
                layers += [nn.ReLU(True)]
            elif act == 'prelu':
                layers += [nn.PReLU()]

        super(UpscaleBlock, self).__init__(*layers)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class RRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(RRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Sequential):
    """
    Discriminator for SRGAN.
    Dense layers are replaced with global poolings and 1x1 convolutions.
    """

    def __init__(self, ndf):

        def ConvBlock(in_channels, out_channels, stride):
            out = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]
            return out

        super(Discriminator, self).__init__(
            nn.Conv2d(3, ndf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            *ConvBlock(ndf, ndf, 2),

            *ConvBlock(ndf, ndf * 2, 1),
            *ConvBlock(ndf * 2, ndf * 2, 2),

            *ConvBlock(ndf * 2, ndf * 4, 1),
            *ConvBlock(ndf * 4, ndf * 4, 2),

            *ConvBlock(ndf * 4, ndf * 8, 1),
            *ConvBlock(ndf * 8, ndf * 8, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )
