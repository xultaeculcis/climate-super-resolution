# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class GBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(GBasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation, groups=4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.Sigmoid())

    def forward(self, x):
        out = self.body(x)
        return out


class GBasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(GBasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class GResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class ConvertBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ConvertBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels * blocks, out_channels * blocks // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * blocks // 2, out_channels * blocks // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * blocks // 4, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        # out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                    nn.ReLU(inplace=True),
                ]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
            ]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel, (channel // reduction) * 3, ksize=1, stride=1, pad=0, dilation=1)
        self.c4 = BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c_out = self.c1(y)
        y = self.c4(c_out)
        return x * y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ResidualBlock(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out


class DRLN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, scaling_factor: int = 4, **kwargs):
        super(DRLN, self).__init__()

        self.scale = scaling_factor
        channels = 64

        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)

        self.b1 = Block(channels, channels)
        self.b2 = Block(channels, channels)
        self.b3 = Block(channels, channels)
        self.b4 = Block(channels, channels)
        self.b5 = Block(channels, channels)
        self.b6 = Block(channels, channels)
        self.b7 = Block(channels, channels)
        self.b8 = Block(channels, channels)
        self.b9 = Block(channels, channels)
        self.b10 = Block(channels, channels)
        self.b11 = Block(channels, channels)
        self.b12 = Block(channels, channels)
        self.b13 = Block(channels, channels)
        self.b14 = Block(channels, channels)
        self.b15 = Block(channels, channels)
        self.b16 = Block(channels, channels)
        self.b17 = Block(channels, channels)
        self.b18 = Block(channels, channels)
        self.b19 = Block(channels, channels)
        self.b20 = Block(channels, channels)

        self.c1 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c2 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c3 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c4 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c5 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c6 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c7 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c8 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c9 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c10 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c11 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c12 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c13 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c14 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c15 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c16 = BasicBlock(channels * 5, channels, 3, 1, 1)
        self.c17 = BasicBlock(channels * 2, channels, 3, 1, 1)
        self.c18 = BasicBlock(channels * 3, channels, 3, 1, 1)
        self.c19 = BasicBlock(channels * 4, channels, 3, 1, 1)
        self.c20 = BasicBlock(channels * 5, channels, 3, 1, 1)

        self.upsample = UpsampleBlock(channels, self.scale, multi_scale=False)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)

        b5 = self.b5(a1)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)

        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3

        b13 = self.b13(a4)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)

        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4

        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)

        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        a6 = o20 + a5

        b_out = a6 + x
        out = self.upsample(b_out, scale=self.scale)

        out = self.tail(out)

        return out
