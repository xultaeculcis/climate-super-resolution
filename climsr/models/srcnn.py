# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)

        return out
