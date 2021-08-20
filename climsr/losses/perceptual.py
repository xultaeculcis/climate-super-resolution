# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19


class PerceptualLoss(nn.Module):
    """
    Represents perceptual loss. Assumes input images to be of shape Nx1xHxW (single channel images).
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, fake_high_resolution, high_resolution):
        with torch.no_grad():
            # create 3-channel inputs for thr VGG network as it was pre-trained on RGB images
            # we assume input of size NxCxHxW, where N = batch size, C = channels, H & W = image height and width
            fake_high_resolution_3d = torch.cat(
                [fake_high_resolution, fake_high_resolution, fake_high_resolution],
                dim=1,
            )
            high_resolution_3d = torch.cat(
                [high_resolution, high_resolution, high_resolution], dim=1
            )

            perception_loss = self.l1_loss(
                self.loss_network(high_resolution_3d),
                self.loss_network(fake_high_resolution_3d),
            )
            return perception_loss
