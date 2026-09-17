"""Building blocks for the demo UNet."""

import torch
import torch.nn.functional as F
from torch import nn

import frereg as fr


class DoubleConv(nn.Module):
    """Two frequency-regularized convolutions with normalization and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            fr.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                minrate=0.001,
                droprate=0.1,
            ),
            nn.BatchNorm2d(mid_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            fr.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                minrate=0.001,
                droprate=0.1,
            ),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block consisting of max pooling and :class:`DoubleConv`."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with a skip connection."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = fr.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
                minrate=0.001,
                droprate=0.1,
                bias=False,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Final frequency-regularized one-by-one convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = fr.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            minrate=0.001,
            droprate=0.1,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)
