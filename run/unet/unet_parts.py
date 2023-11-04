import torch
import torch.nn as nn
import torch.nn.functional as F
import frereg as fr


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            fr.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, minrate=0.001, droprate=0.1),
            nn.BatchNorm2d(mid_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            fr.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, minrate=0.001, droprate=0.1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = fr.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, minrate=0.001, droprate=0.1, bias=False)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = fr.Conv2d(in_channels, out_channels, kernel_size=1, minrate=0.001, droprate=0.1, bias=False)

    def forward(self, x):
        return self.conv(x)
