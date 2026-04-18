"""
encoder.py  —  3-D Dense CNN Encoder
=====================================
Author : Anurag Sharma Josyula
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.

COURSE TOPIC: CNN, Lectures 6-7
"""

import torch
import torch.nn as nn


class ConvBnRelu3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenseBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu3D(in_ch,          out_ch)
        self.conv2 = ConvBnRelu3D(in_ch + out_ch, out_ch)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(torch.cat([x, h1], dim=1))
        return h2


class Encoder3D(nn.Module):
    def __init__(self, in_channels=4, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1       = DenseBlock3D(in_channels, f)
        self.enc2       = DenseBlock3D(f,           f * 2)
        self.enc3       = DenseBlock3D(f * 2,       f * 4)
        self.enc4       = DenseBlock3D(f * 4,       f * 8)
        self.bottleneck = DenseBlock3D(f * 8,       f * 16)
        self.pool       = nn.MaxPool3d(2, 2)
        self.skip_channels       = [f, f*2, f*4, f*8]
        self.bottleneck_channels = f * 16

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return b, [e4, e3, e2, e1]
