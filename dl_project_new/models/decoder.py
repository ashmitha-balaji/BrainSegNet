"""
decoder.py  —  3-D Attention U-Net Decoder + Deep Supervision
==============================================================
Author : Ashmitha Paruchuri Balaji
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.

COURSE TOPIC: CNN, Semantic Segmentation, Lectures 7-8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3D(nn.Module):
    def __init__(self, feat_ch, gate_ch, inter_ch):
        super().__init__()
        self.Wx  = nn.Conv3d(feat_ch,  inter_ch, 1)
        self.Wg  = nn.Conv3d(gate_ch,  inter_ch, 1)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )
        self.bn  = nn.BatchNorm3d(feat_ch)

    def forward(self, x, g):
        g_up  = F.interpolate(g, size=x.shape[2:],
                              mode='trilinear', align_corners=False)
        alpha = self.psi(self.Wx(x) + self.Wg(g_up))
        return self.bn(x * alpha)


class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.attn = AttentionGate3D(skip_ch, in_ch // 2, skip_ch // 2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch // 2 + skip_ch, out_ch,
                      3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        up  = self.up(x)
        att = self.attn(skip, up)
        return self.conv(torch.cat([up, att], dim=1))


class AuxHead(nn.Module):
    def __init__(self, in_ch, n_cls=4):
        super().__init__()
        self.head = nn.Conv3d(in_ch, n_cls, 1)

    def forward(self, x, tgt_size=96):
        logits = self.head(x)
        if logits.shape[2] != tgt_size:
            logits = F.interpolate(
                logits, size=(tgt_size, tgt_size, tgt_size),
                mode='trilinear', align_corners=False)
        return logits


class Decoder3D(nn.Module):
    def __init__(self, bottleneck_ch=512, skip_channels=None,
                 n_classes=4, base_filters=32):
        super().__init__()
        if skip_channels is None:
            f = base_filters
            skip_channels = [f*8, f*4, f*2, f]
        f = base_filters

        self.dec4 = DecoderBlock3D(bottleneck_ch, skip_channels[0], f*8)
        self.dec3 = DecoderBlock3D(f*8,           skip_channels[1], f*4)
        self.dec2 = DecoderBlock3D(f*4,           skip_channels[2], f*2)
        self.dec1 = DecoderBlock3D(f*2,           skip_channels[3], f)
        self.aux3 = AuxHead(f*4, n_classes)
        self.aux2 = AuxHead(f*2, n_classes)
        self.main = nn.Conv3d(f, n_classes, 1)

    def forward(self, bottleneck, skips, training=True):
        d4   = self.dec4(bottleneck, skips[0])
        d3   = self.dec3(d4,         skips[1])
        d2   = self.dec2(d3,         skips[2])
        d1   = self.dec1(d2,         skips[3])
        main = self.main(d1)
        if training:
            sz = main.shape[2]
            return main, self.aux3(d3, sz), self.aux2(d2, sz)
        return main, None, None
