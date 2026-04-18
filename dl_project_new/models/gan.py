"""
gan.py  —  Conditional GAN for Missing-Modality Feature Synthesis
==================================================================
Author : Ashmitha Paruchuri Balaji
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.

COURSE TOPIC: GAN, Lecture 9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    def __init__(self, latent_dim=128, n_modalities=4,
                 feat_channels=512, spatial_size=6):
        super().__init__()
        self.fc   = feat_channels
        self.sp   = spatial_size
        flat      = feat_channels * spatial_size ** 3

        self.mlp  = nn.Sequential(
            nn.Linear(latent_dim + n_modalities, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512),                       nn.ReLU(inplace=True),
            nn.Linear(512, flat),                      nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv3d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(feat_channels), nn.Tanh(),
        )

    def forward(self, z, modality_mask):
        B    = z.shape[0]
        feat = self.mlp(torch.cat([z, modality_mask], dim=1))
        feat = feat.view(B, self.fc, self.sp, self.sp, self.sp)
        return self.refine(feat)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def generator_loss(fake_pred, fake_feat, real_feat, lam=10.0):
    adv = F.binary_cross_entropy_with_logits(
        fake_pred, torch.ones_like(fake_pred))
    return adv + lam * F.l1_loss(fake_feat, real_feat)


def discriminator_loss(real_pred, fake_pred):
    rl = F.binary_cross_entropy_with_logits(
        real_pred, torch.ones_like(real_pred))
    fl = F.binary_cross_entropy_with_logits(
        fake_pred, torch.zeros_like(fake_pred))
    return (rl + fl) * 0.5
