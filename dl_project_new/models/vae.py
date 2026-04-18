"""
vae.py  —  Variational Autoencoder Bottleneck
==============================================
Author : Arpana Singh
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.

COURSE TOPIC: Autoencoder + VAE, Lecture 9
"""

import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    def __init__(self, in_channels=512, latent_dim=128,
                 spatial_size=6, beta=0.1):
        super().__init__()
        self.in_ch = in_channels
        self.lat   = latent_dim
        self.sp    = spatial_size
        self.beta  = beta
        flat       = in_channels * spatial_size ** 3

        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)
        self.refine    = nn.Conv3d(in_channels, in_channels, 1)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def kl_loss(self, mu, logvar):
        return self.beta * (-0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()))

    def forward(self, x):
        B   = x.shape[0]
        xf  = x.flatten(1)
        mu  = self.fc_mu(xf)
        lv  = self.fc_logvar(xf)
        z   = self.reparameterise(mu, lv)
        dec = self.fc_decode(z).view(B, self.in_ch, self.sp, self.sp, self.sp)
        return self.refine(dec), self.kl_loss(mu, lv), mu, lv
