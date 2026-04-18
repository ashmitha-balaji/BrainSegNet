"""
brainsegnet.py  —  BrainSegNet Full Model + Teacher-Student Wrapper
====================================================================
Author : Yuktaa Sri Addanki
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.maca    import MACA3D
from models.encoder import Encoder3D
from models.vae     import VAEBottleneck
from models.gan     import FeatureGenerator, PatchGANDiscriminator
from models.decoder import Decoder3D


class BrainSegNet(nn.Module):
    def __init__(self, in_channels=4, n_classes=4, base_filters=32,
                 crop_size=96, latent_dim=128, beta_vae=0.1, use_gan=True):
        super().__init__()
        f           = base_filters
        sp          = crop_size // 16
        self.use_gan= use_gan

        self.maca        = MACA3D(in_channels, hidden_dim=32)
        self.encoder     = Encoder3D(in_channels, base_filters=f)
        self.vae         = VAEBottleneck(f*16, latent_dim, sp, beta_vae)

        if use_gan:
            self.generator     = FeatureGenerator(latent_dim, in_channels, f*16, sp)
            self.discriminator = PatchGANDiscriminator(f*16)

        self.decoder = Decoder3D(f*16, [f*8, f*4, f*2, f], n_classes, f)

    def forward(self, x, modality_mask, training=True):
        x = self.maca(x, modality_mask)
        bottleneck, skips = self.encoder(x)
        vae_out, kl, mu, logvar = self.vae(bottleneck)

        if self.use_gan and training:
            gen_out = self.generator(mu, modality_mask)
            merged  = (vae_out + gen_out) * 0.5
        else:
            gen_out = vae_out
            merged  = vae_out

        main, aux3, aux2 = self.decoder(merged, skips, training)

        if training:
            return main, aux3, aux2, kl, gen_out, bottleneck
        return main
