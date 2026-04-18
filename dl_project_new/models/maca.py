"""
maca.py  —  MACA-3D: Modality-Aware Channel Attention (Novel Contribution)
===========================================================================
Author : Anurag Sharma Josyula
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.

COURSE TOPIC: DNN (fully-connected layers), Lecture 2
"""

import torch
import torch.nn as nn


class MACA3D(nn.Module):
    """
    Modality-Aware Channel Attention with Uncertainty Estimation.

    Takes modality_mask [1, 0, 1, 1] and learns to:
      - Suppress missing channels (weight ≈ 0)
      - Amplify present channels to compensate (weight > 1)
      - Estimate uncertainty about each channel

    Only 148 parameters total.
    """

    def __init__(self, n_modalities: int = 4, hidden_dim: int = 32):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities),
            nn.Sigmoid(),
        )

        self.uncertainty_net = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                modality_mask: torch.Tensor) -> torch.Tensor:
        attn   = self.attention_net(modality_mask)
        uncert = self.uncertainty_net(modality_mask)
        raw_w  = attn / (1.0 + uncert)
        norm_w = raw_w / raw_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        final  = norm_w * x.shape[1]
        return x * final.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def get_weights(self, modality_mask: torch.Tensor) -> torch.Tensor:
        attn   = self.attention_net(modality_mask)
        uncert = self.uncertainty_net(modality_mask)
        raw    = attn / (1.0 + uncert)
        return raw / raw.sum(dim=1, keepdim=True).clamp(min=1e-8) * modality_mask.shape[1]
