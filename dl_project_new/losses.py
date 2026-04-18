"""
losses.py  —  Loss Functions and BraTS Evaluation Metrics
==========================================================
Author : Yuktaa Sri Addanki
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  None — this module has no file paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        n   = logits.shape[1]
        pr  = F.softmax(logits, dim=1)
        oh  = F.one_hot(target.clamp(0, n-1), n).permute(0,4,1,2,3).float()
        pf  = pr.flatten(2)
        tf  = oh.flatten(2)
        inter = (pf * tf).sum(-1)
        union = pf.sum(-1) + tf.sum(-1)
        return 1.0 - ((2*inter + self.smooth) / (union + self.smooth)).mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss()
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return (self.alpha * self.dice(logits, target) +
                (1 - self.alpha) * self.ce(logits, target))


class DeepSupervisionLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.seg = CombinedSegLoss(alpha)

    def forward(self, main, aux3, aux2, target):
        loss = self.seg(main, target)
        if aux3 is not None:
            loss = loss + 0.3 * self.seg(aux3, target)
        if aux2 is not None:
            loss = loss + 0.5 * self.seg(aux2, target)
        return loss


def total_loss(main, aux3, aux2, target, kl, gen_feat, enc_feat,
               teacher_main=None,
               w_seg=1.0, w_dis=0.3, w_vae=0.1, w_gan=0.05, T=4.0):
    crit  = DeepSupervisionLoss(0.7)
    l_seg = crit(main, aux3, aux2, target)

    l_dis = torch.tensor(0., device=main.device)
    if teacher_main is not None:
        s     = F.log_softmax(main / T, dim=1)
        t     = F.softmax(teacher_main / T, dim=1).detach()
        l_dis = F.kl_div(s, t, reduction='batchmean') * T**2

    l_gan = F.l1_loss(gen_feat, enc_feat.detach())
    tot   = w_seg*l_seg + w_dis*l_dis + w_vae*kl + w_gan*l_gan

    return tot, {
        'seg':   l_seg.item(),
        'dis':   l_dis.item(),
        'vae':   kl.item() if hasattr(kl, 'item') else float(kl),
        'gan':   l_gan.item(),
        'total': tot.item(),
    }


@torch.no_grad()
def dice_brats(logits, target, eps=1e-6):
    pred = logits.argmax(dim=1)

    def _d(p, t):
        p, t = p.float(), t.float()
        return ((2*(p*t).sum() + eps) / (p.sum()+t.sum()+eps)).item()

    return {
        'WT': _d(pred > 0,              target > 0),
        'TC': _d((pred==1)|(pred==3),   (target==1)|(target==3)),
        'ET': _d(pred == 3,             target == 3),
    }


def hausdorff95(logits, target, region='WT'):
    from scipy.ndimage import distance_transform_edt
    pred = logits.argmax(1).cpu().numpy()[0]
    tgt  = target.cpu().numpy()[0]

    if region == 'WT':
        pm, tm = pred > 0, tgt > 0
    elif region == 'TC':
        pm, tm = (pred==1)|(pred==3), (tgt==1)|(tgt==3)
    else:
        pm, tm = pred == 3, tgt == 3

    if not pm.any() or not tm.any():
        return 373.0

    pd = distance_transform_edt(~pm)
    td = distance_transform_edt(~tm)
    return float(np.percentile(np.concatenate([pd[tm], td[pm]]), 95))
