"""
train.py  —  Two-Stage Training Script for BrainSegNet
=======================================================
Author : Yuktaa Sri Addanki
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  - All paths (DATA_ROOT, OUTPUT_DIR, CHECKPOINT_DIR) read from config.py
  - No hardcoded paths in this file

Usage inside Docker container:
  python train.py --mode teacher
  python train.py --mode student

All data and output paths are set in config.py.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# All paths come from config.py
from config import (DATA_ROOT, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
                    TEACHER_CKPT, STUDENT_CKPT,
                    CROP_SIZE, BATCH_SIZE, CROPS_PER_PATIENT,
                    MISSING_PROB, NUM_WORKERS, BASE_FILTERS,
                    LATENT_DIM, TEACHER_EPOCHS, STUDENT_EPOCHS,
                    TEST_MODE, TEST_EPOCHS, SEED)

from dataset import get_dataloaders
from models.brainsegnet import BrainSegNet
from models.gan import discriminator_loss as disc_loss_fn
from losses import total_loss, DeepSupervisionLoss, dice_brats


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='teacher',
                   choices=['teacher', 'student'],
                   help='teacher: Stage 1 full data | student: Stage 2 missing mods')
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, full_mask=False):
    model.eval()
    scores = {'WT': [], 'TC': [], 'ET': []}
    for imgs, segs, masks in loader:
        imgs  = imgs.to(device)
        segs  = segs.to(device)
        masks = (torch.ones_like(masks) if full_mask
                 else masks).to(device)
        out   = model(imgs, masks, training=False)
        m     = dice_brats(out, segs)
        for k in scores:
            scores[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in scores.items()}


def train_teacher(model, tr_loader, va_loader, device, n_epochs):
    print(f'\n=== STAGE 1: Teacher Training ({n_epochs} epochs) ===')
    print(f'Data    : {DATA_ROOT}')
    print(f'Outputs : {CHECKPOINT_DIR}')

    crit  = DeepSupervisionLoss(0.7)
    opt   = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, 2e-6)
    sc    = GradScaler() if torch.cuda.is_available() else None
    best  = 0.0
    hist  = []

    for ep in range(1, n_epochs + 1):
        model.train()
        ep_loss = []

        for imgs, segs, _ in tqdm(tr_loader,
                                   desc=f'Teacher {ep}/{n_epochs}',
                                   leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            full = torch.ones(imgs.shape[0], 4, device=device)
            opt.zero_grad()

            if sc:
                with autocast():
                    main, a3, a2, kl, _, _ = model(imgs, full, training=True)
                    loss = crit(main, a3, a2, segs) + 0.05*kl
                sc.scale(loss).backward()
                sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                sc.step(opt); sc.update()
            else:
                main, a3, a2, kl, _, _ = model(imgs, full, training=True)
                loss = crit(main, a3, a2, segs) + 0.05*kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            ep_loss.append(loss.item())

        sched.step()
        val = validate(model, va_loader, device, full_mask=True)
        avg = np.mean(ep_loss)
        hist.append({'ep': ep, 'loss': avg, **val})
        print(f'ep{ep:3d} | loss {avg:.4f} | '
              f'WT {val["WT"]:.4f} TC {val["TC"]:.4f} ET {val["ET"]:.4f}')

        if val['WT'] > best:
            best = val['WT']
            torch.save({'ep': ep, 'model_state': model.state_dict(),
                        'best_wt': best}, TEACHER_CKPT)
            print(f'  Saved best teacher → {TEACHER_CKPT}  (WT={best:.4f})')

    with open(os.path.join(OUTPUT_DIR, 'teacher_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'\nTeacher done. Best WT: {best:.4f}')


def train_student(student, teacher, tr_loader, va_loader, device, n_epochs):
    print(f'\n=== STAGE 2: Student + Distillation ({n_epochs} epochs) ===')
    print(f'Data    : {DATA_ROOT}')
    print(f'Outputs : {CHECKPOINT_DIR}')

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    opt_s  = optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-5)
    opt_d  = optim.Adam(student.discriminator.parameters(),
                         lr=1e-4, betas=(0.5, 0.999))
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt_s, n_epochs, 2e-6)
    sc     = GradScaler() if torch.cuda.is_available() else None
    best   = 0.0
    hist   = []

    for ep in range(1, n_epochs + 1):
        student.train()
        ep_c = {'total': [], 'seg': [], 'dis': [], 'vae': [], 'gan': []}

        for imgs, segs, masks in tqdm(tr_loader,
                                       desc=f'Student {ep}/{n_epochs}',
                                       leave=False):
            imgs  = imgs.to(device)
            segs  = segs.to(device)
            masks = masks.to(device)

            # Discriminator update
            with torch.no_grad():
                full  = torch.ones_like(masks)
                r_enc = student.encoder(student.maca(imgs, full))[0]
                r_vae, _, mu, _ = student.vae(r_enc)
                fake  = student.generator(mu, masks)
            rp = student.discriminator(r_vae)
            fp = student.discriminator(fake.detach())
            dl = disc_loss_fn(rp, fp)
            opt_d.zero_grad(); dl.backward(); opt_d.step()

            # Student update
            opt_s.zero_grad()
            if sc:
                with autocast():
                    main, a3, a2, kl, gf, ef = student(imgs, masks, True)
                    with torch.no_grad():
                        t_out = teacher(imgs, torch.ones_like(masks), False)
                    loss, comps = total_loss(main, a3, a2, segs,
                                             kl, gf, ef, t_out)
                sc.scale(loss).backward()
                sc.unscale_(opt_s)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                sc.step(opt_s); sc.update()
            else:
                main, a3, a2, kl, gf, ef = student(imgs, masks, True)
                with torch.no_grad():
                    t_out = teacher(imgs, torch.ones_like(masks), False)
                loss, comps = total_loss(main, a3, a2, segs,
                                         kl, gf, ef, t_out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_s.step()

            for k in ep_c:
                ep_c[k].append(comps.get(k, 0))

        sched.step()
        val = validate(student, va_loader, device, full_mask=False)
        avg = np.mean(ep_c['total'])
        hist.append({'ep': ep, 'loss': avg, **val})
        print(f'ep{ep:3d} | loss {avg:.4f} | '
              f'WT {val["WT"]:.4f} TC {val["TC"]:.4f} ET {val["ET"]:.4f}')

        if val['WT'] > best:
            best = val['WT']
            torch.save({'ep': ep, 'model_state': student.state_dict(),
                        'best_wt': best}, STUDENT_CKPT)
            print(f'  Saved best student → {STUDENT_CKPT}  (WT={best:.4f})')

    with open(os.path.join(OUTPUT_DIR, 'student_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'\nStudent done. Best WT: {best:.4f}')


def main():
    args = get_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    n_epochs = TEST_EPOCHS if TEST_MODE else (
        TEACHER_EPOCHS if args.mode == 'teacher' else STUDENT_EPOCHS)

    tr_l, va_l, _ = get_dataloaders()
    kw = dict(base_filters=BASE_FILTERS, crop_size=CROP_SIZE, latent_dim=LATENT_DIM)

    if args.mode == 'teacher':
        model = BrainSegNet(use_gan=False, **kw).to(device)
        train_teacher(model, tr_l, va_l, device, n_epochs)

    else:
        assert os.path.exists(TEACHER_CKPT), (
            f"Teacher checkpoint not found: {TEACHER_CKPT}\n"
            "Run Stage 1 first: python train.py --mode teacher")
        teacher = BrainSegNet(use_gan=False, **kw).to(device)
        teacher.load_state_dict(
            torch.load(TEACHER_CKPT, map_location=device)['model_state'])
        print(f'Teacher loaded from {TEACHER_CKPT}')
        student = BrainSegNet(use_gan=True, **kw).to(device)
        train_student(student, teacher, tr_l, va_l, device, n_epochs)


if __name__ == '__main__':
    main()
