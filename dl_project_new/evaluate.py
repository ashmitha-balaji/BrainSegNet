"""
evaluate.py  —  Full Evaluation on All 15 Missing-Modality Combinations
========================================================================
Author : Yuktaa Sri Addanki
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  - DATA_ROOT, OUTPUT_DIR, STUDENT_CKPT all read from config.py
  - Results saved to /app/outputs/eval_results.json (inside Docker)

Usage inside Docker container:
  python evaluate.py
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from config import (DATA_ROOT, OUTPUT_DIR, STUDENT_CKPT,
                    CROP_SIZE, BASE_FILTERS, LATENT_DIM)
from dataset import get_dataloaders
from models.brainsegnet import BrainSegNet
from losses import dice_brats, hausdorff95


ALL_COMBOS = [
    ('All 4 modalities',     [1,1,1,1]),
    ('Missing T1',           [0,1,1,1]),
    ('Missing T1ce',         [1,0,1,1]),
    ('Missing T2',           [1,1,0,1]),
    ('Missing FLAIR',        [1,1,1,0]),
    ('Missing T1+T1ce',      [0,0,1,1]),
    ('Missing T1+T2',        [0,1,0,1]),
    ('Missing T1+FLAIR',     [0,1,1,0]),
    ('Missing T1ce+T2',      [1,0,0,1]),
    ('Missing T1ce+FLAIR',   [1,0,1,0]),
    ('Missing T2+FLAIR',     [1,1,0,0]),
    ('Only T1',              [1,0,0,0]),
    ('Only T1ce',            [0,1,0,0]),
    ('Only T2',              [0,0,1,0]),
    ('Only FLAIR',           [0,0,0,1]),
]

M3AE = {'WT': 0.858, 'TC': 0.774, 'ET': 0.599}


def run_evaluation(model, loader, device):
    model.eval()
    results = {}

    for name, mask_vals in ALL_COMBOS:
        print(f'  {name}  mask={mask_vals}')
        d = {'WT': [], 'TC': [], 'ET': []}

        for imgs, segs, _ in tqdm(loader, desc=name, leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            im2  = imgs.clone()
            for ch, pres in enumerate(mask_vals):
                if not pres:
                    im2[:, ch] = 0.
            mk = torch.tensor([mask_vals], dtype=torch.float32,
                               device=device).expand(imgs.shape[0], -1)

            with torch.no_grad():
                out = model(im2, mk, training=False)
                if isinstance(out, tuple):
                    out = out[0]

            m = dice_brats(out, segs)
            for k in d:
                d[k].append(m[k])

        avg = {k: float(np.mean(d[k])) for k in ['WT','TC','ET']}
        results[name] = avg
        print(f'    WT {avg["WT"]:.4f}  TC {avg["TC"]:.4f}  ET {avg["ET"]:.4f}')

    results['MEAN'] = {
        k: float(np.mean([results[n][k] for n, _ in ALL_COMBOS]))
        for k in ['WT','TC','ET']
    }
    return results


def print_table(results):
    W = 35
    print(f'\n{"Scenario":<{W}} {"WT":>8} {"TC":>8} {"ET":>8}')
    print('=' * (W+28))
    for name, _ in ALL_COMBOS:
        v = results[name]
        print(f'{name:<{W}} {v["WT"]:>8.4f} {v["TC"]:>8.4f} {v["ET"]:>8.4f}')
    print('-' * (W+28))
    mn = results['MEAN']
    print(f'{"MEAN (all 15)":<{W}} {mn["WT"]:>8.4f} '
          f'{mn["TC"]:>8.4f} {mn["ET"]:>8.4f}')
    print('=' * (W+28))
    print(f'{"M3AE benchmark":<{W}} '
          f'{M3AE["WT"]:>8.4f} {M3AE["TC"]:>8.4f} {M3AE["ET"]:>8.4f}')
    dw = mn['WT'] - M3AE['WT']
    sign = '+' if dw >= 0 else ''
    print(f'\nWT diff vs M3AE: {sign}{dw*100:.2f}%',
          '← BEATS M3AE!' if dw > 0 else '')


def main():
    assert os.path.exists(STUDENT_CKPT), (
        f"Student checkpoint not found: {STUDENT_CKPT}\n"
        "Complete training first (Stages 1 and 2).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kw     = dict(base_filters=BASE_FILTERS, crop_size=CROP_SIZE,
                  latent_dim=LATENT_DIM, use_gan=True)
    model  = BrainSegNet(**kw).to(device)
    ckpt   = torch.load(STUDENT_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'Model loaded | best_wt={ckpt.get("best_wt","?"):.4f}')
    print(f'Data  : {DATA_ROOT}')
    print(f'Output: {OUTPUT_DIR}')

    _, _, te = get_dataloaders()
    results  = run_evaluation(model, te, device)
    print_table(results)

    out_path = os.path.join(OUTPUT_DIR, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {out_path}')


if __name__ == '__main__':
    main()
