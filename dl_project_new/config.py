"""
config.py  —  Central Path Configuration for BrainSegNet
=========================================================
Paths default to the repo workspace (parent folder of `dl_project_new`):

  <workspace>/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/  ← DATA_ROOT
  <workspace>/outputs/                                                     ← OUTPUT_DIR

Docker examples (paths follow wherever `dl_project_new/config.py` lives):

  Mount parent folder (your setup):
    -v "C:\\Users\\arpana\\Desktop\\Arpana":/app
  Repo at /app/BrainSegNet/dl_project_new/ → WORKSPACE_ROOT=/app/BrainSegNet
  → DATA_ROOT=/app/BrainSegNet/data/.../MICCAI_BraTS2020_TrainingData

  Mount repo root instead:
    -v "C:\\Users\\...\\BrainSegNet":/app
  Repo at /app/dl_project_new/ → WORKSPACE_ROOT=/app

Override without editing this file (optional):

  export BRAINSENET_DATA_ROOT=/path/to/MICCAI_BraTS2020_TrainingData
  export BRAINSENET_OUTPUT_DIR=/path/to/outputs

Expected patient folder (example):

  BraTS20_Training_001/
      BraTS20_Training_001_t1.nii
      BraTS20_Training_001_t1ce.nii
      ...
"""

import os

# ============================================================
# Resolve workspace root = parent of this package (dl_project_new)
# ============================================================

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(_PKG_DIR)

_DEFAULT_DATA = os.path.join(
    WORKSPACE_ROOT,
    "data",
    "BraTS2020_TrainingData",
    "MICCAI_BraTS2020_TrainingData",
)
_DEFAULT_OUTPUT = os.path.join(WORKSPACE_ROOT, "outputs")

DATA_ROOT = os.environ.get("BRAINSENET_DATA_ROOT", _DEFAULT_DATA)
OUTPUT_DIR = os.environ.get("BRAINSENET_OUTPUT_DIR", _DEFAULT_OUTPUT)

# ============================================================
# TRAINING HYPERPARAMETERS — safe defaults for the lab GPU
# ============================================================

CROP_SIZE         = 96      # 3D crop size. Use 64 if GPU OOM error appears.
BATCH_SIZE        = 2       # Reduce to 1 if GPU runs out of memory
CROPS_PER_PATIENT = 4       # Random crops per patient per epoch
MISSING_PROB      = 0.5     # Probability each modality is zeroed (student training)
NUM_WORKERS       = 4       # MUST be 0 inside Docker on Windows
BASE_FILTERS      = 32      # Encoder filter count at level 1
LATENT_DIM        = 128     # VAE latent dimension
TEACHER_EPOCHS    = 100     # Stage 1 epochs
STUDENT_EPOCHS    = 150     # Stage 2 epochs
SEED              = 42
# The GAN needs a much lower starting LR
STUDENT_LR      = 5e-5    # was 2e-4 — 4x lower
DISC_LR         = 1e-5    # was 1e-4 — 10x lower
# ============================================================
# 5-PATIENT TEST MODE — flip to True for quick verification
# ============================================================
TEST_MODE         = False   # Set True for 5-patient / 3-epoch test run
TEST_N_TRAIN      = 4       # Patients used for training in test mode
TEST_N_VAL        = 1       # Patients used for validation in test mode
TEST_EPOCHS       = 3       # Epochs in test mode
USE_GAN           = False
# ============================================================
# Derived paths — do not edit below this line
# ============================================================
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
TEACHER_CKPT   = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")
STUDENT_CKPT   = os.path.join(CHECKPOINT_DIR, "student_best.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
