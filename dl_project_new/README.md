# BrainSegNet — GPU Lab Run Guide
## SJSU Docker Lab · Team Ashmitha

---

## Changes Made to This Version

Every change from the original project is documented here.

### NEW FILE: config.py
**This is the most important change.** All paths are now in one place.
```python
DATA_ROOT  = "/app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_DIR = "/app/outputs"
NUM_WORKERS = 0     # MUST be 0 in Docker on Windows
TEST_MODE   = True  # Set False for full training
```

### CHANGED: dataset.py
- Removed all hardcoded paths
- All paths now imported from config.py
- `NUM_WORKERS` defaults to 0 (Docker/Windows requirement)
- `TEST_MODE` automatically uses 5 patients when True

### CHANGED: train.py
- Removed all hardcoded paths
- All paths and hyperparameters imported from config.py
- `TEST_MODE` automatically reduces to 3 epochs when True

### CHANGED: evaluate.py
- Removed all hardcoded paths
- All paths imported from config.py

### UNCHANGED: models/ (all 5 model files)
- maca.py, encoder.py, vae.py, gan.py, decoder.py, brainsegnet.py
- No paths in model files — no changes needed

### CHANGED: QuickStart_Verification.ipynb
- Paths pre-set via config.py import — no manual changes needed
- TEST_MODE=True pre-configured for 5-patient run

### CHANGED: BrainSegNet_Full_Pipeline.ipynb
- All paths read from config.py — no manual DATA_ROOT edit needed
- Automatically uses TEST_MODE setting from config.py

---

## Project Directory Structure

Save everything exactly like this on the lab machine Desktop:

```
C:\Users\ashmitha\Desktop\ashmitha\
│
├── dl_project_new\                    ← YOUR CODE (copy this whole folder)
│   ├── config.py                      ★ PATHS ARE SET HERE — edit if needed
│   ├── dataset.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── requirements.txt
│   ├── QuickStart_Verification.ipynb  ← RUN THIS FIRST
│   ├── BrainSegNet_Full_Pipeline.ipynb
│   └── models\
│       ├── __init__.py
│       ├── maca.py
│       ├── encoder.py
│       ├── vae.py
│       ├── gan.py
│       ├── decoder.py
│       └── brainsegnet.py
│
├── data\                              ← YOUR BRATS DATA (already on machine)
│   ├── BraTS2020_TrainingData\
│   │   └── MICCAI_BraTS2020_TrainingData\
│   │       ├── BraTS20_Training_001\
│   │       │   ├── BraTS20_Training_001_t1.nii
│   │       │   ├── BraTS20_Training_001_t1ce.nii
│   │       │   ├── BraTS20_Training_001_t2.nii
│   │       │   ├── BraTS20_Training_001_flair.nii
│   │       │   └── BraTS20_Training_001_seg.nii
│   │       ├── BraTS20_Training_002\
│   │       └── ...
│   └── BraTS2020_ValidationData\
│       └── MICCAI_BraTS2020_ValidationData\
│           └── ...
│
└── outputs\                           ← CREATED AUTOMATICALLY
    ├── checkpoints\
    │   ├── teacher_best.pth           ← saved after Stage 1
    │   └── student_best.pth           ← saved after Stage 2
    ├── patient_vis.png
    ├── training_curves.png
    ├── predictions.png
    ├── eval_results.json
    ├── teacher_history.json
    └── student_history.json
```

---

## Step-by-Step Lab Instructions

### Step 1 — Log In
```
Username: .\ashmitha   (include the .\  before username)
Password: (from email)
```

### Step 2 — Start Docker Desktop
1. Open Docker Desktop from the taskbar
2. Accept → Continue without signing in → Skip
3. Wait for "Engine running" bottom-left

### Step 3 — Load the Docker Image
Open File Explorer → This PC → OS(C:) → DockerImages
Double-click the **PyTorch.bat** file. Wait for download to finish.

Then open **Command Prompt** (NOT PowerShell) and verify:
```cmd
docker images
```
Note the REPOSITORY and TAG shown (e.g. gdevakumar/pytorch  latest)

### Step 4 — Copy Your Files
Copy the `dl_project_new` folder to:
```
C:\Users\ashmitha\Desktop\ashmitha\dl_project_new\
```
The data is already at:
```
C:\Users\ashmitha\Desktop\ashmitha\data\BraTS2020_TrainingData\...
```

### Step 5 — Launch Docker with Jupyter
Open **Command Prompt** and run (replace IMAGE:TAG with what docker images showed):
```cmd
docker run --gpus all -p 8888:8888 -v "C:\Users\ashmitha\Desktop\ashmitha":/app gdevakumar/pytorch:latest
```

Copy the URL from the terminal output (looks like):
```
http://127.0.0.1:8888/tree?token=abc123...
```
Paste into Chrome.

### Step 6 — Install Dependencies (once per session)
In Jupyter, click **New → Terminal** and run:
```bash
pip install nibabel scipy einops -q
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 7 — Run QuickStart FIRST (5-patient test)
In Jupyter, navigate to `dl_project_new/`
Open **QuickStart_Verification.ipynb**
Click **Kernel → Restart & Run All**

Expected time: **15–25 minutes**
Expected outcome: All 5 checks show PASSED

### Step 8 — Run Full Training
Once QuickStart passes:
1. Open `config.py` in Jupyter text editor
2. Change: `TEST_MODE = False`
3. Save config.py
4. Open **BrainSegNet_Full_Pipeline.ipynb**
5. Click **Kernel → Restart & Run All**

Expected time: **30–40 hours total** (12h teacher + 18h student)

### Step 9 — Save Results Before Leaving
Upload to Google Drive before your session ends:
- `C:\Users\ashmitha\Desktop\ashmitha\outputs\checkpoints\teacher_best.pth`
- `C:\Users\ashmitha\Desktop\ashmitha\outputs\checkpoints\student_best.pth`
- `C:\Users\ashmitha\Desktop\ashmitha\outputs\eval_results.json`

### Step 10 — Clean Up (required by lab rules)
1. Stop Docker: Ctrl+C in Command Prompt
2. Delete your Desktop folder
3. Lock screen: Windows + L

---

## Common Errors

**`DATA_ROOT not found`**
Check the Docker -v path. The `-v` flag must point to your Desktop ashmitha folder.

**`DataLoader worker exited unexpectedly`**
config.py must have `NUM_WORKERS = 0`. Do not change this for Docker on Windows.

**`CUDA out of memory`**
In config.py set `CROP_SIZE = 64` or `BATCH_SIZE = 1`.

**`FileNotFoundError: Cannot find t1 for BraTS20_Training_XXX`**
Open Jupyter terminal and check:
```bash
ls /app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/ | head -5
```
The path you see here must match DATA_ROOT in config.py.

**`ModuleNotFoundError: nibabel`**
Run in Jupyter terminal: `pip install nibabel scipy einops -q`
