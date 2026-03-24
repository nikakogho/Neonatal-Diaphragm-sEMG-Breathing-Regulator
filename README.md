# Neonatal Diaphragm sEMG -> Pump Aux Regression

## What this project is

This repository supports a bachelor-thesis workflow with two connected parts:

1. **Signal preprocessing and inspection** from high-density diaphragm sEMG recordings.
2. **Supervised learning** to predict two ventilator AUX channels from cleaned EMG windows.

The long-term goal is adaptive ventilator support.  
The current practical goal is: **from a cleaned EMG window, predict AUX[0] and AUX[1] robustly on unseen patients**.

## Current reality (as of March 5, 2026)

- Canonical ML pipeline now lives under `src/` + `scripts/`.
- Subject-held-out nested CV is implemented.
- Export + inference scripts are implemented.
- **Model is not yet shipment-ready for unseen patients**:
  - `runs/ship_check_raw300d50/result.json` shows held-out skill `< 0` (worse than baseline).
  - `runs/nested_medium_real/outer_test_pid_4` and `outer_test_pid_5` both selected `constant_mean` (skill `0.0`).
  - The medium sweep was interrupted and is currently partial (`outer_test_pid_7` incomplete).

So the pipeline is clean and operational, but generalization quality is still insufficient.

## Note on earlier 1D CNN "better-than-baseline" runs

Some older runs did show 1D CNN outperforming their recorded baseline (for example, `runs/test_1d_cnn_delta50_fulldata` and `runs/test_1d_cnn_delta_50`).  
However, those gains were **not stable across held-out subjects** and were mixed with many runs where 1D CNN was at or below baseline.

What changed is mainly the evaluation protocol:

- Earlier experiments were mostly ad-hoc train/val/test splits on selected participant combinations.
- Current canonical evaluation uses **nested, subject-held-out CV** and selects by mean subject-level skill.
- Under this stricter protocol, completed folds consistently select `constant_mean` and current CNN candidates do not show reliable positive skill on unseen subjects.

Window/timing combinations tried so far in stored run artifacts:

- `win_ms`: `100`, `300`
- `delta_ms`: `-100`, `-50`, `0`, `50`, `100`, `150`
- Most legacy 1D CNN runs used `win_ms=300` with varying deltas.
- Nested profiles currently exercised:
  - `fast`: `(100, 0)`
  - `medium`: `(100, 0)`, `(300, 0)`, `(300, 50)`
- Full canonical search space in code is larger (`win_ms in {100,200,300,400}`, `delta_ms in {-100,-50,0,50,100}`), but full runs are not complete yet.

Current conclusion: we do not yet have stable cross-subject evidence that 1D CNN beats a constant baseline.

## Repository structure

- `diaphragm_heatmap.py`
  - GUI + preprocessing/export of cleaned EMG + AUX to `.npz`.
- `src/data/`
  - Recording loading, manifest, canonical window dataset, normalization policy.
- `src/features/`
  - Binned feature extraction (`mean|abs|`, RMS per grid/bin).
- `src/models/`
  - Canonical model registry (`mlp_feature`, `cnn1d_feature`, `cnn1d_raw`).
- `src/training/protocol.py`
  - Nested subject-held-out protocol, scoring, report generation.
- `scripts/train.py`
  - Single-config training/evaluation on explicit train/val split.
- `scripts/evaluate.py`
  - Evaluate a config on explicit train/test split.
- `scripts/run_nested_cv.py`
  - Full nested CV runner with `fast|medium|full` profiles and resume support.
- `scripts/export_model.py`
  - Package model weights + normalization stats into one deployable bundle.
- `scripts/infer_window.py`
  - Run one-window inference from an exported bundle.
- `tests/`
  - Canonical pipeline tests.

## Data assumptions

Expected preprocessed `.npz` recording keys:

- `emg`: `(n_t, 6, 8, 8)`
- `aux`: `(n_t, 2)`
- `bad_mask`: `(6, 8, 8)` (`True = bad electrode`)
- `meta`: JSON string with `fs_export_hz`

Canonical participant set currently used for model work:

- `4, 5, 7, 8, 9, 10`

## Quickstart

### 1) Run tests

```powershell
venv\Scripts\python -m pytest -q
```

### 2) Single experiment

```powershell
venv\Scripts\python scripts\train.py `
  --data extra_patients `
  --participants 4,5,7,8,9,10 `
  --train_pids 4,5,7,8 `
  --val_pid 9 `
  --model mlp_feature `
  --win_ms 300 `
  --delta_ms 50 `
  --step_ms 50 `
  --epochs 10 `
  --run_dir runs\example_single
```

### 3) Nested CV

```powershell
# smoke
venv\Scripts\python scripts\run_nested_cv.py --profile fast --output_root runs\nested_fast_real

# reduced sweep (recommended during iteration)
venv\Scripts\python scripts\run_nested_cv.py --profile medium --output_root runs\nested_medium_real

# full grid
venv\Scripts\python scripts\run_nested_cv.py --profile full --output_root runs\nested_full_real
```

Resume behavior:

- Existing completed fold artifacts are reused automatically.
- Add `--no_resume` to force full recomputation.

### 4) Export a trained torch model

```powershell
venv\Scripts\python scripts\export_model.py `
  --run_dir runs\ship_check_raw300d50 `
  --data extra_patients `
  --participants 4,5,7,8,9,10 `
  --train_pids 4,5,7,8 `
  --output models\ship_check_raw300d50_bundle.pt
```

### 5) Inference on one window

```powershell
venv\Scripts\python scripts\infer_window.py `
  --bundle models\ship_check_raw300d50_bundle.pt `
  --npz "extra_patients\participant 9 stuff\par9rec1_processed_1024Hz.npz" `
  --start 0
```

## What is still missing before this can be shipped

1. Complete a full subject-held-out sweep across all outer folds and candidate configs.
2. Demonstrate stable **positive subject-level skill score** versus constant baseline.
3. Lock one final model + timing + normalization configuration.
4. Add calibration/uncertainty checks and failure-handling policy for out-of-distribution subjects.
5. Freeze a reproducible training recipe and regenerate final export bundle from that run.

Until those are done, treat the current model outputs as research-only.
