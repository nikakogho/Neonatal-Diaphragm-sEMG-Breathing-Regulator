# Neonatal Diaphragm sEMG Breathing Regulator

## What this project is

This repository is about learning ventilator-related control signals from **cleaned diaphragm surface EMG**.

The overall workflow has two major parts:

1. **Preprocessing**
   - Raw high-density EMG recordings are processed in [`diaphragm_heatmap.py`](./diaphragm_heatmap.py).
   - The goal there is to keep the diaphragm-relevant sEMG signal, suppress unusable channels / artifacts, and export cleaned recordings as `.npz` files.

2. **Model training**
   - After preprocessing, we train models on cleaned EMG windows and ask them to predict ventilator AUX values.
   - The current modeling code lives in `src/` and `scripts/`, centered on the single-recording workflow.

## Current status

### Step 1 is complete

We have already completed the first modeling milestone:

**1. Confirm that we can overfit one recording of one person.**

The canonical proof run is here:

- [`runs/p4rec6_overfit_demo_raw300_delta-150_20260324_151750/README.md`](./runs/p4rec6_overfit_demo_raw300_delta-150_20260324_151750/README.md)

That run shows that for **participant 4**, recording **`p4rec6_processed_1024Hz.npz`**, the current pipeline can overfit the recording very strongly on GPU from the project venv.

Main result from that run:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -150 ms / 50 ms`
- train/eval RMSE: `0.063667`
- train/eval MAE: `0.049000`
- train/eval Pearson: `0.999026`
- train/eval R2: `0.997743`

Because this is a **same-window** overfit experiment, train and eval metrics are identical by design. That is expected here: the point of step 1 is to show that the signal-to-target mapping is learnable enough for the model to memorize one cleaned recording extremely well.

## Model that worked in step 1

The strongest step-1 model so far is a **raw 1D CNN scalar regressor**:

- input: one cleaned EMG window of shape `(time, 6, 8, 8)`
- internal reshape: `(B, time, 6, 8, 8) -> (B, 384, time)`
- architecture:
  - `Conv1d(384 -> 64, kernel_size=5, padding=2)`
  - `BatchNorm1d(64)`
  - `ReLU`
  - `MaxPool1d(2)`
  - `Conv1d(64 -> 128, kernel_size=5, padding=2)`
  - `BatchNorm1d(128)`
  - `ReLU`
  - `Dropout(0.15)`
  - `AdaptiveAvgPool1d(1)`
  - `Linear(128 -> 64)`
  - `ReLU`
  - `Dropout(0.15)`
  - `Linear(64 -> 1)`

Step-1 hyperparameters that worked best:

- model: `cnn1d_raw_scalar`
- recording: `participant 4 / p4rec6_processed_1024Hz.npz`
- target channel: `AUX[0]`
- `win_ms=300`
- `delta_ms=-150`
- `step_ms=50`
- `epochs=100`
- `batch_size=16`
- `lr=3e-4`
- `weight_decay=1e-5`
- `dropout=0.15`
- `base_channels=16`
- `seed=13`
- device: `cuda`

## What currently looks most promising for the next stages

Given the shape of the data and the task, the most promising direction right now is:

- **Primary candidate:** raw 1D CNN with `300 ms` windows and roughly `-150 ms` target offset
- **Secondary / lighter reference candidate:** feature 1D CNN with the same `300 ms / -150 ms` timing

Why this is our current view:

- each EMG window is short enough for 1D temporal convolutions to be practical
- each window still contains a lot of spatial information (`6 x 8 x 8 = 384` channels), so keeping the raw window is useful
- the raw 1D CNN consistently beat the nearby feature-based alternatives on the participant-4 recordings we compared
- the `300 ms / -150 ms` timing pair remained the strongest candidate across `p4rec3`, `p4rec5`, and `p4rec6`

So for the next stages, the raw 1D CNN is the first model we should keep testing, and the feature 1D CNN is the sensible backup / comparison baseline.

## Next planned modeling steps

The next steps are **not implemented yet**. They are the planned progression after step 1:

2. **Confirm overfit on every recording at the same time for one given person with one model.**
3. **Check predictive ability for one given person with one model.**
4. **Check predictive ability when we train on multiple people, then expose the model to a few recordings of a new person, and see whether later recordings of that new person can be picked up.**

## Repository layout

- [`diaphragm_heatmap.py`](./diaphragm_heatmap.py)
  - preprocessing / inspection tool that exports cleaned `.npz` recordings
- `src/data/`
  - cleaned recording loading, normalization, windowing
- `src/features/`
  - binned feature extraction helpers
- `src/models/`
  - current model registry, including raw and feature 1D CNN variants
- [`src/training/single_recording.py`](./src/training/single_recording.py)
  - single-recording overfit workflow, plotting, and report generation
- [`scripts/train_single_recording.py`](./scripts/train_single_recording.py)
  - main training entrypoint for the current step-1 style runs
- [`scripts/run_single_recording_sweep.py`](./scripts/run_single_recording_sweep.py)
  - timing/model sweep helper
- [`scripts/check_gpu.py`](./scripts/check_gpu.py)
  - venv / CUDA / GPU sanity check

## Data format

Expected cleaned `.npz` recording keys:

- `emg`: `(n_t, 6, 8, 8)`
- `aux`: `(n_t, 2)`
- `bad_mask`: `(6, 8, 8)`
- `time_s`
- `meta` with `fs_export_hz`

## Minimal usage

### 1. Verify venv + GPU

```powershell
venv\Scripts\python scripts\check_gpu.py
```

### 2. Run tests

```powershell
venv\Scripts\python -m pytest -q
```

### 3. Train one single-recording run

```powershell
venv\Scripts\python scripts\train_single_recording.py `
  --npz "extra_patients\participant 4 stuff\p4rec6_processed_1024Hz.npz" `
  --model cnn1d_raw_scalar `
  --win_ms 300 `
  --delta_ms -150 `
  --step_ms 50 `
  --epochs 100 `
  --batch_size 16 `
  --lr 3e-4 `
  --weight_decay 1e-5 `
  --dropout 0.15 `
  --base_channels 16 `
  --device cuda `
  --require_venv `
  --require_cuda `
  --run_dir runs\example_single_recording
```

## Research status

This repository is still a research workflow, not a deployment-ready ventilator controller.

What we have shown so far:

- preprocessing pipeline exists and exports cleaned diaphragm-focused EMG recordings
- step 1 overfit on one recording of one participant works very well

What is still missing:

- multi-recording same-person overfit with one model
- same-person predictive testing
- multi-person training with adaptation to a new person after a few observed recordings
