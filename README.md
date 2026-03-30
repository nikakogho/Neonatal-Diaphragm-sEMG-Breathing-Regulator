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

We completed the first modeling milestone:

**1. Confirm that we can overfit one recording of one person.**

Canonical proof run:

- [`runs/p4rec6_overfit_demo_raw300_delta-150_20260324_151750/README.md`](./runs/p4rec6_overfit_demo_raw300_delta-150_20260324_151750/README.md)

That run shows that for **participant 4**, recording **`p4rec6_processed_1024Hz.npz`**, the current pipeline can overfit a single cleaned recording very strongly on GPU from the project venv.

Main step-1 result:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -150 ms / 50 ms`
- train/eval RMSE: `0.063667`
- train/eval MAE: `0.049000`
- train/eval Pearson: `0.999026`
- train/eval R2: `0.997743`

Loss curve:

![Step 1 loss curve](./runs/p4rec6_overfit_demo_raw300_delta-150_20260324_151750/loss_curve.png)

### Step 2 is complete

We also completed the second modeling milestone:

**2. Confirm that we can overfit every recording of one person at the same time with one model.**

Canonical proof run:

- [`runs/p4_all_recordings_overfit_demo_raw300_delta-150_20260324_171422/README.md`](./runs/p4_all_recordings_overfit_demo_raw300_delta-150_20260324_171422/README.md)

Step-2 tuning comparison:

- [`runs/participant4_step2_hyperparam_comparison/summary.md`](./runs/participant4_step2_hyperparam_comparison/summary.md)

That pooled run trains one shared model on **all 9 cleaned recordings from participant 4**, with windows shuffled across recordings rather than fed recording-by-recording.

Main step-2 result:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -150 ms / 50 ms`
- epochs / batch / lr / dropout / base channels: `250 / 16 / 3e-4 / 0.15 / 16`
- pooled train/eval RMSE: `0.077544`
- pooled train/eval MAE: `0.059050`
- pooled train/eval Pearson: `0.996199`
- pooled train/eval R2: `0.991141`
- dynamic participant-4 recordings (`p4rec2` to `p4rec8`) all reached `R2 >= 0.98`

Loss curve:

![Participant 4 pooled loss curve](./runs/p4_all_recordings_overfit_demo_raw300_delta-150_20260324_171422/loss_curve.png)

Because step 1 and step 2 are both **same-window overfit** experiments, train and eval metrics are identical by design. That is expected here: these stages are about proving the mapping is learnable and that one model can memorize either one cleaned recording or one participant's pooled recordings.

### Additional step-2 validation on participant 9

We also repeated the same pooled step-2 workflow on participant 9 to check whether the participant-4 result generalizes as a modeling setup:

- [`runs/p9_all_recordings_overfit_demo_raw300_delta-150_20260324_181445/README.md`](./runs/p9_all_recordings_overfit_demo_raw300_delta-150_20260324_181445/README.md)
- [`runs/participant9_step2_hyperparam_comparison/summary.md`](./runs/participant9_step2_hyperparam_comparison/summary.md)

Main participant-9 result:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -150 ms / 50 ms`
- epochs / batch / lr / dropout / base channels: `350 / 16 / 3e-4 / 0.15 / 16`
- pooled train/eval RMSE: `0.100551`
- pooled train/eval MAE: `0.077072`
- pooled train/eval Pearson: `0.995591`
- pooled train/eval R2: `0.991010`

Loss curve:

![Participant 9 pooled loss curve](./runs/p9_all_recordings_overfit_demo_raw300_delta-150_20260324_181445/loss_curve.png)

Interpretation:

- the same raw 1D CNN and the same `300 ms / -150 ms` timing pair were still the best participant-9 candidate
- participant 9 was harder than participant 4 in this same-window pooled setting
- the limiting recordings were `par9rec2` and `par9rec6`, so participant 9 supports the same architecture choice but not yet the same per-recording overfit quality as participant 4

### Additional step-2 validation on participant 10

We also repeated the pooled step-2 workflow on participant 10:

- [`runs/p10_all_recordings_overfit_demo_raw300_delta-100_20260328_013950/README.md`](./runs/p10_all_recordings_overfit_demo_raw300_delta-100_20260328_013950/README.md)
- [`runs/participant10_step2_hyperparam_comparison/summary.md`](./runs/participant10_step2_hyperparam_comparison/summary.md)

Main participant-10 result:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -100 ms / 50 ms`
- epochs / batch / lr / dropout / base channels: `350 / 16 / 3e-4 / 0.15 / 16`
- pooled train/eval RMSE: `0.081444`
- pooled train/eval MAE: `0.059245`
- pooled train/eval Pearson: `0.996934`
- pooled train/eval R2: `0.990947`
- dynamic participant-10 recordings (`par10rec1` to `par10rec5`) all reached `R2 >= 0.98`

Loss curve:

![Participant 10 pooled loss curve](./runs/p10_all_recordings_overfit_demo_raw300_delta-100_20260328_013950/loss_curve.png)

Interpretation:

- the same raw 1D CNN remained the best participant-10 model family
- participant 10 preferred a slightly different timing pair than participants 4 and 9: `300 ms / -100 ms` beat the old `300 ms / -150 ms` baseline
- unlike participant 9, participant 10 did achieve the dynamic-recording overfit target after the timing search

### Harder pooled overfit attempt across participants 4, 9, and 10

We also tried the harder next question: can **one shared model** overfit the pooled recordings from participants 4, 9, and 10 at the same time?

- [`runs/p4_p9_p10_combined_overfit_demo_raw300_delta-150_20260328_174523/README.md`](./runs/p4_p9_p10_combined_overfit_demo_raw300_delta-150_20260328_174523/README.md)
- [`runs/participants4_9_10_combined_overfit_comparison/summary.md`](./runs/participants4_9_10_combined_overfit_comparison/summary.md)

Best combined-cohort result:

- model: `cnn1d_raw_scalar`
- target: mean `AUX[0]` over the shifted target window
- window / delta / step: `300 ms / -150 ms / 50 ms`
- epochs / batch / lr / dropout / base channels: `350 / 16 / 3e-4 / 0.15 / 16`
- pooled train/eval RMSE: `0.553339`
- pooled train/eval MAE: `0.359372`
- pooled train/eval Pearson: `0.796952`
- pooled train/eval R2: `0.631759`
- dynamic recordings at `R2 >= 0.98`: `4 / 18`

Loss curve:

![Participants 4, 9, and 10 combined loss curve](./runs/p4_p9_p10_combined_overfit_demo_raw300_delta-150_20260328_174523/loss_curve.png)

Interpretation:

- this **did not** overfit reasonably successfully with one unconditioned shared model
- the initial shared raw 1D CNN run failed mainly on participants 9 and 10, even though participant 4 remained partly learnable inside the combined pool
- a shared `-125 ms` timing probe was worse, and a wider raw CNN probe was also worse
- so the current evidence is that the same simple shared raw-CNN setup that works for one participant at a time does **not** scale cleanly to pooled same-window overfit across participants 4, 9, and 10

## Model that worked in step 1 and step 2

The strongest model so far is a **raw 1D CNN scalar regressor**:

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

Step-2 hyperparameters that worked best:

- model: `cnn1d_raw_scalar`
- recordings: all cleaned `.npz` files in `extra_patients/participant 4 stuff`
- target channel: `AUX[0]`
- `win_ms=300`
- `delta_ms=-150`
- `step_ms=50`
- `epochs=250`
- `batch_size=16`
- `lr=3e-4`
- `weight_decay=1e-5`
- `dropout=0.15`
- `base_channels=16`
- `seed=13`
- device: `cuda`

Best participant-10 validation variant:

- model: `cnn1d_raw_scalar`
- recordings: all cleaned `.npz` files in `extra_patients/participant 10 stuff`
- target channel: `AUX[0]`
- `win_ms=300`
- `delta_ms=-100`
- `step_ms=50`
- `epochs=350`
- `batch_size=16`
- `lr=3e-4`
- `weight_decay=1e-5`
- `dropout=0.15`
- `base_channels=16`
- `seed=13`
- device: `cuda`

## What currently looks most promising for the next stages

Given the shape of the data and the task, the most promising direction right now is:

- **Primary candidate:** raw 1D CNN with `300 ms` windows and a target offset in roughly the `-150 ms` to `-100 ms` range
- **Secondary / lighter reference candidate:** feature 1D CNN with the same `300 ms / -150 ms` timing

Why this is our current view:

- each EMG window is short enough for 1D temporal convolutions to be practical
- each window still contains a lot of spatial information (`6 x 8 x 8 = 384` channels), so keeping the raw window is useful
- the raw 1D CNN consistently beat the nearby feature-based alternatives on the participant-4, participant-9, and participant-10 pooled comparisons
- participants 4 and 9 were best at `300 ms / -150 ms`, while participant 10 was best at `300 ms / -100 ms`
- for pooled same-person memorization, simply training the same raw model longer worked better than widening it or switching to the feature CNN
- participant 9 suggests the same architecture transfers even when one participant is harder, and participant 10 shows that a modest timing adjustment can recover strong per-recording overfit
- the pooled 4+9+10 experiment shows a new limit: one unconditioned shared raw 1D CNN with one global timing pair did **not** memorize the multi-participant cohort well enough, even in same-window mode

So for the next stages, the raw 1D CNN is still the first model we should keep testing, and the feature 1D CNN is still the sensible backup / comparison baseline. But the multi-participant pooled-overfit failure means we should **not** assume that the same simple shared setup will scale to future multi-person training. For future predictive stages, the current best starting point is still the raw architecture with `300 ms` windows and a lag in the `-150 ms` to `-100 ms` range, but multi-person work will likely need a stronger strategy than one plain shared model with one global timing pair.

## Next planned modeling steps

The next steps are **not implemented yet**. They are the planned progression after step 2:

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
- [`src/training/participant_overfit.py`](./src/training/participant_overfit.py)
  - pooled participant-level overfit workflow, per-recording reporting, and shuffle audit generation
- [`src/training/cohort_overfit.py`](./src/training/cohort_overfit.py)
  - pooled multi-participant overfit workflow with per-participant and per-recording reporting
- [`scripts/train_single_recording.py`](./scripts/train_single_recording.py)
  - main training entrypoint for the current step-1 style runs
- [`scripts/train_participant_overfit.py`](./scripts/train_participant_overfit.py)
  - pooled participant-level training entrypoint for the current step-2 style runs
- [`scripts/train_cohort_overfit.py`](./scripts/train_cohort_overfit.py)
  - pooled multi-participant training entrypoint for the harder combined-cohort overfit experiment
- [`scripts/run_single_recording_sweep.py`](./scripts/run_single_recording_sweep.py)
  - timing/model sweep helper
- [`scripts/compare_participant_overfit_runs.py`](./scripts/compare_participant_overfit_runs.py)
  - helper for comparing multiple pooled participant-level runs
- [`scripts/compare_cohort_overfit_runs.py`](./scripts/compare_cohort_overfit_runs.py)
  - helper for comparing multiple pooled multi-participant runs
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

### 4. Train one pooled participant-level run

```powershell
venv\Scripts\python scripts\train_participant_overfit.py `
  --recordings_dir "extra_patients\participant 4 stuff" `
  --model cnn1d_raw_scalar `
  --win_ms 300 `
  --delta_ms -150 `
  --step_ms 50 `
  --epochs 250 `
  --batch_size 16 `
  --lr 3e-4 `
  --weight_decay 1e-5 `
  --dropout 0.15 `
  --base_channels 16 `
  --device cuda `
  --require_venv `
  --require_cuda `
  --run_dir runs\example_participant_overfit
```

### 5. Train one pooled multi-participant run

```powershell
venv\Scripts\python scripts\train_cohort_overfit.py `
  --recordings_dir "extra_patients\participant 4 stuff" `
  --recordings_dir "extra_patients\participant 9 stuff" `
  --recordings_dir "extra_patients\participant 10 stuff" `
  --model cnn1d_raw_scalar `
  --win_ms 300 `
  --delta_ms -150 `
  --step_ms 50 `
  --epochs 350 `
  --batch_size 16 `
  --lr 3e-4 `
  --weight_decay 1e-5 `
  --dropout 0.15 `
  --base_channels 16 `
  --device cuda `
  --require_venv `
  --require_cuda `
  --run_dir runs\example_cohort_overfit
```

## Research status

This repository is still a research workflow, not a deployment-ready ventilator controller.

What we have shown so far:

- preprocessing pipeline exists and exports cleaned diaphragm-focused EMG recordings
- step 1 overfit on one recording of one participant works very well
- step 2 pooled overfit on all recordings of one participant with one shared model also works well
- participant-10 validation shows that the same raw model family remains strongest, and that a small timing shift can materially improve pooled same-person overfit
- pooled overfit across participants 4, 9, and 10 with one unconditioned shared model did **not** work well enough, which is useful evidence that multi-person training is qualitatively harder than same-person pooled memorization

What is still missing:

- same-person predictive testing
- multi-person training with adaptation to a new person after a few observed recordings
