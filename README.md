# Neonatal Diaphragm sEMG Visualizer & Artifact Rejection Pipeline

This repository contains the code for my bachelor thesis project on **surface EMG (sEMG)** of the **diaphragm** in neonates connected to an assisted ventilation machine.

The long-term goal of the project is to help build a **real-time, adaptive ventilator control system** that can:

* Read sEMG from the diaphragm via 6 high-density (8×8) electrode grid arrays.
* Infer **breathing effort / breathing difficulty** from the cleaned signals.
* Output **suggestions to increase or decrease ventilator pressure** to avoid under-assistance (dyspnea) or over-assistance (lung injury).

This repo currently focuses on **Step 1 of that pipeline**:

> **Robust visualization and separation of heart + noise artifacts from the diaphragm sEMG, and inspection of the cleaned diaphragm activity over the torso.**

---

## High-Level Overview

The code implements a **Python GUI application** that lets me:

1. Load `.mat` files containing high-density diaphragm sEMG.
2. Automatically:

   * Detect global heartbeats from chest grids.
   * Run ICA on each of the 6 grids.
   * Classify ICA components as:

     * **HEART artifacts** (cardiac contamination),
     * **High-frequency noise**, or
     * **Spiky artifacts**.
   * Zero out those components and reconstruct:

     * A **“cleaned” multi-channel sEMG time series**.
     * A **rectified, low-passed envelope heatmap** for each grid (8×8).
3. Visualize:

   * Full-torso heatmap animations (6 grids laid out anatomically).
   * Single-grid heatmap animations.
   * Multi-channel time series for raw vs cleaned signals.
   * Inspect, override, and manually correct which ICA components are removed.

Later project stages will use these cleaned signals/envelopes to estimate **breathing difficulty** and suggest ventilator pressure adjustments.

---

## Input Data & Electrode Layout

### Expected `.mat` structure

The app expects a `.mat` file with at least:

* `sEMG` – raw surface EMG recordings, shape: **(n_channels × n_samples)** or similar.

  * The code transposes it to **(n_samples, n_channels)**.
* `fsamp` – sampling rate in Hz (stored as `self.fs`).

Internally, the code assumes:

* **6 grids**, each with **64 channels**, so:

  * `n_channels = 6 × 64 = 384`.
* Channels are ordered per grid as:

  * Grid 1: indices `[0 : 64]`
  * Grid 2: `[64 : 128]`
  * Grid 3: `[128 : 192]`
  * Grid 4: `[192 : 256]`
  * Grid 5: `[256 : 320]`
  * Grid 6: `[320 : 384]`

Within each grid, the 64 channels are reshaped as an **8×8 matrix**, then transposed and flipped vertically for more anatomical visualization.

### Anatomical grid layout

`GRID_CONFIG` maps the 6 grids to torso positions:

```python
GRID_CONFIG = {
    0: {"name": "IN 1 (Right Chest)",   "pos": (0, 0)},
    1: {"name": "MULTI 1 (Right Ribs)", "pos": (1, 0)},
    2: {"name": "MULTI 2 (Right Abs)",  "pos": (2, 0)},
    3: {"name": "MULTI 3 (Left Abs)",   "pos": (2, 1)},
    4: {"name": "MULTI 4 (Left Ribs)",  "pos": (1, 1)},
    5: {"name": "IN 5 (Left Chest)",    "pos": (0, 1)},
}
```

When visualizing the **full torso**, these grids are placed in a 3×2 layout matching this configuration.

---

## Signal Processing Pipeline

### 1. Loading data

Function: `load_mat_any(path)`

* Loads a `.mat` file with `scipy.io.loadmat`.
* Filters out internal metadata keys.
* Expects `sEMG` and `fsamp` to be present.

In the GUI (`EMGControlPanel.load_data`):

* `self.sEMG_data = data['sEMG'].T` → shape `(n_samples, n_channels)`.
* `self.fs = data['fsamp']`.
* Pre-computes `raw_cache`:

  * For each of the 6 grids:

    * Extract 64 channels → reshape into `(n_samples, 8, 8)`.
    * Transpose & flip vertically for display.
    * Store rectified magnitude (`np.abs`) for instant raw heatmap visualization.

### 2. Basic filtering

Function: `filter_data(data, fs)`

Applied to each grid chunk (64 channels) prior to ICA.

1. **50 Hz notch filter** to remove mains interference:

   * `iirnotch(w0=50.0, Q=30.0, fs=fs)`
   * Applied with `filtfilt` over time axis.

2. **20–400 Hz bandpass filter** for EMG:

   * 4th order Butterworth:

     * `butter(N=4, Wn=[20, 400], btype='bandpass', fs=fs)`
   * Applied with `filtfilt`.

This step removes low-frequency drift and high-frequency noise outside typical diaphragm EMG.

### 3. ICA decomposition

Function: `ICA(data)`

* Uses `FastICA` from scikit-learn:

  * `n_components = 20`
  * `whiten='unit-variance'`
  * `max_iter=1000`, `tol=0.005`
* Returns:

  * Fitted ICA model.
  * Matrix of independent component time series: `sources` (shape `(n_samples, n_components)`).

This is the basis for artifact separation.

### 4. Global heartbeat detection (“Captain”)

Function: `detect_global_heartbeats(sEMG, fs)`

Goal: detect a **global heartbeat train** that will act as a synchronization reference for identifying heart-related components.

Steps:

1. **Select chest channels**:

   * Uses channels corresponding to **Grid 1** and **Grid 6**:

     * `sEMG[:, 0:64]` (Right Chest)
     * `sEMG[:, 320:384]` (Left Chest)
   * Horizontally stacks them into `chest_data`.

2. **Bandpass for heart frequencies**:

   * 2nd order Butterworth 10–100 Hz:

     * `butter(N=2, Wn=[10, 100], btype='bandpass', fs=fs)`.
   * Filtered via `filtfilt`.

3. **RMS envelope**:

   * Calculates an RMS across all chest channels per sample:

     * `master_lead = sqrt(mean(filtered**2, axis=1))`.

4. **Peak detection** (QRS-like events):

   * Uses `scipy.signal.find_peaks`:

     * `height = median(master_lead) * 3`
     * `distance = int(fs * 0.35)` (~350 ms minimum, enforcing physiological heart rate range).

The output `global_heart_peaks` (array of sample indices) is used later to check if a component is **synchronized with the global heart rhythm**.

### 5. Component classification (heart vs noise vs artifact)

Function: `analyze_envelope_refined(sources, fs, global_heart_peaks=None)`

Goal: decide which ICs belong to:

* **HEART** (cardiac contamination),
* **HF noise** (dominated by high frequencies),
* **Spiky artifacts** (irregular bursts not synchronized with heart).

Key elements:

* **Temporal downsampling**:

  * Uses a stride of 4: `sig_stat = sig[::stride]`.
  * Effective stats sampling rate: `fs_stat = fs / stride`.

* For each IC:

  1. **Kurtosis** (`k`):

     * Measures how “peaky” the component is.

  2. **Envelope** (`sig_env`):

     * Defined as `abs(sig_stat - median(sig_stat))`.

  3. **Frequency content**:

     * Uses Welch PSD on `sig_env`:

       * `nperseg = int(fs_stat * 4)` (4-second windows).
     * Dominant frequency `dom_freq` is the frequency with max PSD.

  4. **Peak detection with robust threshold**:

     * Computes `signal_iqr = iqr(sig_env)`.
     * Baseline `floor = median(sig_env)`.
     * Peak threshold:

       * `peak_thresh = floor + 2.5 * signal_iqr`.
     * Minimum distance between peaks:

       * `distance = int(fs_stat * 0.4)`.

  5. **Regularity** of peaks:

     * `regularity = std(diff(peaks) / fs_stat)` if enough peaks.
     * Lower std = more regular rhythm.

  6. **Global synchronization (“capture rate”)**:

     * If `global_heart_peaks` is provided:

       * Scales them to the downsampled domain: `scaled_global_peaks`.
       * For each global peak, finds nearest local peak.
       * If distance < `sync_tol_samples = int(0.05 * fs_stat)` (±50 ms), counts a match.
       * `capture_rate = matches / number_of_global_peaks`.

* **Classification logic:**

  * **High-frequency noise**:

    * If `dom_freq > 12 Hz` → IC is classified as **HF noise**.

  * **Heart components**:

    * If `capture_rate > 0.50`:

      * Mark as **HEART** regardless of local stats, and boost `k` by +500 (makes it rank higher).
    * Else if:

      * `0.7 <= dom_freq <= 3.0` (plausible heart/breath rhythm band),
      * `k > 1.5` (peaky),
      * `regularity < 0.20` (periodic),
      * → Mark as **HEART**.

  * **Spiky artifacts**:

    * If `0.7 <= dom_freq <= 3.0` and `k > 1.5`, but no good regularity or sync, mark as **ARTIFACT**.

* The function returns:

  * `final_hearts` – up to 6 heart-related component indices.
  * `hf_noise` – high-frequency noise component indices.
  * `final_artifacts` – artifact indices.

All of these become **“bad components”** to be zeroed out.

### 6. Reconstruction & heatmap computation

Function: `reconstruct_heatmap(ica, sources, bad_indices, fs)`

Given:

* Fitted ICA model,
* All sources `sources`,
* List of indices to remove (`bad_indices`),

it:

1. Copies `sources` to `sources_clean`.
2. Sets `sources_clean[:, bad_indices] = 0.0`.
3. Reconstructs cleaned time series:

   * `clean_signal = ica.inverse_transform(sources_clean)`.
4. Computes **rectified signal**:

   * `rectified = abs(clean_signal)`.
5. Computes **envelope** for each channel:

   * 4th order low-pass at 3 Hz:

     * `butter(N=4, Wn=3.0, btype='low', fs=fs)`.
   * Applied via `filtfilt` → `envelopes`.
6. Reshapes into grid geometry:

   * `envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]`.
   * Returns:

     * `grid_matrix`: (n_time, 8, 8) envelopes suitable for heatmap animation.
     * `clean_signal`: (n_samples, 64) cleaned multi-channel time series.

### 7. Parallel processing per grid

Function: `process_single_grid_wrapper(args)`

This is a worker function for the parallel pipeline:

* Inputs: `(raw_chunk, fs, grid_id, global_heart_peaks)`.
* Steps:

  1. Filter data (`filter_data`).
  2. Run ICA (`ICA`).
  3. Classify components (`analyze_envelope_refined`).
  4. Reconstruct cleaned signals & envelopes (`reconstruct_heatmap`).
* Returns a dictionary per grid with:

  * `grid_id`
  * `heatmap` (envelope 8×8 over time)
  * `clean_signal` (time series)
  * `ica_model`
  * `sources`
  * `bad_indices` (components removed)

In `EMGControlPanel.run_processing_task`, a `ProcessPoolExecutor` runs this worker over all 6 grids in parallel.

---

## GUI & Visualization Features

### EMGControlPanel (Main Window)

Key features:

* **File section**:

  * `Browse...` to select `.mat` file.
  * Displays filename and sampling rate once loaded.

* **Analysis pipeline**:

  * `Run Auto-Processing (All Grids)`:

    * Detects global heartbeats.
    * Spawns parallel processing for all grids.
    * Caches results in `self.grid_states`.

* **Visualization settings**:

  * **Target grid**:

    * `Full Torso (Anatomical View)` or specific `Grid 1...6`.
  * **Inspect & Edit Components**:

    * Open the `ComponentEditor` for a chosen grid.
    * See all ICA components (first 4 seconds).
    * Click on plots to toggle “bad” vs “good” components (background changes).
    * Confirm to recompute `heatmap` and `clean_signal` for that grid.
  * **Signal source**:

    * `Processed (Cleaned)` vs `Original (Raw)`.
  * **View mode**:

    * `Heatmap Animation` (full torso or single grid).
    * `Time Series (Individual Grids Only)`:

      * Opens a scrollable window of 64 subplots (channels) for the selected grid and time range.
  * **Color scaling** (heatmap):

    * `Global` – same vmax across all grids to compare relative intensity.
    * `Local` – per-grid automatic scaling for maximum contrast.

* **Time range selection**:

  * Two horizontal sliders:

    * `Start Time` and `End Time` (in seconds).
    * Enforces at least a 2-second window.
  * All visualizations (heatmaps and time series) only use data in this time window.

* **Visualize Result**:

  * Triggers background visualization:

    * For `Heatmap + Full Torso`:

      * Calls `visualize_torso_animation`.
    * For `Heatmap + Single Grid`:

      * Calls `visualize_grid_animation_scaled`.
    * For `Time Series`:

      * Opens `RawSignalViewer` for raw/processed signals in chosen time window.

### RawSignalViewer

* New window with scrollable panel.
* Creates a grid of subplots (4 columns) for each channel.
* Shows **time on X axis** (with correct offset based on `start_time`) and **amplitude on Y**.
* Works for both raw and processed signals, with all axes visible.

### ComponentEditor

* Scrollable panel with all ICA components for a grid.
* Background color:

  * Green (`#e8f5e9`) → **kept** component.
  * Red (`#ffebee`) → **removed** component.
* Click on a component plot to toggle.
* `Confirm & Save` recomputes that grid’s heatmap and cleaned signals.

---

## Current Status (What’s Already Implemented)

Use these as checkable items to reflect current state:

* [x] Load neonate diaphragm sEMG from `.mat` files with `sEMG` and `fsamp`.
* [x] Map 384-channel recording into 6×(8×8) grids with anatomical torso layout.
* [x] 50 Hz notch + 20–400 Hz bandpass filtering per grid.
* [x] Global heartbeat detection from chest grids using RMS envelope and peak detection.
* [x] ICA decomposition (20 components per grid) for artifact separation.
* [x] Hybrid component classification combining:

  * Local stats (kurtosis, IQR-based envelope peaks, regularity).
  * Global synchronization with heartbeats (“capture rate”).
* [x] Reconstruction of cleaned multi-channel time series and 8×8 envelope heatmaps.
* [x] Full-torso heatmap animation with anatomical overlays and time display.
* [x] Single-grid heatmap animation with optional global or local scaling.
* [x] Time-series viewer for raw vs processed signals on any grid and time window.
* [x] GUI controls for:

  * Selecting grids, modes, time ranges.
  * Inspecting and editing ICA components per grid.
* [x] Parallelized per-grid processing using `ProcessPoolExecutor`.

---

## How to Run

### Requirements

* Python 3.10+ (recommended)
* Packages:

  * `numpy`
  * `scipy`
  * `matplotlib`
  * `scikit-learn`
  * `tkinter` (bundled with standard Python on most systems)
  * `concurrent.futures` (standard library)

Install dependencies (example):

```bash
pip install numpy scipy matplotlib scikit-learn
```

### Running the app

Assuming the main file is called `main.py`:

```bash
python main.py
```

Then:

1. Click **“Browse…”** and select a `.mat` file with `sEMG` and `fsamp`.
2. Wait for **“Status: Raw data ready.”**.
3. Optionally adjust **Start/End time** sliders.
4. Click **“Run Auto-Processing (All Grids)”**.

   * Wait until the status shows **“Processing Complete.”**.
5. Optionally:

   * Use **“Inspect & Edit Components”** to fine-tune which ICA components are removed.
6. Choose:

   * `Signal Source` (Processed vs Original),
   * `View Mode` (Heatmap vs Time Series),
   * `Target Grid` (Full Torso or specific grid).
7. Click **“Visualize Result”** to open the selected view.

---

## How This Fits Into the Thesis

This repository implements the **artifact rejection and visualization layer** of a future **real-time ventilator-adaptive controller** based on diaphragmatic sEMG.

In the thesis, this corresponds to:

* **Data pre-processing**:

  * Filtering, ICA, artifact classification, heart suppression.
* **Spatial-temporal visualization**:

  * Torso heatmaps showing how diaphragm activation spreads over time.
* **Interactive inspection**:

  * Human-in-the-loop adjustment of ICA decisions (expert oversight).

The next stages of the thesis will:

1. **Extract features** from the cleaned signals/heatmaps that correlate with breathing effort and difficulty.
2. **Define a breathing difficulty index** per breath or per time window.
3. **Map that index to ventilator pressure change suggestions**, at least in a simulated offline way.

---

## Roadmap / TODO (Tickable Checklist)

The roadmap is split into phases so I can tick items as I complete them.

### Phase 1 - Preprocessing (Leave Only Diaphragm Signal)
- Removes dead channels
- Remvoes heart signal
- Removes noise

### Phase 2 - CNN to predict what air pump value should be

---

## Notes for Future Me

* The current code is tuned primarily for **visualization and exploratory analysis**.
  For real-time ventilator control, the focus will shift to:

  * Reducing latency,
  * Ensuring deterministic behavior,
  * And validating the pipeline clinically.
* The **component editor** is an important tool to:

  * Build intuition about which components correspond to heart, motion, and diaphragm activity.
  * Validate the automatic classification rules on real data before fully trusting them.
