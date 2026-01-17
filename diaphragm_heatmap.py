import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.io as sio
from scipy.stats import kurtosis, iqr
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import concurrent.futures

GRID_CONFIG = {
    0: {"name": "IN 1 (Right Chest)",   "pos": (0, 0)},
    1: {"name": "MULTI 1 (Right Ribs)", "pos": (1, 0)},
    2: {"name": "MULTI 2 (Right Abs)",  "pos": (2, 0)},
    3: {"name": "MULTI 3 (Left Abs)",   "pos": (2, 1)},
    4: {"name": "MULTI 4 (Left Ribs)",  "pos": (1, 1)},
    5: {"name": "IN 5 (Left Chest)",    "pos": (0, 1)},
}


# BACKEND: SIGNAL PROCESSING
def load_mat_any(path):
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k:v for k,v in d.items() if not k.startswith("__")}
    except Exception as e:
        print(f"Error loading MAT: {e}")
        return None

def filter_data(data, fs):
    # 50Hz Notch
    b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
    data = filtfilt(b_notch, a_notch, data, axis=0)
    # 20-400Hz Bandpass
    b_band, a_band = butter(N=4, Wn=[20, 400], btype='bandpass', fs=fs)
    data = filtfilt(b_band, a_band, data, axis=0)
    return data

def detect_bad_electrodes(filtered_grid, fs):
    """
    Detect obviously bad electrodes in a single 8x8 grid.

    Strategy:
      - Always flag clearly flat channels (almost no RMS).
      - Flag channels that are MUCH louder than their immediate neighbors.
      - Flag channels that are strong global amplitude outliers AND decorrelated
        from their immediate neighbors.
      - Additionally, if there are a few *extremely* loud outliers, mark them
        bad outright (they will dominate ICA otherwise).
    """
    x = np.asarray(filtered_grid)
    n_samples, n_channels = x.shape
    assert n_channels == 64, "Expected 64 channels per grid."

    # -------- 1) RMS-based checks (flat + global outliers) --------
    rms = np.sqrt(np.mean(x**2, axis=0))  # (64,)
    med_rms = np.median(rms)
    mad_rms = np.median(np.abs(rms - med_rms)) + 1e-12  # robust scale

    # Definitely flat (dead / detached electrodes)
    too_flat = np.where(rms < 0.03 * med_rms)[0]   # <3% of median RMS

    # Globally loud (candidate for badness)
    z_rms = (rms - med_rms) / mad_rms
    global_loud = np.where(z_rms > 4.0)[0]         # >4 MAD above median

    # EXTREMELY loud outliers: auto-bad if they are few
    EXTREME_Z = 7.0          # very strong outliers in z-score
    EXTREME_MULT = 8.0       # or >8x median RMS
    extremely_loud = np.where(
        (z_rms > EXTREME_Z) | (rms > EXTREME_MULT * med_rms)
    )[0]

    bad_indices = set(too_flat.tolist())

    # If only a handful of channels are absurdly loud, treat them as bad outright
    MAX_EXTREME_PER_GRID = 5
    if 0 < len(extremely_loud) <= MAX_EXTREME_PER_GRID:
        bad_indices.update(extremely_loud.tolist())

    # -------- neighbor utility --------
    def neighbors(idx):
        """Return indices of up/down/left/right neighbors in the 8x8 layout."""
        r = idx // 8   # 0..7
        c = idx % 8    # 0..7
        neigh = []
        if r > 0: neigh.append((r - 1) * 8 + c)
        if r < 7: neigh.append((r + 1) * 8 + c)
        if c > 0: neigh.append(r * 8 + (c - 1))
        if c < 7: neigh.append(r * 8 + (c + 1))
        return neigh

    # -------- 2) Local amplitude ratio: much louder than neighbors --------
    LOCAL_RATIO_THRESH = 3.0  # >3x neighbor median RMS

    for ch in range(n_channels):
        if ch in bad_indices:
            continue  # already bad (flat or extreme)

        neigh = neighbors(ch)
        if not neigh:
            continue

        neigh_rms = rms[neigh]
        neigh_rms = neigh_rms[neigh_rms > 0]
        if len(neigh_rms) == 0:
            continue

        local_med = np.median(neigh_rms)
        if local_med <= 0:
            continue

        if rms[ch] > LOCAL_RATIO_THRESH * local_med:
            bad_indices.add(ch)
            continue

    # -------- 3) Global loud + decorrelated from neighbors --------
    CORR_THRESH = 0.2  # very low median correlation

    for ch in global_loud:
        if ch in bad_indices:
            continue  # already flagged

        sig = x[:, ch]
        if np.std(sig) < 1e-9:
            continue

        neigh = neighbors(ch)
        if not neigh:
            continue

        corrs = []
        for n in neigh:
            sig_n = x[:, n]
            if np.std(sig_n) < 1e-9:
                continue
            c = np.corrcoef(sig, sig_n)[0, 1]
            if np.isfinite(c):
                corrs.append(c)

        if not corrs:
            continue

        median_corr = np.median(corrs)

        if median_corr < CORR_THRESH:
            bad_indices.add(ch)

    return sorted(bad_indices)

def ICA(data, downsampled):
    max_components = 16
    n_timestamps = downsampled.shape[0]
    n_electrodes = downsampled.shape[1]
    
    n_components = min(max_components, n_timestamps, n_electrodes)

    ica = FastICA(n_components=n_components, random_state=42, 
                  whiten='unit-variance', max_iter=1000, tol=0.005)
    ica.fit(downsampled)

    sources_downsampled = ica.transform(downsampled)
    sources_full_res = ica.transform(data)

    return ica, sources_downsampled, sources_full_res

def detect_global_heartbeats(sEMG, fs):
    """
    [Global Logic] Finds the 'Captain' heartbeat from Chest Grids (1 & 6).
    """
    print("   [Captain] Detecting Global Heartbeat...")
    # Extract Grid 1 and Grid 6 (Indices 0-64 and 320-384)
    chest_data = np.hstack([sEMG[:, 0:64], sEMG[:, 320:384]])
    
    # Filter for Heart Frequencies (10-100Hz)
    b, a = butter(N=2, Wn=[10, 100], btype='bandpass', fs=fs)
    filtered = filtfilt(b, a, chest_data, axis=0)
    
    # RMS Envelope to find QRS
    master_lead = np.sqrt(np.mean(filtered**2, axis=1))
    
    # Peak Detection
    peaks, _ = find_peaks(master_lead, height=np.median(master_lead)*3, distance=int(fs*0.35))
    return peaks

def analyze_envelope_refined(sources, fs, global_heart_peaks=None):
    """
    Hybrid Logic: Combines Robust Local Stats (IQR) with Global Sync (Captain).
    """
    MAX_HEART_COMPONENTS = 6
    candidates = []
    hf_noise = []
    
    stride = 4
    fs_stat = fs / stride
    
    # Prepare Global Sync Data
    sync_tol_samples = int(0.05 * fs_stat) # +/- 50ms tolerance
    if global_heart_peaks is not None and len(global_heart_peaks) > 0:
        scaled_global_peaks = (global_heart_peaks / stride).astype(int)
    else:
        scaled_global_peaks = []

    for i in range(sources.shape[1]):
        sig = sources[:, i]
        sig_stat = sig[::stride]
        
        # 1. Stats
        k = kurtosis(sig_stat)
        sig_env = np.abs(sig_stat - np.median(sig_stat))
        
        # 2. Frequency
        freqs, psd = welch(sig_env, fs_stat, nperseg=int(fs_stat*4))
        dom_freq = freqs[psd.argmax()] if len(psd) > 0 else 0
        
        # 3. Robust Peak Finding (Local State Awareness)
        signal_iqr = iqr(sig_env)
        floor = np.median(sig_env)
        peak_thresh = floor + 2.5 * signal_iqr # Slightly lower to allow Capture Rate to work
        peaks, _ = find_peaks(sig_env, height=peak_thresh, distance=int(fs_stat*0.4))
        
        # 4. Metrics
        regularity = np.std(np.diff(peaks) / fs_stat) if len(peaks) > 3 else 10.0
        
        capture_rate = 0.0
        if len(scaled_global_peaks) > 0 and len(peaks) > 0:
            match_count = 0
            for g_peak in scaled_global_peaks:
                dist = np.min(np.abs(peaks - g_peak))
                if dist < sync_tol_samples: match_count += 1
            capture_rate = match_count / len(scaled_global_peaks)

        # --- CLASSIFICATION ---
        
        if dom_freq > 12.0:
            hf_noise.append(i)
            continue
            
        # LOGIC: 
        # If it syncs with Global Heart (>50%), it IS a heart, even if local stats are messy.
        # If it doesn't sync, fall back to Local Robust Logic.
        
        is_heart = False
        
        if capture_rate > 0.50:
            is_heart = True
            k += 500 # Boost priority
        elif (0.7 <= dom_freq <= 3.0) and (k > 1.5) and (regularity < 0.20):
            is_heart = True
            
        if is_heart:
            candidates.append({'id': i, 'kurt': k, 'type': 'HEART'})
        elif (0.7 <= dom_freq <= 3.0) and (k > 1.5):
             # Spiky but not rhythmic/synced -> Artifact
            candidates.append({'id': i, 'kurt': k, 'type': 'ARTIFACT'})

    candidates.sort(key=lambda x: x['kurt'], reverse=True)
    final_hearts = []
    final_artifacts = []
    
    for c in candidates:
        if c['type'] == 'HEART':
            if len(final_hearts) < MAX_HEART_COMPONENTS:
                final_hearts.append(c['id'])
        else:
            final_artifacts.append(c['id'])

    return final_hearts, hf_noise, final_artifacts

def reconstruct_heatmap(ica, sources, bad_ICA_components, bad_electrodes, fs, good_mask=None, n_channels=64):
    """
    Reconstructs 'Clean' signal and 'Heatmap' envelope.

    Parameters
    ----------
    ica : FastICA
        Fitted ICA model.
    sources : np.ndarray
        Shape (n_samples, n_components), ICA sources.
    bad_ICA_components : list[int]
        Source indices to zero out (heart, noise, artifacts).
    bad_electrodes : list[int]
        Electrode indices to zero out in the final signal.
    fs : float
        Sampling frequency.
    good_mask : np.ndarray[bool] or None
        If not None, boolean mask of length n_channels indicating which
        channels were used in ICA (True = used, False = bad electrode).
        If None, we assume ICA was run on all channels and no expansion
        is required.
    n_channels : int
        Total number of channels in the grid (default 64).

    Returns
    -------
    grid_matrix : np.ndarray
        Shape (n_time, 8, 8) envelope values for heatmap.
    clean_signal : np.ndarray
        Shape (n_samples, n_channels), reconstructed clean signals.
        Bad electrodes are set to 0 if good_mask is provided.
    """
    sources_clean = sources.copy()
    if bad_ICA_components:
        sources_clean[:, bad_ICA_components] = 0.0

    # Reconstruct only the channels that ICA knows about
    clean_subset = ica.inverse_transform(sources_clean)  # (n_samples, n_used)

    if good_mask is None:
        # Old behavior: ICA was run on all channels
        clean_signal = clean_subset
    else:
        # Expand back to full 64-channel layout
        clean_signal = np.zeros((clean_subset.shape[0], n_channels), dtype=clean_subset.dtype)
        clean_signal[:, good_mask] = clean_subset

    if bad_electrodes:
        clean_signal[:, bad_electrodes] = 0.0

    # Envelope & geometry
    rectified = np.abs(clean_signal)
    b_env, a_env = butter(N=4, Wn=3.0, btype='low', fs=fs)
    envelopes = filtfilt(b_env, a_env, rectified, axis=0)

    # This assumes 64 channels = 8x8 layout
    grid_matrix = envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]

    return grid_matrix, clean_signal

def process_single_grid_wrapper(args):
    """Parallel Worker."""
    raw_chunk, fs, grid_id, global_heart_peaks = args
    print(f"[Core Task] Processing Grid {grid_id}...")

    # 1) Classic filtering (notch + 20–400 Hz bandpass)
    filtered = filter_data(raw_chunk, fs)  # (n_samples, 64)

    # 2) Detect bad electrodes on filtered data
    bad_electrodes = detect_bad_electrodes(filtered, fs)
    print(f"Grid {grid_id}: {len(bad_electrodes)} / 64 electrodes flagged bad.")

    good_mask = None
    filtered_for_ica = filtered

    if len(bad_electrodes) > 0:
        mask = np.ones(filtered.shape[1], dtype=bool)
        mask[bad_electrodes] = False

        # Require at least a few good channels; otherwise, fall back to using all
        if np.sum(mask) >= 5:
            good_mask = mask
            filtered_for_ica = filtered[:, good_mask]
        else:
            print(f"[Warning] Grid {grid_id}: too many channels flagged bad; "
                  f"running ICA on all channels instead.")
            good_mask = None
            filtered_for_ica = filtered
            # keep bad electrodes as is

    # 3.0) Downsample for ICA
    ds_stride = 2 if fs >= 1600 else 1

    if ds_stride > 1:
        filtered_ds = filtered_for_ica[::ds_stride, :]
        fs_ds = fs / ds_stride
    else:
        filtered_ds = filtered_for_ica
        fs_ds = fs

    # 3) ICA only on good channels (or all if good_mask is None)
    ica, sources_downsampled, sources_full_res = ICA(filtered_for_ica, filtered_ds)

    # 4) Component classification
    hearts, hf, artifacts = analyze_envelope_refined(sources_downsampled, fs_ds, global_heart_peaks)
    bad_components = hearts + hf + artifacts

    # 5) Reconstruct to full 8x8 grid, zeroing bad electrodes in the clean signal
    heatmap, clean_signal = reconstruct_heatmap(
        ica,
        sources_full_res,
        bad_components,
        bad_electrodes,
        fs,
        good_mask=good_mask,
        n_channels=filtered.shape[1]
    )

    return {
        'grid_id': grid_id,
        'heatmap': heatmap,            # For Heatmap Viz (8x8 envelopes over time)
        'clean_signal': clean_signal,  # For Time Series Viz (bad electrodes = 0)
        'ica_model': ica,
        'sources': sources_full_res,
        'bad_indices': bad_components, # bad ICA components
        'bad_electrodes': bad_electrodes,
        'good_mask': good_mask,
    }

#        WINDOWS & VIEWERS


class RawSignalViewer(tk.Toplevel):
    def __init__(self, parent, data, fs, grid_name, start_time=0.0, bad_channels=None):
        super().__init__(parent)
        self.title(f"Time Series: {grid_name}")
        self.geometry("1400x900")
        self.data = data
        self.fs = fs
        self.start_time = start_time
        self.n_channels = data.shape[1]
        self.bad_channels = set(bad_channels) if bad_channels is not None else set()
        
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.plot_signals()

    def plot_signals(self):
        cols = 4
        rows = int(np.ceil(self.n_channels / cols))
        fig = Figure(figsize=(15, 1.2 * rows), dpi=100)

        n_samples = self.data.shape[0]
        time_vec = np.arange(n_samples) / self.fs + self.start_time

        for i in range(self.n_channels):
            ax = fig.add_subplot(rows, cols, i+1)

            is_bad = i in self.bad_channels
            color = '#e74c3c' if is_bad else '#34495e'
            title_suffix = " (BAD)" if is_bad else ""

            ax.plot(time_vec, self.data[:, i], linewidth=0.8, color=color)
            ax.set_title(f"Ch {i+1}{title_suffix}", fontsize=8, loc='left', pad=1)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlim(time_vec[0], time_vec[-1])

            # Show ticks for all charts
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

            # Background color indicates status
            ax.set_facecolor('#ffebee' if is_bad else '#f8f9fa')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class ComponentEditor(tk.Toplevel):
    def __init__(self, parent, sources, fs, auto_bad_indices, grid_name, callback):
        super().__init__(parent)
        self.title(f"Inspect Components: {grid_name}")
        self.geometry("1400x900")
        
        self.sources = sources
        self.fs = fs
        self.bad_indices = set(auto_bad_indices)
        self.callback = callback
        self.n_components = sources.shape[1]
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="top", fill="x", padx=10, pady=5)
        ttk.Label(btn_frame, text="Click plots to toggle Red (Remove) / Green (Keep).").pack(side="left")
        ttk.Button(btn_frame, text="Confirm & Save", command=self.confirm).pack(side="right", padx=10)

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.plot_components()

    def plot_components(self):
        cols = 2
        rows = int(np.ceil(self.n_components / cols))
        fig = Figure(figsize=(12, 1.5 * rows), dpi=100)
        self.axes = []
        limit_sec = 4.0
        limit = min(int(limit_sec * self.fs), self.sources.shape[0])
        time_vec = np.arange(limit) / self.fs
        
        for i in range(self.n_components):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.plot(time_vec, self.sources[:limit, i], linewidth=1.0, color='#2c3e50')
            ax.set_title(f"Component {i}", fontsize=10, loc='left', pad=2)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0, limit_sec)
            if i >= (self.n_components - cols):
                ax.set_xlabel("Time (s)", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel("Amp", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            self.update_plot_color(ax, i)
            self.axes.append(ax)

        fig.tight_layout()
        self.canvas_agg = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_agg.mpl_connect('button_press_event', self.on_click)

    def update_plot_color(self, ax, idx):
        color = '#ffebee' if idx in self.bad_indices else '#e8f5e9'
        ax.set_facecolor(color)

    def on_click(self, event):
        if event.inaxes in self.axes:
            ax_idx = self.axes.index(event.inaxes)
            if ax_idx in self.bad_indices:
                self.bad_indices.remove(ax_idx)
            else:
                self.bad_indices.add(ax_idx)
            self.update_plot_color(event.inaxes, ax_idx)
            self.canvas_agg.draw()

    def confirm(self):
        self.callback(list(self.bad_indices))
        self.destroy()


#        FRONTEND: VISUALIZATION


def _add_grid_overlay(ax):
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(8, 0, -1)) 
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.tick_params(which='both', length=0)

def _create_grid_lines(ax):
    breaks = np.arange(0.5, 7.5, 1)
    v_lines = ax.vlines(breaks, -0.5, 7.5, colors='white', linewidth=0.5, alpha=0.5)
    h_lines = ax.hlines(breaks, -0.5, 7.5, colors='white', linewidth=0.5, alpha=0.5)
    return [v_lines, h_lines]

def _add_bad_electrode_markers(ax, bad_electrodes):
    """
    Draw red X markers over bad electrodes in an 8x8 grid.

    bad_electrodes: iterable of channel indices (0..63)
    """
    if not bad_electrodes:
        return []

    lines = []
    for ch in bad_electrodes:
        # Map channel index to (row, col) in final geometry.
        # Original mapping: reshape(..., 8, 8).transpose(0,2,1)[:, ::-1, :]
        # That yields:
        #   final_row = 7 - (ch % 8)
        #   final_col = ch // 8
        row = 7 - (ch % 8)
        col = ch // 8

        # Draw an X spanning the cell from (col-0.5, row-0.5) to (col+0.5, row+0.5)
        l1, = ax.plot([col - 0.5, col + 0.5],
                      [row - 0.5, row + 0.5],
                      color='red', linewidth=1.5)
        l2, = ax.plot([col - 0.5, col + 0.5],
                      [row + 0.5, row - 0.5],
                      color='red', linewidth=1.5)
        lines.extend([l1, l2])

    return lines

def visualize_torso_animation(all_grids_video, fs, fps=30, start_s=0.0, bad_electrodes_by_grid=None):
    print("Generating Anatomical Torso Animation...")
    step = int(fs / fps)
    # Downsample first
    downsampled_grids = [g[::step] for g in all_grids_video]
    total_frames = len(downsampled_grids[0])
    
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    global_max = np.percentile(np.array(downsampled_grids), 99.0)
    
    animated_artists = []
    images = []
    
    for idx, conf in GRID_CONFIG.items():
        row, col = conf['pos']
        ax = axes[row, col]
        
        _add_grid_overlay(ax)
        ax.set_title(f"Grid {idx+1}: {conf['name']}", fontsize=10)
        
        im = ax.imshow(downsampled_grids[idx][0], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
        
        lines = _create_grid_lines(ax)
        images.append((im, idx))
        animated_artists.append(im)
        animated_artists.extend(lines)

        bad_for_this = None
        if bad_electrodes_by_grid is not None and idx < len(bad_electrodes_by_grid):
            bad_for_this = bad_electrodes_by_grid[idx]
        marker_lines = _add_bad_electrode_markers(ax, bad_for_this)
        animated_artists.extend(marker_lines)
    
    ref_ax = axes[0, 0]
    time_text = ref_ax.text(0.05, 0.9, f"Time: {start_s:.2f}s", 
                            transform=ref_ax.transAxes, color='white', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    animated_artists.append(time_text)
    
    def update(frame_idx):
        for im_obj, grid_idx in images:
            im_obj.set_data(downsampled_grids[grid_idx][frame_idx])
        # Current time = start offset + frame delta
        current_time = start_s + (frame_idx * step / fs)
        time_text.set_text(f"Time: {current_time:.2f}s")
        return animated_artists

    anim = FuncAnimation(fig, update, frames=total_frames, 
                         interval=1000/fps, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return anim

def visualize_grid_animation_scaled(grid_data, fs, fps=30, grid_idx=0, vmax=None, start_s=0.0, bad_electrodes=None):
    name = GRID_CONFIG[grid_idx]['name']
    print(f"Generating Animation for {name}...")
    
    step = int(fs / fps)
    downsampled_data = grid_data[::step]
    total_frames = len(downsampled_data)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    if vmax is None: vmax = np.percentile(downsampled_data, 99.5)
    
    _add_grid_overlay(ax)
    ax.set_title(f"{name}")
    
    im = ax.imshow(downsampled_data[0], cmap='magma', interpolation='bicubic', 
                   origin='upper', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    lines = _create_grid_lines(ax)
    marker_lines = _add_bad_electrode_markers(ax, bad_electrodes)

    time_text = ax.text(0.05, 0.93, f"Time: {start_s:.2f}s",
                        transform=ax.transAxes, color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    all_artists = [im, time_text] + lines + marker_lines
    
    def update(frame_idx):
        im.set_data(downsampled_data[frame_idx])
        current_time = start_s + (frame_idx * step / fs)
        time_text.set_text(f"Time: {current_time:.2f}s")
        return all_artists

    anim = FuncAnimation(fig, update, frames=total_frames, 
                         interval=1000/fps, blit=True, cache_frame_data=False)
    plt.show()
    return anim


#        GUI CONTROLLER


class EMGControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG Analysis Tool")
        self.root.geometry("550x850") # Taller for Time Range
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.fs = None
        self.sEMG_data = None
        self.raw_cache = None 
        self.grid_states = [None] * 6 
        self.global_heart_peaks = None
        self.total_duration = 0.0
        self.aux_data = None
        self.loaded_file = None

        style = ttk.Style()
        style.theme_use('clam')
        
        ttk.Label(root, text="Neonate sEMG Visualizer", font=("Segoe UI", 14, "bold")).pack(pady=15)
        
        # 1. File
        file_frame = ttk.LabelFrame(root, text="Data File")
        file_frame.pack(fill="x", padx=20, pady=5)
        self.file_lbl = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_lbl.pack(side="left", padx=10, pady=10)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="right", padx=10, pady=10)
        
        # 2. Pipeline
        proc_frame = ttk.LabelFrame(root, text="Analysis Pipeline")
        proc_frame.pack(fill="x", padx=20, pady=5)
        self.proc_btn = ttk.Button(proc_frame, text="Run Auto-Processing (All Grids)", command=self.on_process, state="disabled")
        self.proc_btn.pack(fill="x", padx=10, pady=5)
        self.proc_status_var = tk.StringVar(value="Status: Waiting for data")
        ttk.Label(proc_frame, textvariable=self.proc_status_var).pack(pady=2)

        # 3. Visualization
        vis_frame = ttk.LabelFrame(root, text="Visualization Settings")
        vis_frame.pack(fill="x", padx=20, pady=5)
        
        ttk.Label(vis_frame, text="Target Grid:").pack(anchor="w", padx=10, pady=2)
        self.grid_var = tk.StringVar()
        grid_labels = ["Full Torso (Anatomical View)"]
        for i in range(6):
            grid_labels.append(f"Grid {i+1}: {GRID_CONFIG[i]['name']}")
        self.grid_combo = ttk.Combobox(vis_frame, textvariable=self.grid_var, values=grid_labels, state="readonly")
        self.grid_combo.current(0)
        self.grid_combo.pack(fill="x", padx=10, pady=5)
        
        # Inspection Button
        self.inspect_btn = ttk.Button(vis_frame, text="Inspect & Edit Components", command=self.on_inspect_grid_ICA_components, state="disabled")
        self.inspect_btn.pack(fill="x", padx=10, pady=10)
        
        # Source Selection
        ttk.Label(vis_frame, text="Signal Source:").pack(anchor="w", padx=10, pady=2)
        self.signal_var = tk.StringVar(value="processed")
        ttk.Radiobutton(vis_frame, text="Processed (Cleaned)", variable=self.signal_var, value="processed").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Original (Raw)", variable=self.signal_var, value="original").pack(anchor="w", padx=10)

        # View Mode
        ttk.Label(vis_frame, text="View Mode:").pack(anchor="w", padx=10, pady=2)
        self.view_mode_var = tk.StringVar(value="heatmap")
        ttk.Radiobutton(vis_frame, text="Heatmap Animation", variable=self.view_mode_var, value="heatmap").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Time Series (Individual Grids Only)", variable=self.view_mode_var, value="signal").pack(anchor="w", padx=10)

        # Scaling
        ttk.Label(vis_frame, text="Color Scaling (Heatmap Only):").pack(anchor="w", padx=10, pady=2)
        self.scale_var = tk.StringVar(value="global")
        ttk.Radiobutton(vis_frame, text="Global (Compare all grids)", variable=self.scale_var, value="global").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Local (Maximize contrast)", variable=self.scale_var, value="local").pack(anchor="w", padx=10)
        
        # 4. Time Range Slider (NEW)
        time_frame = ttk.LabelFrame(root, text="Time Range Selection (s)")
        time_frame.pack(fill="x", padx=20, pady=5)
        
        self.start_scale = tk.Scale(time_frame, from_=0, to=100, orient="horizontal", label="Start Time", command=self.on_time_change)
        self.start_scale.pack(fill="x", padx=10)
        
        self.end_scale = tk.Scale(time_frame, from_=0, to=100, orient="horizontal", label="End Time", command=self.on_time_change)
        self.end_scale.pack(fill="x", padx=10)
        
        btn_row = ttk.Frame(root)
        btn_row.pack(pady=(10, 15))

        self.vis_btn = ttk.Button(btn_row, text="Visualize Result", command=self.on_visualize, state="disabled")
        self.vis_btn.pack(side="left", padx=(0, 12), ipadx=10, ipady=5)

        self.export_btn = ttk.Button(
            btn_row,
            text=f"📤 Export Processed Window...",
            command=self.on_export,
            state="disabled"
        )
        self.export_btn.pack(side="left", padx=(12, 0), ipadx=10, ipady=5)

    def on_closing(self):
        self.root.destroy()
        os._exit(0)

    def browse_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat"), ("All Files", "*.*")])
        if fpath:
            self.load_data(fpath)

    def load_data(self, fpath):
        self.file_lbl.config(text="Loading...", foreground="blue")
        self.root.update()
        
        data = load_mat_any(fpath)
        if not (data and 'sEMG' in data):
            messagebox.showerror("Error", "Invalid .mat file")
            return
        
        self.sEMG_data = data['sEMG'].T
        self.fs = data['fsamp']
        self.grid_states = [None] * 6
        self.raw_cache = []
        self.global_heart_peaks = None
        
        self.loaded_file = fpath

        aux = data.get('AUX_signal', None)
        if aux is not None:
            aux = np.asarray(aux)
            if aux.ndim == 1:
                aux = aux.reshape(-1, 1)
            elif aux.ndim == 2 and aux.shape[0] <= 4 and aux.shape[0] < aux.shape[1]:
                # common case: (2, N) -> (N, 2)
                aux = aux.T
            self.aux_data = aux
        else:
            self.aux_data = None

        # disable export until processing is done
        self.export_btn.config(state="disabled")

        # Update Time Sliders
        self.total_duration = self.sEMG_data.shape[0] / self.fs
        self.start_scale.config(to=self.total_duration)
        self.end_scale.config(to=self.total_duration)
        self.start_scale.set(0)
        self.end_scale.set(self.total_duration)
        
        print("Pre-calculating raw geometry...")
        for i in range(6):
            start, end = i * 64, (i + 1) * 64
            chunk = self.sEMG_data[:, start:end]
            raw_geom = chunk.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
            self.raw_cache.append(np.abs(raw_geom))
        
        self.file_lbl.config(text=f"{os.path.basename(fpath)} ({self.fs} Hz)", foreground="black")
        self.proc_btn.config(state="normal", text="Run Auto-Processing")
        self.vis_btn.config(state="normal")
        self.inspect_btn.config(state="disabled")
        self.proc_status_var.set("Status: Raw data ready.")
        
    def on_time_change(self, val):
        # Enforce constraints: End > Start + 2s
        start = self.start_scale.get()
        end = self.end_scale.get()
        min_window = 2.0
        
        # If Start pushes End
        if start + min_window > end:
            if start + min_window <= self.total_duration:
                self.end_scale.set(start + min_window)
            else:
                self.start_scale.set(end - min_window)

    def on_process(self):
        threading.Thread(target=self.run_processing_task, daemon=True).start()

    def run_processing_task(self):
        self.root.after(0, lambda: self.proc_btn.config(state="disabled"))
        self.root.after(0, lambda: self.vis_btn.config(state="disabled"))
        self.root.after(0, lambda: self.proc_status_var.set("Status: Detecting Global Heartbeats..."))
        
        try:
            # Phase 1: Global Heart Detection
            self.global_heart_peaks = detect_global_heartbeats(self.sEMG_data, self.fs)
            print(f"Global Detection Found {len(self.global_heart_peaks)} heartbeats.")
            
            self.root.after(0, lambda: self.proc_status_var.set("Status: Spawning parallel processes..."))

            # Phase 2: Parallel Cleaning
            tasks = []
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG_data[:, start:end]
                tasks.append((chunk, self.fs, i+1, self.global_heart_peaks))

            results = [None] * 6
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = executor.map(process_single_grid_wrapper, tasks)
                results = list(futures)

            self.grid_states = results
            
            self.root.after(0, lambda: self.proc_status_var.set("Status: Processing Complete."))
            self.root.after(0, lambda: self.proc_btn.config(text="Processing Done (Cached)"))
            self.root.after(0, lambda: self.inspect_btn.config(state="normal"))
            self.root.after(0, lambda: self.export_btn.config(state="normal"))
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            self.root.after(
                0,
                lambda msg=error_msg: self.proc_status_var.set(f"Error: {msg}")
            )
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    def on_inspect_grid_ICA_components(self):
        selection = self.grid_combo.get()
        if "Full Torso" in selection:
            messagebox.showinfo("Select Grid", "Please select a specific Grid to inspect.")
            return
        
        grid_id = int(selection.split(":")[0].split(" ")[1])
        state = self.grid_states[grid_id - 1]
        
        if state is None:
            messagebox.showwarning("No Data", "Please run processing first.")
            return

        ComponentEditor(self.root, state['sources'], self.fs, state['bad_indices'], 
                        f"Grid {grid_id}", 
                        lambda bad: self.apply_manual_edit(grid_id, bad))

    def apply_manual_edit(self, grid_id, bad_indices):
        state = self.grid_states[grid_id - 1]
        print(f"Reconstructing Grid {grid_id} with new bad indices: {bad_indices}")
        heatmap, clean_signal = reconstruct_heatmap(
            state['ica_model'],
            state['sources'],
            bad_indices,
            state['bad_electrodes'],
            self.fs,
            good_mask=state.get('good_mask'),
            n_channels=64
        )
        state['clean_signal'] = clean_signal
        state['bad_indices'] = bad_indices
        state['heatmap'] = heatmap
        self.proc_status_var.set(f"Status: Grid {grid_id} Updated Manually.")

    def ask_export_hz(self):
        fs_int = int(round(float(self.fs)))
        divisors = [hz for hz in range(1, fs_int + 1) if fs_int % hz == 0]
        default_hz = 1024 if 1024 in divisors else fs_int
        dlg = tk.Toplevel(self.root)
        dlg.title("Export rate (Hz)")
        dlg.resizable(False, False)

        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=f"Choose export sampling rate (Hz)\n(fs={fs_int} Hz, must divide fs, max 2048 Hz):").pack(anchor="w")

        hz_var = tk.StringVar(value=str(default_hz))
        combo = ttk.Combobox(frm, textvariable=hz_var, values=[str(x) for x in divisors], state="readonly", width=12)
        combo.pack(anchor="w", pady=(8, 12))
        combo.set(str(default_hz))

        result = {"hz": None}

        def ok():
            try:
                result["hz"] = int(hz_var.get())
            except:
                result["hz"] = None
            dlg.destroy()

        def cancel():
            result["hz"] = None
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=cancel).pack(side="right")
        ttk.Button(btns, text="OK", command=ok).pack(side="right", padx=(0, 8))

        dlg.transient(self.root)
        dlg.grab_set()
        dlg.wait_window()
        return result["hz"]

    def on_export(self):
        if self.grid_states[0] is None:
            messagebox.showwarning("Wait", "Please run processing first.")
            return

        export_hz = self.ask_export_hz()
        if export_hz is None:
            return

        default_name = "export_processed.npz"
        if self.loaded_file:
            base = os.path.splitext(os.path.basename(self.loaded_file))[0]
            default_name = f"{base}_processed_{export_hz}Hz.npz"

        out_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")],
            initialfile=default_name
        )
        if not out_path:
            return

        # Use the currently selected window
        start_s = float(self.start_scale.get())
        end_s = float(self.end_scale.get())
        start_idx = int(start_s * self.fs)
        end_idx = int(end_s * self.fs)

        # Clamp to available lengths
        n_emg = self.sEMG_data.shape[0]
        end_idx = min(end_idx, n_emg)
        start_idx = max(0, min(start_idx, end_idx - 1))

        stride = self.fs // export_hz
        fs_export = self.fs / stride

        # Collect EMG in 8x8 geometry: (time, grid, 8, 8)
        grid_slices = []
        bad_mask_grid = np.zeros((6, 8, 8), dtype=bool)

        for gi, state in enumerate(self.grid_states):
            sig = state["clean_signal"]                 # (n_samples, 64)
            sig_win = sig[start_idx:end_idx:stride]     # (n_t, 64)

            sig_grid = sig_win.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]  # (n_t, 8, 8)
            grid_slices.append(sig_grid)

            for ch in state.get("bad_electrodes", []):
                r = 7 - (ch % 8)
                c = ch // 8
                bad_mask_grid[gi, r, c] = True

        emg_export = np.stack(grid_slices, axis=1)  # (n_t, 6, 8, 8)

        # AUX (device) aligned by sample index, if present
        aux_export = None
        if self.aux_data is not None and self.aux_data.ndim == 2 and self.aux_data.shape[0] > 0:
            n_aux = self.aux_data.shape[0]
            end_idx2 = min(end_idx, n_aux)
            start_idx2 = min(start_idx, end_idx2 - 1) if end_idx2 > 0 else 0
            aux_export = self.aux_data[start_idx2:end_idx2:stride]

            # Ensure same length as EMG
            n = min(emg_export.shape[0], aux_export.shape[0])
            emg_export = emg_export[:n]
            aux_export = aux_export[:n]

        # Time vector for exported samples (seconds)
        time_s = start_s + np.arange(emg_export.shape[0]) * (stride / self.fs)

        meta = {
            "source_file": self.loaded_file,
            "fs_original_hz": float(self.fs),
            "fs_export_hz": float(fs_export),
            "stride": int(stride),
            "window_start_s": float(start_s),
            "window_end_s": float(end_s),
            "n_samples_export": int(emg_export.shape[0]),
            "emg_shape": list(emg_export.shape),
            "bad_mask_shape": list(bad_mask_grid.shape),
            "aux_shape": (list(aux_export.shape) if aux_export is not None else None),
            "emg_format": "emg_grid (n_t, 6, 8, 8) using reshape+transpose+flip mapping",
            "bad_mask_format": "bad_mask_grid (6, 8, 8) in same final grid coords"
        }

        np.savez_compressed(
            out_path,
            emg=emg_export,          # (n_t, 6, 8, 8)
            aux=(aux_export if aux_export is not None else np.empty((0, 0))),
            bad_mask=bad_mask_grid,       # (6, 8, 8) bool
            time_s=time_s,           # (n_t,)
            meta=json.dumps(meta)
        )

        messagebox.showinfo("Export", f"Saved:\n{out_path}")

    def on_visualize(self):
        threading.Thread(target=self.run_visualization_task, daemon=True).start()

    def run_visualization_task(self):
        self.root.after(0, lambda: self.vis_btn.config(state="disabled"))
        
        mode = self.signal_var.get()
        view_mode = self.view_mode_var.get()
        selection = self.grid_combo.get()
        
        # Get Time Range
        start_s = self.start_scale.get()
        end_s = self.end_scale.get()
        start_idx = int(start_s * self.fs)
        end_idx = int(end_s * self.fs)
        
        try:
            # Determine Grid ID
            if "Full Torso" in selection:
                target_grid_idx = -1
            else:
                target_grid_idx = int(selection.split(":")[0].split(" ")[1]) - 1

            # BRANCH 1: Time Series View (Single Grid Only)
            if view_mode == "signal":
                if target_grid_idx == -1:
                    self.root.after(0, lambda: messagebox.showwarning("Selection Error", "Time Series view is only available for individual grids. Please select Grid 1-6."))
                    return

                # Get Data
                signal_data = None
                if mode == "original":
                    # Extract Raw Chunk from sEMG_data
                    raw_start, raw_end = target_grid_idx * 64, (target_grid_idx + 1) * 64
                    signal_data = self.sEMG_data[:, raw_start:raw_end]
                else:
                    # Extract Processed Clean Signal
                    if self.grid_states[target_grid_idx] is None:
                        self.root.after(0, lambda: messagebox.showwarning("Wait", "Please run processing first."))
                        return
                    signal_data = self.grid_states[target_grid_idx]['clean_signal']
                
                # SLICE DATA
                signal_slice = signal_data[start_idx:end_idx]
                
                # Launch Viewer (Safe on main thread)
                title = f"Grid {target_grid_idx+1} ({mode.capitalize()})"
                bad_channels = None
                state = self.grid_states[target_grid_idx] if self.grid_states[target_grid_idx] is not None else None
                if state is not None and 'bad_electrodes' in state:
                    bad_channels = state['bad_electrodes']

                self.root.after(
                    0,
                    lambda: RawSignalViewer(
                        self.root,
                        signal_slice,
                        self.fs,
                        title,
                        start_time=start_s,
                        bad_channels=bad_channels
                    )
                )
                return # Done

            # BRANCH 2: Heatmap Animation
            video_data = []
            if mode == "original":
                # Slice raw cache
                video_data = [g[start_idx:end_idx] for g in self.raw_cache]
            else:
                if self.grid_states[0] is None:
                    self.root.after(0, lambda: messagebox.showwarning("Wait", "Please run processing first."))
                    return 
                # Slice processed heatmaps
                video_data = [state['heatmap'][start_idx:end_idx] for state in self.grid_states]

            scale_mode = self.scale_var.get()
            if scale_mode == "global":
                # Compute global max on the SLICED data to be accurate to what is shown
                all_grids_array = np.array(video_data)
                vmax = np.percentile(all_grids_array, 99.0)
                print(f"Global Scale: {vmax:.2f}")
            else:
                vmax = None
                print("Local Scale: Auto")

            self.root.after(0, lambda: self.launch_plot(target_grid_idx, video_data, vmax, start_s))
            
        except Exception as e:
            print(e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    def launch_plot(self, target_idx, data, vmax, start_s):
        # Prepare bad electrode info (if processing has been run)
        bad_electrodes_by_grid = None
        if self.grid_states[0] is not None:
            bad_electrodes_by_grid = []
            for state in self.grid_states:
                if state is not None and 'bad_electrodes' in state:
                    bad_electrodes_by_grid.append(state['bad_electrodes'])
                else:
                    bad_electrodes_by_grid.append([])

        if target_idx == -1:
            visualize_torso_animation(data, self.fs, fps=30, start_s=start_s, bad_electrodes_by_grid=bad_electrodes_by_grid)
        else:
            grid_anim = data[target_idx]
            visualize_grid_animation_scaled(grid_anim, self.fs, fps=30, grid_idx=target_idx, vmax=vmax, start_s=start_s, bad_electrodes=bad_electrodes_by_grid[target_idx])

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = EMGControlPanel(root)
    root.mainloop()