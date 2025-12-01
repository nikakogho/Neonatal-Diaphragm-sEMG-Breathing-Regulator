import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import concurrent.futures

# ==========================================
#        CONFIGURATION & MAPPING
# ==========================================

# Maps Index (0-5) to Anatomical Info
# Pos: (Row, Col) for a 3x2 Grid
GRID_CONFIG = {
    0: {"name": "IN 1 (Right Chest)",   "pos": (0, 0)},
    1: {"name": "MULTI 1 (Right Ribs)", "pos": (1, 0)},
    2: {"name": "MULTI 2 (Right Abs)",  "pos": (2, 0)},
    3: {"name": "MULTI 3 (Left Abs)",   "pos": (2, 1)},
    4: {"name": "MULTI 4 (Left Ribs)",  "pos": (1, 1)},
    5: {"name": "IN 5 (Left Chest)",    "pos": (0, 1)},
}

# ==========================================
#        BACKEND: SIGNAL PROCESSING
# ==========================================

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

def ICA(data):
    n_components = 25 # Robust rank for 64 channels
    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance')
    sources = ica.fit_transform(data) # Returns (Samples, Components)

    return ica, sources

def analyze_envelope_refined(sources, fs):
    """
    Optimized: Uses a strided view for statistical analysis to speed up calculation.
    """
    MAX_HEART_COMPONENTS = 6
    candidates = []
    hf_noise = []
    
    # OPTIMIZATION: Downsample signal for statistics only (Stride 4 -> 512Hz effective)
    # This speeds up Welch and Peak Finding by 4x without losing 1Hz Heart/Breathing data.
    stride = 4
    fs_stat = fs / stride
    
    for i in range(sources.shape[1]):
        sig = sources[:, i]
        sig_stat = sig[::stride] # Lightweight view for math
        
        kurt = kurtosis(sig_stat)
        sig_env = np.abs(sig_stat) - np.mean(np.abs(sig_stat))
        
        # Fast Welch
        freqs, psd = welch(sig_env, fs_stat, nperseg=int(fs_stat*4))
        valid_mask = freqs > 0.5 
        dom_freq = freqs[valid_mask][np.argmax(psd[valid_mask])] if np.sum(valid_mask) > 0 else 0.0
        
        # Fast Peak Finding
        peaks, _ = find_peaks(sig_env, height=np.std(sig_env)*3, distance=int(fs_stat*0.4))
        regularity = np.std(np.diff(peaks) / fs_stat) if len(peaks) > 3 else 10.0

        if dom_freq > 12.0:
            hf_noise.append(i)
            continue
            
        if (0.7 <= dom_freq <= 3.0) and (kurt > 1.5):
            c_type = 'HEART' if regularity < 0.20 else 'ARTIFACT'
            candidates.append({'id': i, 'kurt': kurt, 'type': c_type})

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

def process_single_grid_wrapper(args):
    """
    Unpacks arguments for ProcessPoolExecutor mapping.
    """
    raw_chunk, fs, grid_id = args
    print(f"[Core Task] Processing Grid {grid_id}...")
    
    filtered = filter_data(raw_chunk, fs)
    ica, sources = ICA(filtered)
    hearts, hf, artifacts = analyze_envelope_refined(sources, fs)
    bad_indices = hearts + hf + artifacts
    sources[:, bad_indices] = 0.0
    
    clean = ica.inverse_transform(sources)
    rectified = np.abs(clean)
    b_env, a_env = butter(N=4, Wn=3.0, btype='low', fs=fs)
    envelopes = filtfilt(b_env, a_env, rectified, axis=0)
    
    grid_matrix = envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
    return grid_matrix

# ==========================================
#        FRONTEND: VISUALIZATION
# ==========================================

def _setup_axis_labels(ax):
    """Sets up the 1-8 numbering on the axes (Static Background)."""
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(8, 0, -1)) 
    ax.tick_params(which='both', length=0) # Hide tick marks, keep numbers

def _create_grid_lines(ax):
    """
    Creates manual grid lines that can be passed to the animator.
    Returns a list of LineCollection artists.
    """
    # Grid lines go between pixels (indices -0.5, 0.5, 1.5, etc.)
    # We want lines at 0.5, 1.5, ... 6.5
    breaks = np.arange(0.5, 7.5, 1)
    
    # Create Vertical and Horizontal lines
    # colors='white', linewidth=0.5, alpha=0.5
    v_lines = ax.vlines(breaks, -0.5, 7.5, colors='white', linewidth=0.5, alpha=0.5)
    h_lines = ax.hlines(breaks, -0.5, 7.5, colors='white', linewidth=0.5, alpha=0.5)
    
    return [v_lines, h_lines]

def visualize_torso_animation(all_grids_video, fs, fps=30):
    print("Generating Anatomical Torso Animation...")
    
    step = int(fs / fps)
    downsampled_grids = [g[::step] for g in all_grids_video]
    total_frames = len(downsampled_grids[0])
    
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    global_max = np.percentile(np.array(downsampled_grids), 99.0)
    
    # Store all artists that need updating/redrawing
    animated_artists = [] 
    images = [] # Just the images for easy data updating
    
    for idx, conf in GRID_CONFIG.items():
        row, col = conf['pos']
        ax = axes[row, col]
        
        # 1. Setup Static Labels (Background)
        _setup_axis_labels(ax)
        ax.set_title(f"Grid {idx+1}: {conf['name']}", fontsize=10)
        
        # 2. Create Image (Animated)
        im = ax.imshow(downsampled_grids[idx][0], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
        
        # 3. Create Grid Lines (Animated Overlay)
        # We MUST create them here and add them to the list so they are drawn ON TOP of the image
        grid_lines = _create_grid_lines(ax)
        
        images.append((im, idx))
        
        # Add Image AND Lines to the list of things to redraw
        animated_artists.append(im)
        animated_artists.extend(grid_lines)
    
    # HUD Text
    ref_ax = axes[0, 0]
    time_text = ref_ax.text(0.05, 0.9, "Time: 0.00s", 
                            transform=ref_ax.transAxes, 
                            color='white', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    animated_artists.append(time_text)
    
    def update(frame_idx):
        # Update Data
        for im_obj, grid_idx in images:
            im_obj.set_data(downsampled_grids[grid_idx][frame_idx])
            
        time_text.set_text(f"Time: {frame_idx * step / fs:.2f}s")
        
        # Return EVERYTHING (Images + Grid Lines + Text)
        # Matplotlib will redraw them in this order: Image first, then Lines on top
        return animated_artists

    anim = FuncAnimation(fig, update, frames=total_frames, 
                         interval=1000/fps, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return anim

def visualize_grid_animation_scaled(grid_data, fs, fps=30, grid_idx=0, vmax=None):
    name = GRID_CONFIG[grid_idx]['name']
    print(f"Generating Animation for {name}...")
    
    step = int(fs / fps)
    downsampled_data = grid_data[::step]
    total_frames = len(downsampled_data)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    if vmax is None: vmax = np.percentile(downsampled_data, 99.5)
    
    _setup_axis_labels(ax)
    ax.set_title(f"{name}")
    
    # Image
    im = ax.imshow(downsampled_data[0], cmap='magma', interpolation='bicubic', 
                   origin='upper', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    # Grid Lines (Overlay)
    grid_lines = _create_grid_lines(ax)
    
    # Text
    time_text = ax.text(0.05, 0.93, "Time: 0.00s", 
                        transform=ax.transAxes, 
                        color='white', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    # Combine all animated parts
    all_artists = [im, time_text] + grid_lines
    
    def update(frame_idx):
        im.set_data(downsampled_data[frame_idx])
        time_text.set_text(f"Time: {frame_idx * step / fs:.2f}s")
        return all_artists

    anim = FuncAnimation(fig, update, frames=total_frames, 
                         interval=1000/fps, blit=True, cache_frame_data=False)
    plt.show()
    return anim

# ==========================================
#        GUI CONTROLLER
# ==========================================

class EMGControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG Analysis Tool")
        self.root.geometry("550x680")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.fs = None
        self.sEMG_data = None
        self.filename = None
        self.raw_cache = None 
        self.processed_cache = None

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
        self.proc_btn = ttk.Button(proc_frame, text="Run Processing (ICA)", command=self.on_process, state="disabled")
        self.proc_btn.pack(fill="x", padx=10, pady=10)
        self.proc_status_var = tk.StringVar(value="Status: Waiting for data")
        ttk.Label(proc_frame, textvariable=self.proc_status_var).pack(pady=5)

        # 3. Visualization
        vis_frame = ttk.LabelFrame(root, text="Visualization Settings")
        vis_frame.pack(fill="x", padx=20, pady=5)
        
        ttk.Label(vis_frame, text="Target Grid:").pack(anchor="w", padx=10, pady=2)
        self.grid_var = tk.StringVar()
        
        # Generate Dropdown List from GRID_CONFIG
        grid_labels = ["Full Torso (Anatomical View)"]
        for i in range(6):
            grid_labels.append(f"Grid {i+1}: {GRID_CONFIG[i]['name']}")
            
        self.grid_combo = ttk.Combobox(vis_frame, textvariable=self.grid_var, values=grid_labels, state="readonly")
        self.grid_combo.current(0)
        self.grid_combo.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(vis_frame, text="Signal Source:").pack(anchor="w", padx=10, pady=2)
        self.signal_var = tk.StringVar(value="processed")
        ttk.Radiobutton(vis_frame, text="Processed (Cleaned)", variable=self.signal_var, value="processed").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Original (Raw)", variable=self.signal_var, value="original").pack(anchor="w", padx=10)

        ttk.Label(vis_frame, text="Color Scaling:").pack(anchor="w", padx=10, pady=2)
        self.scale_var = tk.StringVar(value="global")
        ttk.Radiobutton(vis_frame, text="Global (Compare all grids)", variable=self.scale_var, value="global").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Local (Maximize contrast)", variable=self.scale_var, value="local").pack(anchor="w", padx=10)
        
        self.vis_btn = ttk.Button(root, text="Visualize Result", command=self.on_visualize, state="disabled")
        self.vis_btn.pack(pady=20, ipadx=10, ipady=5)

    def on_closing(self):
        self.root.destroy()
        os._exit(0) # Force kill threads

    def browse_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat"), ("All Files", "*.*")])
        if fpath:
            self.load_data(fpath)

    def load_data(self, fpath):
        self.file_lbl.config(text="Loading...", foreground="blue")
        self.root.update()
        
        data = load_mat_any(fpath)
        if data and 'sEMG' in data:
            self.sEMG_data = data['sEMG'].T
            self.fs = data['fsamp']
            self.filename = os.path.basename(fpath)
            self.processed_cache = None
            self.raw_cache = []
            
            print("Pre-calculating raw geometry...")
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG_data[:, start:end]
                raw_geom = chunk.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
                self.raw_cache.append(np.abs(raw_geom))
            
            self.file_lbl.config(text=f"{self.filename} ({self.fs} Hz)", foreground="black")
            self.proc_btn.config(state="normal", text="Run Processing (ICA)")
            self.vis_btn.config(state="normal")
            self.proc_status_var.set("Status: Raw data ready. Processing optional.")
        else:
            messagebox.showerror("Error", "Invalid .mat file")

    def on_process(self):
        threading.Thread(target=self.run_processing_task, daemon=True).start()

    def run_processing_task(self):
        # Initial UI Update must be on main thread
        self.root.after(0, lambda: self.proc_btn.config(state="disabled"))
        self.root.after(0, lambda: self.vis_btn.config(state="disabled"))
        self.root.after(0, lambda: self.proc_status_var.set("Status: Spawning parallel processes..."))
        
        try:
            tasks = []
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG_data[:, start:end]
                tasks.append((chunk, self.fs, i+1))

            results = [None] * 6
            # Use ProcessPoolExecutor to bypass GIL
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = executor.map(process_single_grid_wrapper, tasks)
                results = list(futures)

            self.processed_cache = results
            
            # Safe UI Update
            self.root.after(0, lambda: self.proc_status_var.set("Status: Processing Complete."))
            self.root.after(0, lambda: self.proc_btn.config(text="Processing Done (Cached)"))
            
        except Exception as e:
            print(f"Error: {e}")
            self.root.after(0, lambda: self.proc_status_var.set(f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    def on_visualize(self):
        threading.Thread(target=self.run_visualization_task, daemon=True).start()

    def run_visualization_task(self):
        self.root.after(0, lambda: self.vis_btn.config(state="disabled"))
        mode = self.signal_var.get()
        try:
            video_data = []
            if mode == "original":
                video_data = self.raw_cache
            else:
                if self.processed_cache is None:
                    self.run_processing_task() 
                    if self.processed_cache is None: return 
                video_data = self.processed_cache

            selection = self.grid_combo.get()
            scale_mode = self.scale_var.get()
            
            # PARSE SELECTION USING LIST INDEX or STRING MATCH
            if "Full Torso" in selection:
                target_grid_idx = -1
            else:
                # String format is "Grid X: Name..."
                # We extract the number X, subtract 1 to get index
                target_grid_idx = int(selection.split(":")[0].split(" ")[1]) - 1

            if scale_mode == "global":
                all_grids_array = np.array(video_data)
                vmax = np.percentile(all_grids_array, 99.0)
                print(f"Global Scale: {vmax:.2f}")
            else:
                vmax = None
                print("Local Scale: Auto")

            self.root.after(0, lambda: self.visualize_animation(target_grid_idx, video_data, vmax))
        except Exception as e:
            print(e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    def visualize_animation(self, target_grid_idx, video_data, vmax):
        if target_grid_idx == -1:
            visualize_torso_animation(video_data, self.fs, fps=30)
        else:
            grid_anim = video_data[target_grid_idx]
            visualize_grid_animation_scaled(grid_anim, self.fs, fps=30, grid_idx=target_grid_idx, vmax=vmax)

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGControlPanel(root)
    root.mainloop()