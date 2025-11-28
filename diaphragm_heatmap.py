import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA, PCA
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

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
    pca_estimator = PCA(n_components=0.999, svd_solver='full')
    pca_estimator.fit(data)
    n_components = pca_estimator.n_components_

    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance')
    sources = ica.fit_transform(data) # Returns (Samples, Components)

    return ica, sources

def analyze_envelope_refined(sources, fs):
    hearts, hf_noise, other_noise = [], [], []
    print(f"{'ID':<4} | {'Freq':<6} | {'Kurt':<6} | {'Reg (s)':<8} | {'Verdict'}")
    print("-" * 50)
    
    for i in range(sources.shape[1]):
        sig = sources[:, i]
        kurt = kurtosis(sig)
        
        # Frequency
        sig_env = np.abs(sig) - np.mean(np.abs(sig))
        freqs, psd = welch(sig_env, fs, nperseg=fs*4)
        valid_mask = freqs > 0.5 
        dom_freq = freqs[valid_mask][np.argmax(psd[valid_mask])] if np.sum(valid_mask) > 0 else 0.0

        # Regularity
        peaks, _ = find_peaks(sig_env, height=np.std(sig_env)*3, distance=int(fs*0.4))
        if len(peaks) > 3:
            regularity = np.std(np.diff(peaks) / fs)
        else:
            regularity = 10.0

        # Verdict
        verdict = "Keep"
        if dom_freq > 12.0:
            verdict = "HF NOISE"
            hf_noise.append(i)
        elif (0.7 <= dom_freq <= 3.0) and (kurt > 1.5):
            if regularity < 0.20:
                verdict = "HEART"
                hearts.append(i)
            else:
                verdict = "ARTIFACT" 
                other_noise.append(i)
        
        print(f"{i:<4} | {dom_freq:<6.2f} | {kurt:<6.2f} | {regularity:<8.3f} | {verdict}")
        
    return hearts, hf_noise, other_noise

def process_single_grid(raw_chunk, fs, grid_id):
    """Runs the full cleaning pipeline on one 64-channel chunk."""
    print(f"\n=== PROCESSING GRID {grid_id} ===")
    
    # 1. Filter
    filtered = filter_data(raw_chunk, fs)
    
    # 2. ICA
    ica, sources = ICA(filtered)
    
    # 3. Auto-Detect Artifacts
    hearts, hf, artifacts = analyze_envelope_refined(sources, fs)
    bad_indices = hearts + hf + artifacts
    print(f"   -> Removing: {bad_indices}")
    
    # 4. Zero out
    sources[:, bad_indices] = 0.0
    
    # 5. Reconstruct & Envelope
    clean = ica.inverse_transform(sources)
    rectified = np.abs(clean)
    
    # 3Hz Lowpass for smooth heatmap visualization
    b_env, a_env = butter(N=4, Wn=3.0, btype='low', fs=fs)
    envelopes = filtfilt(b_env, a_env, rectified, axis=0)
    
    # 6. Reshape & Align (Column-Major Bottom-Up correction)
    grid_matrix = envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
    
    return grid_matrix

def _add_grid_overlay(ax):
    """Adds 1-8 numbering and grid lines to an axis."""
    # 1. Major Ticks (The Numbers 1-8) centered on pixels
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(8, 0, -1))
    
    # 2. Minor Ticks (The Grid Lines) between pixels
    # We place them at -0.5, 0.5, 1.5, etc.
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    
    # 3. Draw the Grid
    # alpha=0.3 makes it subtle so it doesn't obscure the data
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 4. Remove the little tick marks sticking out
    ax.tick_params(which='both', length=0)

def visualize_torso_animation(all_grids_video, fs, fps=30):
    print("Generating Full Torso Animation...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    global_max = np.percentile(np.array(all_grids_video), 99.0)
    images = []
    
    for i, ax in enumerate(axes):
        im = ax.imshow(all_grids_video[i][0], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
        _add_grid_overlay(ax)
        ax.set_title(f"Grid {i+1}")
        images.append(im)
    
    time_text = fig.suptitle("Time: 0.00s", fontsize=16)
    step = int(fs / fps) # Skip samples to match Real-Time speed
    total_frames = len(all_grids_video[0])
    
    def update(frame_idx):
        for i, im in enumerate(images):
            im.set_data(all_grids_video[i][frame_idx])
        time_text.set_text(f"Time: {frame_idx/fs:.2f}s")
        return images + [time_text]

    anim = FuncAnimation(fig, update, frames=range(0, total_frames, step), 
                         interval=1000/fps, blit=False)
    plt.tight_layout()
    plt.show()
    return anim

def visualize_grid_animation_scaled(grid_data, fs, fps=30, grid_id=1, vmax=None):
    print(f"Generating Grid {grid_id} Animation with max={vmax if vmax else 'Auto'}...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if vmax is None:
        vmax = np.percentile(grid_data, 99.5)
    
    im = ax.imshow(grid_data[0], cmap='magma', interpolation='bicubic', 
                   origin='upper', vmin=0, vmax=vmax)
    _add_grid_overlay(ax)
    plt.colorbar(im, ax=ax)
    title = ax.set_title(f"Grid {grid_id} - Time: 0.00s")

    step = int(fs / fps)
    
    def update(frame_idx):
        im.set_data(grid_data[frame_idx])
        title.set_text(f"Grid {grid_id} - Time: {frame_idx/fs:.2f}s")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=range(0, len(grid_data), step), 
                         interval=1000/fps, blit=False)
    plt.show()
    return anim

# GUI

class EMGControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG Analysis Tool")
        self.root.geometry("550x650")
        
        # --- STATE ---
        self.fs = None
        self.sEMG = None
        self.filename = None
        
        # Cache for data [List of 6 arrays]
        self.raw_cache = None 
        self.processed_cache = None

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        ttk.Label(root, text="Neonate sEMG Visualizer", font=("Segoe UI", 14, "bold")).pack(pady=15)
        
        # 1. File Selection
        file_frame = ttk.LabelFrame(root, text="Data File")
        file_frame.pack(fill="x", padx=20, pady=5)
        
        self.file_lbl = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_lbl.pack(side="left", padx=10, pady=10)
        
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="right", padx=10, pady=10)
        
        # 2. Processing Control (NEW)
        proc_frame = ttk.LabelFrame(root, text="Analysis Pipeline")
        proc_frame.pack(fill="x", padx=20, pady=5)
        
        self.proc_btn = ttk.Button(proc_frame, text="Run Processing (ICA)", command=self.on_process, state="disabled")
        self.proc_btn.pack(fill="x", padx=10, pady=10)
        
        self.proc_status_var = tk.StringVar(value="Status: Waiting for data")
        ttk.Label(proc_frame, textvariable=self.proc_status_var).pack(pady=5)

        # 3. Visualization Settings
        vis_frame = ttk.LabelFrame(root, text="Visualization Settings")
        vis_frame.pack(fill="x", padx=20, pady=5)
        
        # Grid Selection
        ttk.Label(vis_frame, text="Target Grid:").pack(anchor="w", padx=10, pady=2)
        self.grid_var = tk.StringVar()
        grids = ["Full Torso (All 6)", "Grid 1", "Grid 2", "Grid 3", "Grid 4", "Grid 5", "Grid 6"]
        self.grid_combo = ttk.Combobox(vis_frame, textvariable=self.grid_var, values=grids, state="readonly")
        self.grid_combo.current(0)
        self.grid_combo.pack(fill="x", padx=10, pady=5)
        
        # Source Selection
        ttk.Label(vis_frame, text="Signal Source:").pack(anchor="w", padx=10, pady=2)
        self.signal_var = tk.StringVar(value="processed")
        ttk.Radiobutton(vis_frame, text="Processed (Cleaned)", variable=self.signal_var, value="processed").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Original (Raw)", variable=self.signal_var, value="original").pack(anchor="w", padx=10)

        # Scaling Selection
        ttk.Label(vis_frame, text="Color Scaling:").pack(anchor="w", padx=10, pady=2)
        self.scale_var = tk.StringVar(value="global")
        ttk.Radiobutton(vis_frame, text="Global (Truth - Recommended)", variable=self.scale_var, value="global").pack(anchor="w", padx=10)
        ttk.Radiobutton(vis_frame, text="Local (Debug - Max Contrast)", variable=self.scale_var, value="local").pack(anchor="w", padx=10)
        
        # 4. Launch Button
        self.vis_btn = ttk.Button(root, text="Visualize Result", command=self.on_visualize, state="disabled")
        self.vis_btn.pack(pady=20, ipadx=10, ipady=5)

    def browse_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat"), ("All Files", "*.*")])
        if fpath:
            self.load_data(fpath)

    def load_data(self, fpath):
        self.file_lbl.config(text="Loading...", foreground="blue")
        self.root.update()
        
        data = load_mat_any(fpath)
        if data and 'sEMG' in data:
            self.sEMG = data['sEMG'].T # (Samples, 384)
            self.fs = data['fsamp']
            self.filename = os.path.basename(fpath)
            
            # Reset Caches
            self.processed_cache = None
            self.raw_cache = []
            
            # Pre-calculate Raw Geometry (Fast)
            print("Pre-calculating raw geometry...")
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG[:, start:end]
                # Apply Geometry fix: Reshape -> Transpose -> Flip Y
                raw_geom = chunk.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
                # Rectify for visualization consistency
                self.raw_cache.append(np.abs(raw_geom))
            
            self.file_lbl.config(text=f"{self.filename} ({self.fs} Hz)", foreground="black")
            self.proc_btn.config(state="normal", text="Run Processing (ICA)")
            self.vis_btn.config(state="normal")
            self.proc_status_var.set("Status: Raw data ready. Processing optional.")
        else:
            messagebox.showerror("Error", "Invalid .mat file")

    def on_process(self):
        # Trigger processing thread
        threading.Thread(target=self.run_processing_task, daemon=True).start()

    def run_processing_task(self):
        self.proc_btn.config(state="disabled")
        self.vis_btn.config(state="disabled")
        self.proc_status_var.set("Status: Processing... Please wait...")
        
        try:
            results = []
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG[:, start:end]
                # Full pipeline
                processed = process_single_grid(chunk, self.fs, grid_id=i+1)
                results.append(processed)
            
            self.processed_cache = results
            self.proc_status_var.set("Status: Processing Complete.")
            self.proc_btn.config(text="Processing Done (Cached)")
            
        except Exception as e:
            print(e)
            self.proc_status_var.set(f"Error: {e}")
        finally:
            # Re-enable visualization, keep process disabled (already done)
            self.vis_btn.config(state="normal")

    def on_visualize(self):
        threading.Thread(target=self.run_visualization_task, daemon=True).start()

    def run_visualization_task(self):
        self.vis_btn.config(state="disabled")
        mode = self.signal_var.get()
        
        try:
            video_data = []
            
            # --- DATA RETRIEVAL ---
            if mode == "original":
                video_data = self.raw_cache
            else:
                # User wants Processed. Do we have it?
                if self.processed_cache is None:
                    print("Processing required first...")
                    # Run processing synchronously here if needed
                    self.run_processing_task() 
                    # If it failed, stop
                    if self.processed_cache is None: return 
                
                video_data = self.processed_cache

            # --- TARGET SELECTION ---
            selection = self.grid_combo.get()
            scale_mode = self.scale_var.get()
            
            if "Full Torso" in selection:
                target_grid = 0
            else:
                target_grid = int(selection.split(" ")[1])

            # --- SCALING ---
            if scale_mode == "global":
                all_grids_array = np.array(video_data)
                vmax = np.percentile(all_grids_array, 99.0)
                print(f"Global Scale: {vmax:.2f}")
            else:
                vmax = None
                print("Local Scale: Auto")

            # --- RENDER ---
            if target_grid == 0:
                visualize_torso_animation(video_data, self.fs, fps=30)
            else:
                grid_anim = video_data[target_grid - 1]
                visualize_grid_animation_scaled(grid_anim, self.fs, fps=30, grid_id=target_grid, vmax=vmax)

        except Exception as e:
            print(e)
            messagebox.showerror("Error", str(e))
        finally:
            self.vis_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGControlPanel(root)
    root.mainloop()