import tkinter as tk
from tkinter import ttk, filedialog
import threading
from tkinter import messagebox
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA, PCA
import os

def load_mat_any(path):
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k:v for k,v in d.items() if not k.startswith("__")}
    except Exception as e:
        print(f"Error: {e}")

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
    
    # 6. Reshape (Samples, 8, 8)
    envelopes = envelopes.reshape(-1, 8, 8)

    # 7. Align to match anatomical layout of starting bottom-left and moving up first
    envelopes = envelopes.transpose(0, 2, 1)[:, ::-1, :]

    return envelopes

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

def visualize_full_torso_one_frame_at_a_time(all_grids_video, fs, frames_to_show):
    """Plots 6 grids side-by-side with a shared color scale."""
    # Calculate global max so bright grids look bright and quiet grids look dark
    global_max = np.percentile(np.array(all_grids_video), 99.0)
    
    for t in frames_to_show:
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Full Torso Activation - Frame {t} ({t/fs:.2f}s)", fontsize=16)
        
        for i in range(6):
            ax = plt.subplot(2, 3, i+1) # 2 Rows, 3 Columns
            
            plt.imshow(all_grids_video[i][t], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
            
            _add_grid_overlay(ax)
            plt.title(f"Grid {i+1}")

        plt.tight_layout()
        plt.show()

def visualize_torso_animation(all_grids_video, fs, fps=30):
    """
    Animates ALL 6 grids simultaneously in a 2x3 layout.
    """
    print("Generating Full Torso Animation...")
    
    # 1. Setup Figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration
    
    # 2. Calculate Global Max (Crucial for consistent coloring across grids)
    global_max = np.percentile(np.array(all_grids_video), 99.0)
    
    images = []
    
    # 3. Initialize Plots
    for i, ax in enumerate(axes):
        # Initial frame (0)
        im = ax.imshow(all_grids_video[i][0], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
        _add_grid_overlay(ax)
        ax.set_title(f"Grid {i+1}")
        images.append(im)
    
    # Add a main title for the time
    time_text = fig.suptitle("Time: 0.00s", fontsize=16)
    
    # 4. Animation Update Function
    step = int(fs / fps) # Skip samples to match Real-Time speed
    total_frames = len(all_grids_video[0])
    
    def update(frame_idx):
        # Update all 6 grids
        for i, im in enumerate(images):
            im.set_data(all_grids_video[i][frame_idx])
            
        # Update Time Title
        time_text.set_text(f"Time: {frame_idx/fs:.2f}s")
        
        # Return list of artists to update
        return images + [time_text]

    # 5. Create Animation
    anim = FuncAnimation(fig, update, 
                         frames=range(0, total_frames, step), 
                         interval=1000/fps, blit=False)
    
    plt.tight_layout()
    plt.show()
    return anim

def visualize_grid_animation(grid_data, fs, fps=30, grid_id=1):
    """
    Animates a single grid over time using FuncAnimation.
    """
    print(f"Generating Animation for Grid {grid_id}...")
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(grid_data, 99.5)
    
    # Initial Plot
    im = ax.imshow(grid_data[0], cmap='magma', interpolation='bicubic', 
                   origin='upper', vmin=0, vmax=vmax)
    _add_grid_overlay(ax)
    plt.colorbar(im, ax=ax)
    title = ax.set_title(f"Grid {grid_id} - Time: 0.00s")

    # Calculate Step Size
    # We have fs samples per second. We want fps frames per second.
    # step = fs / fps
    step = int(fs / fps)
    
    def update(frame_idx):
        im.set_data(grid_data[frame_idx])
        title.set_text(f"Grid {grid_id} - Time: {frame_idx/fs:.2f}s")
        return [im, title]

    # Create Animation
    # Frames is a list of indices: [0, 68, 136, ...]
    anim = FuncAnimation(fig, update, 
                         frames=range(0, len(grid_data), step), 
                         interval=1000/fps, blit=False)
    
    plt.show()
    return anim

file_path = 'PARTICIPANT_1/Tube/Relaxed_Breathing_1_tube.otb+.mat'
if not os.path.exists(file_path):
    print("File not found!")
    exit()

data = load_mat_any(file_path)
sEMG = data['sEMG'].T # (63955, 384)
fs = data['fsamp']

# GUI
class EMGControlPanel:
    def __init__(self, root):
        self.root = root
        self.fs = None
        self.sEMG = None
        self.filename = None
        
        self.root.title("sEMG Analysis Tool")
        self.root.geometry("500x550")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        header = ttk.Label(root, text="Neonate sEMG Visualizer", font=("Segoe UI", 14, "bold"))
        header.pack(pady=20)
        
        # 0. File Selection
        file_frame = ttk.LabelFrame(root, text="Data File")
        file_frame.pack(fill="x", padx=20, pady=5)
        
        self.file_lbl = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_lbl.pack(side="left", padx=10, pady=10)
        
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="right", padx=10, pady=10)
        
        # 1. Signal Type
        type_frame = ttk.LabelFrame(root, text="Signal Source")
        type_frame.pack(fill="x", padx=20, pady=5)
        
        self.signal_var = tk.StringVar(value="processed")
        ttk.Radiobutton(type_frame, text="Processed (Filtered + ICA)", 
                        variable=self.signal_var, value="processed").pack(anchor="w", padx=10, pady=5)
        ttk.Radiobutton(type_frame, text="Original (Raw Input)", 
                        variable=self.signal_var, value="original").pack(anchor="w", padx=10, pady=5)
        
        # 2. Grid Selection
        grid_frame = ttk.LabelFrame(root, text="Target Grid")
        grid_frame.pack(fill="x", padx=20, pady=10)
        
        self.grid_var = tk.StringVar()
        grids = ["Full Torso (All 6)", "Grid 1", "Grid 2", "Grid 3", "Grid 4", "Grid 5", "Grid 6"]
        self.grid_combo = ttk.Combobox(grid_frame, textvariable=self.grid_var, values=grids, state="readonly")
        self.grid_combo.current(0)
        self.grid_combo.pack(fill="x", padx=10, pady=10)
        
        # 3. Action Button
        self.btn = ttk.Button(root, text="Launch Visualization", command=self.on_launch)
        self.btn.pack(pady=20, ipadx=10, ipady=5)
        
        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = ttk.Label(root, textvariable=self.status_var, font=("Segoe UI", 9, "italic"))
        self.status_lbl.pack(pady=5)

    def browse_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat"), ("All Files", "*.*")])
        if fpath:
            self.load_data(fpath)

    def load_data(self, fpath):
        self.status_var.set("Loading data...")
        self.root.update()
        
        data = load_mat_any(fpath)
        if data and 'sEMG' in data:
            self.sEMG = data['sEMG'].T # (Samples, 384)
            self.fs = data['fsamp']
            self.filename = os.path.basename(fpath)
            
            self.file_lbl.config(text=f"{self.filename} ({self.fs} Hz)", foreground="black")
            self.status_var.set("Data Loaded. Ready.")
            self.btn.config(state="normal")
        else:
            messagebox.showerror("Error", "Invalid .mat file. Must contain 'sEMG' and 'fsamp'.")
            self.status_var.set("Load Failed")

    def on_launch(self):
        # Run in thread so UI doesn't freeze during calculation
        threading.Thread(target=self.run_visualization, daemon=True).start()

    def run_visualization(self):
        self.btn.config(state="disabled")
        self.status_var.set("Processing... Check Console...")
        
        try:
            # Parse inputs
            mode = self.signal_var.get()
            selection = self.grid_combo.get()
            
            # Determine Grid ID (0 for All, 1-6 for specific)
            if "Full Torso" in selection:
                grid_id = 0
            else:
                grid_id = int(selection.split(" ")[1])

            # --- DATA PREPARATION ---
            video_data = []
            
            # Loop needed to handle geometry correctly for both Raw and Processed
            for i in range(6):
                # Optimization: If user selected specific grid, only process that one
                if grid_id != 0 and (i + 1) != grid_id:
                    video_data.append(None) # Placeholder
                    continue

                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG[:, start:end]

                if mode == "processed":
                    # Run full pipeline
                    processed = process_single_grid(chunk, self.fs, grid_id=i+1)
                    video_data.append(processed)
                else:
                    # Just reshape Raw data to match Anatomy (Flip/Transpose)
                    # We utilize the same geometry fix we discovered earlier
                    raw_geom = chunk.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
                    video_data.append(raw_geom)

            # --- DISPLAY ---
            self.status_var.set("Rendering...")
            
            if grid_id == 0:
                # Filter out Nones if we calculated all
                visualize_torso_animation(video_data, self.fs, fps=30)
            else:
                # Extract the single grid we calculated
                single_grid_data = video_data[grid_id - 1]
                visualize_grid_animation(single_grid_data, self.fs, fps=30, grid_id=grid_id)

            self.status_var.set("Visualization Complete")

        except Exception as e:
            print(e)
            self.status_var.set(f"Error: {str(e)}")
        finally:
            self.btn.config(state="normal")

if __name__ == "__main__":
    print("Starting GUI...")
    root = tk.Tk()
    app = EMGControlPanel(root)
    root.mainloop()