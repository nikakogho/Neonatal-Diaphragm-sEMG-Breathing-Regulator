import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA, PCA
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
    n_components = 20
    ica = FastICA(n_components=n_components, random_state=42,
                  whiten='unit-variance', max_iter=1000, tol=0.005)
    sources = ica.fit_transform(data)
    return ica, sources

def analyze_envelope_refined(sources, fs):
    """
    Returns lists of indices: hearts, hf_noise, artifacts
    """
    MAX_HEART_COMPONENTS = 6
    candidates = []
    hf_noise = []
    
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
    """Parallel wrapper"""
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
    
    # Geometry Fix
    grid_matrix = envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
    return grid_matrix

# ==========================================
#        COMPONENT INSPECTOR WINDOW
# ==========================================

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
        
        # UI Layout
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="top", fill="x", padx=10, pady=5)
        ttk.Label(btn_frame, text="Click plots to toggle Red (Remove) / Green (Keep).").pack(side="left")
        ttk.Button(btn_frame, text="Confirm & Save", command=self.confirm).pack(side="right", padx=10)

        # Scrollable Canvas
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
        cols = 5
        rows = int(np.ceil(self.n_components / cols))
        fig = Figure(figsize=(15, 2.5 * rows), dpi=100)
        self.axes = []
        
        # Plot only first 4 seconds to save render time
        limit = min(int(4 * self.fs), self.sources.shape[0])
        
        for i in range(self.n_components):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.plot(self.sources[:limit, i], linewidth=0.8, color='black')
            ax.set_title(f"Comp {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            self.update_plot_color(ax, i)
            self.axes.append(ax)

        fig.tight_layout()
        self.canvas_agg = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_agg.mpl_connect('button_press_event', self.on_click)

    def update_plot_color(self, ax, idx):
        color = '#ffcccc' if idx in self.bad_indices else '#ccffcc'
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

# ==========================================
#        FRONTEND: VISUALIZATION
# ==========================================

def _add_grid_overlay(ax):
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(8, 0, -1)) 
    ax.tick_params(which='both', length=0) # Hide tick marks, keep numbers

def _create_grid_lines(ax):
    breaks = np.arange(0.5, 7.5, 1)
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
        
        _add_grid_overlay(ax)
        ax.set_title(f"Grid {idx+1}: {conf['name']}", fontsize=10)
        
        im = ax.imshow(downsampled_grids[idx][0], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
        
        # Lines must be re-drawn every frame on top of image
        lines = _create_grid_lines(ax)
        
        images.append((im, idx))
        animated_artists.append(im)
        animated_artists.extend(lines)
    
    # HUD
    ref_ax = axes[0, 0]
    time_text = ref_ax.text(0.05, 0.9, "Time: 0.00s", 
                            transform=ref_ax.transAxes, color='white', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    animated_artists.append(time_text)
    
    def update(frame_idx):
        for im_obj, grid_idx in images:
            im_obj.set_data(downsampled_grids[grid_idx][frame_idx])
        time_text.set_text(f"Time: {frame_idx * step / fs:.2f}s")
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
    
    _add_grid_overlay(ax)
    ax.set_title(f"{name}")
    
    im = ax.imshow(downsampled_data[0], cmap='magma', interpolation='bicubic', 
                   origin='upper', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    lines = _create_grid_lines(ax)
    
    time_text = ax.text(0.05, 0.93, "Time: 0.00s", 
                        transform=ax.transAxes, color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    all_artists = [im, time_text] + lines
    
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
        self.root.geometry("550x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.fs = None
        self.sEMG_data = None
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
        
        self.inspect_btn = ttk.Button(vis_frame, text="Inspect & Edit Components", command=self.on_inspect, state="disabled")
        self.inspect_btn.pack(fill="x", padx=10, pady=10)
        
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
            self.processed_cache = None
            self.raw_cache = []
            
            print("Pre-calculating raw geometry...")
            for i in range(6):
                start, end = i * 64, (i + 1) * 64
                chunk = self.sEMG_data[:, start:end]
                raw_geom = chunk.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
                self.raw_cache.append(np.abs(raw_geom))
            
            self.file_lbl.config(text=f"{os.path.basename(fpath)} ({self.fs} Hz)", foreground="black")
            self.proc_btn.config(state="normal", text="Run Auto-Processing")
            self.vis_btn.config(state="normal")
            self.inspect_btn.config(state="disabled") # Can't inspect until we select a grid
            self.proc_status_var.set("Status: Raw data ready.")
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
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = executor.map(process_single_grid_wrapper, tasks)
                results = list(futures)

            self.processed_cache = results
            
            self.root.after(0, lambda: self.proc_status_var.set("Status: Processing Complete."))
            self.root.after(0, lambda: self.proc_btn.config(text="Processing Done (Cached)"))
            self.root.after(0, lambda: self.inspect_btn.config(state="normal"))
            
        except Exception as e:
            print(f"Error: {e}")
            self.root.after(0, lambda: self.proc_status_var.set(f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    # --- COMPONENT INSPECTION ---
    def on_inspect(self):
        selection = self.grid_combo.get()
        if "Full Torso" in selection:
            messagebox.showinfo("Select Grid", "Please select a specific Grid (1-6) to inspect.")
            return
        
        grid_id = int(selection.split(":")[0].split(" ")[1])
        threading.Thread(target=self.run_inspection_task, args=(grid_id,), daemon=True).start()

    def run_inspection_task(self, grid_id):
        self.root.after(0, lambda: self.proc_status_var.set(f"Status: Analyzing Grid {grid_id} components..."))
        
        # We must re-run ICA locally to get sources (not cached to save RAM)
        start, end = (grid_id-1) * 64, grid_id * 64
        chunk = self.sEMG_data[:, start:end]
        
        filtered = filter_data(chunk, self.fs)
        ica, sources = ICA(filtered)
        hearts, hf, artifacts = analyze_envelope_refined(sources, self.fs)
        auto_bad = hearts + hf + artifacts
        
        # Open Window on Main Thread
        self.root.after(0, lambda: ComponentEditor(self.root, sources, self.fs, auto_bad, 
                                                   f"Grid {grid_id}", 
                                                   lambda bad: self.apply_manual_edit(grid_id, ica, sources, bad)))

    def apply_manual_edit(self, grid_id, ica, sources, bad_indices):
        # Callback from Editor: Re-construct and Update Cache
        print(f"Applying manual edits to Grid {grid_id}. Removing {len(bad_indices)} components.")
        sources[:, bad_indices] = 0.0
        clean = ica.inverse_transform(sources)
        rectified = np.abs(clean)
        b_env, a_env = butter(N=4, Wn=3.0, btype='low', fs=self.fs)
        envelopes = filtfilt(b_env, a_env, rectified, axis=0)
        
        # Geometry
        new_heatmap = envelopes.reshape(-1, 8, 8).transpose(0, 2, 1)[:, ::-1, :]
        
        # Update Cache
        if self.processed_cache is None: self.processed_cache = [None]*6
        self.processed_cache[grid_id-1] = new_heatmap
        
        self.proc_status_var.set(f"Status: Grid {grid_id} Updated Manually.")

    # --- VISUALIZATION ---
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
                    self.root.after(0, lambda: messagebox.showwarning("Wait", "Please run processing first."))
                    return 
                video_data = self.processed_cache

            selection = self.grid_combo.get()
            scale_mode = self.scale_var.get()
            
            if "Full Torso" in selection:
                target_grid_idx = -1
            else:
                target_grid_idx = int(selection.split(":")[0].split(" ")[1]) - 1

            if scale_mode == "global":
                all_grids_array = np.array(video_data)
                vmax = np.percentile(all_grids_array, 99.0)
                print(f"Global Scale: {vmax:.2f}")
            else:
                vmax = None
                print("Local Scale: Auto")

            self.root.after(0, lambda: self.launch_plot(target_grid_idx, video_data, vmax))
        except Exception as e:
            print(e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.vis_btn.config(state="normal"))

    def launch_plot(self, target_idx, data, vmax):
        if target_idx == -1:
            visualize_torso_animation(data, self.fs, fps=30)
        else:
            grid_anim = data[target_idx]
            visualize_grid_animation_scaled(grid_anim, self.fs, fps=30, grid_idx=target_idx, vmax=vmax)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = EMGControlPanel(root)
    root.mainloop()