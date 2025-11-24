from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA
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
    ica = FastICA(n_components=15, random_state=42, whiten='unit-variance')
    sources = ica.fit_transform(data)  # Returns (Samples, Components)

    return ica, sources

def analyze_envelope_refined(sources, fs):
    MAX_HEART_COMPONENTS = 30
    MAX_ARTICAT_COMPONENTS = 30
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
            if regularity < 0.20 and len(hearts) < MAX_HEART_COMPONENTS:
                verdict = "HEART"
                hearts.append(i)
            elif len(other_noise) < MAX_ARTICAT_COMPONENTS:
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
    return envelopes.reshape(-1, 8, 8)

def visualize_full_torso_one_frame_at_a_time(all_grids_video, fs, frames_to_show):
    """Plots 6 grids side-by-side with a shared color scale."""
    # Calculate global max so bright grids look bright and quiet grids look dark
    global_max = np.percentile(np.array(all_grids_video), 99.0)
    
    for t in frames_to_show:
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Full Torso Activation - Frame {t} ({t/fs:.2f}s)", fontsize=16)
        
        for i in range(6):
            plt.subplot(2, 3, i+1) # 2 Rows, 3 Columns
            
            plt.imshow(all_grids_video[i][t], cmap='magma', 
                       interpolation='bicubic', origin='upper', 
                       vmin=0, vmax=global_max)
            
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"Grid {i+1}")
            plt.axis('off')

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
        ax.set_title(f"Grid {i+1}")
        ax.axis('off')
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
    plt.colorbar(im, ax=ax)
    title = ax.set_title(f"Grid {grid_id} - Time: 0.00s")
    ax.axis('off')

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

file_path = 'PARTICIPANT_1/Tube/Fast_Deep_Breathing_4_tube.otb+.mat'
if not os.path.exists(file_path):
    print("File not found!")
    exit()

data = load_mat_any(file_path)
sEMG = data['sEMG'].T # (63955, 384)
fs = data['fsamp']

all_grids_video = []

# Loop through all 6 grids
for i in range(6):
    start = i * 64
    end = (i + 1) * 64
    grid_chunk = sEMG[:, start:end]
    
    # Process independent chunks
    processed_grid = process_single_grid(grid_chunk, fs, grid_id=i+1)
    all_grids_video.append(processed_grid)

#print("\nGenerating Torso Maps...")
## Visualize Frames 2000, 2500, 3000 (adjust as needed)
#visualize_full_torso(all_grids_video, fs, frames_to_show=[2000, 2500, 3000])

grid1anim = visualize_grid_animation(all_grids_video[0], fs, fps=30, grid_id=1)
#anim = visualize_torso_animation(all_grids_video, fs, fps=30)