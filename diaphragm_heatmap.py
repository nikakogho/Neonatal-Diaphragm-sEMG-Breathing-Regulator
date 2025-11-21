import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, welch
from sklearn.decomposition import FastICA

def load_mat_any(path):
    """Load MATLAB .mat (classic or v7.3/HDF5). Returns a dict."""
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        # strip meta keys
        return {k:v for k,v in d.items() if not k.startswith("__")}
    except NotImplementedError:
        print('Not working')

def notch_filter(data, fs):
    b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
    filtered_notch = filtfilt(b_notch, a_notch, data, axis=0)

    return filtered_notch

def bandpass_filter(data, fs):
    b_band, a_band = butter(N=4, Wn=[20, 400], btype='bandpass', fs=fs)
    filtered_final = filtfilt(b_band, a_band, data, axis=0)

    return filtered_final

def filter_data(data, fs):
    data = notch_filter(data, fs)
    data = bandpass_filter(data, fs)
    return data

def ICA(data):
    ica = FastICA(n_components=15, random_state=42, whiten='unit-variance')
    sources = ica.fit_transform(data)  # Returns (Samples, Components)
    mixing_matrix = ica.mixing_        # Returns (Channels, Components)

    return ica, sources, mixing_matrix

def analyze_envelope_refined(sources, fs):
    """
    Refined Logic: Distinguishes between HEART (Regular) and ARTIFACT (Irregular).
    """
    print(f"{'ID':<4} | {'Freq':<6} | {'Kurt':<6} | {'Reg (s)':<8} | {'Verdict'}")
    print("-" * 50)
    
    hearts, hf_noise, other_noise = [], [], []
    
    for i in range(sources.shape[1]):
        sig = sources[:, i]
        
        # 1. Shape (Kurtosis)
        kurt = kurtosis(sig)

        # 2. Frequency (Envelope FFT)
        sig_env = np.abs(sig)
        sig_env = sig_env - np.mean(sig_env)
        freqs, psd = welch(sig_env, fs, nperseg=fs*4)
        
        valid_mask = freqs > 0.5 
        if np.sum(valid_mask) > 0:
            peak_idx = np.argmax(psd[valid_mask])
            dom_freq = freqs[valid_mask][peak_idx]
        else:
            dom_freq = 0.0

        # 3. REGULARITY CHECK (The New Standard)
        # Find peaks to measure time intervals
        # distance=0.4s ensures we don't count double-peaks
        peaks, _ = find_peaks(sig_env, height=np.std(sig_env)*3, distance=int(fs*0.4))
        
        if len(peaks) > 3:
            intervals = np.diff(peaks) / fs
            regularity = np.std(intervals) # Low means very regular
        else:
            regularity = 10.0 # High penalty for not having enough peaks
            
        # --- VERDICT LOGIC ---
        verdict = "Keep"
        
        # RULE A: High Frequency Noise (Fan/Line Noise)
        if dom_freq > 12.0:
            verdict = "HF NOISE"
            hf_noise.append(i)
            
        # RULE B: The Heart (Strictly Regular)
        # Must be ~1Hz, Spiky, AND Regular (std dev < 0.2s)
        elif (0.7 <= dom_freq <= 3.0) and (kurt > 1.5):
            if regularity < 0.20:
                verdict = "HEART"
                hearts.append(i)
            else:
                # It looks like a heart (freq/kurt) but it skips beats or jitters
                verdict = "ARTIFACT" 
                other_noise.append(i) # We still kill it, but we label it correctly.
        
        print(f"{i:<4} | {dom_freq:<6.2f} | {kurt:<6.2f} | {regularity:<8.3f} | {verdict}")
        
    return hearts, hf_noise, other_noise
    """
    Calculates the Dominant Frequency of each component using FFT/Welch.
    """
    print(f"{'ID':<4} | {'Dom Freq (Hz)':<14} | {'Est BPM':<10} | {'Verdict'}")
    print("-" * 55)
    
    bad_indices = []
    
    for i in range(sources.shape[1]):
        sig = np.abs(sources[:, i])
        
        # Remove DC offset (Gravity) so FFT focuses on rhythm
        sig = sig - np.mean(sig)
        
        # 2. RUN FFT (Welch's Method)
        # Higher nperseg = better frequency resolution
        freqs, psd = welch(sig, fs, nperseg=fs*4)
        
        # 3. FIND PEAKS
        # We ignore < 0.5Hz (Slow Breathing)
        valid_mask = freqs > 0.5
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0: continue
            
        peak_idx = np.argmax(valid_psd)
        dom_freq = valid_freqs[peak_idx]
        
        # --- VERDICT LOGIC ---
        verdict = "Keep"
        
        # Heart Range: 0.6Hz (36 BPM) to 2.5Hz (150 BPM)
        if 0.6 <= dom_freq <= 2.5:
            verdict = "HEART"
            # bad_indices.append(i) # only catch noise here
            
        # Machine Noise: Anything distinct above 3Hz
        elif dom_freq > 3.0:
            verdict = "HF NOISE"
            bad_indices.append(i)
            
        print(f"{i:<4} | {dom_freq:<14.2f} | {dom_freq*60:<10.0f} | {verdict}")
        
    return bad_indices

def visualize_zoomed_in(data):
    # VISUALIZE ALL 15 COMPONENTS (ZOOMED IN)
    # We only look at 4000 samples (approx 2 seconds) to see the beat alignment.

    plt.figure(figsize=(15, 20)) # Tall plot to fit everyone
    start_sample = 10000
    end_sample = 14000 # 2 second window

    for i in range(15): # Assumes n_components=15
        plt.subplot(15, 1, i+1)
        plt.plot(data[start_sample:end_sample, i])
        plt.title(f"Component {i}", fontsize=10)
        plt.grid(True)
        # Remove x-ticks for cleanliness except on the last one
        if i < 14:
            plt.xticks([]) 

    plt.tight_layout()
    plt.show()

def heatmap(sources, ica):
    # 1. Reconstruct Clean Data
    clean_data_flat = ica.inverse_transform(sources)

    # 2. Rectify (Absolute Value)
    rectified = np.abs(clean_data_flat)

    # 3. Envelope (Low Pass Filter at 3Hz)
    # We want to smooth out the jitter and just see the "swell" of the breath
    b_env, a_env = butter(N=4, Wn=3.0, btype='low', fs=fs)
    envelopes = filtfilt(b_env, a_env, rectified, axis=0)

    # 4. Reshape to Grid (Samples, 8, 8)
    # CRITICAL: Check your mapping. Usually OT Bioelettronica is linear row-by-row.
    # If the heatmap looks "scrambled", we might need to transpose the 8x8.
    grid_video = envelopes.reshape(-1, 8, 8)

    # 5. Visualize a "Breathing Cycle"
    # Let's plot 16 frames separated by 0.5 seconds to see the lungs fill
    plt.figure(figsize=(15, 16))
    frames_to_show = [500*(i+1) for i in range(16)] # Arbitrary time points

    for i, t in enumerate(frames_to_show):
        plt.subplot(4, 4, i+1)
        # vmin/vmax ensures the color scale stays constant so you can compare frames
        plt.imshow(grid_video[t], cmap='magma', interpolation='bicubic', origin='upper',
                vmin=0, vmax=np.percentile(grid_video, 99))
        plt.colorbar()
        plt.title(f"Frame {t}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

data = load_mat_any('PARTICIPANT_1/Tube/Relaxed_Breathing_1_tube.otb+.mat')
#dict_keys(['AUX_signal', 'ELECTRODEPLACEMENT_INTRA', 'GRIDPLACEMENT_SURFACE', 'fsamp', 'iEMG', 'sEMG'])
sEMG = data['sEMG']
fs = data['fsamp'] # 2048 hz so 2048 frames per second so ~31 seconds of data

# print(sEMG.shape) # (384, 63955) due to being 6 8x8 grids
sEMG = sEMG.T # (63955, 384) so time comes first

grid1 = sEMG[:, 0:64] # only first of the 6 grids
# print(grid1.shape) # (63955, 64)

grid1 = filter_data(grid1, fs)

ica, sources, mixing_matrix = ICA(grid1)

# visualize_zoomed_in(sources) # showed that heart is in 8, 9, 11, and that 3 is noise

#sources[:, 3] = 0
bad_indices = []

detected_heart_indices, hf_noise, other_noise = analyze_envelope_refined(sources, fs)
potential_diaghragm_indices = [i for i in range(sources.shape[1]) if i not in detected_heart_indices + hf_noise + other_noise]

print(f"\nAuto-Detected Heart Indices: {detected_heart_indices}")
print(f"\nAuto-Detected HF Noise Indices: {hf_noise}")
print(f"\nAuto-Detected Other Noise Indices: {other_noise}")
print(f"\nAuto-Detected Potential Diaphragm Indices: {potential_diaghragm_indices}")

# removing non-diaphragm components
# heart_components = [8, 9, 11]
# noise_components = [3]

# components_to_remove = heart_components + noise_components
# for component in components_to_remove:
#     sources[:, component] = 0.0

# heatmap(sources, ica)