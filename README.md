# Neonatal Assisted Breathing Regulator (sEMG-Based)

> ⚠️ **Status: Work in Progress (Research Phase)**

**An automated bio-signal processing and control framework designed to calibrate assisted breathing devices in real-time using High-Density Surface EMG (HD-sEMG).**

This repository hosts the backend logic for a **closed-loop ventilator control system**. It processes raw electrical signals from the diaphragm to assess respiratory effort, detects breathing difficulties (e.g., accessory muscle usage), and provides feedback to mechanical ventilators to prevent lung injury in neonates.

---

## 🗺️ Project Roadmap

This research is divided into four distinct engineering phases.

- [ ] **Phase 1: Signal Processing & Artifact Removal**
    - [x] Import proprietary `.mat` / `.orb` HD-sEMG data.
    - [x] Implement 50Hz Notch and 20-400Hz Bandpass filtering.
    - [x] Develop Blind Source Separation (BSS) using `FastICA`.
    - [x] Create an automated "Hunter-Killer" classifier to identify Heart (ECG) and Noise artifacts.
    - [x] Generate clean, rectifed Bicubic Heatmaps of muscle activation.
    - [ ] Automate same + united heatmap for all grids
- [ ] **Phase 2: Machine Learning (Clinical Assessment)**
    - [ ] Train a classifier to distinguish "Healthy Breathing" vs. "Respiratory Distress".
    - [ ] Detect specific pathologies (e.g., Pneumothorax signatures).
- [ ] **Phase 4: Control Logic (Hypothetical)**
    - [ ] Implement PID-style logic to adjust ventilator pressure based on EMG effort.
    - [ ] Simulate feedback loops for "Patient-Ventilator Asynchrony" prevention.

---

## ⚡ Phase 1: The Engineering Challenge (Signal Cleaning)
*Current State: Completed*

Surface EMG on the torso is chemically noisy. The signal of interest (Diaphragm) is effectively "whispering" while the heart (ECG) is "shouting."
* **Diaphragm Signal:** Stochastic, Gaussian distribution, Low Amplitude (~10µV).
* **ECG Artifact:** Deterministic, High Kurtosis, High Amplitude (~1mV).
* **Motion Artifacts:** Low-frequency cable sway and skin impedance changes.

Standard frequency filters (Bandpass) fail to remove the ECG because its QRS complex shares the same frequency band (20Hz-100Hz) as the muscle signal.

## 🛠️ The Solution: Automated ICA-Based Cleaning

This pipeline utilizes **Blind Source Separation (BSS)** via Independent Component Analysis (ICA) combined with a novel heuristic classification engine.

### The Pipeline Steps
1.  **Preprocessing:** Notch & Bandpass Filtering.
2.  **Source Separation:** `FastICA` decomposition (64 channels → 15 components).
3.  **Automated Classification (The "Hunter-Killer" Logic):**
    * The system analyzes each independent component using three metrics:
        * **Frequency (FFT):** Distinguishes biological signals (<3Hz envelope) from mechanical noise (>12Hz).
        * **Shape (Kurtosis):** Distinguishes spiky sources (Heart) from Gaussian sources (Muscle).
        * **Regularity (Interval Variance):** Distinguishes rhythmic heartbeats from random artifact pops.
4.  **Reconstruction:** Artifact components are zeroed out, and the clean signal is reconstructed.

---

## 🚀 Installation & Usage

### Requirements
* Python 3.8+
* `numpy`, `scipy`, `matplotlib`, `scikit-learn`

### Setup
```bash
# Clone the repository
git clone https://github.com/nikakogho/Neonatal-Diaphragm-sEMG-Breathing-Regulator.git
cd Neonatal-Diaphragm-sEMG-Breathing-Regulator

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
The script expects OT Bioelettronica `.mat` files. Ensure your data is placed in a secure directory (ignored by git).

```python
python main.py
```

## 📊 Algorithm Logic
The core innovation of Phase 1 is the `analyze_envelope_refined` function:

| Metric | Heart (ECG) | Diaphragm (Signal) | HF Noise | Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Dominant Freq** | ~1.0 Hz | N/A (Broadband) | > 12 Hz | ~1.0 Hz |
| **Kurtosis** | High (> 1.5) | Low (< 1.5) | Variable | High |
| **Rhythm** | Regular (σ < 0.2s) | Random | Regular | Irregular |
| **Verdict** | **DELETE** | **KEEP** | **DELETE** | **DELETE** |

## 🔒 Data Privacy Notice
**No patient data is contained in this repository.**
All raw `.mat` and `.orb` files containing biomedical recordings are strictly excluded via `.gitignore` to comply with GDPR/HIPAA regulations.

## 🤝 Contribution
This project is a Bachelor Thesis work.