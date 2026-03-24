"""
Binned window features used by low-data baselines.
"""

from __future__ import annotations

import math

import numpy as np


def extract_binned_features(emg: np.ndarray, good_mask: np.ndarray, fs_hz: float, bin_ms: int = 25) -> np.ndarray:
    """
    Convert a normalized EMG window into per-grid binned features.

    Output shape: (n_bins, 12) where each bin contains
    [grid0_mav, grid0_rms, grid1_mav, grid1_rms, ... grid5_mav, grid5_rms].
    """

    if emg.ndim != 4 or emg.shape[1:] != (6, 8, 8):
        raise ValueError(f"Unexpected EMG window shape: {emg.shape}")
    if good_mask.shape != (6, 8, 8):
        raise ValueError(f"Unexpected mask shape: {good_mask.shape}")

    bin_samples = max(1, int(round(fs_hz * bin_ms / 1000.0)))
    n_bins = int(math.ceil(emg.shape[0] / bin_samples))
    features = np.zeros((n_bins, 12), dtype=np.float32)

    for bin_idx in range(n_bins):
        start = bin_idx * bin_samples
        end = min(emg.shape[0], start + bin_samples)
        chunk = emg[start:end]
        for grid_idx in range(6):
            mask = good_mask[grid_idx].astype(bool)
            grid_values = chunk[:, grid_idx][:, mask]
            if grid_values.size == 0:
                mav = 0.0
                rms = 0.0
            else:
                mav = float(np.mean(np.abs(grid_values)))
                rms = float(np.sqrt(np.mean(np.square(grid_values))))
            features[bin_idx, grid_idx * 2] = mav
            features[bin_idx, grid_idx * 2 + 1] = rms

    return features
