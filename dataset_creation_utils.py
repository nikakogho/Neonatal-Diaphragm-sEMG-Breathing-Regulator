# dataset_creation_utils.py
"""
Build TRAIN/TEST datasets from a folder of exported sEMG recordings (.npz).

Each .npz is expected to contain:
  - emg:      (n_t, 6, 8, 8) float/float32
  - aux:      (n_t, 2)        float/float32   (pump readings per timestamp)
  - bad_mask: (6, 8, 8)       bool           (True = bad electrode)
  - time_s:   (n_t,)          float          (optional but usually present)
  - meta:     json string     (required here; contains fs_export_hz etc.)

This module exports ONE public function:
  create_train_test_datasets_from_folder(...)

Split is RECORDING-WISE to avoid leakage.
Each sample returns:
  X: (T, 12, 8, 8)  where 12 = 6 signal channels + 6 mask channels
  y: (2,)           average aux over label window [start+delta : start+delta+T]

Even if EMG is already zeroed at bad electrodes, we re-apply the mask inside the dataset
as a safety check.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Recording:
    path: str
    name: str
    emg: np.ndarray        # (n_t, 6, 8, 8) float32
    aux: np.ndarray        # (n_t, 2) float32
    good_mask: np.ndarray  # (6, 8, 8) float32 (1 good, 0 bad)
    fs: float              # fs_export_hz

def _load_recording(npz_path: str) -> Recording:
    d = np.load(npz_path, allow_pickle=True)

    if "meta" not in d.files:
        raise ValueError(f"{npz_path} missing required 'meta' json string")

    meta = json.loads(str(d["meta"]))
    fs = float(meta["fs_export_hz"])

    for key in ("emg", "aux", "bad_mask"):
        if key not in d.files:
            raise ValueError(f"{npz_path} missing '{key}'")

    emg = d["emg"].astype(np.float32)        # (n_t,6,8,8)
    aux = d["aux"].astype(np.float32)        # (n_t,2)
    bad = d["bad_mask"].astype(bool)         # (6,8,8)

    if emg.ndim != 4 or emg.shape[1:] != (6, 8, 8):
        raise ValueError(f"{npz_path} bad emg shape {emg.shape}, expected (n_t, 6, 8, 8)")
    if aux.ndim != 2 or aux.shape[1] != 2:
        raise ValueError(f"{npz_path} bad aux shape {aux.shape}, expected (n_t, 2)")
    if bad.shape != (6, 8, 8):
        raise ValueError(f"{npz_path} bad bad_mask shape {bad.shape}, expected (6, 8, 8)")
    if aux.shape[0] == 0:
        raise ValueError(f"{npz_path} aux is empty; pump readings required")

    # Align lengths defensively
    n = min(emg.shape[0], aux.shape[0])
    emg = emg[:n]
    aux = aux[:n]

    good_mask = (~bad).astype(np.float32)  # 1 good, 0 bad
    name = os.path.splitext(os.path.basename(npz_path))[0]
    return Recording(path=npz_path, name=name, emg=emg, aux=aux, good_mask=good_mask, fs=fs)


def _scan_folder(folder: str) -> List[Recording]:
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".npz")
    )
    if not paths:
        raise FileNotFoundError(f"No .npz files found in folder: {folder}")

    recs = [_load_recording(p) for p in paths]

    fs_set = {round(r.fs, 6) for r in recs}
    if len(fs_set) != 1:
        raise ValueError(f"Different fs_export_hz across recordings: {fs_set}")
    return recs

def _build_window_index_and_normalize(
    recs: Sequence[Recording],
    win_ms: float,
    delta_ms: float,
    step_ms: float,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    fs = recs[0].fs
    win = int(round(fs * (win_ms / 1000.0)))
    delta = int(round(fs * (delta_ms / 1000.0)))
    step = int(round(fs * (step_ms / 1000.0)))

    if win <= 0:
        raise ValueError("win_ms too small -> win <= 0 samples")
    if step <= 0:
        raise ValueError("step_ms too small -> step <= 0 samples")
    if delta < 0:
        raise ValueError("delta_ms must be >= 0 for this formulation")

    index: List[Tuple[int, int]] = []
    for rid, r in enumerate(recs):
        n = r.emg.shape[0]
        # need: start+win <= n and start+delta+win <= n
        last_start = n - (delta + win)
        for start in range(0, max(0, last_start) + 1, step):
            if start + win <= n and start + delta + win <= n:
                index.append((rid, start))
    return index, win, delta, step


def _compute_norm_stats(
    recs: Sequence[Recording],
    train_rec_ids: Sequence[int],
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Global train-set normalization stats.
    - X mean/std: one scalar over ALL masked EMG values
    - y mean/std: per-target (2,) over all aux timestamps

    This does NOT remove "heavy breathing" information; it rescales it.
    """
    x_sum = 0.0
    x_sq = 0.0
    x_n = 0

    y_sum = np.zeros((2,), dtype=np.float64)
    y_sq = np.zeros((2,), dtype=np.float64)
    y_n = 0

    for rid in train_rec_ids:
        r = recs[rid]
        m = r.good_mask[None, :, :, :]     # (1,6,8,8)
        x = r.emg * m                      # ensure zeros at bad electrodes
        x_sum += float(x.sum())
        x_sq += float((x * x).sum())
        x_n += int(x.size)

        y = r.aux
        y_sum += y.sum(axis=0)
        y_sq += (y * y).sum(axis=0)
        y_n += int(y.shape[0])

    x_mean = x_sum / max(1, x_n)
    x_var = x_sq / max(1, x_n) - x_mean * x_mean
    x_std = float(np.sqrt(max(1e-12, x_var)))

    y_mean = y_sum / max(1, y_n)
    y_var = y_sq / max(1, y_n) - y_mean * y_mean
    y_std = np.sqrt(np.maximum(1e-12, y_var)).astype(np.float32)

    return float(x_mean), float(x_std), y_mean.astype(np.float32), y_std

class EMGWindowDataset(Dataset):
    """
    Each item:
      X: (T, 12, 8, 8)  = [6 signal chans, 6 mask chans]
      y: (2,)           = mean(aux[t+delta : t+delta+T], axis=0)
    """
    def __init__(
        self,
        recs: Sequence[Recording],
        index: Sequence[Tuple[int, int]],
        win: int,
        delta: int,
        normalize: bool,
        x_mean: float,
        x_std: float,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        include_mask_channels: bool = True,
    ):
        self.recs = recs
        self.index = list(index)
        self.win = int(win)
        self.delta = int(delta)

        self.normalize = bool(normalize)
        self.x_mean = float(x_mean)
        self.x_std = float(x_std) if x_std > 0 else 1.0
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = np.where(y_std > 0, y_std, 1.0).astype(np.float32)

        self.include_mask_channels = bool(include_mask_channels)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        rid, start = self.index[i]
        r = self.recs[rid]

        x = r.emg[start : start + self.win]  # (T,6,8,8)
        m = r.good_mask                      # (6,8,8)

        # Safety: enforce zeroing
        x = x * m[None, :, :, :]

        # Build X
        if self.include_mask_channels:
            m_time = np.broadcast_to(m[None, :, :, :], x.shape).astype(np.float32)  # (T,6,8,8)
            X = np.concatenate([x, m_time], axis=1)  # (T,12,8,8)
        else:
            X = x  # (T,6,8,8)

        # Label: average aux over future window
        y_win = r.aux[start + self.delta : start + self.delta + self.win]  # (T,2)
        y = y_win.mean(axis=0).astype(np.float32)  # (2,)

        if self.normalize:
            X = (X - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(X), torch.from_numpy(y)

# Public API: ONE function
def create_train_test_datasets_from_folder(
    folder: str,
    *,
    test_recording: Optional[Union[int, str]] = None,
    win_ms: float = 200.0,
    delta_ms: float = 100.0,
    step_ms: float = 50.0,
    normalize: bool = True,
    include_mask_channels: bool = True,
) -> Tuple[Dataset, Dataset, Dict]:
    """
    Build TRAIN/TEST datasets from a folder of per-recording .npz files.

    Split is RECORDING-WISE (prevents leakage).
      - If test_recording is None: picks LAST recording by filename (sorted).
      - If int: index into the sorted recordings list.
      - If str: matched against recording base-name (exact match).

    Returns:
      train_ds, test_ds, info dict (names, ids, fs, window params, stats, counts).
    """
    recs = _scan_folder(folder)
    names = [r.name for r in recs]

    if test_recording is None:
        test_id = len(recs) - 1
    elif isinstance(test_recording, int):
        if not (0 <= test_recording < len(recs)):
            raise IndexError(f"test_recording index out of range: {test_recording}")
        test_id = int(test_recording)
    elif isinstance(test_recording, str):
        if test_recording not in names:
            raise ValueError(f"test_recording='{test_recording}' not found. Available: {names}")
        test_id = names.index(test_recording)
    else:
        raise TypeError("test_recording must be None, int, or str")

    train_ids = [i for i in range(len(recs)) if i != test_id]

    index_all, win, delta, step = _build_window_index_and_normalize(recs, win_ms=win_ms, delta_ms=delta_ms, step_ms=step_ms)
    train_index = [(rid, s) for (rid, s) in index_all if rid in train_ids]
    test_index = [(rid, s) for (rid, s) in index_all if rid == test_id]

    if len(train_index) == 0:
        raise ValueError("Train index is empty. Check win_ms/delta_ms/step_ms vs recording lengths.")
    if len(test_index) == 0:
        raise ValueError("Test index is empty. Check win_ms/delta_ms/step_ms vs recording lengths.")

    # Normalization stats (train only)
    if normalize:
        x_mean, x_std, y_mean, y_std = _compute_norm_stats(recs, train_ids)
    else:
        x_mean, x_std = 0.0, 1.0
        y_mean = np.zeros((2,), np.float32)
        y_std = np.ones((2,), np.float32)

    train_ds = EMGWindowDataset(
        recs=recs,
        index=train_index,
        win=win,
        delta=delta,
        normalize=normalize,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        include_mask_channels=include_mask_channels,
    )

    test_ds = EMGWindowDataset(
        recs=recs,
        index=test_index,
        win=win,
        delta=delta,
        normalize=normalize,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        include_mask_channels=include_mask_channels,
    )

    info = {
        "recordings": names,
        "test_recording_id": test_id,
        "test_recording_name": names[test_id],
        "train_recording_ids": train_ids,
        "train_recording_names": [names[i] for i in train_ids],
        "fs_export_hz": float(recs[0].fs),
        "win_ms": float(win_ms),
        "delta_ms": float(delta_ms),
        "step_ms": float(step_ms),
        "win_samples": int(win),
        "delta_samples": int(delta),
        "step_samples": int(step),
        "n_train_windows": int(len(train_ds)),
        "n_test_windows": int(len(test_ds)),
        "normalize": bool(normalize),
        "include_mask_channels": bool(include_mask_channels),
        "x_mean": float(x_mean),
        "x_std": float(x_std),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }

    return train_ds, test_ds, info
