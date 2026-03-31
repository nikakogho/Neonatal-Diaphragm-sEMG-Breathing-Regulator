"""
Participant-level held-out recording prediction workflow for scalar AUX[0].
"""

from __future__ import annotations

import json
import math
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import WindowSpec, build_window_index, compute_window_starts, load_recording
from src.features import extract_binned_features
from src.training.single_recording import (
    _build_scalar_model,
    _plot_loss_curve,
    _plot_residual_hist,
    _plot_residual_vs_true,
    _plot_scatter,
    _plot_timeseries,
    _write_csv,
    build_model_info,
    collect_runtime_info,
    compute_scalar_metrics,
    set_seed,
    validate_runtime_requirements,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _normalize_emg_window(window: np.ndarray, good_mask: np.ndarray, x_mean: float, x_std: float) -> np.ndarray:
    masked = window * good_mask[None, :, :, :]
    normalized = (masked - x_mean) / x_std
    return normalized * good_mask[None, :, :, :]


def _feature_mode_for_model(model_name: str) -> str:
    return "raw" if model_name == "cnn1d_raw_scalar" else "binned"


def _dynamic_recording_name_set() -> set[str]:
    return {
        "p4rec2_processed_1024Hz",
        "p4rec3_processed_1024Hz",
        "p4rec4_processed_1024Hz",
        "p4rec5_processed_1024Hz",
        "p4rec6_processed_1024Hz",
        "p4rec7_processed_1024Hz",
        "p4rec8_processed_1024Hz",
    }


def _context_recording_name_set() -> set[str]:
    return {"p4rec11_processed_1024Hz", "p4rec1_processed_1024Hz"}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _prediction_config_slug(config: Dict[str, object]) -> str:
    parts = [
        str(config["model"]),
        f"win{int(config['win_ms'])}",
        f"delta{int(config['delta_ms'])}",
        f"bs{int(config['batch_size'])}",
        f"lr{float(config['lr']):.0e}".replace("+", ""),
        f"drop{str(config['dropout']).replace('.', '')}",
        f"base{int(config['base_channels'])}",
    ]
    return "_".join(parts)


def _participant_id_from_recordings(recordings) -> int:
    participant_ids = sorted({int(recording.pid) for recording in recordings})
    if len(participant_ids) != 1:
        raise ValueError(f"Expected exactly one participant in held-out prediction workflow, got {participant_ids}")
    return int(participant_ids[0])


def _list_recordings(recordings_dir: str):
    root = Path(recordings_dir)
    recordings = [load_recording(str(path)) for path in sorted(root.glob("*.npz"))]
    if not recordings:
        raise FileNotFoundError(f"No .npz recordings found in {recordings_dir}")
    return recordings


def _compute_scalar_normalization(recordings, target_channel: int) -> Dict[str, float]:
    x_sum = 0.0
    x_sq = 0.0
    x_n = 0
    y_sum = 0.0
    y_sq = 0.0
    y_n = 0
    for recording in recordings:
        mask = recording.good_mask.astype(bool)
        masked = recording.emg[:, mask]
        x_sum += float(masked.sum())
        x_sq += float(np.square(masked).sum())
        x_n += int(masked.size)
        y = recording.aux[:, int(target_channel)].astype(np.float32)
        y_sum += float(y.sum())
        y_sq += float(np.square(y).sum())
        y_n += int(y.shape[0])
    if x_n <= 0 or y_n <= 0:
        raise ValueError("Not enough train-only data to compute normalization statistics")
    x_mean = x_sum / x_n
    x_std = float(np.sqrt(max(1e-12, x_sq / x_n - x_mean * x_mean)))
    y_mean = y_sum / y_n
    y_std = float(np.sqrt(max(1e-12, y_sq / y_n - y_mean * y_mean)))
    return {
        "x_mean": float(x_mean),
        "x_std": float(max(x_std, 1e-6)),
        "y_mean": float(y_mean),
        "y_std": float(max(y_std, 1e-6)),
    }


class ParticipantPredictionDataset(Dataset):
    """Scalar AUX[0] dataset over a supplied recording subset and index subset."""

    def __init__(
        self,
        recordings,
        index,
        window_spec: WindowSpec,
        x_mean: float,
        x_std: float,
        y_mean: float,
        y_std: float,
        target_channel: int = 0,
        feature_mode: str = "raw",
        bin_ms: int = 25,
    ):
        self.recordings = list(recordings)
        self.recordings_by_path = {recording.path: recording for recording in self.recordings}
        self.index = list(index)
        self.window_spec = window_spec
        self.x_mean = float(x_mean)
        self.x_std = float(max(x_std, 1e-6))
        self.y_mean = float(y_mean)
        self.y_std = float(max(y_std, 1e-6))
        self.target_channel = int(target_channel)
        self.feature_mode = str(feature_mode)
        self.bin_ms = int(bin_ms)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.index[idx]
        recording = self.recordings_by_path[entry.path]
        win = int(round(recording.fs_hz * self.window_spec.win_ms / 1000.0))
        delta = int(round(recording.fs_hz * self.window_spec.delta_ms / 1000.0))
        start = int(entry.window_start)
        emg_window = recording.emg[start : start + win]
        target_window = recording.aux[start + delta : start + delta + win, self.target_channel]

        x = _normalize_emg_window(emg_window, recording.good_mask, self.x_mean, self.x_std)
        y_phys = np.asarray([float(target_window.mean())], dtype=np.float32)
        y = np.asarray([(float(y_phys[0]) - self.y_mean) / self.y_std], dtype=np.float32)
        midpoint = start + (win // 2)
        time_s = midpoint / recording.fs_hz

        if self.feature_mode == "raw":
            x_tensor = torch.from_numpy(x)
        elif self.feature_mode == "binned":
            x_tensor = torch.from_numpy(
                extract_binned_features(
                    emg=x,
                    good_mask=recording.good_mask,
                    fs_hz=recording.fs_hz,
                    bin_ms=self.bin_ms,
                ).astype(np.float32)
            )
        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        return {
            "x": x_tensor,
            "y": torch.from_numpy(y),
            "y_phys": torch.from_numpy(y_phys),
            "recording_name": str(recording.recording_name),
            "recording_path": str(recording.path),
            "window_start": start,
            "time_s": float(time_s),
        }


def _collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "y_phys": torch.stack([item["y_phys"] for item in batch], dim=0),
        "recording_name": [str(item["recording_name"]) for item in batch],
        "recording_path": [str(item["recording_path"]) for item in batch],
        "window_start": np.asarray([int(item["window_start"]) for item in batch], dtype=np.int32),
        "time_s": np.asarray([float(item["time_s"]) for item in batch], dtype=np.float32),
    }


def _make_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_batch)


def make_epoch_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    generator = np.random.default_rng(seed + epoch - 1)
    perm = generator.permutation(indices.shape[0])
    return indices[perm]


def build_prediction_shuffle_audit_rows(
    dataset: ParticipantPredictionDataset,
    epoch_order: np.ndarray,
    batch_size: int,
    holdout_recording: str,
    max_batches: int = 25,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_batches = int(math.ceil(len(epoch_order) / batch_size))
    for batch_idx in range(min(total_batches, max_batches)):
        batch_indices = epoch_order[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        names = [dataset.index[int(item)].recording_name for item in batch_indices]
        counts: Dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        rows.append(
            {
                "holdout_recording": holdout_recording,
                "epoch": 1,
                "batch_idx": batch_idx,
                "batch_size": int(len(batch_indices)),
                "unique_recordings": int(len(counts)),
                "recordings_seen": ", ".join(sorted(counts)),
                "recording_mix_json": json.dumps(counts, sort_keys=True),
            }
        )
    return rows


def _predict_scalar_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> Dict[str, np.ndarray]:
    pred_std_rows: List[np.ndarray] = []
    y_std_rows: List[np.ndarray] = []
    y_phys_rows: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    times: List[np.ndarray] = []
    names: List[str] = []
    paths: List[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred_std = model(batch["x"].to(device)).cpu().numpy().reshape(-1)
            pred_std_rows.append(pred_std)
            y_std_rows.append(batch["y"].numpy().reshape(-1))
            y_phys_rows.append(batch["y_phys"].numpy().reshape(-1))
            starts.append(batch["window_start"])
            times.append(batch["time_s"])
            names.extend(batch["recording_name"])
            paths.extend(batch["recording_path"])
    pred_std = np.concatenate(pred_std_rows, axis=0)
    y_std_arr = np.concatenate(y_std_rows, axis=0)
    y_phys = np.concatenate(y_phys_rows, axis=0)
    pred_phys = pred_std * y_std + y_mean
    return {
        "pred_std": pred_std,
        "y_std": y_std_arr,
        "pred_phys": pred_phys,
        "y_phys": y_phys,
        "recording_name": np.asarray(names, dtype=object),
        "recording_path": np.asarray(paths, dtype=object),
        "window_start": np.concatenate(starts, axis=0),
        "time_s": np.concatenate(times, axis=0),
    }


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += float(loss.item()) * int(xb.shape[0])
            total_count += int(xb.shape[0])
    return total_loss / max(total_count, 1)


def _split_recording_entries(recording, spec: WindowSpec, val_fraction: float):
    starts = compute_window_starts(recording.n_samples, recording.fs_hz, spec)
    if starts.shape[0] == 0:
        return [], []
    if starts.shape[0] == 1:
        entries = build_window_index([recording], spec)
        return entries, []
    val_count = max(1, min(starts.shape[0] - 1, int(math.ceil(starts.shape[0] * val_fraction))))
    train_starts = {int(item) for item in starts[:-val_count].tolist()}
    val_starts = {int(item) for item in starts[-val_count:].tolist()}
    entries = build_window_index([recording], spec)
    train_entries = [entry for entry in entries if int(entry.window_start) in train_starts]
    val_entries = [entry for entry in entries if int(entry.window_start) in val_starts]
    return train_entries, val_entries


def make_leave_one_recording_out_folds(
    recordings,
    spec: WindowSpec,
    val_fraction: float,
    target_channel: int = 0,
) -> List[Dict[str, object]]:
    folds: List[Dict[str, object]] = []
    sorted_recordings = sorted(recordings, key=lambda recording: str(recording.recording_name))
    for holdout_recording in sorted_recordings:
        train_recordings = [recording for recording in sorted_recordings if recording.path != holdout_recording.path]
        train_entries = []
        val_entries = []
        for recording in train_recordings:
            fold_train_entries, fold_val_entries = _split_recording_entries(recording, spec, val_fraction)
            train_entries.extend(fold_train_entries)
            val_entries.extend(fold_val_entries)
        test_entries = build_window_index([holdout_recording], spec)
        norm_stats = _compute_scalar_normalization(train_recordings, target_channel=target_channel)
        folds.append(
            {
                "holdout_recording": holdout_recording,
                "train_recordings": train_recordings,
                "train_entries": train_entries,
                "val_entries": val_entries,
                "test_entries": test_entries,
                "norm_stats": norm_stats,
            }
        )
    return folds


def classify_prediction_outcome(fold_rows: Sequence[Dict[str, object]]) -> str:
    dynamic_rows = [row for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    if not dynamic_rows:
        return "failure"
    dynamic_r2 = np.asarray([float(row["r2"]) for row in dynamic_rows], dtype=np.float32)
    median_dynamic_r2 = float(np.median(dynamic_r2))
    positive_count = int(np.sum(dynamic_r2 > 0.0))
    if median_dynamic_r2 >= 0.50 and positive_count >= 5:
        return "works"
    if median_dynamic_r2 <= 0.0 or positive_count < 3:
        return "failure"
    return "mixed"


def _aggregate_fold_metrics(
    fold_rows: Sequence[Dict[str, object]],
    aggregate_predictions: Dict[str, np.ndarray],
) -> Dict[str, object]:
    weighted_all = compute_scalar_metrics(aggregate_predictions["y_true"], aggregate_predictions["y_pred"])
    dynamic_rows = [row for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    context_rows = [row for row in fold_rows if not _as_bool(row["is_dynamic_recording"])]

    def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
        if not rows:
            return {
                "count": 0,
                "mean_r2": 0.0,
                "median_r2": 0.0,
                "mean_rmse": 0.0,
                "median_rmse": 0.0,
                "mean_pearson": 0.0,
                "positive_r2_count": 0,
            }
        r2 = np.asarray([float(row["r2"]) for row in rows], dtype=np.float32)
        rmse = np.asarray([float(row["rmse"]) for row in rows], dtype=np.float32)
        pearson = np.asarray([float(row["pearson"]) for row in rows], dtype=np.float32)
        return {
            "count": int(len(rows)),
            "mean_r2": float(np.mean(r2)),
            "median_r2": float(np.median(r2)),
            "mean_rmse": float(np.mean(rmse)),
            "median_rmse": float(np.median(rmse)),
            "mean_pearson": float(np.mean(pearson)),
            "positive_r2_count": int(np.sum(r2 > 0.0)),
        }

    return {
        "weighted_all_holdouts": weighted_all,
        "dynamic_holdouts": _summary(dynamic_rows),
        "context_holdouts": _summary(context_rows),
        "outcome_label": classify_prediction_outcome(fold_rows),
    }


def _best_and_worst_holdouts(fold_rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    dynamic_rows = [row for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    ordered = sorted(dynamic_rows, key=lambda row: (float(row["r2"]), float(row["rmse"])))
    return {
        "worst": ordered[:3],
        "best": list(reversed(ordered[-3:])),
    }


def summarize_prediction_run_row(
    config: Dict[str, object],
    aggregate_metrics: Dict[str, object],
    fold_rows: Sequence[Dict[str, object]],
    run_dir: Path,
    fixed_first: bool,
) -> Dict[str, object]:
    dynamic_rows = [row for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    dynamic_r2 = (
        np.asarray([float(row["r2"]) for row in dynamic_rows], dtype=np.float32)
        if dynamic_rows
        else np.zeros((0,), dtype=np.float32)
    )
    dynamic_pearson = (
        np.asarray([float(row["pearson"]) for row in dynamic_rows], dtype=np.float32)
        if dynamic_rows
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "config_name": _prediction_config_slug(config),
        "run_dir": str(run_dir),
        "fixed_first": bool(fixed_first),
        "model": str(config["model"]),
        "win_ms": int(config["win_ms"]),
        "delta_ms": int(config["delta_ms"]),
        "step_ms": int(config["step_ms"]),
        "max_epochs": int(config["max_epochs"]),
        "batch_size": int(config["batch_size"]),
        "lr": float(config["lr"]),
        "dropout": float(config["dropout"]),
        "base_channels": int(config["base_channels"]),
        "weighted_r2": float(aggregate_metrics["weighted_all_holdouts"]["r2"]),
        "weighted_rmse": float(aggregate_metrics["weighted_all_holdouts"]["rmse"]),
        "weighted_mae": float(aggregate_metrics["weighted_all_holdouts"]["mae"]),
        "weighted_pearson": float(aggregate_metrics["weighted_all_holdouts"]["pearson"]),
        "dynamic_median_r2": float(np.median(dynamic_r2)) if dynamic_r2.size else 0.0,
        "dynamic_positive_r2_count": int(np.sum(dynamic_r2 > 0.0)) if dynamic_r2.size else 0,
        "dynamic_mean_pearson": float(np.mean(dynamic_pearson)) if dynamic_pearson.size else 0.0,
        "outcome_label": str(aggregate_metrics["outcome_label"]),
    }


def rank_prediction_run_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["dynamic_median_r2"]),
            -int(row["dynamic_positive_r2_count"]),
            -float(row["dynamic_mean_pearson"]),
            float(row["weighted_rmse"]),
        ),
    )


def _plot_holdout_r2_rmse(fold_rows: Sequence[Dict[str, object]], path: Path) -> None:
    names = [str(row["holdout_recording"]) for row in fold_rows]
    r2_values = [float(row["r2"]) for row in fold_rows]
    rmse_values = [float(row["rmse"]) for row in fold_rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(names, r2_values)
    axes[0].set_title("Held-out recording R2")
    axes[0].set_ylim(min(-0.2, float(min(r2_values, default=0.0)) - 0.05), 1.05)
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, rmse_values)
    axes[1].set_title("Held-out recording RMSE")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_holdout_pearson(fold_rows: Sequence[Dict[str, object]], path: Path) -> None:
    names = [str(row["holdout_recording"]) for row in fold_rows]
    values = [float(row["pearson"]) for row in fold_rows]
    plt.figure(figsize=(10, 4.5))
    plt.bar(names, values)
    plt.title("Held-out recording Pearson")
    plt.ylim(min(-0.2, float(min(values, default=0.0)) - 0.05), 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_holdout_window_count(fold_rows: Sequence[Dict[str, object]], path: Path) -> None:
    names = [str(row["holdout_recording"]) for row in fold_rows]
    values = [int(row["test_window_count"]) for row in fold_rows]
    plt.figure(figsize=(10, 4.5))
    plt.bar(names, values)
    plt.title("Held-out window counts")
    plt.ylabel("Window count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_holdout_timeseries_grid(prediction_rows: Sequence[Dict[str, object]], path: Path) -> None:
    unique_names = sorted({str(row["holdout_recording"]) for row in prediction_rows})
    if not unique_names:
        return
    n_cols = 3
    n_rows = int(math.ceil(len(unique_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.1 * n_rows), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, holdout_name in zip(axes.flatten(), unique_names):
        rows = [row for row in prediction_rows if str(row["holdout_recording"]) == holdout_name]
        rows.sort(key=lambda row: float(row["time_s"]))
        ax.axis("on")
        ax.plot([float(row["time_s"]) for row in rows], [float(row["y_true"]) for row in rows], label="true", linewidth=1.35)
        ax.plot([float(row["time_s"]) for row in rows], [float(row["y_pred"]) for row in rows], label="pred", linewidth=1.0)
        ax.set_title(holdout_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AUX[0]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Held-out predicted vs true over time", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_dynamic_vs_context_metrics(fold_rows: Sequence[Dict[str, object]], path: Path) -> None:
    dynamic_r2 = [float(row["r2"]) for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    context_r2 = [float(row["r2"]) for row in fold_rows if not _as_bool(row["is_dynamic_recording"])]
    dynamic_rmse = [float(row["rmse"]) for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    context_rmse = [float(row["rmse"]) for row in fold_rows if not _as_bool(row["is_dynamic_recording"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].boxplot([dynamic_r2 or [0.0], context_r2 or [0.0]], tick_labels=["Dynamic", "Context"])
    axes[0].set_title("R2 by holdout type")
    axes[1].boxplot([dynamic_rmse or [0.0], context_rmse or [0.0]], tick_labels=["Dynamic", "Context"])
    axes[1].set_title("RMSE by holdout type")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_config_metric_heatmap(
    config_rows: Sequence[Dict[str, object]],
    per_config_fold_rows: Dict[str, Sequence[Dict[str, object]]],
    metric_key: str,
    title: str,
    path: Path,
) -> None:
    config_names = [str(row["config_name"]) for row in config_rows]
    fold_names = sorted(
        {
            str(fold_row["holdout_recording"])
            for rows in per_config_fold_rows.values()
            for fold_row in rows
        }
    )
    if not config_names or not fold_names:
        return
    matrix = np.zeros((len(config_names), len(fold_names)), dtype=np.float32)
    for row_idx, config_name in enumerate(config_names):
        fold_rows = {str(item["holdout_recording"]): item for item in per_config_fold_rows[config_name]}
        for col_idx, fold_name in enumerate(fold_names):
            matrix[row_idx, col_idx] = float(fold_rows[fold_name][metric_key])
    plt.figure(figsize=(1.2 * len(fold_names) + 4, 0.65 * len(config_names) + 3))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.xticks(np.arange(len(fold_names)), fold_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(config_names)), config_names)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_config_rank_summary(config_rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not config_rows:
        return
    names = [str(row["config_name"]) for row in config_rows]
    median_r2 = [float(row["dynamic_median_r2"]) for row in config_rows]
    weighted_r2 = [float(row["weighted_r2"]) for row in config_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(names, median_r2)
    axes[0].set_title("Dynamic median held-out R2")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, weighted_r2)
    axes[1].set_title("Weighted all-holdout R2")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fold_dir_name(recording_name: str) -> str:
    return str(recording_name)


def _save_fold_artifacts(
    fold_dir: Path,
    fold_config: Dict[str, object],
    history: Sequence[Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    metrics_row: Dict[str, object],
    best_state: Dict[str, torch.Tensor],
    last_state: Dict[str, torch.Tensor],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "config.json").write_text(json.dumps(fold_config, indent=2), encoding="utf-8")
    (fold_dir / "metrics.json").write_text(json.dumps(metrics_row, indent=2), encoding="utf-8")
    (fold_dir / "history.json").write_text(json.dumps(list(history), indent=2), encoding="utf-8")
    _write_csv(fold_dir / "history.csv", list(history))
    prediction_rows = [
        {
            "holdout_recording": str(metrics_row["holdout_recording"]),
            "recording_name": str(recording_name),
            "window_start": int(window_start),
            "time_s": float(time_s),
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "residual": float(y_pred - y_true),
        }
        for recording_name, window_start, time_s, y_true, y_pred in zip(
            predictions["recording_name"],
            predictions["window_start"],
            predictions["time_s"],
            predictions["y_phys"],
            predictions["pred_phys"],
        )
    ]
    _write_csv(fold_dir / "predictions.csv", prediction_rows)
    np.savez(
        fold_dir / "predictions.npz",
        recording_name=np.asarray(predictions["recording_name"], dtype="U128"),
        window_start=predictions["window_start"],
        time_s=predictions["time_s"],
        y_true=predictions["y_phys"],
        y_pred=predictions["pred_phys"],
        y_std=predictions["y_std"],
        pred_std=predictions["pred_std"],
    )
    torch.save(best_state, fold_dir / "model_best.pt")
    torch.save(last_state, fold_dir / "model_last.pt")
    residual = predictions["pred_phys"] - predictions["y_phys"]
    _plot_loss_curve(history, fold_dir / "loss_curve.png")
    _plot_scatter(predictions["y_phys"], predictions["pred_phys"], metrics_row, fold_dir / "holdout_scatter.png")
    _plot_timeseries(predictions["time_s"], predictions["y_phys"], predictions["pred_phys"], fold_dir / "holdout_timeseries.png")
    _plot_residual_hist(residual, fold_dir / "residual_hist.png")
    _plot_residual_vs_true(predictions["y_phys"], residual, fold_dir / "residual_vs_true.png")


def _artifact_guide_lines(has_comparison: bool) -> List[str]:
    lines = [
        "- `config.json`: exact selected configuration and benchmark settings.",
        "- `metrics.json`: aggregate held-out metrics, fold summaries, and outcome label for this selected config.",
        "- `model_info.json`: exact architecture description and parameter counts.",
        "- `fold_metrics.csv`: one row per held-out recording with RMSE, MAE, R2, Pearson, and checkpoint info.",
        "- `aggregate_predictions.csv` / `aggregate_predictions.npz`: pooled held-out predictions across all folds.",
        "- `model_locations.csv`: exact best-checkpoint path for every held-out recording.",
        "- `shuffle_audit.csv`: evidence that training windows were mixed across training recordings inside each fold.",
        "- `folds/<heldout_recording>/model_best.pt`: best checkpoint for that held-out fold.",
        "- `folds/<heldout_recording>/model_last.pt`: last checkpoint for that held-out fold.",
        "- `folds/<heldout_recording>/history.csv` / `history.json`: train/validation loss for that fold.",
        "- `folds/<heldout_recording>/predictions.csv` / `predictions.npz`: held-out predictions for that fold.",
        "- `holdout_r2_rmse.png`: per-heldout R2 and RMSE chart.",
        "- `holdout_pearson.png`: per-heldout Pearson chart.",
        "- `holdout_window_count.png`: held-out window counts.",
        "- `aggregate_holdout_scatter.png`: all held-out windows pooled together.",
        "- `holdout_timeseries_grid.png`: small-multiple time-series view across all held-out recordings.",
        "- `dynamic_vs_context_metrics.png`: dynamic-vs-context boxplot comparison.",
    ]
    if has_comparison:
        lines.extend(
            [
                "- `../comparison/leaderboard.csv`: ranking of fixed-first and rescue-sweep configs.",
                "- `../comparison/config_vs_fold_r2_heatmap.png`: held-out R2 heatmap across configs.",
                "- `../comparison/config_vs_fold_rmse_heatmap.png`: held-out RMSE heatmap across configs.",
            ]
        )
    return lines


def _example_fold_name(fold_rows: Sequence[Dict[str, object]]) -> str:
    dynamic_rows = [row for row in fold_rows if _as_bool(row["is_dynamic_recording"])]
    if dynamic_rows:
        return str(max(dynamic_rows, key=lambda row: float(row["r2"]))["holdout_recording"])
    return str(fold_rows[0]["holdout_recording"])


def _write_prediction_run_summary(
    run_dir: Path,
    config: Dict[str, object],
    metrics: Dict[str, object],
    fold_rows: Sequence[Dict[str, object]],
    has_comparison: bool,
) -> None:
    lines = [
        "# Participant 4 Held-Out Prediction Summary",
        "",
        "Main presentation document: [README.md](README.md)",
        "",
        "## Outcome",
        f"- Outcome label: `{metrics['outcome_label']}`",
        f"- Weighted all-holdout RMSE: `{metrics['weighted_all_holdouts']['rmse']:.6f}`",
        f"- Weighted all-holdout R2: `{metrics['weighted_all_holdouts']['r2']:.6f}`",
        f"- Dynamic median held-out R2: `{metrics['dynamic_holdouts']['median_r2']:.6f}`",
        f"- Dynamic positive-R2 folds: `{metrics['dynamic_holdouts']['positive_r2_count']} / {metrics['dynamic_holdouts']['count']}`",
        "",
        "## Protocol",
        "- Outer loop: leave one full recording out.",
        "- Inner validation: last 20% of windows from each remaining training recording.",
        "- Checkpoint selection: best validation MSE on training recordings only.",
        "",
        "## Key Charts",
        "![Holdout R2 and RMSE](holdout_r2_rmse.png)",
        "",
        "![Aggregate held-out scatter](aggregate_holdout_scatter.png)",
        "",
        "![Held-out time-series grid](holdout_timeseries_grid.png)",
        "",
        "## Artifact Guide",
        *_artifact_guide_lines(has_comparison=has_comparison),
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prediction_run_readme(
    run_dir: Path,
    config: Dict[str, object],
    runtime_info: Dict[str, object],
    model_info: Dict[str, object],
    metrics: Dict[str, object],
    fold_rows: Sequence[Dict[str, object]],
    model_location_rows: Sequence[Dict[str, object]],
    has_comparison: bool,
    fixed_first_metrics: Optional[Dict[str, object]],
    selected_from_rescue: bool,
) -> None:
    highlights = _best_and_worst_holdouts(fold_rows)
    example_fold = _example_fold_name(fold_rows)
    lines = [
        "# Participant 4 Step 3 Held-Out Recording Prediction",
        "",
        "## Goal",
        "Train on participant 4 recordings except one, then predict `AUX[0]` on the held-out recording.",
        "",
        "This is the first same-person prediction benchmark in the repository. Unlike step 1 and step 2, the evaluation recording is completely unseen during fitting, normalization, and checkpoint selection.",
        "",
        "## Protocol",
        "- Outer benchmark: leave-one-recording-out across all 9 cleaned participant-4 recordings.",
        "- Inner validation: the last 20% of windows from each remaining training recording.",
        "- Held-out recording is excluded from train, validation, normalization, and early stopping.",
        "- Training windows are shuffled globally across the remaining recordings every epoch.",
        "",
        "## Selected Outcome",
        f"- Outcome label: `{metrics['outcome_label']}`",
        f"- Weighted all-holdout RMSE: `{metrics['weighted_all_holdouts']['rmse']:.6f}`",
        f"- Weighted all-holdout MAE: `{metrics['weighted_all_holdouts']['mae']:.6f}`",
        f"- Weighted all-holdout Pearson: `{metrics['weighted_all_holdouts']['pearson']:.6f}`",
        f"- Weighted all-holdout R2: `{metrics['weighted_all_holdouts']['r2']:.6f}`",
        f"- Dynamic median held-out R2: `{metrics['dynamic_holdouts']['median_r2']:.6f}`",
        f"- Dynamic positive-R2 folds: `{metrics['dynamic_holdouts']['positive_r2_count']} / {metrics['dynamic_holdouts']['count']}`",
        "",
    ]
    if fixed_first_metrics is not None:
        lines.extend(
            [
                "## Fixed-First Baseline vs Selected Config",
                f"- Fixed-first baseline outcome: `{fixed_first_metrics['outcome_label']}`",
                f"- Fixed-first weighted R2: `{fixed_first_metrics['weighted_all_holdouts']['r2']:.6f}`",
                f"- Fixed-first dynamic median R2: `{fixed_first_metrics['dynamic_holdouts']['median_r2']:.6f}`",
                f"- Selected config came from rescue sweep: `{selected_from_rescue}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Exact Model Architecture",
            f"- Model family: `{model_info['display_name']}`",
            f"- Input description: {model_info['input_description']}",
            f"- Output: {model_info['output_description']}",
            f"- Trainable parameters: `{model_info['trainable_params']}`",
            f"- Total parameters: `{model_info['total_params']}`",
            "",
            "Architecture, layer by layer:",
            *[f"- {layer}" for layer in model_info["layers"]],
            "",
            "## Hyperparameters",
            f"- Model: `{config['model']}`",
            f"- Window / delta / step (ms): `{config['win_ms']}` / `{config['delta_ms']}` / `{config['step_ms']}`",
            f"- Batch size: `{config['batch_size']}`",
            f"- Learning rate: `{config['lr']}`",
            f"- Weight decay: `{config['weight_decay']}`",
            f"- Dropout: `{config['dropout']}`",
            f"- Base channels: `{config['base_channels']}`",
            f"- Min / max epochs: `{config['min_epochs']}` / `{config['max_epochs']}`",
            f"- Patience: `{config['patience']}`",
            f"- Seed: `{config['seed']}`",
            "",
            "## Runtime Environment",
            f"- Python executable: `{runtime_info['python_executable']}`",
            f"- Running inside venv: `{runtime_info['in_venv']}`",
            f"- Resolved device: `{runtime_info['resolved_device']}`",
            f"- CUDA available: `{runtime_info['cuda_available']}`",
            f"- GPU: `{runtime_info.get('device_name', 'n/a')}`",
            "",
            "## What Worked / What Didn't",
            "- Best dynamic held-outs:",
        ]
    )
    for row in highlights["best"]:
        lines.append(
            f"- `{row['holdout_recording']}`: R2 `{float(row['r2']):.6f}`, RMSE `{float(row['rmse']):.6f}`, Pearson `{float(row['pearson']):.6f}`"
        )
    lines.append("- Weakest dynamic held-outs:")
    for row in highlights["worst"]:
        lines.append(
            f"- `{row['holdout_recording']}`: R2 `{float(row['r2']):.6f}`, RMSE `{float(row['rmse']):.6f}`, Pearson `{float(row['pearson']):.6f}`"
        )
    lines.extend(
        [
            "",
            "## Fold Table",
            "| Held-out recording | Dynamic | Test windows | Best epoch | R2 | RMSE | MAE | Pearson |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['holdout_recording']} | {_as_bool(row['is_dynamic_recording'])} | {row['test_window_count']} | "
            f"{row['best_epoch']} | {float(row['r2']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['mae']):.6f} | {float(row['pearson']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Where The Trained Models Are",
            "There is no single universal checkpoint in leave-one-recording-out evaluation. Each held-out recording has its own best model.",
            "",
            "| Held-out recording | Best checkpoint | Best epoch | R2 | RMSE | Pearson |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in model_location_rows:
        lines.append(
            f"| {row['holdout_recording']} | `{row['checkpoint_path']}` | {row['best_epoch']} | "
            f"{float(row['r2']):.6f} | {float(row['rmse']):.6f} | {float(row['pearson']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Guide",
            *_artifact_guide_lines(has_comparison=has_comparison),
            "",
            "## Charts",
            "### Held-out R2 and RMSE",
            "![Held-out R2 and RMSE](holdout_r2_rmse.png)",
            "",
            "### Aggregate Held-out Scatter",
            "![Aggregate held-out scatter](aggregate_holdout_scatter.png)",
            "",
            "### Held-out Time-Series Grid",
            "![Held-out time-series grid](holdout_timeseries_grid.png)",
            "",
            f"### Example Fold Loss Curve: `{example_fold}`",
            f"![Example fold loss curve](folds/{example_fold}/loss_curve.png)",
            "",
        ]
    )
    if has_comparison:
        lines.extend(
            [
                "### Rescue-Sweep Heatmap",
                "![Config vs fold R2 heatmap](../comparison/config_vs_fold_r2_heatmap.png)",
                "",
            ]
        )
    lines.extend(
        [
            "### Dynamic vs Context Metrics",
            "![Dynamic vs context metrics](dynamic_vs_context_metrics.png)",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_prediction_fold(
    fold: Dict[str, object],
    config: Dict[str, object],
    feature_mode: str,
    torch_device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    holdout_recording = fold["holdout_recording"]
    norm_stats = fold["norm_stats"]
    window_spec = WindowSpec(
        win_ms=int(config["win_ms"]),
        step_ms=int(config["step_ms"]),
        delta_ms=int(config["delta_ms"]),
    )
    train_dataset = ParticipantPredictionDataset(
        recordings=fold["train_recordings"],
        index=fold["train_entries"],
        window_spec=window_spec,
        x_mean=float(norm_stats["x_mean"]),
        x_std=float(norm_stats["x_std"]),
        y_mean=float(norm_stats["y_mean"]),
        y_std=float(norm_stats["y_std"]),
        target_channel=int(config["target_channel"]),
        feature_mode=feature_mode,
    )
    val_dataset = ParticipantPredictionDataset(
        recordings=fold["train_recordings"],
        index=fold["val_entries"],
        window_spec=window_spec,
        x_mean=float(norm_stats["x_mean"]),
        x_std=float(norm_stats["x_std"]),
        y_mean=float(norm_stats["y_mean"]),
        y_std=float(norm_stats["y_std"]),
        target_channel=int(config["target_channel"]),
        feature_mode=feature_mode,
    )
    test_dataset = ParticipantPredictionDataset(
        recordings=[holdout_recording],
        index=fold["test_entries"],
        window_spec=window_spec,
        x_mean=float(norm_stats["x_mean"]),
        x_std=float(norm_stats["x_std"]),
        y_mean=float(norm_stats["y_mean"]),
        y_std=float(norm_stats["y_std"]),
        target_channel=int(config["target_channel"]),
        feature_mode=feature_mode,
    )
    train_indices = np.arange(len(train_dataset), dtype=np.int32)
    val_indices = np.arange(len(val_dataset), dtype=np.int32)
    test_indices = np.arange(len(test_dataset), dtype=np.int32)
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError(f"Fold for {holdout_recording.recording_name} does not have enough train/test windows")

    val_loader = (
        _make_loader(val_dataset, val_indices, batch_size=int(config["batch_size"]), shuffle=False)
        if val_indices.size
        else None
    )
    test_loader = _make_loader(test_dataset, test_indices, batch_size=int(config["batch_size"]), shuffle=False)
    first_epoch_order = make_epoch_order(train_indices, seed=int(config["seed"]), epoch=1)
    shuffle_rows = build_prediction_shuffle_audit_rows(
        train_dataset,
        epoch_order=first_epoch_order,
        batch_size=int(config["batch_size"]),
        holdout_recording=str(holdout_recording.recording_name),
    )

    sample_x = train_dataset[0]["x"]
    model = _build_scalar_model(config["model"], sample_x=sample_x, config=config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.MSELoss()
    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    fit_start = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        epoch_order = make_epoch_order(train_indices, seed=int(config["seed"]), epoch=epoch)
        train_loader = _make_loader(train_dataset, epoch_order, batch_size=int(config["batch_size"]), shuffle=False)
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            xb = batch["x"].to(torch_device)
            yb = batch["y"].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(xb.shape[0])
            total_count += int(xb.shape[0])
        train_loss = total_loss / max(total_count, 1)
        val_loss = _evaluate_loss(model, val_loader, criterion, torch_device) if val_loader is not None else train_loss
        history.append({"epoch": epoch, "train_loss": float(train_loss), "eval_loss": float(val_loss)})
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        elif epoch >= int(config["min_epochs"]):
            stale_epochs += 1
        last_state = deepcopy(model.state_dict())
        if epoch >= int(config["min_epochs"]) and stale_epochs >= int(config["patience"]):
            break

    fit_seconds = time.perf_counter() - fit_start
    if best_state is None or last_state is None:
        raise RuntimeError(f"Training fold {holdout_recording.recording_name} ended without checkpoints")

    model.load_state_dict(best_state)
    predictions = _predict_scalar_model(
        model,
        test_loader,
        torch_device,
        y_mean=float(norm_stats["y_mean"]),
        y_std=float(norm_stats["y_std"]),
    )
    fold_metrics = compute_scalar_metrics(predictions["y_phys"], predictions["pred_phys"])
    metrics_row = {
        "holdout_recording": str(holdout_recording.recording_name),
        "is_dynamic_recording": bool(holdout_recording.recording_name in _dynamic_recording_name_set()),
        "aux0_std": float(holdout_recording.aux[:, int(config["target_channel"])].std()),
        "train_window_count": int(train_indices.size),
        "val_window_count": int(val_indices.size),
        "test_window_count": int(test_indices.size),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "fit_seconds": float(fit_seconds),
        **{key: float(value) for key, value in fold_metrics.items()},
    }
    fold_dir = run_dir / "folds" / _fold_dir_name(str(holdout_recording.recording_name))
    fold_config = {
        **config,
        "holdout_recording": str(holdout_recording.recording_name),
        "train_recording_names": [str(recording.recording_name) for recording in fold["train_recordings"]],
    }
    _save_fold_artifacts(
        fold_dir=fold_dir,
        fold_config=fold_config,
        history=history,
        predictions=predictions,
        metrics_row=metrics_row,
        best_state=best_state,
        last_state=last_state,
    )
    prediction_rows = [
        {
            "holdout_recording": str(holdout_recording.recording_name),
            "recording_name": str(recording_name),
            "window_start": int(window_start),
            "time_s": float(time_s),
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "residual": float(y_pred - y_true),
        }
        for recording_name, window_start, time_s, y_true, y_pred in zip(
            predictions["recording_name"],
            predictions["window_start"],
            predictions["time_s"],
            predictions["y_phys"],
            predictions["pred_phys"],
        )
    ]
    model_location_row = {
        "holdout_recording": str(holdout_recording.recording_name),
        "checkpoint_path": str(Path("folds") / _fold_dir_name(str(holdout_recording.recording_name)) / "model_best.pt"),
        "best_epoch": int(best_epoch),
        "r2": float(metrics_row["r2"]),
        "rmse": float(metrics_row["rmse"]),
        "pearson": float(metrics_row["pearson"]),
    }
    return {
        "metrics_row": metrics_row,
        "prediction_rows": prediction_rows,
        "shuffle_rows": shuffle_rows,
        "model_location_row": model_location_row,
    }


def _aggregate_prediction_rows(prediction_rows: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {
        "holdout_recording": np.asarray([str(row["holdout_recording"]) for row in prediction_rows], dtype=object),
        "recording_name": np.asarray([str(row["recording_name"]) for row in prediction_rows], dtype=object),
        "window_start": np.asarray([int(row["window_start"]) for row in prediction_rows], dtype=np.int32),
        "time_s": np.asarray([float(row["time_s"]) for row in prediction_rows], dtype=np.float32),
        "y_true": np.asarray([float(row["y_true"]) for row in prediction_rows], dtype=np.float32),
        "y_pred": np.asarray([float(row["y_pred"]) for row in prediction_rows], dtype=np.float32),
    }


def _run_prediction_config(
    recordings_dir: str,
    config: Dict[str, object],
    run_dir: Path,
    runtime_info: Dict[str, object],
) -> Dict[str, object]:
    set_seed(int(config["seed"]))
    recordings = _list_recordings(recordings_dir)
    participant_id = _participant_id_from_recordings(recordings)
    spec = WindowSpec(win_ms=int(config["win_ms"]), step_ms=int(config["step_ms"]), delta_ms=int(config["delta_ms"]))
    folds = make_leave_one_recording_out_folds(
        recordings=recordings,
        spec=spec,
        val_fraction=float(config["val_fraction"]),
        target_channel=int(config["target_channel"]),
    )
    feature_mode = _feature_mode_for_model(str(config["model"]))
    torch_device = torch.device(str(runtime_info["resolved_device"]))
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    shuffle_rows: List[Dict[str, object]] = []
    model_location_rows: List[Dict[str, object]] = []
    fit_start = time.perf_counter()

    first_fold_train_dataset = ParticipantPredictionDataset(
        recordings=folds[0]["train_recordings"],
        index=folds[0]["train_entries"],
        window_spec=spec,
        x_mean=float(folds[0]["norm_stats"]["x_mean"]),
        x_std=float(folds[0]["norm_stats"]["x_std"]),
        y_mean=float(folds[0]["norm_stats"]["y_mean"]),
        y_std=float(folds[0]["norm_stats"]["y_std"]),
        target_channel=int(config["target_channel"]),
        feature_mode=feature_mode,
    )
    sample_x = first_fold_train_dataset[0]["x"]
    window_samples = int(round(folds[0]["holdout_recording"].fs_hz * int(config["win_ms"]) / 1000.0))
    probe_model = _build_scalar_model(config["model"], sample_x=sample_x, config=config)
    model_info = build_model_info(
        model_name=str(config["model"]),
        config=config,
        window_samples=window_samples,
        sample_x=sample_x,
        model=probe_model,
    )

    for fold in folds:
        fold_result = _run_single_prediction_fold(
            fold=fold,
            config=config,
            feature_mode=feature_mode,
            torch_device=torch_device,
            run_dir=run_dir,
        )
        fold_rows.append(fold_result["metrics_row"])
        prediction_rows.extend(fold_result["prediction_rows"])
        shuffle_rows.extend(fold_result["shuffle_rows"])
        model_location_rows.append(fold_result["model_location_row"])

    total_fit_seconds = time.perf_counter() - fit_start
    prediction_rows.sort(key=lambda row: (str(row["holdout_recording"]), float(row["time_s"])))
    fold_rows.sort(key=lambda row: str(row["holdout_recording"]))
    model_location_rows.sort(key=lambda row: str(row["holdout_recording"]))
    aggregate_predictions = _aggregate_prediction_rows(prediction_rows)
    aggregate_metrics = _aggregate_fold_metrics(fold_rows, aggregate_predictions)
    metrics = {
        "participant_id": int(participant_id),
        "recording_count": int(len(recordings)),
        "recording_names": [str(recording.recording_name) for recording in sorted(recordings, key=lambda item: str(item.recording_name))],
        "holdout_mode": "leave_one_recording_out",
        "fold_count": int(len(fold_rows)),
        "window_samples": int(window_samples),
        "delta_samples": int(round(folds[0]["holdout_recording"].fs_hz * int(config["delta_ms"]) / 1000.0)),
        "fit_seconds": float(total_fit_seconds),
        **aggregate_metrics,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    _write_csv(run_dir / "fold_metrics.csv", fold_rows)
    _write_csv(run_dir / "model_locations.csv", model_location_rows)
    _write_csv(run_dir / "shuffle_audit.csv", shuffle_rows)
    _write_csv(run_dir / "aggregate_predictions.csv", prediction_rows)
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2), encoding="utf-8")
    np.savez(
        run_dir / "aggregate_predictions.npz",
        holdout_recording=np.asarray(aggregate_predictions["holdout_recording"], dtype="U128"),
        recording_name=np.asarray(aggregate_predictions["recording_name"], dtype="U128"),
        window_start=aggregate_predictions["window_start"],
        time_s=aggregate_predictions["time_s"],
        y_true=aggregate_predictions["y_true"],
        y_pred=aggregate_predictions["y_pred"],
    )
    residual = aggregate_predictions["y_pred"] - aggregate_predictions["y_true"]
    _plot_holdout_r2_rmse(fold_rows, run_dir / "holdout_r2_rmse.png")
    _plot_holdout_pearson(fold_rows, run_dir / "holdout_pearson.png")
    _plot_holdout_window_count(fold_rows, run_dir / "holdout_window_count.png")
    _plot_scatter(
        aggregate_predictions["y_true"],
        aggregate_predictions["y_pred"],
        aggregate_metrics["weighted_all_holdouts"],
        run_dir / "aggregate_holdout_scatter.png",
    )
    _plot_residual_hist(residual, run_dir / "aggregate_residual_hist.png")
    _plot_residual_vs_true(aggregate_predictions["y_true"], residual, run_dir / "aggregate_residual_vs_true.png")
    _plot_holdout_timeseries_grid(prediction_rows, run_dir / "holdout_timeseries_grid.png")
    _plot_dynamic_vs_context_metrics(fold_rows, run_dir / "dynamic_vs_context_metrics.png")
    return {
        "config": config,
        "run_dir": run_dir,
        "runtime_info": runtime_info,
        "model_info": model_info,
        "metrics": metrics,
        "fold_rows": fold_rows,
        "prediction_rows": prediction_rows,
        "model_location_rows": model_location_rows,
    }


def _baseline_config(
    recordings_dir: str,
    device: Optional[str],
    max_epochs: int,
    min_epochs: int,
    patience: int,
) -> Dict[str, object]:
    return {
        "recordings_dir": str(recordings_dir),
        "model": "cnn1d_raw_scalar",
        "target_channel": 0,
        "win_ms": 300,
        "delta_ms": -150,
        "step_ms": 50,
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.15,
        "base_channels": 16,
        "hidden_dim": 128,
        "seed": 13,
        "device": device,
        "val_fraction": 0.2,
        "patience": int(patience),
        "min_epochs": int(min_epochs),
        "max_epochs": int(max_epochs),
    }


def _rescue_configs(recordings_dir: str, device: Optional[str]) -> List[Dict[str, object]]:
    common = {
        "recordings_dir": str(recordings_dir),
        "target_channel": 0,
        "step_ms": 50,
        "seed": 13,
        "device": device,
        "val_fraction": 0.2,
        "patience": 25,
        "min_epochs": 40,
        "weight_decay": 1e-5,
        "hidden_dim": 128,
    }
    return [
        {
            **common,
            "model": "cnn1d_raw_scalar",
            "win_ms": 300,
            "delta_ms": -100,
            "batch_size": 16,
            "lr": 3e-4,
            "dropout": 0.15,
            "base_channels": 16,
            "max_epochs": 250,
        },
        {
            **common,
            "model": "cnn1d_raw_scalar",
            "win_ms": 200,
            "delta_ms": -150,
            "batch_size": 16,
            "lr": 3e-4,
            "dropout": 0.15,
            "base_channels": 16,
            "max_epochs": 250,
        },
        {
            **common,
            "model": "cnn1d_raw_scalar",
            "win_ms": 400,
            "delta_ms": -150,
            "batch_size": 16,
            "lr": 3e-4,
            "dropout": 0.15,
            "base_channels": 16,
            "max_epochs": 250,
        },
        {
            **common,
            "model": "cnn1d_raw_scalar",
            "win_ms": 300,
            "delta_ms": -150,
            "batch_size": 16,
            "lr": 3e-4,
            "dropout": 0.10,
            "base_channels": 24,
            "max_epochs": 250,
        },
        {
            **common,
            "model": "cnn1d_feature_scalar",
            "win_ms": 300,
            "delta_ms": -150,
            "batch_size": 64,
            "lr": 1e-3,
            "dropout": 0.10,
            "base_channels": 32,
            "max_epochs": 200,
        },
    ]


def _copy_selected_artifacts(selected_run_dir: Path, top_level_dir: Path) -> None:
    for filename in [
        "fold_metrics.csv",
        "aggregate_predictions.csv",
        "aggregate_predictions.npz",
        "aggregate_metrics.json",
        "holdout_r2_rmse.png",
        "holdout_pearson.png",
        "holdout_window_count.png",
        "aggregate_holdout_scatter.png",
        "aggregate_residual_hist.png",
        "aggregate_residual_vs_true.png",
        "holdout_timeseries_grid.png",
        "dynamic_vs_context_metrics.png",
    ]:
        shutil.copy2(selected_run_dir / filename, top_level_dir / filename)


def _prefix_selected_model_paths(model_location_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    prefixed = []
    for row in model_location_rows:
        prefixed.append(
            {
                **row,
                "checkpoint_path": str(Path("selected_run") / str(row["checkpoint_path"])),
            }
        )
    return prefixed


def _write_comparison_summary(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Participant 4 Step 3 Rescue Sweep Comparison",
        "",
        "## Ranking Rule",
        "1. Highest median R2 across dynamic held-out recordings",
        "2. Highest count of dynamic held-outs with R2 > 0",
        "3. Highest mean Pearson across dynamic held-outs",
        "4. Lowest weighted RMSE across dynamic held-outs",
        "",
        "## Ranked Configurations",
        "| Rank | Config | Fixed first | Model | win_ms | delta_ms | Dynamic median R2 | Dynamic positive count | Dynamic mean Pearson | Weighted R2 | Weighted RMSE | Outcome |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['config_name']} | {row['fixed_first']} | {row['model']} | {row['win_ms']} | {row['delta_ms']} | "
            f"{row['dynamic_median_r2']:.6f} | {row['dynamic_positive_r2_count']} | {row['dynamic_mean_pearson']:.6f} | "
            f"{row['weighted_r2']:.6f} | {row['weighted_rmse']:.6f} | {row['outcome_label']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `leaderboard.csv`: flat table of all compared configs.",
            "- `leaderboard.json`: same table in JSON form.",
            "- `config_vs_fold_r2_heatmap.png`: fold-wise held-out R2 by config.",
            "- `config_vs_fold_rmse_heatmap.png`: fold-wise held-out RMSE by config.",
            "- `config_rank_summary.png`: dynamic-median-R2 and weighted-R2 summary chart.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_top_level_readme(
    run_dir: Path,
    selected_row: Dict[str, object],
    selected_metrics: Dict[str, object],
    selected_from_rescue: bool,
    baseline_row: Dict[str, object],
    comparison_used: bool,
) -> None:
    lines = [
        "# Participant 4 Step 3 Prediction Benchmark",
        "",
        "## Goal",
        "Test whether one model trained on participant 4 recordings can predict a fully held-out participant-4 recording.",
        "",
        "## Result",
        f"- Outcome label: `{selected_metrics['outcome_label']}`",
        f"- Selected config: `{selected_row['config_name']}`",
        f"- Weighted all-holdout R2: `{selected_metrics['weighted_all_holdouts']['r2']:.6f}`",
        f"- Weighted all-holdout RMSE: `{selected_metrics['weighted_all_holdouts']['rmse']:.6f}`",
        f"- Dynamic median held-out R2: `{selected_metrics['dynamic_holdouts']['median_r2']:.6f}`",
        f"- Dynamic positive-R2 folds: `{selected_metrics['dynamic_holdouts']['positive_r2_count']} / {selected_metrics['dynamic_holdouts']['count']}`",
        f"- Selected from rescue sweep: `{selected_from_rescue}`",
        "",
        "## Fixed-First Baseline",
        f"- Baseline config: `{baseline_row['config_name']}`",
        f"- Baseline weighted R2: `{baseline_row['weighted_r2']:.6f}`",
        f"- Baseline dynamic median R2: `{baseline_row['dynamic_median_r2']:.6f}`",
        "",
        "## Main Links",
        "- [Selected run README](selected_run/README.md)",
        "- [Selected run summary](selected_run/summary.md)",
    ]
    if comparison_used:
        lines.append("- [Comparison summary](comparison/summary.md)")
    lines.extend(
        [
            "",
            "## Main Charts",
            "![Held-out R2 and RMSE](holdout_r2_rmse.png)",
            "",
            "![Aggregate held-out scatter](aggregate_holdout_scatter.png)",
            "",
            "![Held-out time-series grid](holdout_timeseries_grid.png)",
            "",
            "## Where The Trained Models Are",
            "See `selected_model_locations.csv` for the exact fold-specific checkpoint locations.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Participant 4 Step 3 Summary",
                "",
                "Main presentation document: [README.md](README.md)",
                "",
                f"- Outcome label: `{selected_metrics['outcome_label']}`",
                f"- Selected config: `{selected_row['config_name']}`",
                f"- Weighted all-holdout R2: `{selected_metrics['weighted_all_holdouts']['r2']:.6f}`",
                f"- Dynamic median held-out R2: `{selected_metrics['dynamic_holdouts']['median_r2']:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _update_root_readme_with_step3(root_readme_path: Path, step3_run_dir: Path, selected_metrics: Dict[str, object]) -> None:
    content = root_readme_path.read_text(encoding="utf-8")
    marker = "## Next planned modeling steps"
    if marker not in content:
        raise RuntimeError("Could not find insertion marker in root README")
    step3_section = "\n".join(
        [
            "### Step 3 first real same-person prediction benchmark",
            "",
            "We then moved beyond same-window memorization and ran the first real same-person prediction benchmark on participant 4:",
            "",
            f"- [`{step3_run_dir.as_posix()}/README.md`](./{step3_run_dir.as_posix()}/README.md)",
            f"- [`{step3_run_dir.as_posix()}/selected_run/README.md`](./{step3_run_dir.as_posix()}/selected_run/README.md)",
            "",
            "Protocol:",
            "",
            "- outer loop: leave one full participant-4 recording out",
            "- inner validation: last 20% of windows from the remaining training recordings",
            "- held-out recording excluded from fitting, normalization, and checkpoint selection",
            "",
            "Main step-3 result:",
            "",
            f"- outcome label: `{selected_metrics['outcome_label']}`",
            f"- weighted all-holdout RMSE: `{selected_metrics['weighted_all_holdouts']['rmse']:.6f}`",
            f"- weighted all-holdout MAE: `{selected_metrics['weighted_all_holdouts']['mae']:.6f}`",
            f"- weighted all-holdout Pearson: `{selected_metrics['weighted_all_holdouts']['pearson']:.6f}`",
            f"- weighted all-holdout R2: `{selected_metrics['weighted_all_holdouts']['r2']:.6f}`",
            f"- dynamic median held-out R2: `{selected_metrics['dynamic_holdouts']['median_r2']:.6f}`",
            f"- dynamic positive-R2 folds: `{selected_metrics['dynamic_holdouts']['positive_r2_count']} / {selected_metrics['dynamic_holdouts']['count']}`",
            "",
            "Key step-3 charts:",
            "",
            f"![Participant 4 step-3 held-out R2 and RMSE](./{step3_run_dir.as_posix()}/holdout_r2_rmse.png)",
            "",
            f"![Participant 4 step-3 held-out scatter](./{step3_run_dir.as_posix()}/aggregate_holdout_scatter.png)",
            "",
        ]
    )
    updated = content.replace(marker, step3_section + "\n" + marker, 1)
    updated = updated.replace(
        "3. **Check predictive ability for one given person with one model.**\n4. **Check predictive ability when we train on multiple people, then expose the model to a few recordings of a new person, and see whether later recordings of that new person can be picked up.**",
        "4. **Check predictive ability when we train on multiple people, then expose the model to a few recordings of a new person, and see whether later recordings of that new person can be picked up.**",
    )
    root_readme_path.write_text(updated, encoding="utf-8")


def run_participant_prediction_stage(
    recordings_dir: str,
    run_dir: Optional[Path] = None,
    device: Optional[str] = None,
    require_venv: bool = False,
    require_cuda: bool = False,
    val_fraction: float = 0.2,
    patience: int = 25,
    min_epochs: int = 40,
    max_epochs: int = 250,
    update_root_readme: bool = True,
) -> Dict[str, object]:
    runtime_info = collect_runtime_info(requested_device=device)
    validate_runtime_requirements(runtime_info, require_venv=require_venv, require_cuda=require_cuda)
    records = _list_recordings(recordings_dir)
    participant_id = _participant_id_from_recordings(records)
    if participant_id != 4:
        raise ValueError(f"This workflow is currently documented for participant 4 only, got participant {participant_id}")
    stage_run_dir = run_dir or (_repo_root() / "runs" / f"p4_step3_prediction_leave_one_out_{_utc_timestamp_slug()}")
    stage_run_dir.mkdir(parents=True, exist_ok=True)

    stage_config = {
        "recordings_dir": str(recordings_dir),
        "holdout_mode": "leave_one_recording_out",
        "participant_id": int(participant_id),
        "val_fraction": float(val_fraction),
        "patience": int(patience),
        "min_epochs": int(min_epochs),
        "max_epochs": int(max_epochs),
        "require_venv": bool(require_venv),
        "require_cuda": bool(require_cuda),
        "device": device,
    }
    (stage_run_dir / "config.json").write_text(json.dumps(stage_config, indent=2), encoding="utf-8")

    baseline_config = _baseline_config(
        recordings_dir=recordings_dir,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
    )
    baseline_config["val_fraction"] = float(val_fraction)

    comparison_runs_dir = stage_run_dir / "comparison" / "runs"
    baseline_run_dir = comparison_runs_dir / _prediction_config_slug(baseline_config)
    baseline_result = _run_prediction_config(
        recordings_dir=recordings_dir,
        config=baseline_config,
        run_dir=baseline_run_dir,
        runtime_info=runtime_info,
    )
    baseline_row = summarize_prediction_run_row(
        config=baseline_result["config"],
        aggregate_metrics=baseline_result["metrics"],
        fold_rows=baseline_result["fold_rows"],
        run_dir=baseline_run_dir,
        fixed_first=True,
    )
    config_results = [baseline_result]
    config_rows = [baseline_row]

    if str(baseline_result["metrics"]["outcome_label"]) != "works":
        for rescue_config in _rescue_configs(recordings_dir=recordings_dir, device=device):
            rescue_config["val_fraction"] = float(val_fraction)
            rescue_config["patience"] = int(patience)
            rescue_config["min_epochs"] = int(min_epochs)
            rescue_run_dir = comparison_runs_dir / _prediction_config_slug(rescue_config)
            result = _run_prediction_config(
                recordings_dir=recordings_dir,
                config=rescue_config,
                run_dir=rescue_run_dir,
                runtime_info=runtime_info,
            )
            config_results.append(result)
            config_rows.append(
                summarize_prediction_run_row(
                    config=result["config"],
                    aggregate_metrics=result["metrics"],
                    fold_rows=result["fold_rows"],
                    run_dir=rescue_run_dir,
                    fixed_first=False,
                )
            )

    ranked_rows = rank_prediction_run_rows(config_rows)
    result_by_name = {str(_prediction_config_slug(result["config"])): result for result in config_results}
    selected_row = ranked_rows[0]
    selected_result = result_by_name[str(selected_row["config_name"])]
    selected_from_rescue = not bool(selected_row["fixed_first"])

    comparison_used = len(config_rows) > 1
    if comparison_used:
        comparison_dir = stage_run_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(comparison_dir / "leaderboard.csv", ranked_rows)
        (comparison_dir / "leaderboard.json").write_text(json.dumps(ranked_rows, indent=2), encoding="utf-8")
        per_config_fold_rows = {str(_prediction_config_slug(result["config"])): result["fold_rows"] for result in config_results}
        _plot_config_metric_heatmap(
            ranked_rows,
            per_config_fold_rows=per_config_fold_rows,
            metric_key="r2",
            title="Config vs held-out fold R2",
            path=comparison_dir / "config_vs_fold_r2_heatmap.png",
        )
        _plot_config_metric_heatmap(
            ranked_rows,
            per_config_fold_rows=per_config_fold_rows,
            metric_key="rmse",
            title="Config vs held-out fold RMSE",
            path=comparison_dir / "config_vs_fold_rmse_heatmap.png",
        )
        _plot_config_rank_summary(ranked_rows, comparison_dir / "config_rank_summary.png")
        _write_comparison_summary(comparison_dir, ranked_rows)

    selected_run_dir = stage_run_dir / "selected_run"
    if selected_run_dir.exists():
        shutil.rmtree(selected_run_dir)
    shutil.copytree(selected_result["run_dir"], selected_run_dir)
    _copy_selected_artifacts(selected_run_dir, stage_run_dir)

    selected_model_locations = _prefix_selected_model_paths(selected_result["model_location_rows"])
    _write_csv(stage_run_dir / "selected_model_locations.csv", selected_model_locations)
    (stage_run_dir / "selected_config.json").write_text(json.dumps(selected_result["config"], indent=2), encoding="utf-8")
    (stage_run_dir / "runtime.json").write_text(
        json.dumps(
            {
                **runtime_info,
                "stage_generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_config_name": str(selected_row["config_name"]),
                "selected_from_rescue": bool(selected_from_rescue),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_prediction_run_summary(
        selected_run_dir,
        selected_result["config"],
        selected_result["metrics"],
        selected_result["fold_rows"],
        has_comparison=comparison_used,
    )
    _write_prediction_run_readme(
        selected_run_dir,
        selected_result["config"],
        runtime_info,
        selected_result["model_info"],
        selected_result["metrics"],
        selected_result["fold_rows"],
        selected_result["model_location_rows"],
        has_comparison=comparison_used,
        fixed_first_metrics=baseline_result["metrics"],
        selected_from_rescue=selected_from_rescue,
    )
    _write_top_level_readme(
        stage_run_dir,
        selected_row=selected_row,
        selected_metrics=selected_result["metrics"],
        selected_from_rescue=selected_from_rescue,
        baseline_row=baseline_row,
        comparison_used=comparison_used,
    )

    if update_root_readme:
        _update_root_readme_with_step3(_repo_root() / "README.md", stage_run_dir.relative_to(_repo_root()), selected_result["metrics"])

    return {
        "run_dir": str(stage_run_dir),
        "selected_config": selected_result["config"],
        "metrics": selected_result["metrics"],
        "comparison_used": comparison_used,
        "selected_from_rescue": selected_from_rescue,
    }
