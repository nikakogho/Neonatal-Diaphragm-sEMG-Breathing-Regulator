"""
Participant-level pooled overfit workflow for scalar AUX[0] experiments.
"""

from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import WindowSpec, build_window_index, load_recording
from src.features import extract_binned_features
from src.training.single_recording import (
    _build_scalar_model,
    _load_selection_context_rows,
    _plot_loss_curve,
    _plot_residual_hist,
    _plot_scatter,
    _write_csv,
    build_model_info,
    collect_runtime_info,
    compute_scalar_metrics,
    set_seed,
    validate_runtime_requirements,
)


def _normalize_emg_window(window: np.ndarray, good_mask: np.ndarray, x_mean: float, x_std: float) -> np.ndarray:
    masked = window * good_mask[None, :, :, :]
    normalized = (masked - x_mean) / x_std
    return normalized * good_mask[None, :, :, :]


class ParticipantScalarWindowDataset(Dataset):
    """Windowed scalar target dataset pooled across multiple recordings."""

    def __init__(
        self,
        recordings_dir: str,
        window_spec: WindowSpec,
        target_channel: int = 0,
        feature_mode: str = "raw",
        bin_ms: int = 25,
    ):
        root = Path(recordings_dir)
        self.recordings = [load_recording(str(path)) for path in sorted(root.glob("*.npz"))]
        if not self.recordings:
            raise FileNotFoundError(f"No .npz recordings found in {recordings_dir}")
        self.recordings_by_path = {recording.path: recording for recording in self.recordings}
        self.window_spec = window_spec
        self.target_channel = int(target_channel)
        self.feature_mode = feature_mode
        self.bin_ms = int(bin_ms)
        self.index = build_window_index(self.recordings, self.window_spec)
        if self.target_channel < 0 or self.target_channel >= self.recordings[0].aux.shape[1]:
            raise ValueError(f"Invalid target_channel {target_channel}")
        self.x_mean, self.x_std, self.y_mean, self.y_std = self._compute_normalization()
        self.recording_aux_std = {
            recording.recording_name: float(recording.aux[:, self.target_channel].std()) for recording in self.recordings
        }

    def _compute_normalization(self):
        x_sum = 0.0
        x_sq = 0.0
        x_n = 0
        y_sum = 0.0
        y_sq = 0.0
        y_n = 0
        for recording in self.recordings:
            mask = recording.good_mask.astype(bool)
            masked = recording.emg[:, mask]
            x_sum += float(masked.sum())
            x_sq += float(np.square(masked).sum())
            x_n += int(masked.size)
            y = recording.aux[:, self.target_channel].astype(np.float32)
            y_sum += float(y.sum())
            y_sq += float(np.square(y).sum())
            y_n += int(y.shape[0])
        x_mean = x_sum / x_n
        x_std = float(np.sqrt(max(1e-12, x_sq / x_n - x_mean * x_mean)))
        y_mean = y_sum / y_n
        y_std = float(np.sqrt(max(1e-12, y_sq / y_n - y_mean * y_mean)))
        return x_mean, x_std, y_mean, y_std

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
            "recording_name": recording.recording_name,
            "recording_path": recording.path,
            "window_start": start,
            "time_s": float(time_s),
        }


def _collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "y_phys": torch.stack([item["y_phys"] for item in batch], dim=0),
        "recording_name": [str(item["recording_name"]) for item in batch],
        "window_start": np.asarray([int(item["window_start"]) for item in batch], dtype=np.int32),
        "time_s": np.asarray([float(item["time_s"]) for item in batch], dtype=np.float32),
    }


def _make_eval_loader(dataset: Dataset, indices: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)


def make_epoch_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    generator = np.random.default_rng(seed + epoch - 1)
    perm = generator.permutation(indices.shape[0])
    return indices[perm]


def build_shuffle_audit_rows(
    dataset: ParticipantScalarWindowDataset,
    epoch_order: np.ndarray,
    batch_size: int,
    max_batches: int = 50,
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
                "epoch": 1,
                "batch_idx": batch_idx,
                "batch_size": int(len(batch_indices)),
                "unique_recordings": int(len(counts)),
                "recordings_seen": ", ".join(sorted(counts)),
                "recording_mix_json": json.dumps(counts, sort_keys=True),
            }
        )
    return rows


def _make_epoch_train_loader(
    dataset: ParticipantScalarWindowDataset,
    epoch_order: np.ndarray,
    batch_size: int,
) -> DataLoader:
    return DataLoader(Subset(dataset, epoch_order.tolist()), batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)


def _predict_scalar_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: ParticipantScalarWindowDataset,
) -> Dict[str, np.ndarray]:
    pred_std_rows: List[np.ndarray] = []
    y_std_rows: List[np.ndarray] = []
    y_phys_rows: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    times: List[np.ndarray] = []
    names: List[str] = []
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
    pred_std = np.concatenate(pred_std_rows, axis=0)
    y_std = np.concatenate(y_std_rows, axis=0)
    y_phys = np.concatenate(y_phys_rows, axis=0)
    pred_phys = pred_std * dataset.y_std + dataset.y_mean
    return {
        "pred_std": pred_std,
        "y_std": y_std,
        "pred_phys": pred_phys,
        "y_phys": y_phys,
        "recording_name": np.asarray(names, dtype=object),
        "window_start": np.concatenate(starts, axis=0),
        "time_s": np.concatenate(times, axis=0),
    }


def _compute_per_recording_metrics(
    predictions: Dict[str, np.ndarray],
    aux_std_by_recording: Dict[str, float],
    dynamic_std_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    names = np.asarray(predictions["recording_name"], dtype=object)
    for recording_name in sorted({str(name) for name in names.tolist()}):
        mask = names == recording_name
        metrics = compute_scalar_metrics(predictions["y_phys"][mask], predictions["pred_phys"][mask])
        rows.append(
            {
                "recording_name": recording_name,
                "n_windows": int(mask.sum()),
                "aux0_std": float(aux_std_by_recording[recording_name]),
                "is_dynamic_recording": bool(aux_std_by_recording[recording_name] >= dynamic_std_threshold),
                "mse": float(metrics["mse"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "pearson": float(metrics["pearson"]),
                "target_min": float(metrics["target_min"]),
                "target_max": float(metrics["target_max"]),
            }
        )
    return rows


def _plot_per_recording_r2_rmse(per_recording_metrics: Sequence[Dict[str, object]], path: Path) -> None:
    names = [str(row["recording_name"]) for row in per_recording_metrics]
    r2_values = [float(row["r2"]) for row in per_recording_metrics]
    rmse_values = [float(row["rmse"]) for row in per_recording_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(names, r2_values)
    axes[0].set_title("Per-recording R2")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, rmse_values)
    axes[1].set_title("Per-recording RMSE")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_recording_window_counts(per_recording_metrics: Sequence[Dict[str, object]], path: Path) -> None:
    names = [str(row["recording_name"]) for row in per_recording_metrics]
    counts = [int(row["n_windows"]) for row in per_recording_metrics]
    plt.figure(figsize=(10, 4.5))
    plt.bar(names, counts)
    plt.title("Per-recording window counts")
    plt.ylabel("Window count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_per_recording_timeseries(predictions: Dict[str, np.ndarray], path: Path) -> None:
    names = np.asarray(predictions["recording_name"], dtype=object)
    unique_names = sorted({str(name) for name in names.tolist()})
    n_cols = 3
    n_rows = int(math.ceil(len(unique_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, recording_name in zip(axes.flatten(), unique_names):
        mask = names == recording_name
        order = np.argsort(predictions["time_s"][mask])
        x_time = predictions["time_s"][mask][order]
        y_true = predictions["y_phys"][mask][order]
        y_pred = predictions["pred_phys"][mask][order]
        ax.axis("on")
        ax.plot(x_time, y_true, label="true", linewidth=1.4)
        ax.plot(x_time, y_pred, label="pred", linewidth=1.0)
        ax.set_title(recording_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AUX[0]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Per-recording predicted vs true over time", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _participant_artifact_guide_lines(has_checkpoints: bool, has_selection_context: bool) -> List[str]:
    lines = [
        "- `config.json`: exact run configuration and hyperparameters.",
        "- `metrics.json`: pooled metrics, per-recording threshold summary, window counts, and timing.",
        "- `runtime.json`: Python, torch, venv, and GPU runtime information for this run.",
        "- `model_info.json`: exact architecture description and parameter counts.",
        "- `history.csv` / `history.json`: per-epoch training loss.",
        "- `predictions.csv` / `predictions.npz`: pooled per-window predictions across all participant-4 recordings.",
        "- `per_recording_metrics.csv`: one row per recording with RMSE, MAE, R2, Pearson, and variance context.",
        "- `shuffle_audit.csv`: evidence that epoch-1 batches mixed windows from multiple recordings.",
        "- `README.md`: polished presentation document for this run.",
        "- `summary.md`: concise technical companion.",
        "- `loss_curve.png`: pooled training loss over epochs.",
        "- `pred_vs_true_scatter.png`: pooled predicted-vs-true scatter.",
        "- `per_recording_r2_rmse.png`: per-recording R2 and RMSE summary chart.",
        "- `per_recording_window_count.png`: number of windows contributed by each recording.",
        "- `per_recording_timeseries.png`: small-multiple time-series view for every recording.",
        "- `residual_hist.png`: pooled residual distribution.",
    ]
    if has_checkpoints:
        lines.extend(
            [
                "- `model_best.pt`: best checkpoint by epoch-average training loss.",
                "- `model_last.pt`: last checkpoint after the final epoch.",
            ]
        )
    if has_selection_context:
        lines.append("- `selection_context.csv`: supporting step-1 participant-4 comparison used to justify the starting configuration.")
    return lines


def write_participant_overfit_summary(
    run_dir: Path,
    config: Dict[str, object],
    metrics: Dict[str, object],
    has_checkpoints: bool,
    has_selection_context: bool,
) -> None:
    lines = [
        "# Participant-Level Overfit Summary",
        "",
        "Main presentation document: [README.md](README.md)",
        "",
        "## Run Overview",
        f"- Recordings directory: `{config['recordings_dir']}`",
        f"- Recording count: `{metrics['recording_count']}`",
        f"- Model: `{config['model']}`",
        f"- Window / delta / step (ms): `{config['win_ms']}` / `{config['delta_ms']}` / `{config['step_ms']}`",
        f"- Train windows: `{metrics['train_window_count']}`",
        f"- Device: `{metrics['device']}`",
        f"- Training seconds: `{metrics['fit_seconds']:.3f}`",
        "",
        "## Main Result",
        f"- Pooled RMSE: `{metrics['eval_metrics']['rmse']:.6f}`",
        f"- Pooled MAE: `{metrics['eval_metrics']['mae']:.6f}`",
        f"- Pooled R2: `{metrics['eval_metrics']['r2']:.6f}`",
        f"- Pooled Pearson: `{metrics['eval_metrics']['pearson']:.6f}`",
        "",
        "## Dynamic Recording Check",
        f"- Dynamic threshold std: `{metrics['dynamic_std_threshold']:.3f}`",
        f"- Dynamic recordings all meet R2 >= 0.98: `{metrics['dynamic_recordings_meet_r2_threshold']}`",
        "",
        "## Artifact Guide",
        *_participant_artifact_guide_lines(has_checkpoints, has_selection_context),
        "",
        "## Main Charts",
        "![Loss curve](loss_curve.png)",
        "",
        "![Pooled predicted vs true](pred_vs_true_scatter.png)",
        "",
        "![Per-recording R2 and RMSE](per_recording_r2_rmse.png)",
        "",
        "![Per-recording window counts](per_recording_window_count.png)",
        "",
        "![Per-recording time series](per_recording_timeseries.png)",
        "",
        "![Residual histogram](residual_hist.png)",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_participant_overfit_readme(
    run_dir: Path,
    config: Dict[str, object],
    metrics: Dict[str, object],
    runtime_info: Dict[str, object],
    model_info: Dict[str, object],
    per_recording_metrics: Sequence[Dict[str, object]],
    shuffle_rows: Sequence[Dict[str, object]],
    selection_context_rows: Sequence[Dict[str, object]],
    has_checkpoints: bool,
) -> None:
    mean_unique = float(np.mean([int(row["unique_recordings"]) for row in shuffle_rows])) if shuffle_rows else 0.0
    lines = [
        "# Participant 4 Step 2 Overfit Demo",
        "",
        "## Goal",
        "Show that one scalar model can overfit all cleaned recordings from participant 4 at the same time.",
        "",
        "This is still an overfit-stage experiment: the same pooled set of windows is used for training and evaluation. The important difference from step 1 is that the model now sees windows from every participant-4 recording in one shared training run.",
        "",
        "## Run At A Glance",
        f"- Recordings directory: `{config['recordings_dir']}`",
        f"- Recording count: `{metrics['recording_count']}`",
        f"- Total pooled windows: `{metrics['train_window_count']}`",
        f"- Target: mean `AUX[{config['target_channel']}]` over the shifted target window",
        f"- Window / delta / step (ms): `{config['win_ms']}` / `{config['delta_ms']}` / `{config['step_ms']}`",
        f"- Window / delta (samples): `{metrics['window_samples']}` / `{metrics['delta_samples']}`",
        f"- Model: `{model_info['display_name']}`",
        f"- Device used: `{runtime_info['resolved_device']}`",
        f"- GPU: `{runtime_info.get('device_name', 'n/a')}`",
        f"- Python executable: `{runtime_info['python_executable']}`",
        f"- Running inside venv: `{runtime_info['in_venv']}`",
        "",
        "## Main Outcome",
        f"This pooled run overfits participant 4 strongly: pooled RMSE `{metrics['eval_metrics']['rmse']:.6f}`, "
        f"MAE `{metrics['eval_metrics']['mae']:.6f}`, Pearson `{metrics['eval_metrics']['pearson']:.6f}`, "
        f"and R2 `{metrics['eval_metrics']['r2']:.6f}` on all participant-4 windows together.",
        "",
        "| Metric | Train | Eval |",
        "| --- | ---: | ---: |",
        f"| MSE | {metrics['train_metrics']['mse']:.6f} | {metrics['eval_metrics']['mse']:.6f} |",
        f"| RMSE | {metrics['train_metrics']['rmse']:.6f} | {metrics['eval_metrics']['rmse']:.6f} |",
        f"| MAE | {metrics['train_metrics']['mae']:.6f} | {metrics['eval_metrics']['mae']:.6f} |",
        f"| Pearson | {metrics['train_metrics']['pearson']:.6f} | {metrics['eval_metrics']['pearson']:.6f} |",
        f"| R2 | {metrics['train_metrics']['r2']:.6f} | {metrics['eval_metrics']['r2']:.6f} |",
        "",
        "## Recording Coverage",
        f"- Dynamic-recording threshold: `aux0_std >= {metrics['dynamic_std_threshold']:.3f}`",
        f"- Dynamic recordings all meet `R2 >= 0.98`: `{metrics['dynamic_recordings_meet_r2_threshold']}`",
        f"- Dynamic recording names: `{', '.join(metrics['dynamic_recordings'])}`",
        "",
        "| Recording | Windows | AUX0 std | Dynamic | R2 | RMSE | MAE | Pearson |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in per_recording_metrics:
        lines.append(
            f"| {row['recording_name']} | {row['n_windows']} | {float(row['aux0_std']):.6f} | "
            f"{bool(row['is_dynamic_recording'])} | {float(row['r2']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['mae']):.6f} | {float(row['pearson']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Shuffle Evidence",
            "Training windows were not consumed one recording at a time. Each epoch was given a fresh global shuffle over the pooled window index.",
            f"- Audit rows saved: `{len(shuffle_rows)}`",
            f"- Mean unique recordings per audited batch: `{mean_unique:.3f}`",
            "",
            "| Batch idx | Batch size | Unique recordings | Recordings seen |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for row in shuffle_rows[:10]:
        lines.append(
            f"| {row['batch_idx']} | {row['batch_size']} | {row['unique_recordings']} | {row['recordings_seen']} |"
        )
    lines.extend(
        [
            "",
            "## Exact Data and Label Definition",
            "- One pooled dataset is built from all `.npz` recordings in the participant-4 directory.",
            f"- Input representation: `{model_info['feature_mode']}`",
            f"- Each window contributes one scalar label: the mean of `AUX[{config['target_channel']}]` over the shifted target window.",
            "- Windows from all recordings share one global normalization and one shared model.",
            "",
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
            f"- Epochs: `{config['epochs']}`",
            f"- Batch size: `{config['batch_size']}`",
            f"- Learning rate: `{config['lr']}`",
            f"- Weight decay: `{config['weight_decay']}`",
            f"- Dropout: `{config['dropout']}`",
            f"- Base channels: `{config['base_channels']}`",
            f"- Seed: `{config['seed']}`",
            "",
            "## Runtime Environment",
            f"- Torch version: `{runtime_info['torch_version']}`",
            f"- CUDA available: `{runtime_info['cuda_available']}`",
            f"- Device count: `{runtime_info['device_count']}`",
            f"- Resolved device: `{runtime_info['resolved_device']}`",
            f"- GPU capability: `{runtime_info.get('device_capability', 'n/a')}`",
            f"- Generated at UTC: `{runtime_info['generated_at_utc']}`",
            "",
        ]
    )
    if selection_context_rows:
        lines.extend(
            [
                "## Why This Starting Configuration",
                "Step 2 starts from the step-1 winner, because that timing / architecture pair was already best across the single-recording participant-4 experiments.",
                "",
                "| Recording | Model | win_ms | delta_ms | Eval R2 | Eval RMSE |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selection_context_rows:
            lines.append(
                f"| {row['recording']} | {row['model']} | {row['win_ms']} | {row['delta_ms']} | "
                f"{float(row['eval_r2']):.6f} | {float(row['eval_rmse']):.6f} |"
            )
        lines.extend(
            [
                "",
                "- The raw 1D CNN with `300 ms / -150 ms` won on `p4rec3`, `p4rec5`, and `p4rec6`.",
                "- That made it the correct first candidate for pooled participant-level overfit.",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact Guide",
            *_participant_artifact_guide_lines(has_checkpoints, bool(selection_context_rows)),
            "",
            "## Plots",
            "### Loss Curve",
            "![Loss curve](loss_curve.png)",
            "",
            "### Pooled Predicted vs True Scatter",
            "![Pooled predicted vs true](pred_vs_true_scatter.png)",
            "",
            "### Per-recording R2 and RMSE",
            "![Per-recording R2 and RMSE](per_recording_r2_rmse.png)",
            "",
            "### Per-recording Window Counts",
            "![Per-recording window counts](per_recording_window_count.png)",
            "",
            "### Per-recording Time Series",
            "![Per-recording time series](per_recording_timeseries.png)",
            "",
            "### Residual Histogram",
            "![Residual histogram](residual_hist.png)",
            "",
            "## Interpretation",
            "- The pooled same-window result answers step 2 only: can one model memorize all recordings of one participant at the same time?",
            "- It does not yet answer whether the model predicts unseen windows or unseen recordings from that participant.",
            "- The next stage after this is participant-level predictive testing, not more same-window memorization.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_participant_artifacts(
    run_dir: Path,
    config: Dict[str, object],
    history: Sequence[Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, object],
    runtime_info: Dict[str, object],
    model_info: Dict[str, object],
    per_recording_metrics: Sequence[Dict[str, object]],
    shuffle_rows: Sequence[Dict[str, object]],
    selection_context_rows: Sequence[Dict[str, object]],
    best_state: Optional[Dict[str, torch.Tensor]],
    last_state: Optional[Dict[str, torch.Tensor]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps(runtime_info, indent=2), encoding="utf-8")
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(list(history), indent=2), encoding="utf-8")
    _write_csv(run_dir / "history.csv", list(history))
    _write_csv(run_dir / "per_recording_metrics.csv", list(per_recording_metrics))
    _write_csv(run_dir / "shuffle_audit.csv", list(shuffle_rows))
    if selection_context_rows:
        _write_csv(run_dir / "selection_context.csv", list(selection_context_rows))
    prediction_rows = [
        {
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
    _write_csv(run_dir / "predictions.csv", prediction_rows)
    np.savez(
        run_dir / "predictions.npz",
        recording_name=np.asarray(predictions["recording_name"], dtype="U128"),
        window_start=predictions["window_start"],
        time_s=predictions["time_s"],
        y_true=predictions["y_phys"],
        y_pred=predictions["pred_phys"],
        y_std=predictions["y_std"],
        pred_std=predictions["pred_std"],
    )
    if best_state is not None:
        torch.save(best_state, run_dir / "model_best.pt")
    if last_state is not None:
        torch.save(last_state, run_dir / "model_last.pt")
    residual = predictions["pred_phys"] - predictions["y_phys"]
    _plot_loss_curve(history, run_dir / "loss_curve.png")
    _plot_scatter(predictions["y_phys"], predictions["pred_phys"], metrics["eval_metrics"], run_dir / "pred_vs_true_scatter.png")
    _plot_residual_hist(residual, run_dir / "residual_hist.png")
    _plot_per_recording_r2_rmse(per_recording_metrics, run_dir / "per_recording_r2_rmse.png")
    _plot_per_recording_window_counts(per_recording_metrics, run_dir / "per_recording_window_count.png")
    _plot_per_recording_timeseries(predictions, run_dir / "per_recording_timeseries.png")
    has_checkpoints = best_state is not None
    has_selection_context = bool(selection_context_rows)
    write_participant_overfit_summary(
        run_dir,
        config,
        metrics,
        has_checkpoints=has_checkpoints,
        has_selection_context=has_selection_context,
    )
    write_participant_overfit_readme(
        run_dir,
        config,
        metrics,
        runtime_info,
        model_info,
        per_recording_metrics,
        shuffle_rows,
        selection_context_rows,
        has_checkpoints=has_checkpoints,
    )


def run_participant_overfit_experiment(
    recordings_dir: str,
    model_name: str,
    win_ms: int,
    delta_ms: int,
    step_ms: int = 50,
    target_channel: int = 0,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    dropout: float = 0.15,
    base_channels: int = 16,
    hidden_dim: int = 128,
    seed: int = 13,
    device: Optional[str] = None,
    run_dir: Optional[Path] = None,
    dynamic_std_threshold: float = 0.5,
    dynamic_r2_threshold: float = 0.98,
) -> Dict[str, object]:
    set_seed(seed)
    runtime_info = collect_runtime_info(requested_device=device)
    validate_runtime_requirements(runtime_info, require_cuda=False)
    torch_device = torch.device(str(runtime_info["resolved_device"]))
    window_spec = WindowSpec(win_ms=int(win_ms), step_ms=int(step_ms), delta_ms=int(delta_ms))
    feature_mode = "raw" if model_name == "cnn1d_raw_scalar" else "binned"
    dataset = ParticipantScalarWindowDataset(
        recordings_dir=recordings_dir,
        window_spec=window_spec,
        target_channel=target_channel,
        feature_mode=feature_mode,
    )
    selection_context_rows = (
        _load_selection_context_rows()
        if all(int(recording.pid) == 4 for recording in dataset.recordings)
        and model_name in {"cnn1d_raw_scalar", "cnn1d_feature_scalar"}
        else []
    )
    all_indices = np.arange(len(dataset), dtype=np.int32)
    eval_loader = _make_eval_loader(dataset, all_indices, batch_size=int(batch_size))
    first_epoch_order = make_epoch_order(all_indices, seed=int(seed), epoch=1)
    shuffle_rows = build_shuffle_audit_rows(dataset, first_epoch_order, batch_size=int(batch_size))
    window_samples = int(round(dataset.recordings[0].fs_hz * win_ms / 1000.0))
    config = {
        "recordings_dir": str(recordings_dir),
        "model": model_name,
        "win_ms": int(win_ms),
        "delta_ms": int(delta_ms),
        "step_ms": int(step_ms),
        "target_channel": int(target_channel),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "base_channels": int(base_channels),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "device": str(torch_device),
    }
    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_state: Optional[Dict[str, torch.Tensor]] = None
    fit_start = time.perf_counter()

    if model_name == "constant_mean_scalar":
        model_info = build_model_info(model_name, config, window_samples=window_samples)
        y_true = np.asarray([float(dataset[idx]["y_phys"].numpy()[0]) for idx in all_indices], dtype=np.float32)
        pred_phys = np.full_like(y_true, float(y_true.mean()))
        pred_std = ((pred_phys - dataset.y_mean) / dataset.y_std).astype(np.float32)
        fit_seconds = time.perf_counter() - fit_start
        train_loss = float(np.mean(np.square(pred_std - np.asarray([float(dataset[idx]["y"].numpy()[0]) for idx in all_indices], dtype=np.float32))))
        history.append({"epoch": 1, "train_loss": train_loss, "eval_loss": train_loss})
        predictions = {
            "pred_std": pred_std,
            "y_std": np.asarray([float(dataset[idx]["y"].numpy()[0]) for idx in all_indices], dtype=np.float32),
            "pred_phys": pred_phys,
            "y_phys": y_true,
            "recording_name": np.asarray([str(dataset[idx]["recording_name"]) for idx in all_indices], dtype=object),
            "window_start": np.asarray([int(dataset[idx]["window_start"]) for idx in all_indices], dtype=np.int32),
            "time_s": np.asarray([float(dataset[idx]["time_s"]) for idx in all_indices], dtype=np.float32),
        }
    else:
        sample_x = dataset[0]["x"]
        model = _build_scalar_model(model_name, sample_x=sample_x, config=config).to(torch_device)
        model_info = build_model_info(
            model_name=model_name,
            config=config,
            window_samples=window_samples,
            sample_x=sample_x,
            model=model,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = nn.MSELoss()
        best_score = float("inf")

        for epoch in range(1, int(epochs) + 1):
            epoch_order = make_epoch_order(all_indices, seed=int(seed), epoch=epoch)
            train_loader = _make_epoch_train_loader(dataset, epoch_order, batch_size=int(batch_size))
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

            epoch_train_loss = total_loss / max(total_count, 1)
            history.append({"epoch": epoch, "train_loss": float(epoch_train_loss), "eval_loss": float(epoch_train_loss)})
            if epoch_train_loss < best_score:
                best_score = float(epoch_train_loss)
                best_state = deepcopy(model.state_dict())
            last_state = deepcopy(model.state_dict())

        fit_seconds = time.perf_counter() - fit_start
        if best_state is None:
            raise RuntimeError("Training completed without producing a best checkpoint")
        model.load_state_dict(best_state)
        predictions = _predict_scalar_model(model, eval_loader, torch_device, dataset)

    train_metrics = compute_scalar_metrics(predictions["y_phys"], predictions["pred_phys"])
    eval_metrics = compute_scalar_metrics(predictions["y_phys"], predictions["pred_phys"])
    per_recording_metrics = _compute_per_recording_metrics(
        predictions=predictions,
        aux_std_by_recording=dataset.recording_aux_std,
        dynamic_std_threshold=float(dynamic_std_threshold),
    )
    dynamic_rows = [row for row in per_recording_metrics if bool(row["is_dynamic_recording"])]
    metrics = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "fit_seconds": float(fit_seconds),
        "device": str(torch_device),
        "train_window_count": int(all_indices.shape[0]),
        "eval_window_count": int(all_indices.shape[0]),
        "recording_count": int(len(dataset.recordings)),
        "recording_names": [recording.recording_name for recording in dataset.recordings],
        "window_samples": int(window_samples),
        "delta_samples": int(round(dataset.recordings[0].fs_hz * delta_ms / 1000.0)),
        "dynamic_std_threshold": float(dynamic_std_threshold),
        "dynamic_r2_threshold": float(dynamic_r2_threshold),
        "dynamic_recordings": [str(row["recording_name"]) for row in dynamic_rows],
        "dynamic_recordings_meet_r2_threshold": bool(
            dynamic_rows and all(float(row["r2"]) >= float(dynamic_r2_threshold) for row in dynamic_rows)
        ),
        "dynamic_recording_min_r2": float(min((float(row["r2"]) for row in dynamic_rows), default=0.0)),
        "mean_unique_recordings_per_audited_batch": float(
            np.mean([int(row["unique_recordings"]) for row in shuffle_rows]) if shuffle_rows else 0.0
        ),
    }
    if run_dir is not None:
        _save_participant_artifacts(
            run_dir=run_dir,
            config=config,
            history=history,
            predictions=predictions,
            metrics=metrics,
            runtime_info=runtime_info,
            model_info=model_info,
            per_recording_metrics=per_recording_metrics,
            shuffle_rows=shuffle_rows,
            selection_context_rows=selection_context_rows,
            best_state=best_state,
            last_state=last_state,
        )
    return {
        "config": config,
        "history": history,
        "metrics": metrics,
        "runtime_info": runtime_info,
        "model_info": model_info,
        "predictions": predictions,
        "per_recording_metrics": per_recording_metrics,
        "shuffle_rows": shuffle_rows,
    }
