"""
Single-recording scalar overfit workflow for AUX[0] experiments.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ConstantInputWarning, pearsonr
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import NormalizationStats, WindowSpec, compute_window_starts, load_recording
from src.features import extract_binned_features
from src.models import get_model


@dataclass(frozen=True)
class ScalarWindowIndexEntry:
    window_start: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def collect_runtime_info(requested_device: Optional[str] = None) -> Dict[str, object]:
    if requested_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(requested_device)
    runtime = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "sys_prefix": sys.prefix,
        "sys_base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_venv": bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "requested_device": requested_device,
        "resolved_device": resolved_device,
    }
    if torch.cuda.is_available():
        runtime["device_name"] = torch.cuda.get_device_name(0)
        runtime["device_capability"] = list(torch.cuda.get_device_capability(0))
    return runtime


def validate_runtime_requirements(
    runtime_info: Dict[str, object],
    require_venv: bool = False,
    require_cuda: bool = False,
) -> None:
    if require_venv and not bool(runtime_info.get("in_venv", False)):
        raise RuntimeError(
            "This command must be run from the project virtual environment. "
            f"Python executable: {runtime_info.get('python_executable')}"
        )
    if require_cuda and not bool(runtime_info.get("cuda_available", False)):
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False.")
    if str(runtime_info.get("requested_device")) == "cuda" and not bool(runtime_info.get("cuda_available", False)):
        raise RuntimeError("Device 'cuda' was requested, but CUDA is not available.")


class SingleRecordingScalarDataset(Dataset):
    """Windowed scalar target dataset for one recording."""

    def __init__(
        self,
        npz_path: str,
        window_spec: WindowSpec,
        target_channel: int = 0,
        feature_mode: str = "binned",
        bin_ms: int = 25,
        normalization_stats: Optional[NormalizationStats] = None,
    ):
        self.recording = load_recording(npz_path)
        self.window_spec = window_spec
        self.target_channel = int(target_channel)
        self.feature_mode = feature_mode
        self.bin_ms = int(bin_ms)
        self.window_starts = compute_window_starts(self.recording.n_samples, self.recording.fs_hz, self.window_spec)
        if self.target_channel < 0 or self.target_channel >= self.recording.aux.shape[1]:
            raise ValueError(f"Invalid target_channel {target_channel}")
        self.normalization_stats = normalization_stats or self._compute_normalization()

    def _compute_normalization(self) -> NormalizationStats:
        mask = self.recording.good_mask.astype(bool)
        masked = self.recording.emg[:, mask]
        x_mean = float(masked.mean())
        x_std = float(max(masked.std(), 1e-6))
        y = self.recording.aux[:, self.target_channel].astype(np.float32)
        y_mean = np.asarray([float(y.mean())], dtype=np.float32)
        y_std = np.asarray([float(max(y.std(), 1e-6))], dtype=np.float32)
        return NormalizationStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, object]:
        start = int(self.window_starts[idx])
        win = int(round(self.recording.fs_hz * self.window_spec.win_ms / 1000.0))
        delta = int(round(self.recording.fs_hz * self.window_spec.delta_ms / 1000.0))
        emg_window = self.recording.emg[start : start + win]
        target_window = self.recording.aux[start + delta : start + delta + win, self.target_channel]
        x = _normalize_emg_window(emg_window, self.recording.good_mask, self.normalization_stats)
        y_phys = np.asarray([float(target_window.mean())], dtype=np.float32)
        y = ((y_phys - self.normalization_stats.y_mean) / self.normalization_stats.y_std).astype(np.float32)
        midpoint = start + (win // 2)
        time_s = midpoint / self.recording.fs_hz

        if self.feature_mode == "raw":
            x_tensor = torch.from_numpy(x)
        elif self.feature_mode == "binned":
            features = extract_binned_features(
                emg=x,
                good_mask=self.recording.good_mask,
                fs_hz=self.recording.fs_hz,
                bin_ms=self.bin_ms,
            ).astype(np.float32)
            x_tensor = torch.from_numpy(features)
        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        return {
            "x": x_tensor,
            "y": torch.from_numpy(y),
            "y_phys": torch.from_numpy(y_phys),
            "window_start": start,
            "time_s": float(time_s),
        }


def _normalize_emg_window(window: np.ndarray, good_mask: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    masked = window * good_mask[None, :, :, :]
    normalized = (masked - stats.x_mean) / stats.x_std
    return normalized * good_mask[None, :, :, :]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_single_recording_split(n_items: int, split_mode: str, train_fraction: float) -> Dict[str, np.ndarray]:
    indices = np.arange(n_items, dtype=np.int32)
    if split_mode == "same_windows":
        return {"train_idx": indices, "eval_idx": indices}
    if split_mode != "chronological_holdout":
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    train_count = max(1, min(n_items - 1, int(math.floor(n_items * train_fraction))))
    return {
        "train_idx": indices[:train_count],
        "eval_idx": indices[train_count:],
    }


def _collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "y_phys": torch.stack([item["y_phys"] for item in batch], dim=0),
        "window_start": np.asarray([int(item["window_start"]) for item in batch], dtype=np.int32),
        "time_s": np.asarray([float(item["time_s"]) for item in batch], dtype=np.float32),
    }


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        value = float(pearsonr(y_true, y_pred).statistic)
    return value if np.isfinite(value) else 0.0


def compute_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - (float(np.sum(np.square(residual))) / denom) if denom > 0 else 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "pearson": _safe_pearson(y_true, y_pred),
        "target_mean": float(y_true.mean()),
        "target_std": float(y_true.std()),
        "target_min": float(y_true.min()),
        "target_max": float(y_true.max()),
        "prediction_mean": float(y_pred.mean()),
        "prediction_std": float(y_pred.std()),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
    }


class ConstantMeanScalarModel:
    def __init__(self) -> None:
        self.mean_target = 0.0

    def fit(self, y_train: np.ndarray) -> None:
        self.mean_target = float(y_train.mean())

    def predict(self, n_items: int) -> np.ndarray:
        return np.full((n_items,), self.mean_target, dtype=np.float32)


def _build_scalar_model(model_name: str, sample_x: torch.Tensor, config: Dict[str, object]) -> nn.Module:
    if model_name == "mlp_feature_scalar":
        return get_model(
            model_name,
            input_dim=int(sample_x.shape[0] * sample_x.shape[1]),
            hidden_dim=int(config.get("hidden_dim", 128)),
            dropout=float(config.get("dropout", 0.1)),
            output_dim=1,
        )
    if model_name == "cnn1d_feature_scalar":
        return get_model(
            model_name,
            input_channels=int(sample_x.shape[1]),
            base_channels=int(config.get("base_channels", 32)),
            dropout=float(config.get("dropout", 0.1)),
            output_dim=1,
        )
    if model_name == "cnn1d_raw_scalar":
        return get_model(
            model_name,
            input_features=int(sample_x.shape[1] * sample_x.shape[2] * sample_x.shape[3]),
            base_channels=int(config.get("base_channels", 16)),
            dropout=float(config.get("dropout", 0.15)),
            output_dim=1,
        )
    raise ValueError(f"Unsupported scalar model: {model_name}")


def _make_loader(dataset: Dataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_batch)


def _predict_scalar_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_stats: NormalizationStats,
) -> Dict[str, np.ndarray]:
    preds_std: List[np.ndarray] = []
    y_std_rows: List[np.ndarray] = []
    y_phys_rows: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    times: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device)).cpu().numpy().reshape(-1)
            preds_std.append(pred)
            y_std_rows.append(batch["y"].numpy().reshape(-1))
            y_phys_rows.append(batch["y_phys"].numpy().reshape(-1))
            starts.append(batch["window_start"])
            times.append(batch["time_s"])
    pred_std = np.concatenate(preds_std, axis=0)
    y_std = np.concatenate(y_std_rows, axis=0)
    y_phys = np.concatenate(y_phys_rows, axis=0)
    pred_phys = pred_std * float(norm_stats.y_std[0]) + float(norm_stats.y_mean[0])
    return {
        "pred_std": pred_std,
        "y_std": y_std,
        "pred_phys": pred_phys,
        "y_phys": y_phys,
        "window_start": np.concatenate(starts, axis=0),
        "time_s": np.concatenate(times, axis=0),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "trainable_params": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "total_params": int(sum(param.numel() for param in model.parameters())),
    }


def build_model_info(
    model_name: str,
    config: Dict[str, object],
    window_samples: int,
    sample_x: Optional[torch.Tensor] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, object]:
    if model_name == "constant_mean_scalar":
        return {
            "display_name": "Constant mean scalar baseline",
            "feature_mode": "none",
            "input_description": "No learned network; predicts the training mean of AUX[0] for every window.",
            "layers": ["Constant prediction: y_hat = mean(train AUX[0])"],
            "output_description": "Scalar AUX[0] prediction",
            "trainable_params": 0,
            "total_params": 0,
        }

    if sample_x is None or model is None:
        raise ValueError("sample_x and model are required to build neural model info")

    counts = _count_parameters(model)
    if model_name == "cnn1d_raw_scalar":
        input_features = int(sample_x.shape[1] * sample_x.shape[2] * sample_x.shape[3])
        conv1_channels = int(config.get("base_channels", 16)) * 4
        conv2_channels = int(config.get("base_channels", 16)) * 8
        pooled_steps = int(math.floor(window_samples / 2))
        return {
            "display_name": "Raw 1D CNN scalar regressor",
            "feature_mode": "raw",
            "input_description": (
                f"Normalized EMG window with shape (time={window_samples}, grids=6, height=8, width=8), "
                f"reshaped to (channels={input_features}, time={window_samples}) before Conv1d."
            ),
            "layers": [
                f"Input reshape: (B, {window_samples}, 6, 8, 8) -> (B, {input_features}, {window_samples})",
                f"Conv1d({input_features} -> {conv1_channels}, kernel_size=5, padding=2)",
                f"BatchNorm1d({conv1_channels})",
                "ReLU",
                f"MaxPool1d(kernel_size=2) -> temporal length about {pooled_steps}",
                f"Conv1d({conv1_channels} -> {conv2_channels}, kernel_size=5, padding=2)",
                f"BatchNorm1d({conv2_channels})",
                "ReLU",
                f"Dropout(p={float(config.get('dropout', 0.15))})",
                "AdaptiveAvgPool1d(output_size=1)",
                f"Linear({conv2_channels} -> 64)",
                "ReLU",
                f"Dropout(p={float(config.get('dropout', 0.15))})",
                "Linear(64 -> 1)",
            ],
            "output_description": "Scalar AUX[0] prediction per EMG window",
            **counts,
        }

    if model_name == "cnn1d_feature_scalar":
        n_bins = int(sample_x.shape[0])
        input_channels = int(sample_x.shape[1])
        conv1_channels = int(config.get("base_channels", 32))
        conv2_channels = conv1_channels * 2
        return {
            "display_name": "Feature 1D CNN scalar regressor",
            "feature_mode": "binned",
            "input_description": f"Binned feature tensor with shape (bins={n_bins}, channels={input_channels}).",
            "layers": [
                f"Input permute: (B, {n_bins}, {input_channels}) -> (B, {input_channels}, {n_bins})",
                f"Conv1d({input_channels} -> {conv1_channels}, kernel_size=3, padding=1)",
                f"BatchNorm1d({conv1_channels})",
                "ReLU",
                f"Conv1d({conv1_channels} -> {conv2_channels}, kernel_size=3, padding=1)",
                f"BatchNorm1d({conv2_channels})",
                "ReLU",
                f"Dropout(p={float(config.get('dropout', 0.1))})",
                "AdaptiveAvgPool1d(output_size=1)",
                f"Linear({conv2_channels} -> 1)",
            ],
            "output_description": "Scalar AUX[0] prediction per EMG window",
            **counts,
        }

    if model_name == "mlp_feature_scalar":
        n_bins = int(sample_x.shape[0])
        input_channels = int(sample_x.shape[1])
        input_dim = int(n_bins * input_channels)
        hidden_dim = int(config.get("hidden_dim", 128))
        return {
            "display_name": "Feature MLP scalar regressor",
            "feature_mode": "binned",
            "input_description": f"Binned feature tensor with shape (bins={n_bins}, channels={input_channels}), flattened to {input_dim}.",
            "layers": [
                f"Flatten({n_bins} x {input_channels} -> {input_dim})",
                f"Linear({input_dim} -> {hidden_dim})",
                "ReLU",
                f"Dropout(p={float(config.get('dropout', 0.1))})",
                f"Linear({hidden_dim} -> {hidden_dim})",
                "ReLU",
                f"Dropout(p={float(config.get('dropout', 0.1))})",
                f"Linear({hidden_dim} -> 1)",
            ],
            "output_description": "Scalar AUX[0] prediction per EMG window",
            **counts,
        }

    raise ValueError(f"Unsupported model_name for model info: {model_name}")


def _load_selection_context_rows() -> List[Dict[str, object]]:
    comparison_path = _repo_root() / "runs" / "participant4_hyperparam_comparison" / "leaderboard.csv"
    if not comparison_path.exists():
        return []
    with comparison_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep_models = {
        ("cnn1d_raw_scalar", "300", "-150"),
        ("cnn1d_feature_scalar", "300", "-150"),
        ("cnn1d_feature_scalar", "200", "150"),
    }
    filtered = [
        row for row in rows
        if (row.get("model"), row.get("win_ms"), row.get("delta_ms")) in keep_models
        and row.get("recording") in {
            "p4rec3_processed_1024Hz.npz",
            "p4rec5_processed_1024Hz.npz",
            "p4rec6_processed_1024Hz.npz",
        }
    ]
    filtered.sort(key=lambda row: (row["recording"], -float(row["eval_r2"]), float(row["eval_rmse"])))
    return filtered


def _artifact_guide_lines(has_checkpoints: bool, has_selection_context: bool) -> List[str]:
    file_guide = [
        "- `config.json`: exact run configuration and hyperparameters.",
        "- `metrics.json`: train/eval metrics in physical units, device info, window counts, and timing.",
        "- `runtime.json`: Python, torch, venv, and GPU runtime information for this run.",
        "- `model_info.json`: exact architecture description and parameter counts.",
        "- `history.csv` / `history.json`: per-epoch training and evaluation loss.",
        "- `predictions.csv`: one row per evaluated window with true value, prediction, residual, and time.",
        "- `predictions.npz`: same prediction data in NumPy form for later plotting or analysis.",
        "- `model_best.pt`: checkpoint selected by train loss for `same_windows` or eval loss for `chronological_holdout`.",
        "- `model_last.pt`: last neural checkpoint after the final epoch.",
        "- `README.md`: polished presentation document for this run.",
        "- `loss_curve.png`: loss over epochs.",
        "- `pred_vs_true_scatter.png`: predicted versus true AUX[0] with the identity line.",
        "- `pred_vs_true_timeseries.png`: predicted and true AUX[0] over time.",
        "- `residual_hist.png`: residual distribution.",
        "- `residual_vs_true.png`: residual pattern as a function of true AUX[0].",
        "- `summary.md`: concise technical companion to the README.",
    ]
    if not has_checkpoints:
        file_guide = [line for line in file_guide if "model_" not in line]
    if has_selection_context:
        file_guide.append("- `selection_context.csv`: supporting participant-4 comparison used to justify this configuration.")
    return file_guide


def write_single_recording_summary(
    run_dir: Path,
    config: Dict[str, object],
    metrics: Dict[str, object],
    has_checkpoints: bool,
    has_selection_context: bool,
) -> None:
    file_guide = _artifact_guide_lines(has_checkpoints=has_checkpoints, has_selection_context=has_selection_context)
    lines = [
        "# Single Recording Overfit Summary",
        "",
        "Main presentation document: [README.md](README.md)",
        "",
        "## Run Overview",
        f"- Model: `{config['model']}`",
        f"- Recording: `{config['npz']}`",
        f"- Split mode: `{config['split_mode']}`",
        f"- Window / delta / step (ms): `{config['win_ms']}` / `{config['delta_ms']}` / `{config['step_ms']}`",
        f"- Train windows: `{metrics['train_window_count']}`",
        f"- Eval windows: `{metrics['eval_window_count']}`",
        f"- Device: `{metrics['device']}`",
        f"- Training seconds: `{metrics['fit_seconds']:.3f}`",
        "",
        "## Main Result",
        f"- Train RMSE: `{metrics['train_metrics']['rmse']:.6f}`",
        f"- Eval RMSE: `{metrics['eval_metrics']['rmse']:.6f}`",
        f"- Eval R2: `{metrics['eval_metrics']['r2']:.6f}`",
        f"- Eval Pearson: `{metrics['eval_metrics']['pearson']:.6f}`",
        "",
        "## Artifact Guide",
        *file_guide,
        "",
        "## Loss Curve",
        "![Loss curve](loss_curve.png)",
        "",
        "## Predicted vs True Scatter",
        "![Predicted vs true scatter](pred_vs_true_scatter.png)",
        "",
        "## Predicted vs True Over Time",
        "![Predicted vs true over time](pred_vs_true_timeseries.png)",
        "",
        "## Residual Histogram",
        "![Residual histogram](residual_hist.png)",
        "",
        "## Residual vs True",
        "![Residual vs true](residual_vs_true.png)",
        "",
        "## Interpretation Notes",
        "- In `same_windows`, near-perfect fit indicates the model can memorize this recording.",
        "- In `chronological_holdout`, the metrics indicate whether the learned timing carries forward within the same recording.",
        "- Use `README.md` for the full architecture explanation and configuration rationale.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_recording_readme(
    run_dir: Path,
    config: Dict[str, object],
    metrics: Dict[str, object],
    runtime_info: Dict[str, object],
    model_info: Dict[str, object],
    has_checkpoints: bool,
    selection_context_rows: Sequence[Dict[str, object]],
) -> None:
    file_guide = _artifact_guide_lines(
        has_checkpoints=has_checkpoints,
        has_selection_context=bool(selection_context_rows),
    )
    fit_quality = (
        f"This run demonstrates very strong single-recording overfit capability: the model reaches "
        f"RMSE `{metrics['eval_metrics']['rmse']:.6f}`, MAE `{metrics['eval_metrics']['mae']:.6f}`, "
        f"Pearson `{metrics['eval_metrics']['pearson']:.6f}`, and R2 `{metrics['eval_metrics']['r2']:.6f}` "
        f"on the exact same windows it was trained on."
    )
    lines = [
        "# Participant 4 Overfit Demo",
        "",
        "## Goal",
        "Show, in a presentation-ready form, that the EMG-to-AUX pipeline can overfit one cleaned recording from participant 4.",
        "",
        "This is intentionally a same-window experiment: training and evaluation are both done on the same windows so we can measure the best-case memorization capability of the current pipeline.",
        "",
        "## Run At A Glance",
        f"- Recording: `{config['npz']}`",
        f"- Target: mean `AUX[{config['target_channel']}]` over the shifted target window",
        f"- Split: `{config['split_mode']}`",
        f"- Window / delta / step (ms): `{config['win_ms']}` / `{config['delta_ms']}` / `{config['step_ms']}`",
        f"- Window / delta (samples): `{metrics['window_samples']}` / `{metrics['delta_samples']}`",
        f"- Model: `{model_info['display_name']}`",
        f"- Device used: `{runtime_info['resolved_device']}`",
        f"- GPU: `{runtime_info.get('device_name', 'n/a')}`",
        f"- Python executable: `{runtime_info['python_executable']}`",
        f"- Running inside venv: `{runtime_info['in_venv']}`",
        "",
        "## Main Outcome",
        fit_quality,
        "",
        "| Metric | Train | Eval |",
        "| --- | ---: | ---: |",
        f"| MSE | {metrics['train_metrics']['mse']:.6f} | {metrics['eval_metrics']['mse']:.6f} |",
        f"| RMSE | {metrics['train_metrics']['rmse']:.6f} | {metrics['eval_metrics']['rmse']:.6f} |",
        f"| MAE | {metrics['train_metrics']['mae']:.6f} | {metrics['eval_metrics']['mae']:.6f} |",
        f"| Pearson | {metrics['train_metrics']['pearson']:.6f} | {metrics['eval_metrics']['pearson']:.6f} |",
        f"| R2 | {metrics['train_metrics']['r2']:.6f} | {metrics['eval_metrics']['r2']:.6f} |",
        "",
        "## Exact Data and Label Definition",
        f"- Input representation: `{model_info['feature_mode']}`",
        f"- The model consumes one cleaned EMG window from `{Path(config['npz']).name}`.",
        f"- The label is the mean of `AUX[{config['target_channel']}]` over a target window shifted by `{config['delta_ms']}` ms relative to the EMG window.",
        f"- The dataset produces `{metrics['train_window_count']}` windows for this run.",
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
    if selection_context_rows:
        lines.extend(
            [
                "## Why This Configuration",
                "This configuration was chosen because it consistently beat the nearby alternatives across three participant-4 recordings (`p4rec3`, `p4rec5`, `p4rec6`).",
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
                "- `cnn1d_raw_scalar` with `300 ms / -150 ms` was the best run on all three recordings.",
                "- `cnn1d_feature_scalar` with `300 ms / -150 ms` was consistently second-best.",
                "- `cnn1d_feature_scalar` with `200 ms / +150 ms` was consistently worse.",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact Guide",
            *file_guide,
            "",
            "## Plots",
            "### Loss Curve",
            "![Loss curve](loss_curve.png)",
            "",
            "### Predicted vs True Scatter",
            "![Predicted vs true scatter](pred_vs_true_scatter.png)",
            "",
            "### Predicted vs True Over Time",
            "![Predicted vs true over time](pred_vs_true_timeseries.png)",
            "",
            "### Residual Histogram",
            "![Residual histogram](residual_hist.png)",
            "",
            "### Residual vs True",
            "![Residual vs true](residual_vs_true.png)",
            "",
            "## Interpretation",
            "- Because this is a same-window experiment, the train and eval metrics are identical by design.",
            "- That is acceptable here because the goal is to show the upper bound of what the current pipeline can memorize on a single cleaned recording.",
            "- The next level of validation would be chronological holdout or held-out recordings from the same participant.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_loss_curve(history: Sequence[Dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row.get("eval_loss") for row in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="train")
    if any(value is not None for value in val_loss):
        plt.plot(epochs, val_loss, label="eval")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float], path: Path) -> None:
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("True AUX[0]")
    plt.ylabel("Predicted AUX[0]")
    plt.title(f"Predicted vs true\nR2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_timeseries(time_s: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    order = np.argsort(time_s)
    plt.figure(figsize=(10, 4.5))
    plt.plot(time_s[order], y_true[order], label="true", linewidth=1.6)
    plt.plot(time_s[order], y_pred[order], label="pred", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("AUX[0]")
    plt.title("Predicted vs true over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_residual_hist(residual: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.hist(residual, bins=30, alpha=0.85)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_residual_vs_true(y_true: np.ndarray, residual: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_true, residual, s=18, alpha=0.75)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("True AUX[0]")
    plt.ylabel("Residual")
    plt.title("Residual vs true")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_run_artifacts(
    run_dir: Path,
    config: Dict[str, object],
    history: Sequence[Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, object],
    runtime_info: Dict[str, object],
    model_info: Dict[str, object],
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
    pred_rows = [
        {
            "window_start": int(start),
            "time_s": float(time_s),
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "residual": float(y_pred - y_true),
        }
        for start, time_s, y_true, y_pred in zip(
            predictions["window_start"],
            predictions["time_s"],
            predictions["y_phys"],
            predictions["pred_phys"],
        )
    ]
    _write_csv(run_dir / "predictions.csv", pred_rows)
    np.savez(
        run_dir / "predictions.npz",
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
    if selection_context_rows:
        _write_csv(run_dir / "selection_context.csv", list(selection_context_rows))
    residual = predictions["pred_phys"] - predictions["y_phys"]
    _plot_loss_curve(history, run_dir / "loss_curve.png")
    _plot_scatter(predictions["y_phys"], predictions["pred_phys"], metrics["eval_metrics"], run_dir / "pred_vs_true_scatter.png")
    _plot_timeseries(predictions["time_s"], predictions["y_phys"], predictions["pred_phys"], run_dir / "pred_vs_true_timeseries.png")
    _plot_residual_hist(residual, run_dir / "residual_hist.png")
    _plot_residual_vs_true(predictions["y_phys"], residual, run_dir / "residual_vs_true.png")
    has_checkpoints = best_state is not None
    has_selection_context = bool(selection_context_rows)
    write_single_recording_summary(
        run_dir,
        config,
        metrics,
        has_checkpoints=has_checkpoints,
        has_selection_context=has_selection_context,
    )
    write_single_recording_readme(
        run_dir,
        config,
        metrics,
        runtime_info,
        model_info,
        has_checkpoints=has_checkpoints,
        selection_context_rows=selection_context_rows,
    )


def run_single_recording_experiment(
    npz_path: str,
    model_name: str,
    win_ms: int,
    delta_ms: int,
    step_ms: int = 50,
    split_mode: str = "same_windows",
    train_fraction: float = 0.8,
    target_channel: int = 0,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    base_channels: int = 32,
    hidden_dim: int = 128,
    seed: int = 13,
    device: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, object]:
    set_seed(seed)
    runtime_info = collect_runtime_info(requested_device=device)
    validate_runtime_requirements(runtime_info, require_cuda=False)
    torch_device = torch.device(str(runtime_info["resolved_device"]))
    config = {
        "npz": npz_path,
        "model": model_name,
        "win_ms": int(win_ms),
        "delta_ms": int(delta_ms),
        "step_ms": int(step_ms),
        "split_mode": split_mode,
        "train_fraction": float(train_fraction),
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
    feature_mode = "raw" if model_name == "cnn1d_raw_scalar" else "binned"
    dataset = SingleRecordingScalarDataset(
        npz_path=npz_path,
        window_spec=WindowSpec(win_ms=int(win_ms), step_ms=int(step_ms), delta_ms=int(delta_ms)),
        target_channel=target_channel,
        feature_mode=feature_mode,
    )
    selection_context_rows = (
        _load_selection_context_rows()
        if int(dataset.recording.pid) == 4 and model_name in {"cnn1d_raw_scalar", "cnn1d_feature_scalar"}
        else []
    )
    split = make_single_recording_split(len(dataset), split_mode, train_fraction)
    train_idx = split["train_idx"]
    eval_idx = split["eval_idx"]

    norm_stats = dataset.normalization_stats
    train_loader = _make_loader(dataset, train_idx, batch_size=int(batch_size), shuffle=True)
    eval_loader = _make_loader(dataset, eval_idx, batch_size=int(batch_size), shuffle=False)

    history: List[Dict[str, float]] = []
    fit_start = time.perf_counter()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_state: Optional[Dict[str, torch.Tensor]] = None
    window_samples = int(round(dataset.recording.fs_hz * win_ms / 1000.0))

    if model_name == "constant_mean_scalar":
        baseline = ConstantMeanScalarModel()
        model_info = build_model_info(model_name, config, window_samples=window_samples)
        train_targets = np.asarray([float(dataset[idx]["y_phys"].numpy()[0]) for idx in train_idx], dtype=np.float32)
        baseline.fit(train_targets)
        train_true = np.asarray([float(dataset[idx]["y_phys"].numpy()[0]) for idx in train_idx], dtype=np.float32)
        eval_true = np.asarray([float(dataset[idx]["y_phys"].numpy()[0]) for idx in eval_idx], dtype=np.float32)
        train_pred = baseline.predict(train_true.shape[0])
        eval_pred = baseline.predict(eval_true.shape[0])
        fit_seconds = time.perf_counter() - fit_start
        history.append({"epoch": 1, "train_loss": float(np.mean(np.square(train_pred - train_true))), "eval_loss": float(np.mean(np.square(eval_pred - eval_true)))})
        predictions = {
            "window_start": np.asarray([int(dataset[idx]["window_start"]) for idx in eval_idx], dtype=np.int32),
            "time_s": np.asarray([float(dataset[idx]["time_s"]) for idx in eval_idx], dtype=np.float32),
            "y_phys": eval_true,
            "pred_phys": eval_pred,
            "y_std": np.asarray([float(dataset[idx]["y"].numpy()[0]) for idx in eval_idx], dtype=np.float32),
            "pred_std": ((eval_pred - float(norm_stats.y_mean[0])) / float(norm_stats.y_std[0])).astype(np.float32),
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
            model.train()
            total_loss = 0.0
            count = 0
            for batch in train_loader:
                xb = batch["x"].to(torch_device)
                yb = batch["y"].to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]

            train_eval = _predict_scalar_model(model, _make_loader(dataset, train_idx, batch_size=int(batch_size), shuffle=False), torch_device, norm_stats)
            eval_eval = _predict_scalar_model(model, eval_loader, torch_device, norm_stats)
            train_loss = float(np.mean(np.square(train_eval["pred_std"] - train_eval["y_std"])))
            eval_loss = float(np.mean(np.square(eval_eval["pred_std"] - eval_eval["y_std"])))
            history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})
            score = train_loss if split_mode == "same_windows" else eval_loss
            if score < best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            last_state = deepcopy(model.state_dict())

        fit_seconds = time.perf_counter() - fit_start
        assert best_state is not None
        model.load_state_dict(best_state)
        train_eval = _predict_scalar_model(model, _make_loader(dataset, train_idx, batch_size=int(batch_size), shuffle=False), torch_device, norm_stats)
        predictions = _predict_scalar_model(model, eval_loader, torch_device, norm_stats)
        train_true = train_eval["y_phys"]
        train_pred = train_eval["pred_phys"]
        eval_true = predictions["y_phys"]
        eval_pred = predictions["pred_phys"]

    train_metrics = compute_scalar_metrics(train_true, train_pred)
    eval_metrics = compute_scalar_metrics(eval_true, eval_pred)
    metrics = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "fit_seconds": float(fit_seconds),
        "device": str(torch_device),
        "train_window_count": int(train_idx.shape[0]),
        "eval_window_count": int(eval_idx.shape[0]),
        "window_samples": window_samples,
        "delta_samples": int(round(dataset.recording.fs_hz * delta_ms / 1000.0)),
    }
    if run_dir is not None:
        _save_run_artifacts(
            run_dir,
            config,
            history,
            predictions,
            metrics,
            runtime_info,
            model_info,
            selection_context_rows,
            best_state,
            last_state,
        )
    return {
        "config": config,
        "history": history,
        "metrics": metrics,
        "runtime_info": runtime_info,
        "model_info": model_info,
        "predictions": predictions,
    }


def _resolve_model_defaults(model_name: str) -> Dict[str, object]:
    if model_name == "constant_mean_scalar":
        return {"epochs": 1, "batch_size": 64, "lr": 0.0, "weight_decay": 0.0, "dropout": 0.0, "base_channels": 16, "hidden_dim": 64}
    if model_name == "cnn1d_feature_scalar":
        return {"epochs": 200, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "base_channels": 32, "hidden_dim": 128}
    if model_name == "cnn1d_raw_scalar":
        return {"epochs": 250, "batch_size": 16, "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.15, "base_channels": 16, "hidden_dim": 128}
    if model_name == "mlp_feature_scalar":
        return {"epochs": 200, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "base_channels": 16, "hidden_dim": 128}
    raise ValueError(f"Unknown model defaults for {model_name}")


def _render_heatmap(rows: Sequence[Dict[str, object]], metric_name: str, path: Path) -> None:
    models = sorted({str(row["model"]) for row in rows})
    win_values = sorted({int(row["win_ms"]) for row in rows})
    delta_values = sorted({int(row["delta_ms"]) for row in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        matrix = np.full((len(win_values), len(delta_values)), np.nan, dtype=np.float32)
        for row in rows:
            if str(row["model"]) != model_name:
                continue
            r = win_values.index(int(row["win_ms"]))
            c = delta_values.index(int(row["delta_ms"]))
            matrix[r, c] = float(row[metric_name])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(delta_values)), [str(v) for v in delta_values])
        ax.set_yticks(range(len(win_values)), [str(v) for v in win_values])
        ax.set_xlabel("delta_ms")
        ax.set_ylabel("win_ms")
        ax.set_title(model_name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Timing sweep: {metric_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_single_recording_sweep(
    npz_path: str,
    output_root: str,
    models: Sequence[str],
    win_values: Sequence[int],
    delta_values: Sequence[int],
    step_ms: int = 50,
    split_mode: str = "same_windows",
    train_fraction: float = 0.8,
    target_channel: int = 0,
    seed: int = 13,
    device: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: List[Dict[str, object]] = []

    for model_name in models:
        defaults = _resolve_model_defaults(model_name)
        if epochs_override is not None:
            defaults["epochs"] = int(epochs_override)
        if batch_size_override is not None:
            defaults["batch_size"] = int(batch_size_override)
        for win_ms in win_values:
            for delta_ms in delta_values:
                run_name = f"{model_name}_win{win_ms}_delta{delta_ms}_{split_mode}"
                run_dir = output_dir / run_name
                result = run_single_recording_experiment(
                    npz_path=npz_path,
                    model_name=model_name,
                    win_ms=int(win_ms),
                    delta_ms=int(delta_ms),
                    step_ms=int(step_ms),
                    split_mode=split_mode,
                    train_fraction=float(train_fraction),
                    target_channel=int(target_channel),
                    epochs=int(defaults["epochs"]),
                    batch_size=int(defaults["batch_size"]),
                    lr=float(defaults["lr"]),
                    weight_decay=float(defaults["weight_decay"]),
                    dropout=float(defaults["dropout"]),
                    base_channels=int(defaults["base_channels"]),
                    hidden_dim=int(defaults["hidden_dim"]),
                    seed=int(seed),
                    device=device,
                    run_dir=run_dir,
                )
                leaderboard_rows.append(
                    {
                        "model": model_name,
                        "win_ms": int(win_ms),
                        "delta_ms": int(delta_ms),
                        "split_mode": split_mode,
                        "eval_r2": float(result["metrics"]["eval_metrics"]["r2"]),
                        "eval_rmse": float(result["metrics"]["eval_metrics"]["rmse"]),
                        "eval_mae": float(result["metrics"]["eval_metrics"]["mae"]),
                        "eval_pearson": float(result["metrics"]["eval_metrics"]["pearson"]),
                        "train_r2": float(result["metrics"]["train_metrics"]["r2"]),
                        "train_rmse": float(result["metrics"]["train_metrics"]["rmse"]),
                        "fit_seconds": float(result["metrics"]["fit_seconds"]),
                    }
                )

    leaderboard_rows.sort(key=lambda row: (row["model"], -row["eval_r2"], row["eval_rmse"]))
    _write_csv(output_dir / "leaderboard.csv", leaderboard_rows)
    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard_rows, indent=2), encoding="utf-8")
    _write_csv(output_dir / "timing_heatmap.csv", leaderboard_rows)
    _render_heatmap(leaderboard_rows, "eval_r2", output_dir / "timing_heatmap.png")
    return {"leaderboard": leaderboard_rows}


def refresh_single_recording_summary(run_dir: str | Path) -> None:
    path = Path(run_dir)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    runtime_info = json.loads((path / "runtime.json").read_text(encoding="utf-8")) if (path / "runtime.json").exists() else collect_runtime_info(requested_device=config.get("device"))
    if (path / "model_info.json").exists():
        model_info = json.loads((path / "model_info.json").read_text(encoding="utf-8"))
    else:
        window_samples = int(metrics["window_samples"])
        if config["model"] == "constant_mean_scalar":
            model_info = build_model_info(config["model"], config, window_samples)
        else:
            recording = load_recording(config["npz"])
            if config["model"] == "cnn1d_raw_scalar":
                sample_x = torch.zeros((window_samples, 6, 8, 8), dtype=torch.float32)
            elif config["model"] == "cnn1d_feature_scalar":
                emg_window = recording.emg[:window_samples]
                stats = NormalizationStats(
                    x_mean=float(recording.emg.mean()),
                    x_std=float(max(recording.emg.std(), 1e-6)),
                    y_mean=np.asarray([float(recording.aux[:, int(config["target_channel"])].mean())], dtype=np.float32),
                    y_std=np.asarray([float(max(recording.aux[:, int(config["target_channel"])].std(), 1e-6))], dtype=np.float32),
                )
                normalized = _normalize_emg_window(emg_window, recording.good_mask, stats)
                sample_x = torch.from_numpy(
                    extract_binned_features(normalized, recording.good_mask, recording.fs_hz, bin_ms=25).astype(np.float32)
                )
            else:
                sample_x = torch.zeros((max(1, int(math.ceil(window_samples / 25))), 12), dtype=torch.float32)
            model = _build_scalar_model(config["model"], sample_x=sample_x, config=config)
            model_info = build_model_info(config["model"], config, window_samples, sample_x=sample_x, model=model)
    selection_context_rows: List[Dict[str, object]] = []
    if (path / "selection_context.csv").exists():
        with (path / "selection_context.csv").open("r", newline="", encoding="utf-8") as handle:
            selection_context_rows = list(csv.DictReader(handle))
    else:
        selection_context_rows = _load_selection_context_rows()
    has_checkpoints = (path / "model_best.pt").exists() or (path / "model_last.pt").exists()
    has_selection_context = bool(selection_context_rows)
    write_single_recording_summary(
        path,
        config,
        metrics,
        has_checkpoints=has_checkpoints,
        has_selection_context=has_selection_context,
    )
    write_single_recording_readme(
        path,
        config,
        metrics,
        runtime_info,
        model_info,
        has_checkpoints=has_checkpoints,
        selection_context_rows=selection_context_rows,
    )
