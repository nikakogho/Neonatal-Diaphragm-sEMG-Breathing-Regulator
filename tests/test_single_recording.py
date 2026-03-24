"""
Tests for scalar single-recording overfit workflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.training import (
    SingleRecordingScalarDataset,
    make_single_recording_split,
    run_single_recording_experiment,
    run_single_recording_sweep,
)
from src.data import WindowSpec


def _write_recording(path: Path, fs: float = 100.0) -> None:
    n_t = 200
    t = np.arange(n_t, dtype=np.float32) / fs
    emg = np.zeros((n_t, 6, 8, 8), dtype=np.float32)
    carrier = np.sin(2.0 * np.pi * 2.0 * t).astype(np.float32)
    for grid in range(6):
        emg[:, grid] = carrier[:, None, None] * (1.0 + 0.05 * grid)
    aux0 = 0.5 * carrier + 2.0
    aux1 = 0.25 * np.cos(2.0 * np.pi * 1.0 * t) + 3.0
    aux = np.stack([aux0, aux1], axis=1).astype(np.float32)
    bad_mask = np.zeros((6, 8, 8), dtype=bool)
    meta = json.dumps({"fs_export_hz": fs})
    np.savez(path, emg=emg, aux=aux, bad_mask=bad_mask, time_s=t, meta=meta)


def test_scalar_dataset_returns_aux0_window_mean(tmp_path: Path):
    npz_path = tmp_path / "p4rec6_processed_1024Hz.npz"
    _write_recording(npz_path, fs=100.0)
    dataset = SingleRecordingScalarDataset(
        npz_path=str(npz_path),
        window_spec=WindowSpec(win_ms=100, step_ms=50, delta_ms=50),
        target_channel=0,
        feature_mode="binned",
    )
    sample = dataset[0]
    start = int(dataset.window_starts[0])
    expected = dataset.recording.aux[start + 5 : start + 15, 0].mean()
    recovered = float(sample["y_phys"].numpy()[0])
    assert np.isclose(recovered, expected)


def test_single_recording_split_modes():
    same = make_single_recording_split(10, "same_windows", 0.8)
    holdout = make_single_recording_split(10, "chronological_holdout", 0.8)
    assert np.array_equal(same["train_idx"], same["eval_idx"])
    assert len(np.intersect1d(holdout["train_idx"], holdout["eval_idx"])) == 0
    assert holdout["train_idx"][-1] < holdout["eval_idx"][0]


def test_single_recording_experiment_writes_artifacts(tmp_path: Path):
    npz_path = tmp_path / "p4rec6_processed_1024Hz.npz"
    _write_recording(npz_path, fs=100.0)
    run_dir = tmp_path / "single_run"
    result = run_single_recording_experiment(
        npz_path=str(npz_path),
        model_name="constant_mean_scalar",
        win_ms=100,
        delta_ms=0,
        step_ms=50,
        split_mode="same_windows",
        epochs=1,
        run_dir=run_dir,
    )
    assert "metrics" in result
    for filename in [
        "README.md",
        "config.json",
        "metrics.json",
        "runtime.json",
        "model_info.json",
        "history.csv",
        "history.json",
        "predictions.csv",
        "predictions.npz",
        "summary.md",
        "loss_curve.png",
        "pred_vs_true_scatter.png",
        "pred_vs_true_timeseries.png",
        "residual_hist.png",
        "residual_vs_true.png",
    ]:
        assert (run_dir / filename).exists()


def test_raw_run_readme_includes_architecture_and_runtime(tmp_path: Path):
    npz_path = tmp_path / "p4rec6_processed_1024Hz.npz"
    _write_recording(npz_path, fs=100.0)
    run_dir = tmp_path / "raw_run"
    run_single_recording_experiment(
        npz_path=str(npz_path),
        model_name="cnn1d_raw_scalar",
        win_ms=100,
        delta_ms=0,
        step_ms=50,
        split_mode="same_windows",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-5,
        dropout=0.15,
        base_channels=8,
        device="cpu",
        run_dir=run_dir,
    )
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    model_info = json.loads((run_dir / "model_info.json").read_text(encoding="utf-8"))
    assert "Exact Model Architecture" in readme
    assert "Conv1d" in readme
    assert "R2" in readme
    assert runtime["resolved_device"] == "cpu"
    assert model_info["display_name"] == "Raw 1D CNN scalar regressor"


def test_single_recording_sweep_writes_leaderboard(tmp_path: Path):
    npz_path = tmp_path / "p4rec6_processed_1024Hz.npz"
    _write_recording(npz_path, fs=100.0)
    output_root = tmp_path / "sweep"
    result = run_single_recording_sweep(
        npz_path=str(npz_path),
        output_root=str(output_root),
        models=["constant_mean_scalar"],
        win_values=[100],
        delta_values=[0],
        step_ms=50,
    )
    assert len(result["leaderboard"]) == 1
    assert (output_root / "leaderboard.csv").exists()
    assert (output_root / "leaderboard.json").exists()
    assert (output_root / "timing_heatmap.csv").exists()
    assert (output_root / "timing_heatmap.png").exists()


def test_check_gpu_script_runs(repo_root: Path):
    cmd = [sys.executable, str(repo_root / "scripts" / "check_gpu.py")]
    completed = subprocess.run(cmd, check=True, cwd=repo_root, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert "torch_version" in payload
    assert "python_executable" in payload
    assert "in_venv" in payload
