"""
Tests for participant-level pooled overfit workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import WindowSpec, compute_window_starts
from src.training import (
    ParticipantScalarWindowDataset,
    build_shuffle_audit_rows,
    make_epoch_order,
    run_participant_overfit_experiment,
)


def _write_recording(path: Path, fs: float = 100.0, phase: float = 0.0, scale: float = 1.0) -> None:
    n_t = 220
    t = np.arange(n_t, dtype=np.float32) / fs
    emg = np.zeros((n_t, 6, 8, 8), dtype=np.float32)
    carrier = (scale * np.sin(2.0 * np.pi * 2.0 * t + phase)).astype(np.float32)
    for grid in range(6):
        emg[:, grid] = carrier[:, None, None] * (1.0 + 0.05 * grid)
    aux0 = 0.75 * carrier + 2.0 + 0.1 * phase
    aux1 = 0.20 * np.cos(2.0 * np.pi * 1.0 * t + phase) + 3.0
    aux = np.stack([aux0, aux1], axis=1).astype(np.float32)
    bad_mask = np.zeros((6, 8, 8), dtype=bool)
    meta = json.dumps({"fs_export_hz": fs})
    np.savez(path, emg=emg, aux=aux, bad_mask=bad_mask, time_s=t, meta=meta)


def _write_participant_dir(root: Path) -> Path:
    participant_dir = root / "participant 4 stuff"
    participant_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("p4rec3_processed_1024Hz.npz", 0.0, 1.0),
        ("p4rec5_processed_1024Hz.npz", 0.5, 1.1),
        ("p4rec6_processed_1024Hz.npz", 1.0, 0.9),
    ]
    for name, phase, scale in specs:
        _write_recording(participant_dir / name, phase=phase, scale=scale)
    return participant_dir


def test_participant_dataset_length_matches_sum_of_recordings(tmp_path: Path):
    participant_dir = _write_participant_dir(tmp_path)
    spec = WindowSpec(win_ms=100, step_ms=50, delta_ms=0)
    dataset = ParticipantScalarWindowDataset(
        recordings_dir=str(participant_dir),
        window_spec=spec,
        target_channel=0,
        feature_mode="binned",
    )
    expected = sum(compute_window_starts(recording.n_samples, recording.fs_hz, spec).shape[0] for recording in dataset.recordings)
    assert len(dataset) == expected


def test_shuffle_audit_shows_mixed_recordings(tmp_path: Path):
    participant_dir = _write_participant_dir(tmp_path)
    dataset = ParticipantScalarWindowDataset(
        recordings_dir=str(participant_dir),
        window_spec=WindowSpec(win_ms=100, step_ms=50, delta_ms=0),
        target_channel=0,
        feature_mode="binned",
    )
    indices = np.arange(len(dataset), dtype=np.int32)
    epoch_order = make_epoch_order(indices, seed=13, epoch=1)
    rows = build_shuffle_audit_rows(dataset, epoch_order, batch_size=8)
    assert rows
    assert any(int(row["unique_recordings"]) > 1 for row in rows)


def test_participant_overfit_experiment_writes_artifacts(tmp_path: Path):
    participant_dir = _write_participant_dir(tmp_path)
    run_dir = tmp_path / "participant_run"
    result = run_participant_overfit_experiment(
        recordings_dir=str(participant_dir),
        model_name="cnn1d_feature_scalar",
        win_ms=100,
        delta_ms=0,
        step_ms=50,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
        base_channels=8,
        device="cpu",
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
        "per_recording_metrics.csv",
        "shuffle_audit.csv",
        "summary.md",
        "loss_curve.png",
        "pred_vs_true_scatter.png",
        "per_recording_r2_rmse.png",
        "per_recording_window_count.png",
        "per_recording_timeseries.png",
        "residual_hist.png",
        "model_best.pt",
        "model_last.pt",
    ]:
        assert (run_dir / filename).exists()
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Shuffle Evidence" in readme
    assert "Exact Model Architecture" in readme
