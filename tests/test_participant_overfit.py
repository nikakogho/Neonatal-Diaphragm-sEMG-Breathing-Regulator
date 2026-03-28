"""
Tests for participant-level pooled overfit workflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
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


def _write_participant_dir(root: Path, participant_id: int = 4) -> Path:
    participant_dir = root / f"participant {participant_id} stuff"
    participant_dir.mkdir(parents=True, exist_ok=True)
    if participant_id == 4:
        specs = [
            ("p4rec3_processed_1024Hz.npz", 0.0, 1.0),
            ("p4rec5_processed_1024Hz.npz", 0.5, 1.1),
            ("p4rec6_processed_1024Hz.npz", 1.0, 0.9),
        ]
    else:
        specs = [
            (f"par{participant_id}rec1_processed_1024Hz.npz", 0.0, 1.0),
            (f"par{participant_id}rec2_processed_1024Hz.npz", 0.5, 1.1),
            (f"par{participant_id}rec3_processed_1024Hz.npz", 1.0, 0.9),
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


def test_participant_readme_uses_inferred_participant_id(tmp_path: Path):
    participant_dir = _write_participant_dir(tmp_path, participant_id=9)
    run_dir = tmp_path / "participant9_run"
    run_single = run_participant_overfit_experiment(
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
    assert int(run_single["metrics"]["participant_id"]) == 9
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Participant 9 Step 2 Overfit Demo" in readme
    assert "participant 9" in readme.lower()
    assert "Participant 9 Overfit Summary" in summary
    assert "participant 4 at the same time" not in readme.lower()
    assert "selection_context.csv" not in readme


def test_compare_participant_runs_uses_generic_participant_labels(tmp_path: Path, repo_root: Path):
    participant_dir = _write_participant_dir(tmp_path, participant_id=9)
    run_dir = tmp_path / "participant9_run"
    run_participant_overfit_experiment(
        recordings_dir=str(participant_dir),
        model_name="constant_mean_scalar",
        win_ms=100,
        delta_ms=0,
        step_ms=50,
        epochs=1,
        batch_size=8,
        device="cpu",
        run_dir=run_dir,
    )
    output_dir = tmp_path / "comparison"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_participant_overfit_runs.py"),
        "--output_dir",
        str(output_dir),
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root, capture_output=True, text=True)
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Participant 9 Step 2 Hyperparameter Comparison" in summary
    assert "participant-4 overfit" not in summary.lower()


def test_compare_participant_runs_ranks_by_dynamic_min_r2_before_pooled_r2(tmp_path: Path, repo_root: Path):
    output_dir = tmp_path / "comparison"
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    for run_dir, eval_r2, dynamic_min_r2 in [
        (run_a, 0.992, 0.970),
        (run_b, 0.991, 0.985),
    ]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "participant_id": 10,
                    "recordings_dir": "extra_patients/participant 10 stuff",
                    "model": "cnn1d_raw_scalar",
                    "win_ms": 300,
                    "delta_ms": -150,
                    "epochs": 350,
                    "batch_size": 16,
                    "lr": 3e-4,
                    "dropout": 0.15,
                    "base_channels": 16,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "participant_id": 10,
                    "eval_metrics": {
                        "r2": eval_r2,
                        "rmse": 0.1,
                        "mae": 0.08,
                        "pearson": 0.99,
                    },
                    "dynamic_recordings_meet_r2_threshold": False,
                    "dynamic_recording_min_r2": dynamic_min_r2,
                    "fit_seconds": 100.0,
                    "recording_count": 7,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "per_recording_metrics.csv").write_text(
            "recording_name,is_dynamic_recording,r2\npar10rec1_processed_1024Hz,True,0.99\n",
            encoding="utf-8",
        )

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_participant_overfit_runs.py"),
        "--output_dir",
        str(output_dir),
        str(run_a),
        str(run_b),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root, capture_output=True, text=True)
    leaderboard = (output_dir / "leaderboard.csv").read_text(encoding="utf-8").splitlines()
    assert "run_b" in leaderboard[1]
