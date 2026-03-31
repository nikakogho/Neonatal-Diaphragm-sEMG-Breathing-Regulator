"""
Tests for participant held-out recording prediction workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import WindowSpec, build_window_index, load_recording
from src.training import make_leave_one_recording_out_folds, run_participant_prediction_stage
from src.training.participant_prediction import rank_prediction_run_rows, summarize_prediction_run_row


def _write_recording(path: Path, fs: float = 100.0, phase: float = 0.0, scale: float = 1.0) -> None:
    n_t = 260
    t = np.arange(n_t, dtype=np.float32) / fs
    emg = np.zeros((n_t, 6, 8, 8), dtype=np.float32)
    carrier = (scale * np.sin(2.0 * np.pi * 1.5 * t + phase)).astype(np.float32)
    for grid in range(6):
        emg[:, grid] = carrier[:, None, None] * (1.0 + 0.03 * grid)
    aux0 = 0.9 * carrier + 2.0 + 0.15 * phase
    aux1 = 0.2 * np.cos(2.0 * np.pi * 0.8 * t + phase) + 3.0
    aux = np.stack([aux0, aux1], axis=1).astype(np.float32)
    bad_mask = np.zeros((6, 8, 8), dtype=bool)
    meta = json.dumps({"fs_export_hz": fs})
    np.savez(path, emg=emg, aux=aux, bad_mask=bad_mask, time_s=t, meta=meta)


def _write_participant4_dir(root: Path) -> Path:
    participant_dir = root / "participant 4 stuff"
    participant_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("p4rec1_processed_1024Hz.npz", 0.0, 0.6),
        ("p4rec2_processed_1024Hz.npz", 0.4, 1.0),
        ("p4rec3_processed_1024Hz.npz", 0.8, 1.1),
    ]
    for name, phase, scale in specs:
        _write_recording(participant_dir / name, phase=phase, scale=scale)
    return participant_dir


def test_leave_one_recording_out_excludes_holdout_from_train_val_and_normalization(tmp_path: Path):
    participant_dir = _write_participant4_dir(tmp_path)
    recordings = [load_recording(str(path)) for path in sorted(participant_dir.glob("*.npz"))]
    spec = WindowSpec(win_ms=100, step_ms=50, delta_ms=0)
    folds = make_leave_one_recording_out_folds(recordings, spec, val_fraction=0.2, target_channel=0)
    assert len(folds) == 3
    for fold in folds:
        holdout_path = str(fold["holdout_recording"].path)
        assert all(str(entry.path) != holdout_path for entry in fold["train_entries"])
        assert all(str(entry.path) != holdout_path for entry in fold["val_entries"])
        assert all(str(entry.path) == holdout_path for entry in fold["test_entries"])
        train_recordings = fold["train_recordings"]
        holdout_aux = fold["holdout_recording"].aux[:, 0]
        train_y = np.concatenate([recording.aux[:, 0] for recording in train_recordings], axis=0)
        assert not np.isclose(train_y.mean(), holdout_aux.mean())
        assert np.isclose(float(fold["norm_stats"]["y_mean"]), float(train_y.mean()), atol=1e-6)


def test_leave_one_recording_out_validation_uses_last_windows_from_training_recordings(tmp_path: Path):
    participant_dir = _write_participant4_dir(tmp_path)
    recordings = [load_recording(str(path)) for path in sorted(participant_dir.glob("*.npz"))]
    spec = WindowSpec(win_ms=100, step_ms=50, delta_ms=0)
    folds = make_leave_one_recording_out_folds(recordings, spec, val_fraction=0.2, target_channel=0)
    fold = next(item for item in folds if item["holdout_recording"].recording_name == "p4rec3_processed_1024Hz")
    train_paths = {recording.path for recording in fold["train_recordings"]}
    for recording in fold["train_recordings"]:
        all_entries = [entry for entry in build_window_index([recording], spec)]
        val_entries = [entry for entry in fold["val_entries"] if entry.path == recording.path]
        if not val_entries:
            continue
        assert all(entry.path in train_paths for entry in val_entries)
        assert int(val_entries[0].window_start) >= int(all_entries[-len(val_entries)].window_start)


def test_prediction_stage_writes_selected_model_locations_and_fold_artifacts(tmp_path: Path):
    participant_dir = _write_participant4_dir(tmp_path)
    run_dir = tmp_path / "prediction_run"
    result = run_participant_prediction_stage(
        recordings_dir=str(participant_dir),
        run_dir=run_dir,
        device="cpu",
        val_fraction=0.2,
        patience=1,
        min_epochs=1,
        max_epochs=2,
        update_root_readme=False,
    )
    assert "metrics" in result
    for filename in [
        "README.md",
        "summary.md",
        "config.json",
        "runtime.json",
        "selected_config.json",
        "selected_model_locations.csv",
        "aggregate_metrics.json",
        "fold_metrics.csv",
        "aggregate_predictions.csv",
        "aggregate_predictions.npz",
        "holdout_r2_rmse.png",
        "aggregate_holdout_scatter.png",
        "holdout_timeseries_grid.png",
    ]:
        assert (run_dir / filename).exists()
    selected_run = run_dir / "selected_run"
    assert (selected_run / "README.md").exists()
    assert (selected_run / "model_locations.csv").exists()
    fold_dirs = list((selected_run / "folds").iterdir())
    assert fold_dirs
    for fold_dir in fold_dirs:
        for filename in [
            "config.json",
            "metrics.json",
            "history.csv",
            "history.json",
            "model_best.pt",
            "model_last.pt",
            "predictions.csv",
            "predictions.npz",
            "loss_curve.png",
            "holdout_scatter.png",
            "holdout_timeseries.png",
            "residual_hist.png",
            "residual_vs_true.png",
        ]:
            assert (fold_dir / filename).exists()


def test_prediction_ranking_uses_dynamic_median_r2_then_positive_count_then_pearson_then_rmse():
    ranked = rank_prediction_run_rows(
        [
            {
                "config_name": "baseline",
                "dynamic_median_r2": 0.10,
                "dynamic_positive_r2_count": 5,
                "dynamic_mean_pearson": 0.70,
                "weighted_rmse": 0.40,
            },
            {
                "config_name": "candidate_a",
                "dynamic_median_r2": 0.20,
                "dynamic_positive_r2_count": 3,
                "dynamic_mean_pearson": 0.60,
                "weighted_rmse": 0.50,
            },
            {
                "config_name": "candidate_b",
                "dynamic_median_r2": 0.20,
                "dynamic_positive_r2_count": 4,
                "dynamic_mean_pearson": 0.55,
                "weighted_rmse": 0.60,
            },
        ]
    )
    assert [row["config_name"] for row in ranked] == ["candidate_b", "candidate_a", "baseline"]


def test_prediction_summary_parses_csv_bool_strings_for_dynamic_rows(tmp_path: Path):
    config = {
        "model": "cnn1d_feature_scalar",
        "win_ms": 300,
        "delta_ms": -100,
        "step_ms": 50,
        "max_epochs": 60,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "base_channels": 32,
    }
    aggregate_metrics = {
        "weighted_all_holdouts": {
            "r2": -1.0,
            "rmse": 1.0,
            "mae": 1.0,
            "pearson": 0.5,
        },
        "outcome_label": "mixed",
    }
    fold_rows = [
        {"holdout_recording": "p4rec2", "is_dynamic_recording": "True", "r2": "0.4", "pearson": "0.8"},
        {"holdout_recording": "p4rec3", "is_dynamic_recording": "True", "r2": "0.2", "pearson": "0.6"},
        {"holdout_recording": "p4rec1", "is_dynamic_recording": "False", "r2": "-20.0", "pearson": "0.1"},
    ]
    row = summarize_prediction_run_row(config, aggregate_metrics, fold_rows, tmp_path / "run", fixed_first=False)
    assert np.isclose(float(row["dynamic_median_r2"]), 0.3)
    assert int(row["dynamic_positive_r2_count"]) == 2
