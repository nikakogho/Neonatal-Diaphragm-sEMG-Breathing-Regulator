"""
Single-recording training exports.
"""

from .single_recording import (
    SingleRecordingScalarDataset,
    compute_scalar_metrics,
    make_single_recording_split,
    refresh_single_recording_summary,
    run_single_recording_experiment,
    run_single_recording_sweep,
)
from .participant_overfit import (
    ParticipantScalarWindowDataset,
    build_shuffle_audit_rows,
    make_epoch_order,
    run_participant_overfit_experiment,
)

__all__ = [
    "compute_scalar_metrics",
    "build_shuffle_audit_rows",
    "make_single_recording_split",
    "make_epoch_order",
    "ParticipantScalarWindowDataset",
    "refresh_single_recording_summary",
    "run_participant_overfit_experiment",
    "run_single_recording_experiment",
    "run_single_recording_sweep",
    "SingleRecordingScalarDataset",
]
