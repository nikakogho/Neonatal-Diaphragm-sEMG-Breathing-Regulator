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
from .cohort_overfit import (
    CohortScalarWindowDataset,
    build_cohort_shuffle_audit_rows,
    run_cohort_overfit_experiment,
)
from .participant_prediction import (
    ParticipantPredictionDataset,
    build_prediction_shuffle_audit_rows,
    make_leave_one_recording_out_folds,
    run_participant_prediction_stage,
)

__all__ = [
    "compute_scalar_metrics",
    "build_shuffle_audit_rows",
    "build_cohort_shuffle_audit_rows",
    "build_prediction_shuffle_audit_rows",
    "make_single_recording_split",
    "make_epoch_order",
    "make_leave_one_recording_out_folds",
    "CohortScalarWindowDataset",
    "ParticipantPredictionDataset",
    "ParticipantScalarWindowDataset",
    "run_cohort_overfit_experiment",
    "run_participant_prediction_stage",
    "refresh_single_recording_summary",
    "run_participant_overfit_experiment",
    "run_single_recording_experiment",
    "run_single_recording_sweep",
    "SingleRecordingScalarDataset",
]
