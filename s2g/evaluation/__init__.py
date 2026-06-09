"""
evaluation package — metrics and training callbacks for S2G.
"""

from .callbacks import (
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    StepTrackingCallback,
    load_run_metadata,
)
from .metrics import (
    # Task-specific dispatch (Experiment 1)
    compute_metrics_for_task,
    corpus_ner_boundary_f1,
    corpus_ner_strict_f1,
    corpus_rel_boundary_f1,
    corpus_rel_strict_f1,
    macro_ner_boundary_f1,
    macro_ner_strict_f1,
    macro_rel_boundary_f1,
    macro_rel_strict_f1,
    # Backward-compatible interface (pretrain.py / REBEL)
    compute_metrics,
    corpus_boundary_f1,
    corpus_ner_f1,
    corpus_strict_f1,
)

__all__ = [
    # callbacks
    "GenerateTextSamplesCallback",
    "PeriodicCheckpointCallback",
    "StepTrackingCallback",
    "load_run_metadata",
    # metrics — new interface
    "compute_metrics_for_task",
    "corpus_ner_boundary_f1",
    "corpus_ner_strict_f1",
    "corpus_rel_boundary_f1",
    "corpus_rel_strict_f1",
    "macro_ner_boundary_f1",
    "macro_ner_strict_f1",
    "macro_rel_boundary_f1",
    "macro_rel_strict_f1",
    # metrics — backward-compatible interface
    "compute_metrics",
    "corpus_boundary_f1",
    "corpus_ner_f1",
    "corpus_strict_f1",
]