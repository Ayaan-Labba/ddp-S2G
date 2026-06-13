"""
Evaluation package — metrics and training callbacks for S2G.
"""
from .callbacks import (
    GenerateTextSamplesCallback, PeriodicCheckpointCallback, 
    StepTrackingCallback, S2GEarlyStoppingCallback, load_run_metadata
)
from .metrics import (
    compute_metrics_for_task, corpus_ner_boundary_f1, corpus_ner_strict_f1,
    corpus_rel_boundary_f1, corpus_rel_strict_f1, macro_ner_boundary_f1,
    macro_ner_strict_f1, macro_rel_boundary_f1, macro_rel_strict_f1,
)

__all__ = [
    "GenerateTextSamplesCallback", "PeriodicCheckpointCallback", 
    "StepTrackingCallback", "S2GEarlyStoppingCallback", "load_run_metadata",
    "compute_metrics_for_task", "corpus_ner_boundary_f1", "corpus_ner_strict_f1",
    "corpus_rel_boundary_f1", "corpus_rel_strict_f1", "macro_ner_boundary_f1",
    "macro_ner_strict_f1", "macro_rel_boundary_f1", "macro_rel_strict_f1",
]