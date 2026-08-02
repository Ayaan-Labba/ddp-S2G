"""
Evaluation package — metrics and training callbacks for S2G.
"""
from .callbacks import (
    GenerateTextSamplesCallback, PeriodicCheckpointCallback, 
    StepTrackingCallback, S2GEarlyStoppingCallback, load_run_metadata
)
from .metrics import (
    compute_metrics_for_variant
)
from.evaluator import S2GEvaluator

__all__ = [
    "GenerateTextSamplesCallback", "PeriodicCheckpointCallback", 
    "StepTrackingCallback", "S2GEarlyStoppingCallback", "load_run_metadata",
    "compute_metrics_for_variant", "S2GEvaluator"
]