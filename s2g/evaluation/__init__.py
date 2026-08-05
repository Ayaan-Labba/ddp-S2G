"""
Evaluation package — metrics and training callbacks for S2G.
"""
from .metrics import compute_metrics_for_variant
from.evaluator import S2GEvaluator

__all__ = ['compute_metrics_for_variant', 'S2GEvaluator']