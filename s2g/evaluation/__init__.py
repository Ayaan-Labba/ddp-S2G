"""
Evaluation package — metrics and training callbacks for S2G.
"""
from .metrics import compute_metrics_for_variant, score_bundles
from .gold import build_gold_blocks, build_gold_offsets
from .offsets import OffsetResolver, project_blocks
from .evaluator import S2GEvaluator

__all__ = [
    'compute_metrics_for_variant', 'score_bundles',
    'build_gold_blocks', 'build_gold_offsets',
    'OffsetResolver', 'project_blocks',
    'S2GEvaluator',
]