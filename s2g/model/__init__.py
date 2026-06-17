"""
Model package — constraint decoder for FSM-constrained generation.
"""
from .constraint_decoder import build_constraint_processor, ConstraintDecodingProcessor

__all__ = ["build_constraint_processor", "ConstraintDecodingProcessor"]