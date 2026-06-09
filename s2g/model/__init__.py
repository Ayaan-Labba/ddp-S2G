"""
model package — S2G model wrapper and constraint decoder.
"""

from .constraint_decoder import build_constraint_processor, ConstraintDecodingProcessor
from .model import S2GModel

__all__ = [
    "build_constraint_processor",
    "ConstraintDecodingProcessor",
    "S2GModel",
]