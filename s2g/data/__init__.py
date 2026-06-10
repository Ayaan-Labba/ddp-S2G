"""
Data package — dataset loading and batch collation for S2G fine-tuning.
"""
from .collator import S2GCollator
from .dataset  import S2GDataset

__all__ = ["S2GCollator", "S2GDataset"]