"""
Data package — dataset loading and batch collation for S2G fine-tuning.
"""
from .collator import S2GCollator
from .dataset  import S2GDataset
from .worker_init import attach_parent_death_signal, set_parent_death_signal

__all__ = [
    'S2GCollator', 'S2GDataset',
    'attach_parent_death_signal', 'set_parent_death_signal',
]