"""
S2G Dataset — memory-mapped JSONL reader for training and evaluation.
"""
from __future__ import annotations

import logging
import mmap
from pathlib import Path
from typing import Optional, Union

# Fast-path JSON parser bypasses Python GIL bottlenecks
try:
    import orjson as json
except ImportError:
    import json

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
_SCAN_CHUNK_BYTES = 64 * 1024 * 1024


class S2GDataset(Dataset):
    def __init__(
            self, 
            filepath: Union[str, Path], 
            subset_fraction: Optional[float] = None, 
            seed: int = 0
        ) -> None:
        self._filepath = Path(filepath)
        if not self._filepath.exists(): 
            raise FileNotFoundError(f"Dataset file not found: {self._filepath}")

        logger.info("Indexing %s …", self._filepath)
        self._offsets = _build_offset_index(self._filepath)
        logger.info("Indexed %d instances from %s (offset table: %.1f MB)", 
                    len(self._offsets), self._filepath.name, self._offsets.nbytes / 1e6)

        if subset_fraction and 0.0 < subset_fraction < 1.0:
            n = max(1, int(len(self._offsets) * subset_fraction))
            self._offsets = self._offsets[
                np.sort(np.random.default_rng(seed).choice(len(self._offsets), size=n, replace=False))
            ]
            logger.info("Subsetted to %d instances (%.0f%%)", n, subset_fraction * 100)

        self._file, self._mmap = _open_mmap(self._filepath)

    def __len__(self) -> int: 
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        return json.loads(self._mmap[int(self._offsets[idx, 0]):int(self._offsets[idx, 1])])

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in {"_mmap", "_file"}}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._file, self._mmap = _open_mmap(self._filepath)

    def __del__(self) -> None:
        try: 
            self._mmap.close()
            self._file.close()
        except Exception: 
            pass


def _open_mmap(filepath: Path) -> tuple[object, mmap.mmap]:
    f = open(filepath, "rb")
    return f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def _build_offset_index(filepath: Path) -> np.ndarray:
    """Vectorized offset indexer - runs 10-100x faster than looping over newline bytes."""
    starts, ends = [], []
    chunk_base, line_start = 0, 0

    with open(filepath, "rb") as f:
        while chunk := f.read(_SCAN_CHUNK_BYTES):
            arr = np.frombuffer(chunk, dtype=np.uint8)
            nl_indices = np.where(arr == 10)[0]
            
            if len(nl_indices) > 0:
                nl_abs = chunk_base + nl_indices
                
                # Ends map exactly to the absolute newline positions
                ends.append(nl_abs)
                
                # Starts map to the current running line_start, followed by pos right after previous newlines
                chunk_starts = np.empty_like(nl_abs)
                chunk_starts[0] = line_start
                chunk_starts[1:] = nl_abs[:-1] + 1
                starts.append(chunk_starts)
                
                line_start = nl_abs[-1] + 1
                
            chunk_base += len(chunk)
            
        if line_start < chunk_base:
            starts.append(np.array([line_start], dtype=np.int64))
            ends.append(np.array([chunk_base], dtype=np.int64))

    if not starts:
        return np.empty((0, 2), dtype=np.int64)
        
    return np.column_stack([np.concatenate(starts), np.concatenate(ends)])