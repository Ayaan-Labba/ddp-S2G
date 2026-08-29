"""
DataLoader worker initialisation.

Workers outlive a parent that dies abruptly: ``SIGKILL`` runs no cleanup, so the
OOM killer taking a training process leaves its workers reparented to init, still
holding their memory. Those orphans accumulate across failed runs until the host
has none left, at which point every subsequent run is killed on sight — a ratchet
where each crash makes the next one more likely.
"""
from __future__ import annotations

import ctypes
import logging
import signal
import sys

logger = logging.getLogger(__name__)

PR_SET_PDEATHSIG = 1


def set_parent_death_signal(worker_id: int = 0) -> None:
    """
    Ask the kernel to ``SIGKILL`` this worker when its parent dies.

    Linux-only (``prctl(PR_SET_PDEATHSIG)``); a no-op everywhere else. Must be a
    module-level function so it stays picklable for the ``forkserver`` and
    ``spawn`` start methods.
    """
    if not sys.platform.startswith('linux'):
        return

    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:                                   # pragma: no cover - platform dependent
        logger.debug("Could not set PR_SET_PDEATHSIG for worker %s.", worker_id, exc_info=True)


def attach_parent_death_signal(dataloader):
    """
    Install :func:`set_parent_death_signal` on a DataLoader, unless it already has
    an initialiser of its own.

    Set after construction rather than passed in, because ``Trainer`` builds its
    own loaders. ``worker_init_fn`` is read when the iterator is created, not when
    the DataLoader is, so this still takes effect.
    """
    if getattr(dataloader, 'num_workers', 0) and dataloader.worker_init_fn is None:
        dataloader.worker_init_fn = set_parent_death_signal
    return dataloader
