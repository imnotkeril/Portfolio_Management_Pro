"""Per-portfolio locks to prevent concurrent ledger/position sync races."""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager

_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
_meta = threading.Lock()


@contextmanager
def portfolio_maintenance_lock(portfolio_id: str):
    """Serialize maintain and position sync for one portfolio."""
    with _meta:
        lock = _locks[portfolio_id]
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
