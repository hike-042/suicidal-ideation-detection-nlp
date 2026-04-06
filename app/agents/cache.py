"""
cache.py
========
Token-free result cache for the agent orchestrator.

A cache hit costs zero API tokens. Even a 20% hit rate on repeated or
near-identical posts cuts effective token spend significantly.

Strategy
--------
- Key  : MD5 hash of the lowercased, whitespace-normalised text
- Store: in-process dict (survives the request lifetime of the process)
- TTL  : configurable (default 1 hour) — results expire so the cache
         doesn't serve stale data indefinitely
- Size : LRU eviction when max_size is reached (oldest entry removed)
"""

import hashlib
import time
import re
from collections import OrderedDict
from typing import Optional


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace for stable cache keying."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _make_key(text: str) -> str:
    return hashlib.md5(_normalise(text).encode("utf-8")).hexdigest()


class ResultCache:
    """
    Thread-safe (GIL-protected for CPython) LRU cache with TTL.

    Parameters
    ----------
    max_size : int
        Maximum number of cached results (default 512).
    ttl_seconds : int
        Time-to-live in seconds (default 3600 = 1 hour).
        Pass 0 to disable expiry.
    """

    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[dict]:
        """
        Return cached result for *text* or None if absent / expired.
        Moves the entry to the end (most-recently-used) on a hit.
        """
        key = _make_key(text)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if self.ttl > 0 and (time.time() - entry["_cached_at"]) > self.ttl:
            del self._store[key]
            self._misses += 1
            return None

        # Move to end (LRU)
        self._store.move_to_end(key)
        self._hits += 1

        # Return a copy tagged as a cache hit
        result = dict(entry)
        result["cache_hit"] = True
        return result

    def set(self, text: str, result: dict) -> None:
        """Store *result* under the key derived from *text*."""
        key = _make_key(text)

        # Evict oldest if full
        if len(self._store) >= self.max_size and key not in self._store:
            self._store.popitem(last=False)

        entry = dict(result)
        entry["_cached_at"] = time.time()
        self._store[key] = entry
        self._store.move_to_end(key)

    def invalidate(self, text: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        key = _make_key(text)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "cached_entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }

    def __repr__(self) -> str:
        return (
            f"ResultCache(size={self.size}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.1%}, ttl={self.ttl}s)"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all requests in a process
# ---------------------------------------------------------------------------

_cache = ResultCache(max_size=512, ttl_seconds=3600)


def get_cache() -> ResultCache:
    """Return the process-level cache singleton."""
    return _cache
