"""Caching system for price data and validation results."""

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


class Cache:
    """
    Multi-level caching system: in-memory + disk cache.

    Level 1: In-memory cache with TTL (fast access)
    Level 2: Disk cache (parquet/pickle files) for persistence
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize cache system.

        Args:
            cache_dir: Directory for disk cache
                      (defaults to settings.price_cache_dir)
        """
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_dir = cache_dir or settings.price_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if key in self._memory_cache:
            value, expiry = self._memory_cache[key]
            if time.time() < expiry:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit (memory): {key}")
                return value
            else:
                # Expired, remove from memory
                del self._memory_cache[key]

        # Check disk cache
        disk_path = self._cache_dir / f"{self._sanitize_key(key)}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    cached_data = pickle.load(f)
                    value, expiry_timestamp = cached_data

                    # Check if expired
                    if datetime.now().timestamp() < expiry_timestamp:
                        # Cache in memory for faster access
                        ttl = expiry_timestamp - datetime.now().timestamp()
                        self._memory_cache[key] = (value, time.time() + ttl)
                        self._stats["hits"] += 1
                        logger.debug(f"Cache hit (disk): {key}")
                        return value
                    else:
                        # Expired, remove file
                        disk_path.unlink()
                        logger.debug(f"Cache expired (disk): {key}")

            except Exception as e:
                logger.warning(f"Error reading cache file {disk_path}: {e}")
                # Remove corrupted file
                try:
                    disk_path.unlink()
                except Exception:
                    pass

        self._stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 3600 = 1 hour)
        """
        expiry = time.time() + ttl
        expiry_timestamp = datetime.now().timestamp() + ttl

        # Store in memory
        self._memory_cache[key] = (value, expiry)

        # Store on disk
        disk_path = self._cache_dir / f"{self._sanitize_key(key)}.pkl"
        try:
            with open(disk_path, "wb") as f:
                pickle.dump((value, expiry_timestamp), f)

            self._stats["sets"] += 1
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")

        except Exception as e:
            logger.warning(f"Error writing cache file {disk_path}: {e}")

    def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key to delete
        """
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]

        # Remove from disk
        disk_path = self._cache_dir / f"{self._sanitize_key(key)}.pkl"
        if disk_path.exists():
            try:
                disk_path.unlink()
                logger.debug(f"Cache deleted: {key}")
            except Exception as e:
                logger.warning(f"Error deleting cache file {disk_path}: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()

        # Remove all disk cache files
        try:
            for cache_file in self._cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - sets: Number of cache sets
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - size: Number of items in memory cache
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "hit_rate": hit_rate,
            "size": len(self._memory_cache),
        }

    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize cache key for filesystem use.

        Args:
            key: Original cache key

        Returns:
            Sanitized key safe for use as filename
        """
        # Replace invalid filesystem characters
        sanitized = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = (
            sanitized.replace("*", "_").replace("?", "_").replace('"', "_")
        )
        sanitized = (
            sanitized.replace("<", "_").replace(">", "_").replace("|", "_")
        )
        return sanitized

