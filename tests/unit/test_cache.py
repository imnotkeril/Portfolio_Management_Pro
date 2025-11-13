"""Unit tests for cache system."""

import time
from pathlib import Path

import pytest

from core.data_manager.cache import Cache


def test_cache_set_and_get(temp_cache_dir: Path) -> None:
    """Test basic cache set and get operations."""
    cache = Cache(cache_dir=temp_cache_dir)

    # Set value
    cache.set("test_key", "test_value", ttl=3600)

    # Get value
    value = cache.get("test_key")
    assert value == "test_value"


def test_cache_expiration(temp_cache_dir: Path) -> None:
    """Test cache expiration."""
    cache = Cache(cache_dir=temp_cache_dir)

    # Set value with short TTL
    cache.set("expires_soon", "value", ttl=1)

    # Should be available immediately
    assert cache.get("expires_soon") == "value"

    # Wait for expiration
    time.sleep(2)

    # Should be expired
    assert cache.get("expires_soon") is None


def test_cache_delete(temp_cache_dir: Path) -> None:
    """Test cache deletion."""
    cache = Cache(cache_dir=temp_cache_dir)

    cache.set("to_delete", "value")
    assert cache.get("to_delete") == "value"

    cache.delete("to_delete")
    assert cache.get("to_delete") is None


def test_cache_clear(temp_cache_dir: Path) -> None:
    """Test cache clearing."""
    cache = Cache(cache_dir=temp_cache_dir)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_stats(temp_cache_dir: Path) -> None:
    """Test cache statistics."""
    cache = Cache(cache_dir=temp_cache_dir)

    # Initial stats
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["sets"] == 0
    assert stats["hit_rate"] == 0.0

    # Set and get
    cache.set("key1", "value1")
    cache.get("key1")

    stats = cache.get_stats()
    assert stats["sets"] == 1
    assert stats["hits"] == 1
    assert stats["hit_rate"] == 1.0


def test_cache_sanitize_key() -> None:
    """Test key sanitization."""
    cache = Cache()

    # Test various problematic characters
    problematic_key = "test/key:with*chars?"
    sanitized = cache._sanitize_key(problematic_key)
    assert "/" not in sanitized
    assert ":" not in sanitized
    assert "*" not in sanitized
    assert "?" not in sanitized


def test_cache_disk_persistence(temp_cache_dir: Path) -> None:
    """Test that cache persists to disk."""
    cache1 = Cache(cache_dir=temp_cache_dir)
    cache1.set("persistent_key", "persistent_value", ttl=3600)

    # Create new cache instance - should load from disk
    cache2 = Cache(cache_dir=temp_cache_dir)
    value = cache2.get("persistent_key")

    assert value == "persistent_value"


def test_cache_miss(temp_cache_dir: Path) -> None:
    """Test cache miss behavior."""
    cache = Cache(cache_dir=temp_cache_dir)

    value = cache.get("non_existent_key")
    assert value is None

    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0


def test_cache_hit_rate_calculation(temp_cache_dir: Path) -> None:
    """Test cache hit rate calculation."""
    cache = Cache(cache_dir=temp_cache_dir)

    # Set and get multiple times
    cache.set("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.01)


def test_cache_expired_disk_entry(temp_cache_dir: Path) -> None:
    """Test that expired disk cache entries are removed."""
    import time
    cache = Cache(cache_dir=temp_cache_dir)

    # Set with very short TTL
    cache.set("expires_soon", "value", ttl=1)
    
    # Wait for expiration
    time.sleep(2)

    # Should be None and file should be removed
    value = cache.get("expires_soon")
    assert value is None

    # Check that file was removed
    disk_path = temp_cache_dir / "expires_soon.pkl"
    assert not disk_path.exists()
