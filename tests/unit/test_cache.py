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

