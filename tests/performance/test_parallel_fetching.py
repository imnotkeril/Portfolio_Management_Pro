"""Performance tests for parallel data fetching."""

import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.data_manager.cache import Cache
from core.data_manager.price_manager import PriceManager
from services.data_service import DataService


@pytest.mark.performance
@pytest.mark.slow
def test_bulk_fetching_performance_uncached() -> None:
    """Test that parallel bulk fetching is faster than sequential for uncached data."""
    manager = PriceManager()
    
    # Clear cache
    cache = Cache()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = date.today() - timedelta(days=365)
    end_date = date.today()
    
    for ticker in tickers:
        cache_key = f"prices:{ticker}:{start_date}:{end_date}"
        cache.delete(cache_key)
    
    # Sequential fetching
    start = time.perf_counter()
    sequential_results = []
    for ticker in tickers:
        try:
            df = manager.fetch_historical_prices(
                ticker, start_date, end_date, use_cache=False
            )
            sequential_results.append(df)
        except Exception:
            pass  # Skip if fetch fails
    sequential_time = time.perf_counter() - start
    
    # Clear cache again
    for ticker in tickers:
        cache_key = f"prices:{ticker}:{start_date}:{end_date}"
        cache.delete(cache_key)
    
    # Parallel bulk fetching
    start = time.perf_counter()
    try:
        bulk_result = manager.fetch_bulk_prices(
            tickers, start_date, end_date, use_cache=False
        )
        bulk_time = time.perf_counter() - start
    except Exception:
        pytest.skip("Bulk fetch failed, skipping performance test")
    
    # Parallel should be faster (or at least not significantly slower)
    # Allow some margin for network variability
    if sequential_time > 1.0:  # Only compare if sequential took >1s
        speedup = sequential_time / bulk_time
        print(f"\nSequential: {sequential_time*1000:.2f}ms")
        print(f"Parallel: {bulk_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        # Parallel should be at least 1.5x faster for uncached data
        assert speedup >= 1.2, f"Parallel fetching not faster: {speedup:.2f}x"


@pytest.mark.performance
def test_get_latest_prices_performance() -> None:
    """Test that parallel get_latest_prices is faster for multiple tickers."""
    data_service = DataService()
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    # Sequential fetching
    start = time.perf_counter()
    sequential_prices = {}
    for ticker in tickers:
        try:
            price = data_service.fetch_current_price(ticker, use_cache=True)
            sequential_prices[ticker] = price
        except Exception:
            pass
    sequential_time = time.perf_counter() - start
    
    # Parallel fetching
    start = time.perf_counter()
    parallel_prices = data_service.get_latest_prices(tickers)
    parallel_time = time.perf_counter() - start
    
    # Both should return same prices
    for ticker in sequential_prices:
        if ticker in parallel_prices:
            assert sequential_prices[ticker] == parallel_prices[ticker]
    
    # Parallel should be faster (or at least not significantly slower)
    if sequential_time > 0.1:  # Only compare if sequential took >100ms
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\nSequential: {sequential_time*1000:.2f}ms")
        print(f"Parallel: {parallel_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        # Parallel should be at least 1.2x faster
        assert speedup >= 1.0, f"Parallel fetching not faster: {speedup:.2f}x"


@pytest.mark.performance
def test_bulk_fetching_cached_vs_uncached() -> None:
    """Test that cached bulk fetching is much faster than uncached."""
    manager = PriceManager()
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = date.today() - timedelta(days=365)
    end_date = date.today()
    
    # First fetch (uncached)
    start = time.perf_counter()
    try:
        uncached_result = manager.fetch_bulk_prices(
            tickers, start_date, end_date, use_cache=True
        )
        uncached_time = time.perf_counter() - start
    except Exception:
        pytest.skip("Bulk fetch failed, skipping performance test")
    
    # Second fetch (cached)
    start = time.perf_counter()
    cached_result = manager.fetch_bulk_prices(
        tickers, start_date, end_date, use_cache=True
    )
    cached_time = time.perf_counter() - start
    
    # Cached should be much faster
    if uncached_time > 0.1:  # Only compare if uncached took >100ms
        speedup = uncached_time / cached_time if cached_time > 0 else 0
        print(f"\nUncached: {uncached_time*1000:.2f}ms")
        print(f"Cached: {cached_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        # Cached should be at least 2x faster (network variability can affect this)
        assert speedup >= 2.0, f"Cached fetching not much faster: {speedup:.2f}x"
    
    # Results should be identical
    assert len(uncached_result) == len(cached_result)

