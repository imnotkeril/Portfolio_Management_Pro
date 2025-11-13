"""Performance profiling script for Phase 9 optimization.

This script profiles key operations to identify bottlenecks:
- Portfolio creation
- Data fetching (cached/uncached)
- Metrics calculation
- Optimization
"""

import cProfile
import pstats
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analytics_engine.engine import AnalyticsEngine
from core.data_manager.portfolio import Portfolio
from services.analytics_service import AnalyticsService
from services.data_service import DataService
from services.portfolio_service import PortfolioService


def profile_portfolio_creation(iterations: int = 10) -> Dict[str, float]:
    """Profile portfolio creation performance."""
    print("\n=== Profiling Portfolio Creation ===")
    
    portfolio_service = PortfolioService()
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        
        portfolio = Portfolio(
            name=f"Test Portfolio {i}",
            starting_capital=100000.0,
        )
        portfolio.add_position(ticker="AAPL", shares=100.0)
        portfolio.add_position(ticker="MSFT", shares=50.0)
        
        portfolio_service.create_portfolio(portfolio)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print(f"  Target: <100ms | Acceptable: <500ms")
    
    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
    }


def profile_data_fetching_cached(iterations: int = 10) -> Dict[str, float]:
    """Profile cached data fetching performance."""
    print("\n=== Profiling Data Fetching (Cached) ===")
    
    data_service = DataService()
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    # First fetch to populate cache
    data_service.fetch_historical_prices("AAPL", start_date, end_date, use_cache=True)
    
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        
        prices = data_service.fetch_historical_prices(
            "AAPL", start_date, end_date, use_cache=True
        )
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms ({len(prices)} rows)")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print(f"  Target: <10ms | Acceptable: <50ms")
    
    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
    }


def profile_metrics_calculation(iterations: int = 5) -> Dict[str, float]:
    """Profile metrics calculation performance."""
    print("\n=== Profiling Metrics Calculation ===")
    
    portfolio_service = PortfolioService()
    analytics_service = AnalyticsService(
        portfolio_service=portfolio_service,
    )
    
    # Create test portfolio
    portfolio = Portfolio(
        name="Performance Test Portfolio",
        starting_capital=100000.0,
    )
    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.add_position(ticker="MSFT", shares=50.0)
    portfolio.add_position(ticker="GOOGL", shares=30.0)
    
    try:
        created_portfolio = portfolio_service.create_portfolio(portfolio)
        portfolio_id = created_portfolio.id
    except Exception as e:
        # Portfolio might already exist, try to find it
        portfolios = portfolio_service.list_portfolios()
        test_portfolio = next((p for p in portfolios if p.name == "Performance Test Portfolio"), None)
        if test_portfolio:
            portfolio_id = test_portfolio.id
        else:
            raise e
    
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        
        metrics = analytics_service.calculate_portfolio_metrics(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
        )
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print(f"  Target: <500ms | Acceptable: <1000ms")
    
    # Cleanup
    portfolio_service.delete_portfolio(portfolio_id)
    
    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
    }


def profile_bulk_data_fetching() -> Dict[str, float]:
    """Profile bulk data fetching performance."""
    print("\n=== Profiling Bulk Data Fetching ===")
    
    data_service = DataService()
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    # Clear cache for uncached test
    print("  Testing with CACHED data:")
    # Sequential fetching (cached)
    print("    Sequential fetching:")
    start = time.perf_counter()
    for ticker in tickers:
        data_service.fetch_historical_prices(
            ticker, start_date, end_date, use_cache=True
        )
    sequential_cached = time.perf_counter() - start
    print(f"      Time: {sequential_cached*1000:.2f}ms")
    
    # Bulk fetching (cached)
    print("    Bulk fetching:")
    start = time.perf_counter()
    bulk_data = data_service.fetch_bulk_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )
    bulk_cached = time.perf_counter() - start
    print(f"      Time: {bulk_cached*1000:.2f}ms")
    print(f"      Rows: {len(bulk_data)}")
    
    speedup_cached = sequential_cached / bulk_cached if bulk_cached > 0 else 0
    print(f"    Speedup: {speedup_cached:.2f}x")
    
    # Test with uncached data (clear cache first)
    print("\n  Testing with UNCACHED data (parallel fetching):")
    from core.data_manager.cache import Cache
    cache = Cache()
    for ticker in tickers:
        cache_key = f"prices:{ticker}:{start_date}:{end_date}"
        cache.delete(cache_key)
    
    # Sequential fetching (uncached)
    print("    Sequential fetching:")
    start = time.perf_counter()
    for ticker in tickers:
        data_service.fetch_historical_prices(
            ticker, start_date, end_date, use_cache=False
        )
    sequential_uncached = time.perf_counter() - start
    print(f"      Time: {sequential_uncached*1000:.2f}ms")
    
    # Clear cache again
    for ticker in tickers:
        cache_key = f"prices:{ticker}:{start_date}:{end_date}"
        cache.delete(cache_key)
    
    # Bulk fetching (uncached - should use parallel)
    print("    Bulk fetching (parallel):")
    start = time.perf_counter()
    bulk_data_uncached = data_service.fetch_bulk_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=False,
    )
    bulk_uncached = time.perf_counter() - start
    print(f"      Time: {bulk_uncached*1000:.2f}ms")
    print(f"      Rows: {len(bulk_data_uncached)}")
    
    speedup_uncached = sequential_uncached / bulk_uncached if bulk_uncached > 0 else 0
    print(f"    Speedup: {speedup_uncached:.2f}x")
    
    return {
        "sequential_cached": sequential_cached,
        "bulk_cached": bulk_cached,
        "speedup_cached": speedup_cached,
        "sequential_uncached": sequential_uncached,
        "bulk_uncached": bulk_uncached,
        "speedup_uncached": speedup_uncached,
    }


def profile_analytics_engine_directly() -> Dict[str, float]:
    """Profile analytics engine directly (without service overhead)."""
    print("\n=== Profiling Analytics Engine (Direct) ===")
    
    engine = AnalyticsEngine()
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range(start=date.today() - timedelta(days=365), periods=252, freq="D")
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    times = []
    
    for i in range(10):
        start = time.perf_counter()
        
        metrics = engine.calculate_all_metrics(
            portfolio_returns=returns,
            start_date=date.today() - timedelta(days=365),
            end_date=date.today(),
        )
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print(f"  Target: <500ms | Acceptable: <1000ms")
    
    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
    }


def main():
    """Run all performance profiles."""
    print("=" * 60)
    print("PERFORMANCE PROFILING - Phase 9 Optimization")
    print("=" * 60)
    
    results = {}
    
    try:
        results["portfolio_creation"] = profile_portfolio_creation(iterations=5)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    try:
        results["data_fetching_cached"] = profile_data_fetching_cached(iterations=10)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    try:
        results["bulk_fetching"] = profile_bulk_data_fetching()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    try:
        results["metrics_calculation"] = profile_metrics_calculation(iterations=3)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    try:
        results["analytics_engine"] = profile_analytics_engine_directly()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for operation, metrics in results.items():
        print(f"\n{operation.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if value < 1:
                    print(f"  {metric}: {value*1000:.2f}ms")
                else:
                    print(f"  {metric}: {value:.2f}s")
            else:
                print(f"  {metric}: {value}")
    
    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()

