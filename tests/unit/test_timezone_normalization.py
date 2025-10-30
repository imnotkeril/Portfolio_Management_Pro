import pandas as pd
from datetime import date

from services.analytics_service import AnalyticsService


class _FakeDataService:
    def __init__(self, frames_by_ticker):
        self._frames_by_ticker = frames_by_ticker

    def fetch_historical_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        # Return prebuilt frames ignoring dates for unit-level behavior
        return self._frames_by_ticker.get(ticker, pd.DataFrame())


def test_mixed_timezone_dates_are_normalized_and_filtered():
    # Build two small frames: one tz-naive, one tz-aware (UTC)
    dates_naive = pd.to_datetime(["2025-01-02", "2025-01-03"])  # naive
    df_naive = pd.DataFrame({
        "Date": dates_naive,
        "Adjusted_Close": [100.0, 101.0],
    })

    dates_aware = pd.to_datetime([
        "2025-01-02",
        "2025-01-03",
    ]).tz_localize("UTC")
    df_aware = pd.DataFrame({
        "Date": dates_aware,
        "Adjusted_Close": [200.0, 202.0],
    })

    fake_ds = _FakeDataService({
        "AAA": df_naive.copy(),
        "BBB": df_aware.copy(),
    })

    svc = AnalyticsService(data_service=fake_ds)

    start = date(2025, 1, 2)
    end = date(2025, 1, 3)

    # Should not raise and should return tz-naive index inside pivot result
    pivot = svc._fetch_portfolio_prices([
        "AAA",
        "BBB",
    ], start, end)
    assert not pivot.empty

    # Index must be DatetimeIndex and tz-naive
    assert isinstance(pivot.index, pd.DatetimeIndex)
    assert pivot.index.tz is None

    # Ensure filtering applied inclusively
    assert pivot.index.min().date() >= start
    assert pivot.index.max().date() <= end

