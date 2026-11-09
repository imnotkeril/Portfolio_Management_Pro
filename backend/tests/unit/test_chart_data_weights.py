"""Weights for asset impact charts use end-of-window market values."""

import pandas as pd
import pytest

from core.analytics_engine.chart_data import _market_value_weights


class _Pos:
    def __init__(self, ticker: str, shares: float) -> None:
        self.ticker = ticker
        self.shares = shares


def test_market_value_weights_use_last_price_not_first() -> None:
    """Rising stock prices must not inflate cash weight vs market allocation."""
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 200.0],
            "CASH": [1.0, 1.0],
        },
        index=pd.to_datetime(["2020-01-02", "2024-01-02"]),
    )
    positions = [_Pos("AAA", 10.0), _Pos("CASH", 40_000.0)]
    weights = _market_value_weights(positions, prices)
    assert weights is not None
    assert weights["CASH"] == pytest.approx(40_000 / (40_000 + 2_000), rel=1e-4)
    assert weights["AAA"] == pytest.approx(2_000 / (40_000 + 2_000), rel=1e-4)
