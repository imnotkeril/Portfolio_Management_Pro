"""S-03: request size fields are bounded (422 before any computation)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

_BASE = {
    "portfolio_id": "p1",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
}


def _headers(client: TestClient) -> dict[str, str]:
    email = f"limits-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "securepass1"})
    token = client.post(
        "/auth/login/json", json={"email": email, "password": "securepass1"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "path",
    ["/risk/monte-carlo", "/risk/monte-carlo/full"],
)
@pytest.mark.parametrize(
    "extra",
    [
        {"num_simulations": 1_500_000, "time_horizon": 252},
        {"num_simulations": 10_000, "time_horizon": 100_000},
        {"num_simulations": 50_000, "time_horizon": 2_000},  # product cap
        {"num_simulations": 0},
        {"time_horizon": -5},
        {"time_horizon": 0},
    ],
)
def test_monte_carlo_rejects_oversized(
    api_client: TestClient, path: str, extra: dict
) -> None:
    resp = api_client.post(path, headers=_headers(api_client), json={**_BASE, **extra})
    assert resp.status_code == 422, resp.text


_VAR_CASES = [
    (path, extra)
    for path in ("/risk/var", "/risk/var/full")
    for extra in (
        {"num_simulations": 5_000_000},
        {"time_horizon": 100_000},
        {"rolling_window": 10_000_000},
        {"confidence_level": 5},
    )
    # rolling_window exists only on the full request
    if not ("rolling_window" in extra and not path.endswith("/full"))
]


@pytest.mark.parametrize(("path", "extra"), _VAR_CASES)
def test_var_rejects_oversized(api_client: TestClient, path: str, extra: dict) -> None:
    resp = api_client.post(path, headers=_headers(api_client), json={**_BASE, **extra})
    assert resp.status_code == 422, resp.text


def test_validate_tickers_rejects_long_list_and_long_symbols(
    api_client: TestClient,
) -> None:
    too_many = [f"T{i}" for i in range(101)]
    assert api_client.post("/validate-tickers", json=too_many).status_code == 422
    assert api_client.post("/validate-tickers", json=["A" * 50]).status_code == 422


def test_forecast_and_frontier_bounds(api_client: TestClient) -> None:
    headers = _headers(api_client)
    body = {**_BASE, "method": "arima", "horizon": 100_000}
    for path in ("/forecasting/portfolio", "/forecasting/asset"):
        payload = dict(body)
        if path.endswith("asset"):
            payload["ticker"] = "AAPL"
        resp = api_client.post(path, headers=headers, json=payload)
        # 403 (not Pro) is fine only if validation did not run first; the
        # bound must hold regardless of plan, so accept 422 or 403 here and
        # assert the model itself below.
        assert resp.status_code in (403, 422)

    from pydantic import ValidationError

    from api.main import ForecastPortfolioRequest, OptimizationFullRequest

    with pytest.raises(ValidationError):
        ForecastPortfolioRequest(**_BASE, method="arima", horizon=100_000)
    with pytest.raises(ValidationError):
        OptimizationFullRequest(**_BASE, method="max_sharpe", frontier_n_points=10**6)
