"""S-03 remainder: bounded Monte Carlo response, lower work cap, concurrency cap."""

from __future__ import annotations

import json
import uuid
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main
from services.risk_service import RiskService

BASE = {"portfolio_id": "p1", "start_date": "2024-01-01", "end_date": "2024-12-31"}


def _headers(client: TestClient) -> dict[str, str]:
    email = f"mc-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "securepass1"})
    tok = client.post(
        "/auth/login/json", json={"email": email, "password": "securepass1"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def test_service_returns_at_most_100_sample_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 250))
    svc = RiskService()
    monkeypatch.setattr(svc, "_get_portfolio_returns", lambda *a, **k: returns)
    out = svc.run_monte_carlo_simulation(
        "p1", date(2024, 1, 1), date(2024, 12, 31), 252, 5000, 1.0, "gbm"
    )
    paths = out["simulated_paths"]
    assert 0 < len(paths) <= 100
    assert len(out["final_values"]) == 5000
    env = out["path_envelope"]
    assert len(env["max"]) == len(env["min"]) == len(env["median"]) == len(paths[0])
    assert len(json.dumps(out)) < 3_000_000  # was ~376 MB at the old caps
    assert "percentiles" in out and "statistics" in out


def test_work_cap_is_two_million_cells(api_client: TestClient) -> None:
    h = _headers(api_client)
    over = {**BASE, "num_simulations": 10_000, "time_horizon": 250}  # 2.5M
    for path in ("/risk/monte-carlo", "/risk/monte-carlo/full"):
        assert api_client.post(path, headers=h, json=over).status_code == 422
    assert main.MAX_SIMULATED_CELLS == 2_000_000


def test_default_request_is_within_cap() -> None:
    main.RiskMonteCarloRequest(**BASE)  # defaults must validate


def test_busy_server_returns_503(api_client: TestClient) -> None:
    h = _headers(api_client)
    held = 0
    try:
        while main._MC_SLOTS.acquire(blocking=False):
            held += 1
        r = api_client.post("/risk/monte-carlo", headers=h, json=BASE)
        assert r.status_code == 503
        assert r.headers.get("retry-after")
    finally:
        for _ in range(held):
            main._MC_SLOTS.release()


def test_display_bundle_keeps_paths_chart_data_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.risk_ui_bundle import build_monte_carlo_display_bundle

    rng = np.random.default_rng(1)
    svc = RiskService()
    monkeypatch.setattr(
        svc,
        "_get_portfolio_returns",
        lambda *a, **k: pd.Series(rng.normal(0, 0.01, 250)),
    )
    out = build_monte_carlo_display_bundle(
        svc,
        portfolio_id="p1",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        time_horizon=60,
        num_simulations=3000,
        initial_value=1.0,
        model="gbm",
        include_sample_paths=True,
    )
    env = out["path_envelope"]
    assert len(env["max"]) == len(env["min"]) == len(env["median"]) > 1
    assert 0 < len(out["sample_paths"]) <= 40
    assert len(out["simulated_paths"]) <= 100
    assert "Simulation Paths Analysis" in out["interpretation_paths"]
