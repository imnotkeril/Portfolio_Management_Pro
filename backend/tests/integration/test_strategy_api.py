"""Integration tests for Phase 5 strategy API."""

import uuid

from fastapi.testclient import TestClient


def _register_and_headers(client: TestClient) -> dict[str, str]:
    email = f"st-{uuid.uuid4().hex[:8]}@example.com"
    password = "securepass1"
    client.post("/auth/register", json={"email": email, "password": password})
    login = client.post(
        "/auth/login/json",
        json={"email": email, "password": password},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_portfolio(client: TestClient, headers: dict[str, str]) -> str:
    resp = client.post(
        "/portfolios",
        json={
            "name": f"S-{uuid.uuid4().hex[:6]}",
            "starting_capital": 100000.0,
            "rebalance_interval_months": 3,
            "positions": [
                {"ticker": "AAPL", "shares": 10.0, "weight_target": 0.6},
                {"ticker": "MSFT", "shares": 5.0, "weight_target": 0.3},
                {"ticker": "CASH", "shares": 10000.0, "weight_target": 0.1},
            ],
        },
        headers=headers,
    )
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def test_strategy_get_put_and_auth(api_client: TestClient) -> None:
    headers_a = _register_and_headers(api_client)
    pid = _create_portfolio(api_client, headers_a)

    get_resp = api_client.get(f"/portfolios/{pid}/strategy", headers=headers_a)
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["rebalance_interval_months"] == 3
    assert abs(body["total_weight"] - 1.0) < 0.01

    put_resp = api_client.put(
        f"/portfolios/{pid}/strategy",
        json={
            "rebalance_interval_months": 6,
            "targets": {"AAPL": 0.55, "MSFT": 0.35, "CASH": 0.1},
            "replace_targets": True,
        },
        headers=headers_a,
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["strategy"]["rebalance_interval_months"] == 6

    headers_b = _register_and_headers(api_client)
    forbidden = api_client.get(f"/portfolios/{pid}/strategy", headers=headers_b)
    assert forbidden.status_code == 404
