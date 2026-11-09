"""Integration tests for Phase 5 Stripe billing."""

import uuid

from fastapi.testclient import TestClient


def _register_and_headers(client: TestClient) -> dict[str, str]:
    email = f"bill-{uuid.uuid4().hex[:8]}@example.com"
    password = "securepass1"
    client.post("/auth/register", json={"email": email, "password": password})
    login = client.post(
        "/auth/login/json",
        json={"email": email, "password": password},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_portfolio(client: TestClient, headers: dict[str, str], name: str) -> str:
    resp = client.post(
        "/portfolios",
        json={
            "name": name,
            "starting_capital": 50000.0,
            "positions": [
                {
                    "ticker": "CASH",
                    "shares": 50000.0,
                    "weight_target": 1.0,
                    "purchase_price": 1.0,
                }
            ],
        },
        headers=headers,
    )
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def test_billing_status_defaults_free(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    resp = api_client.get("/billing/status", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] == "free"
    assert body["is_pro"] is False


def test_free_user_second_portfolio_forbidden(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    _create_portfolio(api_client, headers, f"P1-{uuid.uuid4().hex[:4]}")
    resp = api_client.post(
        "/portfolios",
        json={
            "name": f"P2-{uuid.uuid4().hex[:4]}",
            "starting_capital": 10000.0,
            "positions": [
                {
                    "ticker": "CASH",
                    "shares": 10000.0,
                    "weight_target": 1.0,
                    "purchase_price": 1.0,
                }
            ],
        },
        headers=headers,
    )
    assert resp.status_code == 403
    assert "Free plan" in resp.json()["detail"]


def test_free_user_optimization_forbidden(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    pid = _create_portfolio(api_client, headers, f"Opt-{uuid.uuid4().hex[:4]}")
    resp = api_client.post(
        "/optimization/run",
        json={
            "portfolio_id": pid,
            "method": "max_sharpe",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        },
        headers=headers,
    )
    assert resp.status_code == 403
    assert "Pro" in resp.json()["detail"]
