"""Integration tests for portfolio access isolation between users."""

import uuid

from fastapi.testclient import TestClient


def _register_and_login(client: TestClient, label: str) -> dict[str, str]:
    email = f"{label}-{uuid.uuid4().hex[:8]}@example.com"
    password = "securepass1"
    client.post("/auth/register", json={"email": email, "password": password})
    login = client.post(
        "/auth/login/json",
        json={"email": email, "password": password},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_portfolio(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/portfolios",
        headers=headers,
        json={
            "name": f"Portfolio {uuid.uuid4().hex[:6]}",
            "starting_capital": 10000.0,
            "positions": [
                {"ticker": "CASH", "shares": 10000.0, "weight_target": 1.0},
            ],
        },
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_user_a_cannot_read_user_b_portfolio(api_client: TestClient) -> None:
    """User B receives 404 when accessing user A portfolio by id."""
    headers_a = _register_and_login(api_client, "usera")
    headers_b = _register_and_login(api_client, "userb")

    portfolio_id = _create_portfolio(api_client, headers_a)

    response = api_client.get(f"/portfolios/{portfolio_id}", headers=headers_b)
    assert response.status_code == 404


def test_list_portfolios_only_own(api_client: TestClient) -> None:
    """New user sees empty portfolio list; after create, only own item."""
    headers = _register_and_login(api_client, "solo")

    empty = api_client.get("/portfolios", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == []

    portfolio_id = _create_portfolio(api_client, headers)

    listed = api_client.get("/portfolios", headers=headers)
    assert listed.status_code == 200
    ids = [p["id"] for p in listed.json()]
    assert portfolio_id in ids
    assert len(ids) == 1


def test_portfolios_require_auth(api_client: TestClient) -> None:
    """Portfolio endpoints return 401 without token when auth is enabled."""
    response = api_client.get("/portfolios")
    assert response.status_code == 401
