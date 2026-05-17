"""Integration tests for Phase 3 transaction ledger API."""

import uuid

from fastapi.testclient import TestClient


def _register_and_headers(client: TestClient) -> dict[str, str]:
    email = f"tx-{uuid.uuid4().hex[:8]}@example.com"
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
            "name": f"P-{uuid.uuid4().hex[:6]}",
            "starting_capital": 100000.0,
            "positions": [
                {
                    "ticker": "CASH",
                    "shares": 100000.0,
                    "weight_target": 1.0,
                    "purchase_price": 1.0,
                }
            ],
        },
        headers=headers,
    )
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def test_buy_sell_pnl_fifo(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    pid = _create_portfolio(api_client, headers)

    api_client.post(
        f"/portfolios/{pid}/transactions",
        json={
            "transaction_date": "2024-01-01",
            "transaction_type": "DEPOSIT",
            "ticker": "CASH",
            "shares": 10000.0,
            "price": 1.0,
        },
        headers=headers,
    )
    api_client.post(
        f"/portfolios/{pid}/transactions",
        json={
            "transaction_date": "2024-01-02",
            "transaction_type": "BUY",
            "ticker": "AAPL",
            "shares": 10.0,
            "price": 100.0,
        },
        headers=headers,
    )
    api_client.post(
        f"/portfolios/{pid}/transactions",
        json={
            "transaction_date": "2024-02-01",
            "transaction_type": "SELL",
            "ticker": "AAPL",
            "shares": 5.0,
            "price": 120.0,
        },
        headers=headers,
    )

    pnl = api_client.get(f"/portfolios/{pid}/pnl", headers=headers)
    assert pnl.status_code == 200
    body = pnl.json()
    assert body["realized_pnl"] == 100.0


def test_filter_transactions_by_ticker(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    pid = _create_portfolio(api_client, headers)

    for ticker in ("AAPL", "MSFT"):
        api_client.post(
            f"/portfolios/{pid}/transactions",
            json={
                "transaction_date": "2024-01-15",
                "transaction_type": "BUY",
                "ticker": ticker,
                "shares": 1.0,
                "price": 100.0,
            },
            headers=headers,
        )

    resp = api_client.get(
        f"/portfolios/{pid}/transactions",
        params={"ticker": "AAPL"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["ticker"] == "AAPL"


def test_nested_delete_and_isolation(api_client: TestClient) -> None:
    headers_a = _register_and_headers(api_client)
    headers_b = _register_and_headers(api_client)
    pid_a = _create_portfolio(api_client, headers_a)
    pid_b = _create_portfolio(api_client, headers_b)

    tx = api_client.post(
        f"/portfolios/{pid_a}/transactions",
        json={
            "transaction_date": "2024-01-01",
            "transaction_type": "BUY",
            "ticker": "AAPL",
            "shares": 1.0,
            "price": 100.0,
        },
        headers=headers_a,
    ).json()

    wrong = api_client.delete(
        f"/portfolios/{pid_b}/transactions/{tx['id']}",
        headers=headers_b,
    )
    assert wrong.status_code == 404

    ok = api_client.delete(
        f"/portfolios/{pid_a}/transactions/{tx['id']}",
        headers=headers_a,
    )
    assert ok.status_code == 200
    assert ok.json()["deleted"] is True


def test_holdings_endpoint(api_client: TestClient) -> None:
    headers = _register_and_headers(api_client)
    pid = _create_portfolio(api_client, headers)

    api_client.post(
        f"/portfolios/{pid}/transactions",
        json={
            "transaction_date": "2024-01-01",
            "transaction_type": "BUY",
            "ticker": "AAPL",
            "shares": 3.0,
            "price": 50.0,
        },
        headers=headers,
    )

    holdings = api_client.get(f"/portfolios/{pid}/holdings", headers=headers)
    assert holdings.status_code == 200
    rows = holdings.json()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["quantity"] == 3.0
