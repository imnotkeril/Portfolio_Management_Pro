"""Integration tests for DELETE /auth/me (account deletion)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from core.data_manager.subscription_repository import SubscriptionRepository
from core.data_manager.user_repository import UserRepository
from database.session import get_db_session
from models.portfolio import Portfolio
from models.subscription import Subscription
from models.transaction import Transaction
from services.billing_service import BillingService

PASSWORD = "securepass1"


def _register(client: TestClient) -> tuple[str, str, dict[str, str]]:
    email = f"del-{uuid.uuid4().hex[:8]}@example.com"
    user = client.post(
        "/auth/register", json={"email": email, "password": PASSWORD}
    ).json()
    token = client.post(
        "/auth/login/json", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]
    return user["id"], email, {"Authorization": f"Bearer {token}"}


def _portfolio(client: TestClient, headers: dict[str, str]) -> str:
    resp = client.post(
        "/portfolios",
        json={
            "name": "p",
            "starting_capital": 1000.0,
            "positions": [
                {
                    "ticker": "CASH",
                    "shares": 1000.0,
                    "weight_target": 1.0,
                    "purchase_price": 1.0,
                }
            ],
        },
        headers=headers,
    )
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def _delete(client: TestClient, headers: dict[str, str], password: str):
    return client.request(
        "DELETE", "/auth/me", json={"password": password}, headers=headers
    )


def test_delete_requires_authentication(api_client: TestClient) -> None:
    resp = api_client.request("DELETE", "/auth/me", json={"password": PASSWORD})
    assert resp.status_code == 401


def test_delete_wrong_password_keeps_account(api_client: TestClient) -> None:
    user_id, _, headers = _register(api_client)
    resp = _delete(api_client, headers, "not-the-password")
    assert resp.status_code == 403
    assert UserRepository().find_by_id(user_id) is not None
    assert api_client.get("/auth/me", headers=headers).status_code == 200


def test_delete_removes_account_and_all_owned_data(api_client: TestClient) -> None:
    user_id, email, headers = _register(api_client)
    portfolio_id = _portfolio(api_client, headers)
    tx = api_client.post(
        f"/portfolios/{portfolio_id}/transactions",
        json={
            "transaction_date": "2024-01-01",
            "transaction_type": "DEPOSIT",
            "ticker": "CASH",
            "shares": 10.0,
            "price": 1.0,
        },
        headers=headers,
    )
    assert tx.status_code in (200, 201), tx.text

    with get_db_session() as session:
        assert (
            session.query(Transaction)
            .filter(Transaction.portfolio_id == portfolio_id)
            .count()
            > 0
        )

    resp = _delete(api_client, headers, PASSWORD)
    assert resp.status_code == 204

    assert UserRepository().find_by_id(user_id) is None
    with get_db_session() as session:
        assert (
            session.query(Portfolio).filter(Portfolio.user_id == user_id).count() == 0
        )
        assert (
            session.query(Transaction)
            .filter(Transaction.portfolio_id == portfolio_id)
            .count()
            == 0
        )
        assert (
            session.query(Subscription).filter(Subscription.user_id == user_id).count()
            == 0
        )
    # Old token no longer works, credentials no longer log in.
    assert api_client.get("/auth/me", headers=headers).status_code == 401
    login = api_client.post(
        "/auth/login/json", json={"email": email, "password": PASSWORD}
    )
    assert login.status_code == 401


def test_delete_does_not_touch_other_users(api_client: TestClient) -> None:
    _, _, headers_a = _register(api_client)
    user_b, _, headers_b = _register(api_client)
    pid_b = _portfolio(api_client, headers_b)

    assert _delete(api_client, headers_a, PASSWORD).status_code == 204

    assert UserRepository().find_by_id(user_b) is not None
    assert api_client.get(f"/portfolios/{pid_b}", headers=headers_b).status_code == 200


def test_delete_cancels_stripe_subscription_first(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, _, headers = _register(api_client)
    cancelled: list[str] = []
    monkeypatch.setattr(
        BillingService,
        "cancel_subscription_for_deletion",
        lambda self, uid: cancelled.append(uid),
    )
    assert _delete(api_client, headers, PASSWORD).status_code == 204
    assert cancelled == [user_id]


def test_delete_aborts_when_stripe_cancel_fails(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.exceptions import BillingError

    user_id, _, headers = _register(api_client)

    def boom(self, uid):
        raise BillingError("stripe down")

    monkeypatch.setattr(BillingService, "cancel_subscription_for_deletion", boom)
    resp = _delete(api_client, headers, PASSWORD)
    assert resp.status_code == 502
    assert UserRepository().find_by_id(user_id) is not None


class _FakeSubscription:
    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self._calls = calls
        self._error = error

    def cancel(self, sub_id: str) -> None:
        self._calls.append(sub_id)
        if self._error:
            raise self._error


class _FakeStripe:
    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self.Subscription = _FakeSubscription(calls, error)


def _service_with_subscription(
    monkeypatch: pytest.MonkeyPatch, stripe: _FakeStripe | None
) -> tuple[BillingService, str]:
    import services.billing_service as module

    user_id = str(uuid.uuid4())
    from models.user import User

    with get_db_session() as session:
        session.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="x"))
    SubscriptionRepository().sync_stripe_subscription(
        user_id,
        plan="pro",
        status="active",
        stripe_subscription_id="sub_123",
        current_period_end=None,
    )
    monkeypatch.setattr(module.settings, "stripe_secret_key", "sk_test_" + "a" * 30)
    monkeypatch.setattr(module, "_stripe_client", lambda: stripe)
    return BillingService(), user_id


def test_service_cancels_live_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    service, user_id = _service_with_subscription(monkeypatch, _FakeStripe(calls))
    service.cancel_subscription_for_deletion(user_id)
    assert calls == ["sub_123"]


def test_service_ignores_already_missing_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Missing(Exception):
        code = "resource_missing"

    calls: list[str] = []
    service, user_id = _service_with_subscription(
        monkeypatch, _FakeStripe(calls, Missing("gone"))
    )
    service.cancel_subscription_for_deletion(user_id)
    assert calls == ["sub_123"]


def test_service_raises_on_other_stripe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.exceptions import BillingError

    service, user_id = _service_with_subscription(
        monkeypatch, _FakeStripe([], RuntimeError("timeout"))
    )
    with pytest.raises(BillingError):
        service.cancel_subscription_for_deletion(user_id)


def test_service_noop_without_stripe_subscription() -> None:
    service = BillingService()
    service.cancel_subscription_for_deletion(str(uuid.uuid4()))
