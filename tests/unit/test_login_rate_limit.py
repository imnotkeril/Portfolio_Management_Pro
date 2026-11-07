"""S-04: login throttling is per client IP (one shared bucket) and per account."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import rate_limit as rl
from config.settings import settings


@pytest.fixture(autouse=True)
def _limiter_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "trusted_proxy_count", 0)
    rl._storage.reset()
    yield
    rl._storage.reset()


def _register(client: TestClient) -> tuple[str, str]:
    email = f"rl-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/auth/register", json={"email": email, "password": "securepass1"})
    assert r.status_code == 201
    return email, "securepass1"


def _bad_json(client: TestClient, email: str, **kw):
    return client.post(
        "/auth/login/json", json={"email": email, "password": "wrong-pass-1"}, **kw
    )


def _bad_form(client: TestClient, email: str, **kw):
    return client.post(
        "/auth/login", data={"username": email, "password": "wrong-pass-1"}, **kw
    )


def test_per_email_lockout_after_repeated_failures(api_client: TestClient) -> None:
    email, pw = _register(api_client)
    for _ in range(5):
        assert _bad_json(api_client, email).status_code == 401
    assert _bad_json(api_client, email).status_code == 429
    # even the right password is refused while the account bucket is exhausted
    good = api_client.post("/auth/login/json", json={"email": email, "password": pw})
    assert good.status_code == 429


def test_one_users_failures_do_not_lock_another_user(api_client: TestClient) -> None:
    victim, _ = _register(api_client)
    other, other_pw = _register(api_client)
    for _ in range(6):
        _bad_json(api_client, victim)
    assert _bad_json(api_client, victim).status_code == 429
    ok = api_client.post(
        "/auth/login/json", json={"email": other, "password": other_pw}
    )
    assert ok.status_code == 200


def test_email_bucket_is_case_insensitive_and_shared_by_both_routes(
    api_client: TestClient,
) -> None:
    email, _ = _register(api_client)
    for i in range(5):
        r = (_bad_json if i % 2 else _bad_form)(
            api_client, email.upper() if i % 2 else email
        )
        assert r.status_code == 401
    assert _bad_json(api_client, email).status_code == 429
    assert _bad_form(api_client, email).status_code == 429


def test_successful_login_resets_email_failures(api_client: TestClient) -> None:
    email, pw = _register(api_client)
    for _ in range(4):
        assert _bad_json(api_client, email).status_code == 401
    assert (
        api_client.post(
            "/auth/login/json", json={"email": email, "password": pw}
        ).status_code
        == 200
    )
    for _ in range(4):
        assert _bad_json(api_client, email).status_code == 401


def test_ip_bucket_is_shared_between_both_login_routes(
    api_client: TestClient,
) -> None:
    statuses = []
    for i in range(25):
        email = f"nobody-{i}@example.com"  # distinct, so no per-email lock
        r = (_bad_json if i % 2 else _bad_form)(api_client, email)
        statuses.append(r.status_code)
    # 20/min total across BOTH routes (previously 20 each = 40)
    assert statuses.count(401) == 20
    assert statuses[20:] == [429] * 5


def test_forwarded_ip_used_only_when_proxy_trusted(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Untrusted (default 0): XFF ignored, all requests share the peer bucket.
    for i in range(20):
        _bad_json(
            api_client, f"a{i}@example.com", headers={"X-Forwarded-For": f"9.9.9.{i}"}
        )
    assert (
        _bad_json(
            api_client, "z@example.com", headers={"X-Forwarded-For": "8.8.8.8"}
        ).status_code
        == 429
    )

    # Trusted (1 hop): the BFF-supplied client IP separates users.
    rl._storage.reset()
    monkeypatch.setattr(settings, "trusted_proxy_count", 1)
    for i in range(20):
        _bad_json(
            api_client, f"b{i}@example.com", headers={"X-Forwarded-For": "1.1.1.1"}
        )
    assert (
        _bad_json(
            api_client, "y@example.com", headers={"X-Forwarded-For": "1.1.1.1"}
        ).status_code
        == 429
    )
    assert (
        _bad_json(
            api_client, "x@example.com", headers={"X-Forwarded-For": "2.2.2.2"}
        ).status_code
        == 401
    )


def test_attacker_ip_failures_do_not_lock_victim_on_another_ip(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-18: failures are keyed by (email, client IP), not by email alone."""
    monkeypatch.setattr(settings, "trusted_proxy_count", 1)
    victim, pw = _register(api_client)
    attacker = {"X-Forwarded-For": "6.6.6.6"}
    for _ in range(5):
        assert _bad_json(api_client, victim, headers=attacker).status_code == 401
    # attacker is now throttled for this account ...
    assert _bad_json(api_client, victim, headers=attacker).status_code == 429
    # ... but the victim, from another IP, still logs in with the right password
    ok = api_client.post(
        "/auth/login/json",
        json={"email": victim, "password": pw},
        headers={"X-Forwarded-For": "7.7.7.7"},
    )
    assert ok.status_code == 200


def test_same_ip_repeated_failures_for_one_email_get_429(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "trusted_proxy_count", 1)
    victim, _ = _register(api_client)
    hdr = {"X-Forwarded-For": "5.5.5.5"}
    codes = [_bad_json(api_client, victim, headers=hdr).status_code for _ in range(7)]
    assert codes == [401] * 5 + [429] * 2
