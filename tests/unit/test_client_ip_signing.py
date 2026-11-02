"""S-19/S-21: signed client IP, soft per-account brake, per-user MC concurrency."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from api import rate_limit as rl
from config.settings import settings

KEY = "k" * 32


@pytest.fixture(autouse=True)
def _on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "trusted_proxy_count", 0)
    monkeypatch.setattr(settings, "client_ip_signing_key", KEY, raising=False)
    rl._storage.reset()
    yield
    rl._storage.reset()


def signed(ip: str, *, key: str = KEY, ts: int | None = None, sig: str | None = None):
    ts = int(time.time()) if ts is None else ts
    mac = (
        sig or hmac.new(key.encode(), f"{ip}|{ts}".encode(), hashlib.sha256).hexdigest()
    )
    return {
        "X-Client-IP": ip,
        "X-Client-IP-Timestamp": str(ts),
        "X-Client-IP-Signature": mac,
    }


def _register(client: TestClient) -> tuple[str, str]:
    email = f"sg-{uuid.uuid4().hex[:8]}@example.com"
    assert (
        client.post(
            "/auth/register", json={"email": email, "password": "securepass1"}
        ).status_code
        == 201
    )
    return email, "securepass1"


def _bad(client: TestClient, email: str, headers: dict | None = None):
    return client.post(
        "/auth/login/json",
        json={"email": email, "password": "wrong-pass-1"},
        headers=headers or {},
    )


def test_valid_signature_is_honored_and_separates_users(api_client: TestClient) -> None:
    victim, pw = _register(api_client)
    for _ in range(5):
        assert _bad(api_client, victim, signed("6.6.6.6")).status_code == 401
    assert _bad(api_client, victim, signed("6.6.6.6")).status_code == 429
    ok = api_client.post(
        "/auth/login/json",
        json={"email": victim, "password": pw},
        headers=signed("7.7.7.7"),
    )
    assert ok.status_code == 200


def test_wrong_signature_is_ignored(api_client: TestClient) -> None:
    victim, _ = _register(api_client)
    # attacker "rotates" IPs with garbage signatures: all keyed on the peer
    codes = [
        _bad(api_client, victim, signed(f"9.9.9.{i}", sig="00" * 32)).status_code
        for i in range(7)
    ]
    assert codes == [401] * 5 + [429] * 2


def test_signature_with_other_key_or_stale_timestamp_is_ignored(
    api_client: TestClient,
) -> None:
    victim, _ = _register(api_client)
    hdrs = [
        signed("1.1.1.1", key="x" * 32),
        signed("2.2.2.2", ts=int(time.time()) - 3600),
        signed("3.3.3.3", ts=int(time.time()) + 3600),
    ]
    for h in hdrs * 2:
        _bad(api_client, victim, h)
    assert _bad(api_client, victim, signed("4.4.4.4", key="x" * 32)).status_code == 429


def test_no_key_configured_ignores_signed_headers(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "client_ip_signing_key", None)
    victim, _ = _register(api_client)
    codes = [
        _bad(api_client, victim, signed(f"5.5.5.{i}")).status_code for i in range(6)
    ]
    assert codes == [401] * 5 + [429]


def test_rotating_xff_does_not_bypass_lockout(api_client: TestClient) -> None:
    victim, _ = _register(api_client)
    codes = [
        _bad(
            api_client,
            victim,
            {"X-Forwarded-For": f"8.8.8.{i}", "X-Real-IP": f"7.7.7.{i}"},
        ).status_code
        for i in range(7)
    ]
    assert codes == [401] * 5 + [429] * 2
    # and the IP bucket cannot be dodged either: 20/min per real peer
    rl._storage.reset()
    codes = [
        _bad(
            api_client, f"nobody{i}@example.com", {"X-Forwarded-For": f"8.8.4.{i}"}
        ).status_code
        for i in range(22)
    ]
    assert codes[:20] == [401] * 20 and codes[20:] == [429, 429]


def test_soft_brake_delays_failures_only(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(rl, "_sleep", delays.append)
    victim, pw = _register(api_client)
    # failures from many distinct (signed) IPs: no per-pair lockout, but the
    # account-wide count grows
    for i in range(14):
        assert _bad(api_client, victim, signed(f"10.0.0.{i}")).status_code == 401
    assert delays and all(0 < d <= 2.0 for d in delays)
    assert delays == sorted(delays) and delays[-1] > delays[0]
    before = len(delays)
    ok = api_client.post(
        "/auth/login/json",
        json={"email": victim, "password": pw},
        headers=signed("10.9.9.9"),
    )
    assert ok.status_code == 200  # correct password never refused
    assert len(delays) == before  # and never delayed


def test_one_concurrent_monte_carlo_run_per_user(api_client: TestClient) -> None:
    import api.main as main

    email, pw = _register(api_client)
    tok = api_client.post(
        "/auth/login/json",
        json={"email": email, "password": pw},
        headers=signed("3.3.3.3"),
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    uid = api_client.get("/auth/me", headers=h).json()["id"]
    body = {"portfolio_id": "p1", "start_date": "2024-01-01", "end_date": "2024-12-31"}

    with main._MC_LOCK:
        main._MC_ACTIVE_USERS.add(uid)
    try:
        r = api_client.post("/risk/monte-carlo", headers=h, json=body)
        assert r.status_code == 429
    finally:
        with main._MC_LOCK:
            main._MC_ACTIVE_USERS.discard(uid)
    # slot is released after a normal request (404: unknown portfolio)
    for _ in range(3):
        assert (
            api_client.post("/risk/monte-carlo", headers=h, json=body).status_code
            != 429
        )
    assert uid not in main._MC_ACTIVE_USERS
