"""Tests for security hardening: production JWT guard + rate limiting."""

import pytest
from fastapi.testclient import TestClient


def test_production_refuses_default_jwt_secret() -> None:
    """Production must not boot with the placeholder JWT secret."""
    from config.settings import Settings

    with pytest.raises(RuntimeError):
        Settings(environment="production", jwt_secret="change-me-in-production")


def test_production_accepts_strong_jwt_secret() -> None:
    from config.settings import Settings

    settings = Settings(environment="production", jwt_secret="x" * 64)
    assert settings.environment == "production"


def test_development_tolerates_default_secret() -> None:
    from config.settings import Settings

    settings = Settings(environment="development", jwt_secret="change-me-in-production")
    assert settings.jwt_secret == "change-me-in-production"


def test_login_is_rate_limited(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exceeding the per-IP login limit returns HTTP 429."""
    from config.settings import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", True)

    api_client.post(
        "/auth/register",
        json={"email": "rl@example.com", "password": "securepass1"},
    )
    codes = [
        api_client.post(
            "/auth/login/json",
            json={"email": "rl@example.com", "password": "securepass1"},
        ).status_code
        for _ in range(25)  # limit is 20/minute
    ]
    assert 429 in codes


# --- B1: production guards -------------------------------------------------


def test_production_refuses_auth_disabled() -> None:
    from config.settings import Settings

    with pytest.raises(RuntimeError, match="AUTH_DISABLED"):
        Settings(environment="production", jwt_secret="x" * 64, auth_disabled=True)


def test_production_refuses_short_jwt_secret() -> None:
    from config.settings import Settings

    with pytest.raises(RuntimeError, match="32"):
        Settings(environment="production", jwt_secret="x" * 31)


def test_production_accepts_32_char_secret() -> None:
    from config.settings import Settings

    assert Settings(environment="Production", jwt_secret="x" * 32).jwt_secret


def test_development_allows_auth_disabled_and_short_secret() -> None:
    from config.settings import Settings

    s = Settings(environment="development", jwt_secret="short", auth_disabled=True)
    assert s.auth_disabled is True


# --- B2: trusted-proxy aware client IP ------------------------------------


def _request(xff: str | None, peer: str = "10.0.0.9"):
    from starlette.requests import Request

    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (peer, 1234),
    }
    return Request(scope)


def test_client_ip_ignores_xff_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.rate_limit import _client_ip
    from config.settings import settings

    monkeypatch.setattr(settings, "trusted_proxy_count", 0)
    assert _client_ip(_request("6.6.6.6")) == "10.0.0.9"


def test_client_ip_uses_rightmost_with_one_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.rate_limit import _client_ip
    from config.settings import settings

    monkeypatch.setattr(settings, "trusted_proxy_count", 1)
    # Attacker prepends fake IPs; the proxy appended the real one last.
    assert _client_ip(_request("1.1.1.1, 2.2.2.2, 203.0.113.7")) == "203.0.113.7"


def test_client_ip_two_proxies_and_short_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.rate_limit import _client_ip
    from config.settings import settings

    monkeypatch.setattr(settings, "trusted_proxy_count", 2)
    assert _client_ip(_request("9.9.9.9, 203.0.113.7, 10.1.1.1")) == "203.0.113.7"
    # Header shorter than the trusted chain: cannot trust it, use peer.
    assert _client_ip(_request("203.0.113.7")) == "10.0.0.9"
    assert _client_ip(_request(None)) == "10.0.0.9"


def test_spoofed_xff_does_not_bypass_login_limit(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "trusted_proxy_count", 0)
    codes = [
        api_client.post(
            "/auth/login/json",
            json={"email": "nobody@example.com", "password": "wrongpass1"},
            headers={"X-Forwarded-For": f"8.8.{i // 250}.{i % 250}"},
        ).status_code
        for i in range(30)
    ]
    assert 429 in codes


# --- B3: transactions for foreign / missing portfolio ---------------------


def test_transactions_missing_portfolio_is_404(
    api_client: TestClient, auth_headers
) -> None:
    headers = auth_headers("tx404@example.com", "securepass1")
    resp = api_client.get("/portfolios/does-not-exist/transactions", headers=headers)
    assert resp.status_code == 404
    resp = api_client.get(
        "/portfolios/does-not-exist/transactions?sync_ledger=false", headers=headers
    )
    assert resp.status_code == 404


def test_transactions_foreign_portfolio_is_404(
    api_client: TestClient, auth_headers
) -> None:
    owner = auth_headers("txowner@example.com", "securepass1")
    created = api_client.post(
        "/portfolios",
        headers=owner,
        json={
            "name": "Owner PF",
            "starting_capital": 1000.0,
            "positions": [{"ticker": "CASH", "shares": 1000.0, "weight_target": 1.0}],
        },
    )
    assert created.status_code == 200
    pid = created.json()["id"]
    other = auth_headers("txother@example.com", "securepass1")
    resp = api_client.get(f"/portfolios/{pid}/transactions", headers=other)
    assert resp.status_code == 404


# --- B4: security headers + docs hiding ------------------------------------


def test_security_headers_present_on_api(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "strict-transport-security" not in resp.headers  # non-production


def test_hsts_only_in_production_and_docs_hidden() -> None:
    from fastapi import FastAPI

    from api.security_headers import SecurityHeadersMiddleware, docs_kwargs

    def build(env: str) -> TestClient:
        app = FastAPI(**docs_kwargs(env))
        app.add_middleware(SecurityHeadersMiddleware, production=env == "production")

        @app.get("/ping")
        def ping() -> dict[str, str]:
            return {"ok": "1"}

        return TestClient(app)

    prod = build("production")
    assert "max-age=" in prod.get("/ping").headers["strict-transport-security"]
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert prod.get(path).status_code == 404
    # 404 responses are decorated too
    assert prod.get("/docs").headers["x-content-type-options"] == "nosniff"

    dev = build("development")
    assert "strict-transport-security" not in dev.get("/ping").headers
    assert dev.get("/openapi.json").status_code == 200
