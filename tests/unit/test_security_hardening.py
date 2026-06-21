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
