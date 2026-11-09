"""Integration tests for authentication API."""

import uuid

from fastapi.testclient import TestClient


def test_register_login_me(api_client: TestClient) -> None:
    """Register, login, and fetch profile."""
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    password = "securepass1"

    register = api_client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert register.status_code == 201
    assert register.json()["email"] == email

    login = api_client.post(
        "/auth/login/json",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200
    body = login.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]

    me = api_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_me_without_token_returns_401(api_client: TestClient) -> None:
    """Protected profile endpoint requires Bearer token."""
    response = api_client.get("/auth/me")
    assert response.status_code == 401


def test_login_invalid_password(api_client: TestClient) -> None:
    """Wrong password returns 401."""
    email = f"bad-{uuid.uuid4().hex[:8]}@example.com"
    api_client.post(
        "/auth/register",
        json={"email": email, "password": "correctpass1"},
    )
    response = api_client.post(
        "/auth/login/json",
        json={"email": email, "password": "wrongpass1"},
    )
    assert response.status_code == 401


def test_register_duplicate_email(api_client: TestClient) -> None:
    """Duplicate registration returns 409."""
    email = f"dup-{uuid.uuid4().hex[:8]}@example.com"
    api_client.post(
        "/auth/register",
        json={"email": email, "password": "securepass1"},
    )
    response = api_client.post(
        "/auth/register",
        json={"email": email, "password": "securepass1"},
    )
    assert response.status_code == 409
