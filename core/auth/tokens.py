"""JWT access token helpers."""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt

from config.settings import settings


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT with ``sub`` = user id."""
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    return jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT; raises ``JWTError`` on failure."""
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
