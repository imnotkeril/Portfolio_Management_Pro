"""IP-based rate limiting exposed as a FastAPI dependency.

Implemented with the ``limits`` library and in-process fixed-window storage.
Keyed by client IP (see ``_client_ip``; X-Forwarded-For is honoured only per
``TRUSTED_PROXY_COUNT``). For BFF-proxied traffic that is the proxy's egress IP
(so limits are generous), while direct calls to the public API carry the real
IP — the brute-force / abuse vector we most want to throttle.

A dependency (not the SlowAPI decorator) is used on purpose: ``api.main`` uses
``from __future__ import annotations``, and wrapping the route function breaks
FastAPI's forward-reference resolution. A dependency leaves the route signature
untouched.

Disabled via ``RATE_LIMIT_ENABLED=false`` (tests share one client IP).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from fastapi import HTTPException, Request, status
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from config.settings import settings

_storage = MemoryStorage()
_limiter = FixedWindowRateLimiter(_storage)


def _client_ip(request: Request) -> str:
    """Client IP, trusting X-Forwarded-For only behind known proxies.

    ``TRUSTED_PROXY_COUNT`` = number of reverse proxies we control in front of
    the API. Each appends the address it saw to X-Forwarded-For, so the real
    client is the N-th entry from the right; anything further left is
    attacker-controlled and ignored. With 0 (default) the header is ignored
    and the socket peer address is used.
    """
    peer = request.client.host if request.client else "anonymous"
    hops = settings.trusted_proxy_count
    if hops <= 0:
        return peer
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return peer
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if len(parts) < hops:
        return peer
    return parts[-hops]


def rate_limit(limit: str, scope: str | None = None) -> Callable[[Request], None]:
    """Build a dependency enforcing ``limit`` (e.g. ``"20/minute"``) per IP.

    Routes passing the same ``scope`` share one bucket (default: one bucket
    per URL path).
    """
    item = parse(limit)

    def _dependency(request: Request) -> None:
        if not settings.rate_limit_enabled:
            return
        client_ip = _client_ip(request)
        if not _limiter.hit(item, client_ip, scope or request.url.path):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests, please slow down.",
            )

    return _dependency


# --- login: per-(account, client IP) failure throttle (S-04, S-18) ------------
# The IP bucket alone is not enough: behind a proxy many users share an IP.
# Failed attempts are therefore also counted per (normalized email, client IP).
# Keying on the pair, not the email alone, means an attacker's failures from
# their own IP cannot lock the real owner out from a different IP; a correct
# password is never refused because of OTHER IPs' failures. A success clears
# the counter for that pair.
LOGIN_IP_LIMIT = "20/minute"
LOGIN_IP_SCOPE = "login"  # shared by /auth/login and /auth/login/json
_login_email_item = parse("5 per 15 minute")


def client_ip(request: Request) -> str:
    return _client_ip(request)


def _email_key(email: str, ip: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()
    return f"login-email:{digest}:{ip}"


def login_attempt_allowed(email: str, ip: str) -> bool:
    """False while this account has used up its failed-login budget."""
    if not settings.rate_limit_enabled:
        return True
    return _limiter.test(_login_email_item, _email_key(email, ip))


def record_login_failure(email: str, ip: str) -> None:
    if settings.rate_limit_enabled:
        _limiter.hit(_login_email_item, _email_key(email, ip))


def clear_login_failures(email: str, ip: str) -> None:
    if settings.rate_limit_enabled:
        _limiter.clear(_login_email_item, _email_key(email, ip))
