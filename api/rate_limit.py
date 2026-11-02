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
import hmac
import time
from collections.abc import Callable

from fastapi import HTTPException, Request, status
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from config.settings import settings

_storage = MemoryStorage()
_limiter = FixedWindowRateLimiter(_storage)


def _signed_client_ip(request: Request) -> str | None:
    """Client IP asserted by the BFF, only if its HMAC verifies.

    The BFF sends ``X-Client-IP``, ``X-Client-IP-Timestamp`` and
    ``X-Client-IP-Signature`` = hex HMAC-SHA256(key, "<ip>|<timestamp>").
    Without a configured ``CLIENT_IP_SIGNING_KEY`` or with a bad / stale
    signature the headers are ignored.
    """
    key = settings.client_ip_signing_key
    if not key:
        return None
    ip = request.headers.get("x-client-ip", "").strip()
    ts = request.headers.get("x-client-ip-timestamp", "").strip()
    sig = request.headers.get("x-client-ip-signature", "").strip()
    if not (ip and ts and sig) or len(ip) > 64:
        return None
    try:
        age = abs(time.time() - int(ts))
    except ValueError:
        return None
    if age > settings.client_ip_max_age_seconds:
        return None
    expected = hmac.new(key.encode(), f"{ip}|{ts}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return ip


def _client_ip(request: Request) -> str:
    """Client IP: signed BFF header, else (opt-in) trusted XFF, else the peer.

    Order: (1) a validly signed ``X-Client-IP``; (2) with
    ``TRUSTED_PROXY_COUNT`` > 0, X-Forwarded-For (see below); (3) the socket
    peer. ``X-Real-IP`` is never read.

    ``TRUSTED_PROXY_COUNT`` = number of reverse proxies we control in front of
    the API. Each appends the address it saw to X-Forwarded-For, so the real
    client is the N-th entry from the right; anything further left is
    attacker-controlled and ignored. With 0 (default) the header is ignored
    and the socket peer address is used.
    """
    signed_ip = _signed_client_ip(request)
    if signed_ip:
        return signed_ip
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
    """Count a failed login and slow the response as the account is hammered."""
    if not settings.rate_limit_enabled:
        return
    _limiter.hit(_login_email_item, _email_key(email, ip))
    key = _account_key(email)
    _limiter.hit(_account_item, key)
    _, remaining = _limiter.get_window_stats(_account_item, key)
    failures = _account_item.amount - remaining
    if failures > BRAKE_AFTER:
        _sleep(min(BRAKE_MAX_SECONDS, (failures - BRAKE_AFTER) * BRAKE_STEP_SECONDS))


# Soft brake: after BRAKE_AFTER failures for one email from ANY IP within the
# window, each further failed attempt is delayed (never successful ones).
_account_item = parse("1000 per 15 minute")
BRAKE_AFTER = 5
BRAKE_STEP_SECONDS = 0.2
BRAKE_MAX_SECONDS = 2.0


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _account_key(email: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()
    return f"login-account:{digest}"


def clear_login_failures(email: str, ip: str) -> None:
    if settings.rate_limit_enabled:
        _limiter.clear(_login_email_item, _email_key(email, ip))
