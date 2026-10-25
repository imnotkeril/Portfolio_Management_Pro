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


def rate_limit(limit: str) -> Callable[[Request], None]:
    """Build a dependency enforcing ``limit`` (e.g. ``"20/minute"``) per IP."""
    item = parse(limit)

    def _dependency(request: Request) -> None:
        if not settings.rate_limit_enabled:
            return
        client_ip = _client_ip(request)
        if not _limiter.hit(item, client_ip, request.url.path):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests, please slow down.",
            )

    return _dependency
