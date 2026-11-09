"""Security response headers and production docs-hiding helpers."""

from __future__ import annotations

from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_BASE_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
)
_HSTS: tuple[bytes, bytes] = (
    b"strict-transport-security",
    b"max-age=31536000; includeSubDomains",
)


def is_production(environment: str) -> bool:
    return environment.strip().lower() == "production"


def docs_kwargs(environment: str) -> dict[str, Any]:
    """FastAPI kwargs that disable Swagger/ReDoc/OpenAPI schema in production."""
    if is_production(environment):
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {}


class SecurityHeadersMiddleware:
    """Adds nosniff / frame deny / referrer policy (and HSTS in production)."""

    def __init__(self, app: ASGIApp, production: bool = False) -> None:
        self.app = app
        self.production = production

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                extra = list(_BASE_HEADERS)
                if self.production:
                    extra.append(_HSTS)
                headers.extend(h for h in extra if h[0] not in present)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
