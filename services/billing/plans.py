"""Subscription tier rules (Phase 5 — Stripe)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.subscription import Subscription

PLAN_FREE = "free"
PLAN_PRO = "pro"

PRO_ACTIVE_STATUSES = frozenset({"active", "trialing"})

FREE_PORTFOLIO_LIMIT = 1


def is_pro_subscription(sub: Subscription | None) -> bool:
    """True when user has an active Pro entitlement."""
    if sub is None:
        return False
    return sub.plan == PLAN_PRO and sub.status in PRO_ACTIVE_STATUSES
