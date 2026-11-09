"""Subscription tier rules (Phase 5 — Stripe)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.subscription import Subscription

PLAN_FREE = "free"
PLAN_PRO = "pro"

PRO_ACTIVE_STATUSES = frozenset({"active", "trialing"})

FREE_PORTFOLIO_LIMIT = 1

# Renewal webhooks can trail the period end by minutes; avoid flapping.
PRO_EXPIRY_GRACE = timedelta(hours=1)


def is_pro_subscription(sub: Subscription | None) -> bool:
    """True when the user has an active, unexpired Pro entitlement."""
    if sub is None:
        return False
    if sub.plan != PLAN_PRO or sub.status not in PRO_ACTIVE_STATUSES:
        return False
    end = sub.current_period_end
    if end is None:
        return False
    if end.tzinfo is None:  # SQLite returns naive datetimes
        end = end.replace(tzinfo=timezone.utc)
    return end + PRO_EXPIRY_GRACE > datetime.now(timezone.utc)
