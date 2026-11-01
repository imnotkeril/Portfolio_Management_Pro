"""Stripe billing: Checkout, Portal, webhooks (Phase 5)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from core.data_manager.subscription_repository import SubscriptionRepository
from core.exceptions import BillingNotConfiguredError
from models.subscription import Subscription
from models.user import User
from services.billing.plans import (
    PLAN_FREE,
    PLAN_PRO,
    PRO_ACTIVE_STATUSES,
    is_pro_subscription,
)

logger = logging.getLogger(__name__)


def _stripe_client():
    key = settings.stripe_secret_key or ""
    if not key.startswith("sk_") or len(key) < 20 or "..." in key:
        raise BillingNotConfiguredError(
            "STRIPE_SECRET_KEY is missing or still a placeholder in .env"
        )
    import stripe

    stripe.api_key = key
    return stripe


def _resolve_price_id(stripe, price_or_product: str) -> str:
    """Accept price_... or prod_... (uses first active recurring price)."""
    pid = (price_or_product or "").strip()
    if pid.startswith("price_"):
        return pid
    if pid.startswith("prod_"):
        prices = stripe.Price.list(product=pid, active=True, limit=10)
        for item in prices.data:
            if getattr(item, "recurring", None):
                return str(item.id)
        if prices.data:
            return str(prices.data[0].id)
        raise BillingNotConfiguredError(f"No prices found for product {pid}")
    raise BillingNotConfiguredError(
        "STRIPE_PRICE_ID_PRO must start with price_ or prod_"
    )


def _as_dict(obj: Any) -> dict[str, Any]:
    """Plain dict from a StripeObject (stripe>=16 is not a dict) or a dict."""
    if obj is None:
        return {}
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(obj)


def _period_end(sub: dict[str, Any]) -> datetime | None:
    """Current period end of a subscription.

    Newer Stripe API versions moved ``current_period_end`` from the
    subscription to its items; older ones keep it on the subscription.
    """
    ends = [
        int(i["current_period_end"])
        for i in (sub.get("items") or {}).get("data") or []
        if i.get("current_period_end")
    ]
    top = sub.get("current_period_end")
    if top:
        ends.append(int(top))
    if not ends:
        return None
    return datetime.fromtimestamp(max(ends), tz=timezone.utc)


def _plan_for_status(status: str) -> str:
    return PLAN_PRO if status in PRO_ACTIVE_STATUSES else PLAN_FREE


def _invoice_subscription_id(invoice: dict[str, Any]) -> str | None:
    """Subscription id of an invoice across Stripe API versions."""
    direct = invoice.get("subscription")
    if isinstance(direct, dict):
        direct = direct.get("id")
    if direct:
        return str(direct)
    parent = (invoice.get("parent") or {}).get("subscription_details") or {}
    if parent.get("subscription"):
        return str(parent["subscription"])
    for line in (invoice.get("lines") or {}).get("data") or []:
        item = (line.get("parent") or {}).get("subscription_item_details") or {}
        sid = line.get("subscription") or item.get("subscription")
        if sid:
            return str(sid)
    return None


class BillingService:
    """Orchestrates subscription rows and Stripe API calls."""

    def __init__(self, repository: SubscriptionRepository | None = None) -> None:
        self._repo = repository or SubscriptionRepository()

    def get_or_create_subscription(self, user_id: str) -> Subscription:
        return self._repo.ensure_free(user_id)

    def sync_from_stripe(self, user: User) -> Subscription:
        """Pull subscription state from Stripe (works without webhooks locally)."""
        sub = self._repo.ensure_free(user.id)
        if not settings.stripe_configured or not sub.stripe_customer_id:
            return sub
        try:
            stripe = _stripe_client()
            subs = stripe.Subscription.list(
                customer=sub.stripe_customer_id,
                status="all",
                limit=10,
            )
            items = [_as_dict(x) for x in subs.data]
            if items:
                # Prefer a live subscription over a newer canceled one.
                s0 = next(
                    (x for x in items if x.get("status") in PRO_ACTIVE_STATUSES),
                    items[0],
                )
                status = str(s0.get("status") or "canceled")
                return self._repo.sync_stripe_subscription(
                    user.id,
                    plan=_plan_for_status(status),
                    status=status,
                    stripe_subscription_id=str(s0.get("id") or "") or None,
                    current_period_end=_period_end(s0),
                )
        except Exception as exc:
            logger.warning("Stripe sync failed for %s: %s", user.id, exc)
        return sub

    def get_status(self, user: User) -> dict[str, Any]:
        sub = self.sync_from_stripe(user)
        return {
            "plan": sub.plan,
            "status": sub.status,
            "is_pro": is_pro_subscription(sub),
            "current_period_end": (
                sub.current_period_end.isoformat() if sub.current_period_end else None
            ),
            "stripe_configured": settings.stripe_configured,
            "free_portfolio_limit": 1,
        }

    def create_checkout_session(self, user: User) -> str:
        if not settings.stripe_configured:
            raise BillingNotConfiguredError(
                "Stripe is not configured (STRIPE_SECRET_KEY, STRIPE_PRICE_ID_PRO)"
            )
        stripe = _stripe_client()
        sub = self._repo.ensure_free(user.id)

        customer_id = sub.stripe_customer_id
        if not customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={"user_id": user.id},
            )
            customer_id = customer.id
            sub.stripe_customer_id = customer_id
            self._repo.save(sub)

        price_id = _resolve_price_id(stripe, settings.stripe_price_id_pro or "")
        base = settings.frontend_url.rstrip("/")
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{base}/billing?success=1",
            cancel_url=f"{base}/billing?canceled=1",
            metadata={"user_id": user.id},
        )
        if not session.url:
            raise BillingNotConfiguredError("Stripe Checkout session has no URL")
        return session.url

    def create_portal_session(self, user: User) -> str:
        if not settings.stripe_configured:
            raise BillingNotConfiguredError("Stripe is not configured")
        stripe = _stripe_client()
        sub = self._repo.ensure_free(user.id)
        if not sub.stripe_customer_id:
            raise BillingNotConfiguredError(
                "No Stripe customer yet — complete checkout first"
            )
        base = settings.frontend_url.rstrip("/")
        portal = stripe.billing_portal.Session.create(
            customer=sub.stripe_customer_id,
            return_url=f"{base}/billing",
        )
        return portal.url

    def handle_webhook_event(
        self, payload: bytes, signature: str | None
    ) -> dict[str, str]:
        if not settings.stripe_webhook_configured:
            raise BillingNotConfiguredError("STRIPE_WEBHOOK_SECRET is not set")
        stripe = _stripe_client()
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, settings.stripe_webhook_secret
            )
        except Exception as exc:
            logger.warning("Stripe webhook verification failed: %s", exc)
            raise

        event_id = event.id
        if not self._repo.mark_event_processed(event_id, event.type):
            return {"status": "already_processed"}

        handler = {
            "checkout.session.completed": self._on_checkout_completed,
            "invoice.paid": self._on_invoice_paid,
            "customer.subscription.updated": self._on_subscription_updated,
            "customer.subscription.deleted": self._on_subscription_deleted,
        }.get(event.type)

        if handler:
            try:
                handler(event.data.object)
            except Exception:
                # The event was marked before handling (dedupe for concurrent
                # deliveries); un-mark it so Stripe's retry is not dropped.
                self._repo.unmark_event(event_id)
                raise
        else:
            logger.info("Unhandled Stripe event type: %s", event.type)

        return {"status": "ok"}

    def _fetch_subscription(self, sub_id: str) -> dict[str, Any] | None:
        """Authoritative subscription state from Stripe (None if it is gone)."""
        import stripe

        try:
            return _as_dict(stripe.Subscription.retrieve(sub_id))
        except stripe.InvalidRequestError as exc:
            if getattr(exc, "code", None) == "resource_missing" or "No such" in str(
                exc
            ):
                return None
            raise

    def _reconcile(self, row: Subscription | None, sub_id: str | None) -> None:
        """Write Stripe's current state of ``sub_id`` onto the user's row.

        The event body is never trusted for entitlement: stale or out-of-order
        events therefore cannot re-grant (or revoke) Pro. Events for a
        subscription other than the one on the row are ignored.
        """
        if row is None:
            return
        current = row.stripe_subscription_id
        sub_id = sub_id or current
        if not sub_id:
            logger.info("Stripe event without subscription id; ignored")
            return
        if current and current != sub_id:
            logger.info("Ignoring event for non-current subscription %s", sub_id)
            return
        state = self._fetch_subscription(sub_id)
        status = str(state.get("status") or "") if state else "canceled"
        if _plan_for_status(status) == PLAN_PRO:
            self._repo.sync_stripe_subscription(
                row.user_id,
                plan=PLAN_PRO,
                status=status,
                stripe_subscription_id=sub_id,
                current_period_end=_period_end(state or {}),
            )
        elif current == sub_id:
            self._repo.downgrade_to_free(row.user_id)

    def _row_for_customer(self, customer_id: Any) -> Subscription | None:
        if not customer_id:
            return None
        return self._repo.find_by_stripe_customer_id(str(customer_id))

    def _on_checkout_completed(self, session: Any) -> None:
        data = _as_dict(session)
        user_id = (data.get("metadata") or {}).get("user_id")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")
        if isinstance(subscription_id, dict):
            subscription_id = subscription_id.get("id")
        row = self._row_for_customer(customer_id)
        if row is None and user_id:
            row = self._repo.find_by_user_id(str(user_id))
        if row is None:
            logger.warning("checkout.session.completed without a known user")
            return
        if customer_id and row.stripe_customer_id != str(customer_id):
            row.stripe_customer_id = str(customer_id)
            row = self._repo.save(row)
        if data.get("payment_status") not in ("paid", "no_payment_required"):
            logger.info("Checkout not paid yet; Pro is granted on a later event")
            return
        # A paid checkout starts a new subscription: a non-Pro row (e.g. one
        # canceled earlier) must accept the new id.
        if (
            subscription_id
            and row.stripe_subscription_id != str(subscription_id)
            and not is_pro_subscription(row)
        ):
            row.stripe_subscription_id = None
            row = self._repo.save(row)
        self._reconcile(row, str(subscription_id) if subscription_id else None)

    def _on_invoice_paid(self, invoice: Any) -> None:
        data = _as_dict(invoice)
        row = self._row_for_customer(data.get("customer"))
        self._reconcile(row, _invoice_subscription_id(data))

    def _on_subscription_updated(self, stripe_sub: Any) -> None:
        data = _as_dict(stripe_sub)
        row = self._row_for_customer(data.get("customer"))
        self._reconcile(row, str(data.get("id") or "") or None)

    def _on_subscription_deleted(self, stripe_sub: Any) -> None:
        self._on_subscription_updated(stripe_sub)
