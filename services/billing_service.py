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
from services.billing.plans import PLAN_FREE, PLAN_PRO, is_pro_subscription

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
                limit=1,
            )
            if subs.data:
                s0 = subs.data[0]
                status = str(s0.get("status") or "active")
                plan = PLAN_PRO if status in ("active", "trialing") else PLAN_FREE
                end_ts = s0.get("current_period_end")
                period_dt = (
                    datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
                    if end_ts
                    else None
                )
                return self._repo.sync_stripe_subscription(
                    user.id,
                    plan=plan,
                    status=status,
                    stripe_subscription_id=str(s0.get("id") or "") or None,
                    current_period_end=period_dt,
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
            handler(event.data.object)
        else:
            logger.info("Unhandled Stripe event type: %s", event.type)

        return {"status": "ok"}

    def _on_checkout_completed(self, session: Any) -> None:
        user_id = (session.get("metadata") or {}).get("user_id")
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        if not user_id and customer_id:
            sub = self._repo.find_by_stripe_customer_id(str(customer_id))
            if sub:
                user_id = sub.user_id
        if not user_id:
            logger.warning("checkout.session.completed without user_id")
            return
        self._repo.update_pro_from_checkout(
            user_id,
            stripe_customer_id=str(customer_id) if customer_id else "",
            stripe_subscription_id=(str(subscription_id) if subscription_id else None),
            status="active",
        )

    def _on_invoice_paid(self, invoice: Any) -> None:
        customer_id = invoice.get("customer")
        period_end = invoice.get("lines", {}).get("data", [{}])
        end_ts = None
        if period_end:
            end_ts = period_end[0].get("period", {}).get("end")
        period_dt = None
        if end_ts:
            period_dt = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
        sub = (
            self._repo.find_by_stripe_customer_id(str(customer_id))
            if customer_id
            else None
        )
        if sub is None:
            return
        self._repo.sync_stripe_subscription(
            sub.user_id,
            plan=PLAN_PRO,
            status="active",
            stripe_subscription_id=sub.stripe_subscription_id,
            current_period_end=period_dt,
        )

    def _on_subscription_updated(self, stripe_sub: Any) -> None:
        customer_id = stripe_sub.get("customer")
        sub = (
            self._repo.find_by_stripe_customer_id(str(customer_id))
            if customer_id
            else None
        )
        if sub is None:
            return
        status = str(stripe_sub.get("status") or "active")
        plan = PLAN_PRO if status in ("active", "trialing") else PLAN_FREE
        end_ts = stripe_sub.get("current_period_end")
        period_dt = (
            datetime.fromtimestamp(int(end_ts), tz=timezone.utc) if end_ts else None
        )
        self._repo.sync_stripe_subscription(
            sub.user_id,
            plan=plan,
            status=status,
            stripe_subscription_id=str(stripe_sub.get("id") or "") or None,
            current_period_end=period_dt,
        )

    def _on_subscription_deleted(self, stripe_sub: Any) -> None:
        customer_id = stripe_sub.get("customer")
        sub = (
            self._repo.find_by_stripe_customer_id(str(customer_id))
            if customer_id
            else None
        )
        if sub is None:
            return
        self._repo.downgrade_to_free(sub.user_id)
