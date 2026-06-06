"""Repository for subscription persistence."""

from __future__ import annotations

from datetime import datetime

from database.session import get_db_session
from models.stripe_event import StripeEvent as StripeEventORM
from models.subscription import Subscription as SubscriptionORM
from services.billing.plans import PLAN_FREE


class SubscriptionRepository:
    """CRUD for user subscriptions and Stripe webhook idempotency."""

    def ensure_free(self, user_id: str) -> SubscriptionORM:
        """Create a free subscription row if the user has none."""
        with get_db_session() as session:
            existing = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.user_id == user_id)
                .first()
            )
            if existing:
                session.expunge(existing)
                return existing

            sub = SubscriptionORM(
                user_id=user_id,
                plan=PLAN_FREE,
                status="active",
            )
            session.add(sub)
            session.flush()
            session.refresh(sub)
            session.expunge(sub)
            return sub

    def find_by_user_id(self, user_id: str) -> SubscriptionORM | None:
        with get_db_session() as session:
            sub = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.user_id == user_id)
                .first()
            )
            if sub:
                session.expunge(sub)
            return sub

    def find_by_stripe_customer_id(self, customer_id: str) -> SubscriptionORM | None:
        with get_db_session() as session:
            sub = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.stripe_customer_id == customer_id)
                .first()
            )
            if sub:
                session.expunge(sub)
            return sub

    def save(self, sub: SubscriptionORM) -> SubscriptionORM:
        with get_db_session() as session:
            merged = session.merge(sub)
            session.flush()
            session.refresh(merged)
            session.expunge(merged)
            return merged

    def mark_event_processed(self, stripe_event_id: str, event_type: str) -> bool:
        """Return False if event was already processed."""
        with get_db_session() as session:
            exists = (
                session.query(StripeEventORM)
                .filter(StripeEventORM.stripe_event_id == stripe_event_id)
                .first()
            )
            if exists:
                return False
            session.add(
                StripeEventORM(
                    stripe_event_id=stripe_event_id,
                    event_type=event_type,
                )
            )
            return True

    def update_pro_from_checkout(
        self,
        user_id: str,
        *,
        stripe_customer_id: str,
        stripe_subscription_id: str | None,
        status: str = "active",
        current_period_end: datetime | None = None,
    ) -> SubscriptionORM:
        with get_db_session() as session:
            sub = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.user_id == user_id)
                .first()
            )
            if sub is None:
                sub = SubscriptionORM(user_id=user_id, plan=PLAN_FREE, status="active")
                session.add(sub)
            sub.stripe_customer_id = stripe_customer_id
            sub.stripe_subscription_id = stripe_subscription_id
            sub.plan = "pro"
            sub.status = status
            sub.current_period_end = current_period_end
            session.flush()
            session.refresh(sub)
            session.expunge(sub)
            return sub

    def downgrade_to_free(self, user_id: str) -> SubscriptionORM:
        with get_db_session() as session:
            sub = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.user_id == user_id)
                .first()
            )
            if sub is None:
                sub = SubscriptionORM(user_id=user_id, plan=PLAN_FREE, status="active")
                session.add(sub)
            else:
                sub.plan = PLAN_FREE
                sub.status = "active"
                sub.stripe_subscription_id = None
                sub.current_period_end = None
            session.flush()
            session.refresh(sub)
            session.expunge(sub)
            return sub

    def sync_stripe_subscription(
        self,
        user_id: str,
        *,
        plan: str,
        status: str,
        stripe_subscription_id: str | None,
        current_period_end: datetime | None,
    ) -> SubscriptionORM:
        with get_db_session() as session:
            sub = (
                session.query(SubscriptionORM)
                .filter(SubscriptionORM.user_id == user_id)
                .first()
            )
            if sub is None:
                sub = SubscriptionORM(user_id=user_id, plan=plan, status=status)
                session.add(sub)
            sub.plan = plan
            sub.status = status
            sub.stripe_subscription_id = stripe_subscription_id
            sub.current_period_end = current_period_end
            session.flush()
            session.refresh(sub)
            session.expunge(sub)
            return sub
