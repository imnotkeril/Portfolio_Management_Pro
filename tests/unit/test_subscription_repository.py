"""Unit tests for SubscriptionRepository (shared in-memory SQLite)."""

import uuid
from datetime import datetime, timezone

import pytest

from core.data_manager.subscription_repository import SubscriptionRepository
from core.data_manager.user_repository import UserRepository
from database.session import get_db_session
from models.subscription import Subscription


def _new_user() -> str:
    email = f"sub-{uuid.uuid4().hex[:8]}@example.com"
    return UserRepository().create(email, "securepass1").id


def _user_without_subscription() -> str:
    uid = _new_user()
    with get_db_session() as session:
        session.query(Subscription).filter(Subscription.user_id == uid).delete()
    return uid


@pytest.fixture
def repo() -> SubscriptionRepository:
    return SubscriptionRepository()


@pytest.fixture
def user_id() -> str:
    return _new_user()


def test_free_subscription_created_with_user(
    repo: SubscriptionRepository, user_id: str
) -> None:
    sub = repo.find_by_user_id(user_id)
    assert sub is not None
    assert sub.plan == "free"
    assert sub.status == "active"


def test_ensure_free_is_idempotent(repo: SubscriptionRepository, user_id: str) -> None:
    first = repo.ensure_free(user_id)
    second = repo.ensure_free(user_id)
    assert first.id == second.id


def test_find_missing_returns_none(repo: SubscriptionRepository) -> None:
    assert repo.find_by_user_id("nope") is None
    assert repo.find_by_stripe_customer_id("cus_missing") is None


def test_update_pro_from_checkout_and_lookup_by_customer(
    repo: SubscriptionRepository, user_id: str
) -> None:
    end = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sub = repo.update_pro_from_checkout(
        user_id,
        stripe_customer_id="cus_abc",
        stripe_subscription_id="sub_abc",
        current_period_end=end,
    )
    assert sub.plan == "pro"
    assert sub.stripe_subscription_id == "sub_abc"
    found = repo.find_by_stripe_customer_id("cus_abc")
    assert found is not None and found.user_id == user_id


def test_update_pro_creates_row_when_missing(repo: SubscriptionRepository) -> None:
    uid = _user_without_subscription()
    sub = repo.update_pro_from_checkout(
        uid, stripe_customer_id="cus_new", stripe_subscription_id=None
    )
    assert sub.plan == "pro" and sub.stripe_customer_id == "cus_new"


def test_downgrade_to_free(repo: SubscriptionRepository, user_id: str) -> None:
    repo.update_pro_from_checkout(
        user_id, stripe_customer_id="cus_d", stripe_subscription_id="sub_d"
    )
    sub = repo.downgrade_to_free(user_id)
    assert sub.plan == "free"
    assert sub.stripe_subscription_id is None
    assert sub.current_period_end is None


def test_downgrade_creates_row_when_missing(repo: SubscriptionRepository) -> None:
    assert repo.downgrade_to_free(_user_without_subscription()).plan == "free"


def test_sync_stripe_subscription_updates_existing(
    repo: SubscriptionRepository, user_id: str
) -> None:
    sub = repo.sync_stripe_subscription(
        user_id,
        plan="pro",
        status="past_due",
        stripe_subscription_id="sub_s",
        current_period_end=None,
    )
    assert (sub.plan, sub.status) == ("pro", "past_due")


def test_sync_stripe_subscription_creates_row(repo: SubscriptionRepository) -> None:
    created = repo.sync_stripe_subscription(
        _user_without_subscription(),
        plan="pro",
        status="active",
        stripe_subscription_id="s2",
        current_period_end=None,
    )
    assert created.plan == "pro"


def test_save_merges_changes(repo: SubscriptionRepository, user_id: str) -> None:
    sub = repo.find_by_user_id(user_id)
    assert sub is not None
    sub.status = "canceled"
    saved = repo.save(sub)
    assert saved.status == "canceled"
    reloaded = repo.find_by_user_id(user_id)
    assert reloaded is not None and reloaded.status == "canceled"


def test_mark_event_processed_idempotent(repo: SubscriptionRepository) -> None:
    evt = f"evt_{uuid.uuid4().hex}"
    assert repo.mark_event_processed(evt, "checkout.session.completed") is True
    assert repo.mark_event_processed(evt, "checkout.session.completed") is False
