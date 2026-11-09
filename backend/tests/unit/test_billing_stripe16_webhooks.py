"""Webhook handling with REAL stripe-python 16 objects (S-01, S-02).

Events are built as signed JSON payloads (dummy ``whsec_``) and verified through
``Webhook.construct_event``, so handlers receive real ``StripeObject`` instances.
Only the outbound ``Subscription.retrieve/list`` calls are faked. No real keys.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
import stripe

from core.data_manager.subscription_repository import SubscriptionRepository
from core.data_manager.user_repository import UserRepository
from models.user import User
from services.billing.plans import is_pro_subscription
from services.billing_service import BillingService

WHSEC = "whsec_" + "t" * 24
SK = "sk_test_" + "x" * 30
FUTURE = int(time.time()) + 20 * 86400
PAST = int(time.time()) - 40 * 86400


@pytest.fixture(autouse=True)
def _stripe_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.billing_service as mod

    monkeypatch.setattr(mod.settings, "stripe_secret_key", SK, raising=False)
    monkeypatch.setattr(mod.settings, "stripe_webhook_secret", WHSEC, raising=False)
    monkeypatch.setattr(
        mod.settings, "stripe_price_id_pro", "price_test123", raising=False
    )


@pytest.fixture
def repo() -> SubscriptionRepository:
    return SubscriptionRepository()


@pytest.fixture
def svc(repo: SubscriptionRepository) -> BillingService:
    return BillingService(repository=repo)


@pytest.fixture
def user() -> User:
    return UserRepository().create(
        f"s16-{uuid.uuid4().hex[:8]}@example.com", "securepass1"
    )


def _sign(payload: bytes) -> str:
    ts = int(time.time())
    mac = hmac.new(
        WHSEC.encode(), f"{ts}.".encode() + payload, hashlib.sha256
    ).hexdigest()
    return f"t={ts},v1={mac}"


def _deliver(svc: BillingService, event_type: str, obj: dict[str, Any]) -> dict:
    body = json.dumps(
        {
            "id": f"evt_{uuid.uuid4().hex[:14]}",
            "object": "event",
            "type": event_type,
            "created": int(time.time()),
            "data": {"object": obj},
        }
    ).encode()
    return svc.handle_webhook_event(body, _sign(body))


def _sub_obj(
    sub_id: str, customer: str, status: str, end: int | None = FUTURE
) -> dict[str, Any]:
    """Subscription as in the API version pinned by stripe 16 (period on items)."""
    item: dict[str, Any] = {"id": "si_1", "object": "subscription_item"}
    if end is not None:
        item["current_period_end"] = end
    return {
        "id": sub_id,
        "object": "subscription",
        "customer": customer,
        "status": status,
        "items": {"object": "list", "data": [item]},
    }


@pytest.fixture
def stripe_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the outbound Stripe API; returns a mutable {sub_id: sub_dict}."""
    store: dict[str, dict[str, Any]] = {}

    def retrieve(sub_id: str, **_: Any) -> stripe.Subscription:
        if sub_id not in store:
            raise stripe.InvalidRequestError("No such subscription", "id")
        return stripe.Subscription.construct_from(store[sub_id], SK)

    def list_(**params: Any) -> SimpleNamespace:
        cust = params["customer"]
        data = [
            stripe.Subscription.construct_from(v, SK)
            for v in store.values()
            if v["customer"] == cust
        ]
        return SimpleNamespace(data=data)

    monkeypatch.setattr(stripe.Subscription, "retrieve", staticmethod(retrieve))
    monkeypatch.setattr(stripe.Subscription, "list", staticmethod(list_))
    return store


def _link(repo: SubscriptionRepository, user: User, customer: str) -> None:
    sub = repo.ensure_free(user.id)
    sub.stripe_customer_id = customer
    repo.save(sub)


def _row(repo: SubscriptionRepository, user: User):
    row = repo.find_by_user_id(user.id)
    assert row is not None
    return row


def _checkout(user: User, customer: str, sub_id: str, pay: str = "paid") -> dict:
    return {
        "id": "cs_1",
        "object": "checkout.session",
        "customer": customer,
        "subscription": sub_id,
        "payment_status": pay,
        "metadata": {"user_id": user.id},
    }


def _invoice(customer: str, sub_id: str | None, end: int = FUTURE) -> dict:
    inv: dict[str, Any] = {
        "id": "in_1",
        "object": "invoice",
        "customer": customer,
        "lines": {"data": [{"period": {"end": end}}]},
    }
    if sub_id:
        inv["parent"] = {
            "type": "subscription_details",
            "subscription_details": {"subscription": sub_id},
        }
    return inv


def test_stripeobject_is_not_a_dict_regression() -> None:
    """Documents the S-01 root cause so a pin change is noticed."""
    obj = stripe.Event.construct_from(
        {"id": "e", "type": "x", "data": {"object": {"id": "a"}}}, SK
    ).data.object
    assert not hasattr(obj, "get")


def test_checkout_completed_grants_pro(svc, repo, user, stripe_state) -> None:
    _link(repo, user, "cus_a")
    stripe_state["sub_a"] = _sub_obj("sub_a", "cus_a", "active")
    res = _deliver(svc, "checkout.session.completed", _checkout(user, "cus_a", "sub_a"))
    assert res == {"status": "ok"}
    row = _row(repo, user)
    assert (row.plan, row.status) == ("pro", "active")
    assert row.stripe_subscription_id == "sub_a"
    assert row.stripe_customer_id == "cus_a"
    assert is_pro_subscription(row)
    end = row.current_period_end.replace(tzinfo=timezone.utc)
    assert int(end.timestamp()) == FUTURE


def test_checkout_without_metadata_resolves_user_by_customer(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_b")
    stripe_state["sub_b"] = _sub_obj("sub_b", "cus_b", "active")
    obj = _checkout(user, "cus_b", "sub_b")
    obj["metadata"] = {}
    _deliver(svc, "checkout.session.completed", obj)
    assert is_pro_subscription(_row(repo, user))


def test_unpaid_checkout_does_not_grant_pro(svc, repo, user, stripe_state) -> None:
    _link(repo, user, "cus_c")
    stripe_state["sub_c"] = _sub_obj("sub_c", "cus_c", "incomplete")
    obj = _checkout(user, "cus_c", "sub_c", "unpaid")
    _deliver(svc, "checkout.session.completed", obj)
    assert not is_pro_subscription(_row(repo, user))


def test_checkout_paid_but_stripe_says_incomplete_does_not_grant(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_c2")
    stripe_state["sub_c2"] = _sub_obj("sub_c2", "cus_c2", "incomplete")
    _deliver(svc, "checkout.session.completed", _checkout(user, "cus_c2", "sub_c2"))
    assert not is_pro_subscription(_row(repo, user))


def test_invoice_paid_grants_pro_with_current_period(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_d")
    stripe_state["sub_d"] = _sub_obj("sub_d", "cus_d", "active")
    _deliver(svc, "invoice.paid", _invoice("cus_d", "sub_d"))
    row = _row(repo, user)
    assert is_pro_subscription(row)
    assert row.stripe_subscription_id == "sub_d"


def test_late_invoice_paid_does_not_resurrect_canceled(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_e")
    stripe_state["sub_e"] = _sub_obj("sub_e", "cus_e", "active")
    _deliver(svc, "checkout.session.completed", _checkout(user, "cus_e", "sub_e"))
    assert is_pro_subscription(_row(repo, user))

    stripe_state["sub_e"] = _sub_obj("sub_e", "cus_e", "canceled", PAST)
    _deliver(svc, "customer.subscription.deleted", stripe_state["sub_e"])
    assert not is_pro_subscription(_row(repo, user))

    # stale, out-of-order invoice.paid for the old period arrives afterwards
    _deliver(svc, "invoice.paid", _invoice("cus_e", "sub_e", PAST))
    row = _row(repo, user)
    assert not is_pro_subscription(row)
    assert row.plan == "free"


def test_invoice_paid_without_subscription_id_does_not_grant(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_f")
    _deliver(svc, "invoice.paid", _invoice("cus_f", None))
    assert not is_pro_subscription(_row(repo, user))


def test_subscription_updated_downgrades_on_cancel(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_g")
    stripe_state["sub_g"] = _sub_obj("sub_g", "cus_g", "active")
    _deliver(svc, "customer.subscription.updated", stripe_state["sub_g"])
    assert is_pro_subscription(_row(repo, user))

    stripe_state["sub_g"] = _sub_obj("sub_g", "cus_g", "canceled", PAST)
    _deliver(svc, "customer.subscription.updated", stripe_state["sub_g"])
    assert not is_pro_subscription(_row(repo, user))


def test_old_subscription_deleted_does_not_downgrade_new_one(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_h")
    stripe_state["sub_new"] = _sub_obj("sub_new", "cus_h", "active")
    stripe_state["sub_old"] = _sub_obj("sub_old", "cus_h", "canceled", PAST)
    _deliver(svc, "checkout.session.completed", _checkout(user, "cus_h", "sub_new"))
    _deliver(svc, "customer.subscription.deleted", stripe_state["sub_old"])
    row = _row(repo, user)
    assert is_pro_subscription(row)
    assert row.stripe_subscription_id == "sub_new"


def test_subscription_deleted_when_stripe_no_longer_has_it(
    svc, repo, user, stripe_state
) -> None:
    _link(repo, user, "cus_i")
    stripe_state["sub_i"] = _sub_obj("sub_i", "cus_i", "active")
    _deliver(svc, "checkout.session.completed", _checkout(user, "cus_i", "sub_i"))
    gone = stripe_state.pop("sub_i")
    _deliver(svc, "customer.subscription.deleted", gone)
    assert not is_pro_subscription(_row(repo, user))


def test_bad_signature_rejected(svc) -> None:
    body = json.dumps(
        {"id": "evt_x", "type": "invoice.paid", "data": {"object": {}}}
    ).encode()
    with pytest.raises(stripe.SignatureVerificationError):
        svc.handle_webhook_event(body, "t=1,v1=deadbeef")


def test_sync_from_stripe_uses_real_objects(svc, repo, user, stripe_state) -> None:
    _link(repo, user, "cus_j")
    stripe_state["sub_j"] = _sub_obj("sub_j", "cus_j", "active")
    row = svc.sync_from_stripe(user)
    assert is_pro_subscription(row)
    assert row.stripe_subscription_id == "sub_j"

    stripe_state["sub_j"] = _sub_obj("sub_j", "cus_j", "canceled", PAST)
    row = svc.sync_from_stripe(user)
    assert not is_pro_subscription(row)


def test_is_pro_requires_unexpired_period() -> None:
    now = datetime.now(timezone.utc)
    day = timedelta(days=3)

    def mk(end, plan="pro"):
        return SimpleNamespace(plan=plan, status="active", current_period_end=end)

    assert is_pro_subscription(mk(now + day))
    assert not is_pro_subscription(mk(now - day))
    assert not is_pro_subscription(mk(None))
    # naive datetimes (SQLite) are treated as UTC
    assert is_pro_subscription(mk((now + day).replace(tzinfo=None)))
    assert not is_pro_subscription(mk((now - day).replace(tzinfo=None)))
    assert not is_pro_subscription(mk(now + day, plan="free"))
