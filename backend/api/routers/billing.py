"""Billing and Stripe webhook routes (Phase 5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.dependencies import get_current_user
from config.settings import settings
from core.exceptions import BillingNotConfiguredError
from models.user import User
from services.billing_service import BillingService

router = APIRouter(tags=["billing"])
_billing = BillingService()


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    url: str


@router.get("/billing/status")
def billing_status(current_user: User = Depends(get_current_user)) -> dict:
    return _billing.get_status(current_user)


@router.post("/billing/checkout", response_model=CheckoutResponse)
def billing_checkout(
    current_user: User = Depends(get_current_user),
) -> CheckoutResponse:
    try:
        url = _billing.create_checkout_session(current_user)
        return CheckoutResponse(url=url)
    except BillingNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/billing/portal", response_model=PortalResponse)
def billing_portal(
    current_user: User = Depends(get_current_user),
) -> PortalResponse:
    try:
        url = _billing.create_portal_session(current_user)
        return PortalResponse(url=url)
    except BillingNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        return _billing.handle_webhook_event(payload, signature)
    except BillingNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        if settings.debug:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook processing failed",
        ) from exc
