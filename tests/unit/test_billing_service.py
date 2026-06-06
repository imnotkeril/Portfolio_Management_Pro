"""Unit tests for billing webhook idempotency."""

from unittest.mock import MagicMock, patch

from core.data_manager.subscription_repository import SubscriptionRepository
from services.billing_service import BillingService


def test_webhook_skips_duplicate_event_id() -> None:
    repo = MagicMock(spec=SubscriptionRepository)
    repo.mark_event_processed.return_value = False
    svc = BillingService(repository=repo)

    with patch("services.billing_service.settings") as mock_settings:
        mock_settings.stripe_webhook_configured = True
        mock_settings.stripe_webhook_secret = "whsec_test"
        with patch("services.billing_service._stripe_client") as mock_stripe:
            mock_stripe.return_value.Webhook.construct_event.return_value = MagicMock(
                id="evt_123",
                type="invoice.paid",
                data=MagicMock(object={}),
            )
            result = svc.handle_webhook_event(b"{}", "sig")
    assert result["status"] == "already_processed"
    repo.mark_event_processed.assert_called_once()
