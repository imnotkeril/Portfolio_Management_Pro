# Stripe CLI: webhooks on your machine

Test mode only. You need the Stripe CLI (`stripe login` once) and the stack running
(`docker compose up -d`, API on `localhost:8000`).

## Forward events

```powershell
.\scripts\start-stripe-listen.ps1
```

This runs `stripe listen --forward-to localhost:8000/stripe/webhook` and prints a signing
secret `whsec_...`. Put it in the project `.env` as `STRIPE_WEBHOOK_SECRET` and restart the
API (`docker compose up -d api`). Without it the webhook returns 503.

## Try the full flow (recommended)

1. Open http://localhost:3000, log in, go to Billing -> Upgrade to Pro.
2. Pay with the test card `4242 4242 4242 4242`, any future date and CVC.
3. The terminal running `stripe listen` shows the four events; the API answers 200.
4. Billing now shows Pro. Send the same event again with
   `stripe events resend evt_...`: the API answers `already_processed`.

## `stripe trigger` limits

`stripe trigger checkout.session.completed` creates a Stripe customer that does not belong to
any app user, so the handler logs `checkout.session.completed without a known user` and
changes nothing. That is expected: the app maps events to users by Stripe customer id
(set at checkout), and re-fetches the subscription from Stripe instead of trusting the event
body. Use the UI flow above for end-to-end checks; use the unit tests
(`tests/unit/test_billing_stripe16_webhooks.py`) for event edge cases.

## Without the CLI

Opening `/billing` after paying syncs the plan from Stripe directly (`sync_from_stripe`), so
local development works without webhooks.
