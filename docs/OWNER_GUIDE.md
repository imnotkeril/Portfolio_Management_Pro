# Owner guide - Portfolio Management Pro

For the person who runs the product, not the developer. Everything here is checked against
`services/billing_service.py`, `services/billing/plans.py` and `docs/RELEASE.md`. Live Stripe
keys are never given to the agent; every step below is yours to do in the Stripe or hosting
dashboards.

## 1. What the plans are

| Plan | Limit | Defined in |
|---|---|---|
| Free | 1 portfolio; no optimization | `FREE_PORTFOLIO_LIMIT` in `services/billing/plans.py`; optimization gated by `require_pro` |
| Pro | unlimited portfolios, optimization (USER_GUIDE also lists forecasting, see note) | active or trialing subscription whose `current_period_end` is in the future (+1 hour grace) |

Note (open finding P-01): USER_GUIDE.md says Forecasting is Pro-only, but the forecasting
endpoints only require a logged-in user (`api/llms.txt`: auth `user`), so a Free user can call
them. Either gate them with `require_pro` or change the guide: your decision.

The price is not stored in the app. It is the recurring Price in Stripe that
`STRIPE_PRICE_ID_PRO` points to (`price_...`, or a `prod_...` product, in which case the first
active recurring price is used).

## 2. Change the price or the plan

1. Stripe Dashboard -> Product catalogue -> Pro -> add a new recurring Price (a Price cannot be
   edited, only archived).
2. Put the new `price_...` id into `STRIPE_PRICE_ID_PRO` (HF Space secrets), restart the Space.
3. New checkouts use the new price. Existing subscribers stay on their old Price until you
   migrate them in Stripe; the app does not care which Price they are on, only the status.
4. Do not mix a live key with a test price or the reverse: checkout fails.

Changing the Free limit or which features are Pro-only is a code change (`plans.py`,
`require_pro` in `api/dependencies.py`).

## 3. Refund a customer

The app has no refund button. Use Stripe Dashboard -> Payments -> the payment -> Refund.

A refund alone does not remove Pro: access follows the subscription status, not payments. To
end access as well, cancel the subscription in Stripe (Customers -> Subscriptions -> Cancel,
immediately). Stripe sends `customer.subscription.deleted`, the app re-fetches the
subscription and downgrades the user to Free. Check with the customer's `/billing` page.

## 4. Read revenue (MRR)

The app does not compute revenue. Use Stripe Dashboard -> Billing overview / Reports
(Stripe's own MRR, churn and revenue figures). For a head count inside the app database:

```sql
SELECT plan, status, COUNT(*) FROM subscriptions GROUP BY plan, status;
```

Run it on the Supabase SQL editor (read-only). A user counts as Pro only if `plan = 'pro'`,
`status` is `active` or `trialing`, and `current_period_end` is in the future.

## 5. Rotate keys

| Secret | Where | Effect of rotating |
|---|---|---|
| `JWT_SECRET` | HF secrets | Everyone is logged out. >= 32 random chars; the API refuses to boot otherwise |
| `CLIENT_IP_SIGNING_KEY` | HF secrets **and** Vercel, same value | Required in production, >= 32 chars; change both, then redeploy both. A mismatch puts all visitors into one rate-limit bucket |
| `STRIPE_SECRET_KEY` | HF secrets | Roll the key in Stripe, paste the new one, restart. Old key keeps working until you expire it in Stripe |
| `STRIPE_WEBHOOK_SECRET` | HF secrets | Stripe -> Webhooks -> endpoint -> Roll secret (optionally keep the old one for 24h), paste the new `whsec_...`, restart. Until then webhooks return 400 |
| `DATABASE_URL` | HF secrets | Reset the Supabase DB password, update, restart |

Generate a random value: `python -c "import secrets; print(secrets.token_urlsafe(64))"`.

## 6. Webhooks in production

Endpoint: `https://<hf-api-host>/stripe/webhook` (the API host, not the Vercel URL). Events:
`checkout.session.completed`, `invoice.paid`, `customer.subscription.updated`,
`customer.subscription.deleted`. Turn on Stripe's failed-webhook email alerts; the app has no
error tracking yet (RELEASE.md B4). A failed delivery is retried by Stripe; the app un-marks
the event on a handler error so the retry is processed.

## 7. Decisions still open for you

See the list in `docs/RELEASE.md` section 1 and `docs/HANDOFF.md`: Stripe live setup, prod
URLs health, account deletion endpoint (F-5), tracking of `docs/` in git.
