# Owner guide - Portfolio Management Pro

For the person who runs the product, not the developer. Everything here is checked against
`services/billing_service.py` and `services/billing/plans.py`. Live Stripe
keys are never given to the agent; every step below is yours to do in the Stripe or hosting
dashboards.

## 0. Turn on payments, click by click

You need three places open in your browser: **Stripe** (dashboard.stripe.com), **Hugging Face**
(the server of the app) and **Vercel** (the website). Button names may differ a little from
what is written here; look for the closest word. Never send a key by chat or email, and never
paste it into a file in the project. Keys only go into the "secret" boxes below.

Do everything first in **test mode** (the "Test mode" switch at the top right of Stripe is
on, the dashboard shows an orange bar). Repeat the same steps in live mode when the test works.

### Step 1. Make the product and the price (Stripe)
1. Left menu -> **Product catalogue** -> **Add product**.
2. Name: `Pro`. Under "Pricing" choose **Recurring**, type the monthly amount, currency, billing
   period **Monthly**. Press **Add product**.
3. On the product page find the price row, click the three dots `...` -> **Copy price ID**.
   It looks like `price_1Abc...`. Keep it in a note for step 4.

### Step 2. Get the secret key (Stripe)
1. Left menu -> **Developers** -> **API keys**.
2. In the box **Secret key** press **Reveal** and copy it. It starts with `sk_test_` in test mode
   (`sk_live_` in live mode). This is a password: only paste it into the boxes in step 4.

### Step 3. Tell Stripe where to send payment news (webhook)
1. **Developers** -> **Webhooks** -> **Add destination** (older dashboards: **Add endpoint**).
2. Endpoint URL: `https://<your Hugging Face app address>/stripe/webhook`
   (for example `https://imnotkeril-portfolio-management-pro-api.hf.space/stripe/webhook`).
   It is the address of the **server**, not of the website.
3. Choose these four events (search by name and tick them):
   `checkout.session.completed`, `invoice.paid`, `customer.subscription.updated`,
   `customer.subscription.deleted`.
4. Press **Add**. On the next page find **Signing secret**, press **Reveal**, copy it. It starts
   with `whsec_`.

### Step 4. Put the three values on the server (Hugging Face)
1. Open your Space -> **Settings** -> **Variables and secrets**.
2. Press **New secret** three times. Name on the left, value on the right:
   - `STRIPE_SECRET_KEY` = the `sk_...` key from step 2
   - `STRIPE_WEBHOOK_SECRET` = the `whsec_...` value from step 3
   - `STRIPE_PRICE_ID_PRO` = the `price_...` id from step 1
3. Press **New variable** (not secret): `FRONTEND_URL` = the address of your website, for
   example `https://portfolio-lenz.vercel.app`.
4. Also check that `CLIENT_IP_SIGNING_KEY` exists as a secret here. It must be the same
   long text (32+ characters) that is on Vercel in step 5. If it is missing the server will
   not start.
5. Press the menu with three dots at the top right of the Space -> **Restart Space** (or
   **Factory reboot**). Wait until the status says **Running**.

### Step 5. Same key on the website (Vercel)
1. Vercel -> project `portfolio-management-pro` -> **Settings** -> **Environment Variables**.
2. Add `CLIENT_IP_SIGNING_KEY` with exactly the same value as on Hugging Face, for
   **Production**. Press **Save**. The site must be deployed again for it to take effect
   (the developer does this).

### Step 6. Allow customers to cancel (Stripe)
1. **Settings** (gear) -> **Billing** -> **Customer portal**.
2. Turn the portal on; make sure **Customers can cancel subscriptions** is on. **Save**.

### Step 7. Try it with a test card
1. Open the website, register, press **Billing** -> upgrade.
2. On the Stripe page pay with card `4242 4242 4242 4242`, any future date, any 3 digits, any
   name and ZIP.
3. You return to the site. After a few seconds the plan shows **Pro**. If not, reload once.
4. Press **Manage subscription**, cancel. The plan goes back to Free when the paid period ends.
5. In Stripe -> **Developers** -> **Webhooks** -> your endpoint: the latest events should have a
   green 200. A red 400 means the signing secret in step 4 is wrong.

### Going live
Switch Stripe to live mode (top right). Redo steps 1, 2 and 3 there (live mode has its own
product, key and webhook), replace the three secrets in step 4 with the live ones, restart the
Space. Do not mix test and live values: checkout will fail.

## 1. What the plans are

| Plan | Limit | Defined in |
|---|---|---|
| Free | 1 portfolio; no optimization or forecasting | `FREE_PORTFOLIO_LIMIT` in `services/billing/plans.py`; optimization and forecasting endpoints use `require_pro` |
| Pro | unlimited portfolios, optimization and forecasting | active or trialing subscription whose `current_period_end` is in the future (+1 hour grace) |

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

## 7. Before going live

Stripe live keys and webhook endpoint (section 1), a legal review of the privacy, terms and refund pages, and the same `CLIENT_IP_SIGNING_KEY` on the API and the web host.
