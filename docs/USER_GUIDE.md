# Portfolio Management Terminal — User Guide

This guide explains how to use the web app (branded **WILD MARKET**) to build portfolios, analyse them, optimize allocations, stress-test risk, forecast returns, and manage your plan. It describes the Next.js interface as it is in this repository. For installation, see the [README](README.md).

---

## Contents

1. [Getting started](#getting-started)
2. [Navigation](#navigation)
3. [Dashboard](#dashboard)
4. [Creating a portfolio](#creating-a-portfolio)
5. [Portfolio list and portfolio overview](#portfolio-list-and-portfolio-overview)
6. [Transactions](#transactions)
7. [Analysis](#analysis)
8. [Optimization (Pro)](#optimization-pro)
9. [Risk](#risk)
10. [Forecasting (Pro)](#forecasting-pro)
11. [Plans and billing](#plans-and-billing)
12. [Settings](#settings)
13. [Troubleshooting](#troubleshooting)

---

## Getting started

1. Open the home page and choose **Register** (or **Sign in** if you already have an account).
2. Enter your email and password. After sign in you return to the page you came from, or to the **Dashboard** if there is none. Registration opens the Dashboard.
3. Your session is kept in a cookie. Use **Sign out** in the top bar to end it.

Public pages: home (`/`), sign in (`/login`), register (`/register`). Every other page needs a signed-in session.

---

## Navigation

The top bar shows the pages of the app:

| Item | Page | What it is for |
|------|------|----------------|
| Dashboard | `/dashboard` | Market indices and an overview |
| Portfolio List | `/portfolios` | All your portfolios |
| Create | `/create` | Build a new portfolio |
| Analysis | `/analysis` | Metrics and charts for one portfolio |
| Optimization | `/optimization` | Optimal allocations (Pro) |
| Risk | `/risk` | VaR, Monte Carlo, stress tests, scenarios |
| Forecasting | `/forecasting` | Price and return forecasts (Pro) |
| Billing | `/billing` | Your plan |
| Settings | `/settings` | Account email, billing link, sign out |

Portfolio pages carry their own tabs (see [Portfolio list and portfolio overview](#portfolio-list-and-portfolio-overview)). `/portfolio/new` redirects to `/create`.

---

## Dashboard

The dashboard shows the major market indices fetched from the market-data provider. Index values depend on the provider, so they may be delayed or temporarily unavailable.

---

## Creating a portfolio

Open **Create**. Choose one of four input methods:

- **Text** (fastest): type tickers with weights, for example `AAPL 40%, MSFT 30%, GOOGL 30%`.
- **File**: upload a CSV with positions.
- **Manual**: add one ticker at a time. Each ticker is validated against the market-data provider before you add it.
- **Template**: start from a predefined template and customize it.

Set the starting capital and the rebalance frequency, review the summary, and create the portfolio. Free accounts can hold **one** portfolio; see [Plans and billing](#plans-and-billing).

---

## Portfolio list and portfolio overview

**Portfolio List** shows every portfolio with a sort control. Open a portfolio to reach its overview. From the list you can also edit a portfolio and delete it.

Each portfolio has these tabs:

- **Overview**: holdings, realized and unrealized P&L, dividend income, and the starting capital.
- **Transactions**: the transaction ledger (see [Transactions](#transactions)).
- **Analysis**, **Risk**, **Optimization**, **Forecasting**: open the matching page with that portfolio already selected.

A portfolio runs in one of two modes:

- **Buy-and-hold**: the positions are held as fixed shares from the start date.
- **With transactions (ledger)**: returns follow your transactions (buys, sells, deposits, withdrawals) and the rebalance schedule.

---

## Transactions

Open a portfolio and choose **Transactions**. The page shows total invested and the number of transactions, and an **Add transaction** form.

Supported transaction types include:

- **BUY** and **SELL** of a ticker, with quantity and price
- **DEPOSIT** and **WITHDRAWAL** of cash
- **DIVIDEND** and stock splits (these are synchronized from market data for the portfolio's tickers)

Delete a transaction from the list if it was entered by mistake. Changes update the holdings and the P&L.

---

## Analysis

**Analysis** computes metrics for one portfolio over a period.

1. Pick the **portfolio**, the **start date** and the **end date**.
2. Choose a **comparison**: none, an index ETF (for example SPY), or another portfolio.
3. Set the **risk-free rate** (in percent, for example 4.35).
4. Click **Calculate Metrics**. **Update Prices** refreshes market prices first.

Results are split into five tabs:

- **Overview**: key metrics (total return, CAGR, volatility, maximum drawdown, Sharpe, Sortino, beta, alpha) and the cumulative return chart.
- **Performance**: returns over time, periodic returns, and benchmark comparison.
- **Risk**: risk metrics and drawdowns.
- **Assets & Correlations**: per-asset metrics and the correlation matrix.
- **Export & Reports**: download the analytics as JSON.

---

## Optimization (Pro)

**Optimization** finds allocations for a portfolio. It requires the Pro plan.

1. Choose a **method** from the dropdown (for example **Max Sharpe** or **Min Vol**; the dropdown also offers the other methods in the engine, such as HRP and Black–Litterman). Each method has its own parameters under **Method Parameters**.
2. Set the **covariance estimator**: shrinkage, Ledoit–Wolf, or sample covariance.
3. Set **Constraints** (weight bounds and similar limits).
4. Run the optimization. Results show:
   - **Allocation**: the optimized weights.
   - **Efficient frontier (training window)** and **Correlation (training window)**.
   - **Performance (test window)** and **Optimized vs current (test window)**: the out-of-sample comparison, so the optimized weights are judged on data they were not fitted to.
   - **Suggested trades**: the buys and sells that would move the portfolio to the optimized weights.

Suggested trades are advice, not orders. Nothing is executed from this page.

---

## Risk

**Risk** measures how much the portfolio could lose. Set the **analysis period** first.

- **Key risk metrics** and **VaR & CVaR** at several confidence levels.
- **VaR methods comparison**: historical, parametric and Monte Carlo estimates side by side, with rolling VaR.
- **Monte Carlo simulation**: the distribution of final values with percentiles and a recovery timeline.
- **Historical scenarios** and **Extreme scenarios**: how the portfolio behaved in past crises.
- **Custom scenario**: define your own shocks to the assets and see the impact, including the per-position breakdown.
- **Scenario chain**: apply several scenarios in sequence.

---

## Forecasting (Pro)

**Forecasting** projects future prices or returns. It requires the Pro plan.

1. Select an **asset** (one ticker) or the **portfolio**.
2. Set the **forecast horizon** and the **analysis / validation** window. Out-of-sample testing scores the model on data held back from fitting.
3. Choose a **forecasting method** family:
   - **Classical** (for example ARIMA and GARCH-style models)
   - **Machine Learning**
   - **Deep Learning** (these need the full dependency set; see the README)
   - **Simple** baselines
4. Read **Forecast results** and **All forecasts vs history**. Forecasts are estimates with uncertainty, not guarantees.

---

## Plans and billing

**Billing** (`/billing`) shows your current plan and status.

| | Free | Pro |
|--|------|-----|
| Portfolios | 1 | Unlimited |
| Analysis, Risk, Create | yes | yes |
| Optimization | no | yes |
| Forecasting | no | yes |

- **Upgrade**: use the checkout button on the billing page. Payment is handled by Stripe Checkout.
- **Manage**: use the Customer Portal button to change or cancel the subscription and to update payment details.
- After a successful payment the plan updates when Stripe notifies the server; if it does not change right away, reload the page.

If the billing page says Stripe is not configured, billing is not set up on this server; the rest of the app still works.

---

## Settings

**Settings** shows the email of the signed-in account, links to **Billing & plan**, and has a **Sign out** button.

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| Redirected to sign in | Your session expired. Sign in again. |
| Creating a portfolio is refused | The Free plan allows one portfolio. Delete an existing portfolio or upgrade to Pro. |
| Optimization or Forecasting is locked | These need the Pro plan. See [Plans and billing](#plans-and-billing). |
| Ticker is rejected | The ticker is not known to the market-data provider. Check the symbol (for example `BRK-B`). |
| Prices or metrics are missing | The market-data provider may be unreachable or rate-limiting. Wait and use **Update Prices** or recalculate. |
| Billing page does not show a change after paying | Reload the page. The plan updates after Stripe confirms the payment. |
| The page does not load at all | Check with the operator that the web app and the API are both running. |

---

## Notes

- The app has no keyboard shortcuts of its own. Charts respond to hovering (tooltips) and the standard browser keys apply to forms and tables.
- Figures are computed from market data and your transactions. They are for analysis and are not investment advice.
