"""Generate api/llms.txt from the FastAPI route declarations (static, AST-based).

Run: python scripts/gen_llms_txt.py          (rewrites api/llms.txt)
     python scripts/gen_llms_txt.py --check  (exit 1 if the file is stale)

tests/unit/test_llms_txt.py also compares the file with the live app routes.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    (ROOT / "api" / "main.py", ""),
    (ROOT / "api" / "routers" / "auth.py", None),
    (ROOT / "api" / "routers" / "billing.py", None),
]
MODEL_FILES = [ROOT / "api" / "main.py", ROOT / "api" / "schemas" / "auth.py"]
METHODS = {"get", "post", "put", "patch", "delete"}
OUT = ROOT / "api" / "llms.txt"
LIMIT_NAMES = {"LOGIN_IP_LIMIT": "20/minute"}

DESCRIPTIONS = {
    "POST /analytics/calculate": "Portfolio performance and risk metrics for a date range, optionally against a benchmark or comparison",
    "POST /billing/checkout": "Create a Stripe Checkout session for the Pro plan; returns {url}. 503 if billing is not configured",
    "POST /billing/portal": "Create a Stripe customer-portal session; returns {url}. 503 if billing is not configured",
    "GET /billing/status": "Current user's plan, subscription status and period end",
    "GET /dashboard/indices": "Market index quotes for the dashboard strip",
    "POST /forecasting/asset": "Forecast one asset's price path",
    "POST /forecasting/batch": "Forecast several assets in one call",
    "POST /forecasting/portfolio": "Forecast a whole portfolio's value",
    "GET /health": "Liveness probe; returns {status: ok}",
    "POST /optimization/full": "Run an optimization method and return weights, frontier and metrics for a saved portfolio",
    "POST /optimization/run": "Run one optimization method on a list of tickers",
    "GET /portfolios": "List the user's portfolios",
    "POST /portfolios": "Create a portfolio",
    "DELETE /portfolios/{portfolio_id}": "Delete a portfolio and its ledger",
    "GET /portfolios/{portfolio_id}": "Get one portfolio",
    "PATCH /portfolios/{portfolio_id}": "Update portfolio fields",
    "GET /portfolios/{portfolio_id}/dividends": "List dividends received by the portfolio",
    "POST /portfolios/{portfolio_id}/dividends/sync": "Fetch and record missing dividends from the market data provider",
    "GET /portfolios/{portfolio_id}/holdings": "Current holdings with quantities and values",
    "POST /portfolios/{portfolio_id}/ledger/maintain": "Sync splits, dividends and scheduled rebalances; partial failures are reported in maintain_error",
    "GET /portfolios/{portfolio_id}/pnl": "Realized and unrealized profit and loss",
    "POST /portfolios/{portfolio_id}/positions": "Add a position to the portfolio",
    "DELETE /portfolios/{portfolio_id}/positions/{ticker}": "Remove a position",
    "PATCH /portfolios/{portfolio_id}/positions/{ticker}": "Update a position",
    "POST /portfolios/{portfolio_id}/rebalance/execute": "Apply a rebalance and record its transactions",
    "POST /portfolios/{portfolio_id}/rebalance/preview": "Show the trades a rebalance would make without saving",
    "POST /portfolios/{portfolio_id}/splits/sync": "Fetch and apply missing stock splits",
    "GET /portfolios/{portfolio_id}/strategy": "Get the portfolio's target-weight strategy",
    "PUT /portfolios/{portfolio_id}/strategy": "Save the portfolio's strategy and rebalance schedule",
    "POST /portfolios/{portfolio_id}/strategy/preview": "Preview a strategy against current holdings without saving",
    "GET /portfolios/{portfolio_id}/transactions": "List the portfolio's transactions",
    "POST /portfolios/{portfolio_id}/transactions": "Record a transaction in the portfolio ledger",
    "DELETE /portfolios/{portfolio_id}/transactions/{transaction_id}": "Delete a transaction of this portfolio",
    "POST /risk/custom-scenario": "Apply user-defined shocks to a portfolio",
    "POST /risk/monte-carlo": "Monte Carlo simulation on a ticker list",
    "POST /risk/monte-carlo/full": "Monte Carlo simulation for a saved portfolio",
    "POST /risk/scenario-chain": "Apply a chain of scenarios in sequence",
    "GET /risk/scenarios": "Catalog of built-in stress scenarios",
    "POST /risk/stress-tests": "Stress-test a ticker list against scenarios",
    "POST /risk/var": "Value at Risk and CVaR for a ticker list",
    "POST /risk/var/full": "Value at Risk and CVaR for a saved portfolio",
    "POST /stripe/webhook": "Stripe webhook. Verifies the signature, dedups by event id, re-fetches the subscription and updates plan state",
    "GET /ticker-price/{ticker}": "Latest price for one ticker",
    "POST /validate-tickers": "Check which tickers exist and return their latest price",
}

HEADER = """# WMC Portfolio API

> REST API (FastAPI) behind the Next.js BFF. Portfolio ledger, analytics, optimization,
> risk, forecasting and Stripe billing. JSON in, JSON out. Generated from the route
> declarations by `scripts/gen_llms_txt.py`; `tests/unit/test_llms_txt.py` fails if it
> drifts from the live routes.

## Conventions

- Auth: `Authorization: Bearer <JWT>` from `POST /auth/login`. Auth column: `public`
  (no token), `user` (valid token), `pro` (valid token and an active Pro subscription).
- Every portfolio route is scoped to the token's user; a foreign or unknown
  `portfolio_id` returns 404, never 403.
- Errors: `{"detail": "<message>"}` with 400/401/402/403/404/409/422/429/502/503.
  429 carries `Retry-After`. 422 is request validation.
- Rate limits are per client IP (some per user); the limit shown is the route-level one.
- Interactive OpenAPI docs are disabled when `ENVIRONMENT=production`; this file is the
  reference there.
- Plans and prices live in `services/billing/plans.py`; the owner guide is
  `OWNER_GUIDE.md`.

## Endpoints
"""


def _const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _models() -> dict[str, list[str]]:
    own: dict[str, list[str]] = {}
    bases: dict[str, list[str]] = {}
    for f in MODEL_FILES + [ROOT / "api" / "routers" / "billing.py"]:
        for n in ast.parse(f.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.ClassDef):
                own[n.name] = [
                    f"{s.target.id}: {ast.unparse(s.annotation)}"
                    for s in n.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                ]
                bases[n.name] = [b.id for b in n.bases if isinstance(b, ast.Name)]

    def fields(name: str) -> list[str]:
        inherited = [x for b in bases.get(name, []) for x in fields(b)]
        return inherited + own.get(name, [])

    return {n: f for n in own if (f := fields(n))}


def _auth(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    text = ast.unparse(fn)
    if "require_pro" in text:
        return "pro"
    if "get_current_user" in text:
        return "user"
    return "public"


def _limits(deco: ast.Call) -> str:
    for kw in deco.keywords:
        if kw.arg == "dependencies":
            for sub in ast.walk(kw.value):
                if (
                    isinstance(sub, ast.Call)
                    and getattr(sub.func, "id", "") == "rate_limit"
                ):
                    arg = sub.args[0] if sub.args else None
                    if arg is None:
                        return ""
                    return _const(arg) or LIMIT_NAMES.get(
                        ast.unparse(arg), ast.unparse(arg)
                    )
                if isinstance(sub, ast.Name) and sub.id == "_monte_carlo_slot":
                    return "1 concurrent run per user"
    return ""


def collect() -> list[dict]:
    models = _models()
    rows: list[dict] = []
    for path, base_prefix in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        prefix = base_prefix or ""
        if base_prefix is None:
            for n in tree.body:
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                    if getattr(n.value.func, "id", "") == "APIRouter":
                        for kw in n.value.keywords:
                            if kw.arg == "prefix":
                                prefix = _const(kw.value) or ""
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in fn.decorator_list:
                if not (
                    isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr in METHODS
                    and isinstance(deco.func.value, ast.Name)
                    and deco.func.value.id in {"app", "router"}
                ):
                    continue
                route = _const(deco.args[0]) if deco.args else None
                if route is None or "include_in_schema=False" in ast.unparse(deco):
                    continue
                params, body = [], []
                args = fn.args.args + fn.args.kwonlyargs
                defaults = (
                    [None] * (len(fn.args.args) - len(fn.args.defaults))
                    + list(fn.args.defaults)
                    + list(fn.args.kw_defaults)
                )
                for a, d in zip(args, defaults):
                    if a.arg in {"current_user", "request", "self"} or (
                        d is not None and "Depends" in ast.unparse(d)
                    ):
                        continue
                    ann = ast.unparse(a.annotation) if a.annotation else "str"
                    name = ann.split("[")[0]
                    if f"{{{a.arg}}}" in route:
                        params.append(f"path `{a.arg}`: {ann}")
                    elif name in models:
                        body.append(f"`{ann}` {{ {', '.join(models[name])} }}")
                    elif (
                        d is not None
                        and "Query" in ast.unparse(d)
                        or ann
                        in {
                            "str",
                            "int",
                            "float",
                            "bool",
                            "str | None",
                            "int | None",
                        }
                    ):
                        params.append(f"query `{a.arg}`: {ann}")
                    else:
                        body.append(f"`{ann}` (JSON body)")
                if "Depends()" in ast.unparse(
                    fn
                ) and "OAuth2PasswordRequestForm" in ast.unparse(fn):
                    body.append("form-urlencoded `username` (email), `password`")
                doc = (ast.get_docstring(fn) or "").strip().split("\n")[0]
                key = f"{deco.func.attr.upper()} {prefix + route}"
                doc = DESCRIPTIONS.get(key, doc)
                rows.append(
                    {
                        "method": deco.func.attr.upper(),
                        "path": prefix + route,
                        "auth": _auth(fn),
                        "limit": _limits(deco),
                        "params": params,
                        "body": body,
                        "doc": doc or fn.name.replace("_", " "),
                        "fn": fn.name,
                    }
                )
    rows.sort(key=lambda r: (r["path"], r["method"]))
    return rows


def render() -> str:
    parts = [HEADER]
    for r in collect():
        parts.append(f"### {r['method']} {r['path']}\n")
        parts.append(
            f"{r['doc'][:1].upper() + r['doc'][1:].rstrip('.')}. Auth: {r['auth']}."
            + (f" Rate limit: {r['limit']}." if r["limit"] else "")
            + "\n"
        )
        for p in r["params"]:
            parts.append(f"- {p}")
        for b in r["body"]:
            parts.append(f"- body {b}")
        if r["path"] == "/stripe/webhook":
            parts.append(
                "- raw body + `Stripe-Signature` header; called by Stripe only"
            )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text(encoding="utf-8") == text else 1)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({text.count('### ')} operations)")
