"""
Build a JSON-serializable bundle for the Next.js optimization page (Streamlit parity).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.analytics_engine.performance import calculate_annualized_return
from core.analytics_engine.ratios import calculate_sharpe_ratio, calculate_sortino_ratio
from core.analytics_engine.risk_metrics import (
    calculate_max_drawdown,
    calculate_volatility,
)
from core.exceptions import CalculationError, InsufficientDataError
from core.optimization_engine.base import OptimizationResult
from services.analytics_service import AnalyticsService
from services.data_service import DataService
from services.optimization_service import OptimizationService
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

RISK_FREE = 0.0435
TRADING_DAYS = 252


def _optimization_period_bounds(
    start_date: date,
    end_date: date,
    out_of_sample: bool,
    training_ratio: float,
) -> tuple[date, date]:
    if out_of_sample:
        analysis_days = (end_date - start_date).days
        training_days = int(analysis_days * training_ratio)
        opt_start = start_date - timedelta(days=training_days)
        opt_end = start_date
        return opt_start, opt_end
    return start_date, end_date


def _metrics_from_returns(
    rets: pd.Series | None,
    risk_free: float = RISK_FREE,
) -> dict[str, float]:
    if rets is None or rets.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    total = float((1 + rets).prod() - 1)
    ann = float(calculate_annualized_return(rets))
    vol_r = calculate_volatility(rets)
    vol = float(vol_r.get("annual", 0.0) if isinstance(vol_r, dict) else vol_r)
    sharpe = float(calculate_sharpe_ratio(rets, risk_free_rate=risk_free) or 0.0)
    sortino = float(calculate_sortino_ratio(rets, risk_free_rate=risk_free) or 0.0)
    dd = calculate_max_drawdown(rets)
    mdd = float(dd[0] if isinstance(dd, tuple) else dd)
    return {
        "total_return": total,
        "annualized_return": ann,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
    }


def _interpret_comparison(opt: dict[str, float], cur: dict[str, float]) -> str:
    lines = ["**Optimization Results Analysis:**"]
    sd = opt.get("sharpe_ratio", 0) - cur.get("sharpe_ratio", 0)
    rd = opt.get("annualized_return", 0) - cur.get("annualized_return", 0)
    vd = opt.get("volatility", 0) - cur.get("volatility", 0)
    dd = opt.get("max_drawdown", 0) - cur.get("max_drawdown", 0)
    if abs(sd) < 0.1:
        lines.append(
            f"Sharpe ratio: Similar ({opt.get('sharpe_ratio', 0):.2f} vs {cur.get('sharpe_ratio', 0):.2f})"
        )
    elif sd > 0:
        lines.append(
            f"✓ Sharpe ratio improved by {sd:.2f} ({opt.get('sharpe_ratio', 0):.2f} vs {cur.get('sharpe_ratio', 0):.2f})"
        )
    else:
        lines.append(
            f"⚠ Sharpe ratio decreased by {abs(sd):.2f} ({opt.get('sharpe_ratio', 0):.2f} vs {cur.get('sharpe_ratio', 0):.2f})"
        )
    if abs(rd) < 0.01:
        lines.append(
            f"Annual return: Similar ({opt.get('annualized_return', 0)*100:.2f}% vs {cur.get('annualized_return', 0)*100:.2f}%)"
        )
    elif rd > 0:
        lines.append(
            f"✓ Annual return increased by {rd*100:.2f}% ({opt.get('annualized_return', 0)*100:.2f}% vs {cur.get('annualized_return', 0)*100:.2f}%)"
        )
    else:
        lines.append(
            f"Annual return decreased by {abs(rd)*100:.2f}% ({opt.get('annualized_return', 0)*100:.2f}% vs {cur.get('annualized_return', 0)*100:.2f}%)"
        )
    if abs(vd) < 0.01:
        lines.append(
            f"Volatility: Similar ({opt.get('volatility', 0)*100:.2f}% vs {cur.get('volatility', 0)*100:.2f}%)"
        )
    elif vd < 0:
        lines.append(
            f"✓ Volatility reduced by {abs(vd)*100:.2f}% ({opt.get('volatility', 0)*100:.2f}% vs {cur.get('volatility', 0)*100:.2f}%)"
        )
    else:
        lines.append(
            f"⚠ Volatility increased by {vd*100:.2f}% ({opt.get('volatility', 0)*100:.2f}% vs {cur.get('volatility', 0)*100:.2f}%)"
        )
    if abs(dd) < 0.01:
        lines.append(
            f"Max drawdown: Similar ({opt.get('max_drawdown', 0)*100:.2f}% vs {cur.get('max_drawdown', 0)*100:.2f}%)"
        )
    elif dd < 0:
        lines.append(
            f"✓ Max drawdown reduced by {abs(dd)*100:.2f}% ({opt.get('max_drawdown', 0)*100:.2f}% vs {cur.get('max_drawdown', 0)*100:.2f}%)"
        )
    else:
        lines.append(
            f"⚠ Max drawdown increased by {dd*100:.2f}% ({opt.get('max_drawdown', 0)*100:.2f}% vs {cur.get('max_drawdown', 0)*100:.2f}%)"
        )
    improvements = sum(
        [
            1 if sd > 0.1 else 0,
            1 if rd > 0.01 else 0,
            1 if vd < -0.01 else 0,
            1 if dd < -0.01 else 0,
        ]
    )
    if improvements >= 3:
        lines.append(
            "\n✓ Optimization shows significant improvements across multiple metrics"
        )
    elif improvements >= 2:
        lines.append("\nOptimization shows moderate improvements")
    elif improvements >= 1:
        lines.append("\nOptimization shows limited improvements")
    else:
        lines.append(
            "\n⚠ Optimization did not improve metrics - consider adjusting constraints or method"
        )
    return "\n".join(lines)


def _interpret_allocation(
    current_w: dict[str, float], optimal_w: dict[str, float]
) -> str:
    if not optimal_w:
        return ""
    lines = ["**Allocation Changes Analysis:**"]
    changes = []
    for t in optimal_w:
        d = optimal_w[t] - current_w.get(t, 0.0)
        if abs(d) > 0.01:
            changes.append((t, current_w.get(t, 0.0), optimal_w[t], d))
    if not changes:
        lines.append(
            "No significant weight changes required - portfolio is already close to optimal"
        )
        return "\n".join(lines)
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    inc = [c for c in changes if c[3] > 0.01]
    dec = [c for c in changes if c[3] < -0.01]
    if inc:
        lines.append(
            f"Largest increase: {inc[0][0]} ({inc[0][1]:.1%} → {inc[0][2]:.1%}, +{inc[0][3]:.1%})"
        )
        if len(inc) > 1:
            lines.append(f"{len(inc)} asset(s) need weight increases")
    if dec:
        lines.append(
            f"Largest decrease: {dec[0][0]} ({dec[0][1]:.1%} → {dec[0][2]:.1%}, {dec[0][3]:.1%})"
        )
        if len(dec) > 1:
            lines.append(f"{len(dec)} asset(s) need weight decreases")
    mx = max(optimal_w.values()) if optimal_w else 0
    if mx > 0.5:
        lines.append(
            f"⚠ High concentration: Max weight is {mx:.1%} - Consider diversification"
        )
    elif mx > 0.3:
        lines.append(f"Moderate concentration: Max weight is {mx:.1%}")
    else:
        lines.append(f"✓ Well-diversified: Max weight is {mx:.1%}")
    tot = sum(abs(c[3]) for c in changes)
    if tot > 0.5:
        lines.append(f"⚠ Large rebalancing required: Total weight changes = {tot:.1%}")
    elif tot > 0.2:
        lines.append(f"Moderate rebalancing needed: Total weight changes = {tot:.1%}")
    else:
        lines.append(f"✓ Small rebalancing: Total weight changes = {tot:.1%}")
    return "\n".join(lines)


def _interpret_trades(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return ""
    lines = ["**Trade List Analysis:**"]
    buys = [t for t in trades if t.get("action") == "BUY"]
    sells = [t for t in trades if t.get("action") == "SELL"]
    total_value = sum(float(t.get("value", 0) or 0) for t in trades)
    lines.append(f"Total trades: {len(trades)} ({len(buys)} buys, {len(sells)} sells)")
    lines.append(f"Total trade value: ${total_value:,.2f}")
    sorted_trades = sorted(
        trades, key=lambda x: abs(float(x.get("value", 0) or 0)), reverse=True
    )
    if sorted_trades:
        top = sorted_trades[0]
        lines.append(
            f"Largest trade: {top.get('ticker')} - {top.get('action')} ${abs(float(top.get('value', 0) or 0)):,.2f}"
        )
    if total_value < 1000:
        lines.append("✓ Small rebalancing - Low transaction costs expected")
    elif total_value < 10000:
        lines.append("Moderate rebalancing - Consider transaction costs")
    else:
        lines.append("⚠ Large rebalancing - Significant transaction costs may apply")
    return "\n".join(lines)


def portfolio_snapshot_rows(
    portfolio_service: PortfolioService,
    data_service: DataService,
    portfolio_id: str,
) -> list[dict[str, Any]]:
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    if not portfolio:
        return []
    positions = portfolio.get_all_positions()
    tickers = [p.ticker for p in positions if p.ticker != "CASH"]
    prices: dict[str, float] = {}
    for t in tickers:
        p = data_service.fetch_current_price(t)
        if p:
            prices[t] = float(p)
    if any(p.ticker == "CASH" for p in positions):
        prices["CASH"] = 1.0
    current_value = portfolio.calculate_current_value(prices)
    rows = []
    for pos in positions:
        if pos.ticker == "CASH":
            price = 1.0
            pos_value = pos.shares
        else:
            price = prices.get(pos.ticker)
            if not price:
                continue
            pos_value = price * pos.shares
        w = (pos_value / current_value) if current_value > 0 else 0.0
        rows.append(
            {
                "ticker": pos.ticker,
                "shares": float(pos.shares),
                "price": float(price),
                "value": float(pos_value),
                "weight": w,
            }
        )
    return rows


def _current_weights_for_result(
    portfolio_service: PortfolioService,
    data_service: DataService,
    portfolio_id: str,
    result_tickers: list[str],
) -> dict[str, float]:
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    if not portfolio:
        return {}
    positions = portfolio.get_all_positions()
    tickers = [p.ticker for p in positions if p.ticker != "CASH"]
    prices: dict[str, float] = {}
    for t in tickers:
        p = data_service.fetch_current_price(t)
        if p:
            prices[t] = float(p)
    if any(p.ticker == "CASH" for p in positions):
        prices["CASH"] = 1.0
    current_value = portfolio.calculate_current_value(prices)
    cw: dict[str, float] = {}
    for pos in positions:
        if pos.ticker not in result_tickers:
            continue
        if pos.ticker == "CASH":
            pos_value = pos.shares
        else:
            pr = prices.get(pos.ticker)
            if not pr:
                continue
            pos_value = pr * pos.shares
        cw[pos.ticker] = pos_value / current_value if current_value > 0 else 0.0
    return cw


def _build_optimized_returns(
    analytics_service: AnalyticsService,
    result: OptimizationResult,
    start_date: date,
    end_date: date,
    current_returns: pd.Series | None,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """
    Backtest optimal weights using the same buy-and-hold mechanics as the live
    portfolio in AnalyticsService (fixed share counts from day-0 weights).

    Using sum(w_i * r_{i,t}) with constant w would imply daily rebalancing and
    unfairly depress or inflate performance vs the current portfolio series.
    """
    optimal_weights = result.get_weights_dict()
    try:
        optimized_returns = (
            analytics_service.simulate_buy_and_hold_returns_from_weights(
                optimal_weights,
                start_date,
                end_date,
            )
        )
    except Exception:
        logger.exception("simulate_buy_and_hold_returns_from_weights failed")
        return None, pd.DataFrame()

    if optimized_returns is None or optimized_returns.empty:
        return None, pd.DataFrame()

    if current_returns is not None and not current_returns.empty:
        aligned = current_returns.index.intersection(optimized_returns.index)
        optimized_returns = optimized_returns.reindex(aligned).dropna()
    return optimized_returns, pd.DataFrame()


def _drawdown_pct(values: pd.Series) -> pd.Series:
    if values is None or values.empty:
        return pd.Series(dtype=float)
    peak = values.expanding().max()
    return (values - peak) / peak * 100.0


def _notebook_constant_weight_returns(
    returns_subset: pd.DataFrame,
    weights: dict[str, float],
    ticker_order: list[str],
) -> pd.Series:
    """
    Daily portfolio return as r_p = r_matrix @ w (constant weights), notebook-style.
    Rows with missing asset returns are dropped so optimized/current stay aligned.
    """
    cols = [c for c in ticker_order if c in returns_subset.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = returns_subset[cols].dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)
    w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
    if w.sum() > 1e-12:
        w = w / w.sum()
    return pd.Series(sub.values @ w, index=sub.index, dtype=float)


def _clean_frontier_portfolio_point(p: Any) -> dict[str, Any] | None:
    if not p or not isinstance(p, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in p.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {
                str(kk): (
                    float(vv)
                    if isinstance(vv, (np.floating, np.integer, float))
                    else vv
                )
                for kk, vv in v.items()
            }
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v
            ]
    return out


def _build_frontier_payload_from_fd(
    fd: dict[str, Any],
    *,
    frontier_analytics: dict[str, Any],
    benchmark_for_charts: str | None,
    result: OptimizationResult,
    current_metrics: dict[str, float],
    fallback_validation_period: bool = False,
) -> dict[str, Any]:
    vols = [float(v) for v in (fd.get("volatilities") or [])]
    rets = [float(r) for r in (fd.get("returns") or [])]
    portfolios = fd.get("portfolios") or []

    tangency_portfolio = None
    min_variance_portfolio = None
    if portfolios:
        cleaned_ports = []
        for p in portfolios:
            cp = _clean_frontier_portfolio_point(p)
            if cp:
                cleaned_ports.append(cp)
        if cleaned_ports:

            def _port_sharpe(p: dict[str, Any]) -> float:
                r = float(p.get("expected_return") or 0.0)
                v = float(p.get("volatility") or 0.0)
                if v <= 1e-12:
                    return -np.inf
                return (r - RISK_FREE) / v

            tangency_portfolio = max(cleaned_ports, key=_port_sharpe)
            min_variance_portfolio = min(
                cleaned_ports,
                key=lambda p: float(p.get("volatility") or np.inf),
            )

    if tangency_portfolio is None:
        tangency_portfolio = _clean_frontier_portfolio_point(
            fd.get("tangency_portfolio")
        )
    if min_variance_portfolio is None:
        min_variance_portfolio = _clean_frontier_portfolio_point(
            fd.get("min_variance_portfolio")
        )

    frontier_current_returns = frontier_analytics.get("portfolio_returns")
    if hasattr(frontier_current_returns, "empty") and frontier_current_returns.empty:
        frontier_current_returns = None
    frontier_current_metrics = (
        _metrics_from_returns(frontier_current_returns, RISK_FREE)
        if frontier_current_returns is not None and not frontier_current_returns.empty
        else current_metrics
    )

    payload: dict[str, Any] = {
        "volatilities_pct": [v * 100 for v in vols],
        "returns_pct": [r * 100 for r in rets],
        "tangency_portfolio": tangency_portfolio,
        "min_variance_portfolio": min_variance_portfolio,
        "optimized_point": {
            "volatility_pct": float(result.volatility or 0) * 100,
            "return_pct": float(result.expected_return or 0) * 100,
            "sharpe": float(result.sharpe_ratio or 0),
        },
        "current_point": {
            "volatility_pct": float(frontier_current_metrics.get("volatility", 0))
            * 100,
            "return_pct": float(frontier_current_metrics.get("annualized_return", 0))
            * 100,
        },
    }

    frontier_benchmark_returns = frontier_analytics.get("benchmark_returns")
    if (
        frontier_benchmark_returns is not None
        and not frontier_benchmark_returns.empty
        and benchmark_for_charts
    ):
        br = calculate_annualized_return(frontier_benchmark_returns)
        bv = calculate_volatility(frontier_benchmark_returns)
        bvol = float(bv.get("annual", 0) if isinstance(bv, dict) else bv)
        payload["benchmark_point"] = {
            "volatility_pct": bvol * 100,
            "return_pct": float(br) * 100 if br is not None else 0.0,
        }

    if fallback_validation_period:
        payload["fallback_validation_period"] = True
    return payload


def _build_notebook_split_bundle(
    optimization_service: OptimizationService,
    portfolio_service: PortfolioService,
    *,
    portfolio_id: str,
    method: str,
    start_date: date,
    end_date: date,
    constraints: dict[str, Any] | None = None,
    benchmark_ticker: str | None = None,
    method_params: dict[str, Any] | None = None,
    benchmark_for_charts: str | None = None,
    include_efficient_frontier: bool = True,
    frontier_n_points: int = 150,
    include_sensitivity: bool = False,
    sensitivity_analysis_type: str = "returns",
    notebook_train_fraction: float = 0.7,
) -> dict[str, Any]:
    """
    Row-based train/validation/test split on the user window:
    - train fraction is user-provided (notebook_train_fraction)
    - remaining rows are split equally into validation and test
    - optimization fits on train rows
    - performance charts and primary comparison metrics use test rows
    """
    data_service = optimization_service._data_service  # noqa: SLF001
    analytics_service = AnalyticsService()

    bench_for_load = (
        benchmark_ticker if method in ("min_tracking_error", "max_alpha") else None
    )
    returns_full, _ = optimization_service.load_portfolio_returns_and_benchmark(
        portfolio_id,
        start_date,
        end_date,
        benchmark_ticker=bench_for_load,
    )

    n = len(returns_full)
    min_train, min_validation, min_test = 30, 20, 20
    if n < min_train + min_validation + min_test:
        raise InsufficientDataError(
            f"3-way split needs at least {min_train + min_validation + min_test} return rows, got {n}"
        )

    train_end_idx = int(n * float(notebook_train_fraction))
    train_end_idx = max(
        min_train,
        min(train_end_idx, n - (min_validation + min_test)),
    )

    remaining = n - train_end_idx
    validation_len = remaining // 2
    test_len = remaining - validation_len

    if validation_len < min_validation:
        need = min_validation - validation_len
        validation_len += need
        test_len -= need
    if test_len < min_test:
        need = min_test - test_len
        test_len += need
        validation_len -= need

    if validation_len < min_validation or test_len < min_test:
        raise InsufficientDataError(
            "Unable to build validation/test windows with minimum row requirements."
        )

    validation_start_idx = train_end_idx
    validation_end_idx = validation_start_idx + validation_len

    returns_train = returns_full.iloc[:train_end_idx]
    returns_validation = returns_full.iloc[validation_start_idx:validation_end_idx]
    returns_test = returns_full.iloc[validation_end_idx:]

    train_start = returns_train.index.min().date()
    train_end = returns_train.index.max().date()
    validation_start = returns_validation.index.min().date()
    validation_end = returns_validation.index.max().date()
    test_start = returns_test.index.min().date()
    test_end = returns_test.index.max().date()

    result = optimization_service.optimize_portfolio(
        portfolio_id=portfolio_id,
        method=method,
        start_date=train_start,
        end_date=train_end,
        constraints=constraints,
        benchmark_ticker=benchmark_ticker,
        method_params=method_params,
        out_of_sample=False,
        training_ratio=0.3,
    )

    base: dict[str, Any] = {
        "optimization": result.to_dict(),
        "success": result.success,
        "message": result.message or "",
        "optimization_period": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
        },
        "validation_period": {
            "start": validation_start.isoformat(),
            "end": validation_end.isoformat(),
        },
        "test_period": {
            "start": test_start.isoformat(),
            "end": test_end.isoformat(),
        },
        "full_data_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "notebook_split": True,
        "split_mode": "train_validation_test",
        "notebook_train_fraction": notebook_train_fraction,
        "out_of_sample": False,
        "training_ratio": notebook_train_fraction,
        "split_fractions": {
            "train": float(len(returns_train)) / float(n),
            "validation": float(len(returns_validation)) / float(n),
            "test": float(len(returns_test)) / float(n),
        },
        "portfolio_snapshot": portfolio_snapshot_rows(
            portfolio_service, data_service, portfolio_id
        ),
        "constraints_applied": constraints or {},
        "method": method,
        "method_params": method_params or {},
    }

    if not result.success:
        return base

    warnings_list: list[str] = []
    min_ret = (constraints or {}).get("min_return")
    if min_ret is not None and result.expected_return is not None:
        if result.expected_return < min_ret:
            warnings_list.append(
                f"Expected return ({result.expected_return:.2%}) is below minimum ({min_ret:.2%}). "
                "Try relaxing constraints."
            )

    if method in ("cvar_optimization", "mean_cvar"):
        tr_n = len(returns_train)
        if 0 < tr_n < 500:
            warnings_list.append(
                f"CVaR-based methods are more stable with larger training samples "
                f"(recommended 500+, training rows {tr_n})."
            )

    optimal_w = result.get_weights_dict()
    current_w = _current_weights_for_result(
        portfolio_service, data_service, portfolio_id, list(result.tickers)
    )

    r_opt = _notebook_constant_weight_returns(
        returns_test, optimal_w, list(result.tickers)
    )
    r_cur = _notebook_constant_weight_returns(
        returns_test, current_w, list(result.tickers)
    )

    validation_metrics: dict[str, dict[str, float]] | None = None
    if not returns_validation.empty:
        r_opt_val = _notebook_constant_weight_returns(
            returns_validation, optimal_w, list(result.tickers)
        )
        r_cur_val = _notebook_constant_weight_returns(
            returns_validation, current_w, list(result.tickers)
        )
        common_val = r_opt_val.index.intersection(r_cur_val.index)
        if not common_val.empty:
            r_opt_val = r_opt_val.reindex(common_val).dropna()
            r_cur_val = r_cur_val.reindex(common_val).dropna()
            validation_metrics = {
                "optimized": _metrics_from_returns(r_opt_val, RISK_FREE),
                "current": _metrics_from_returns(r_cur_val, RISK_FREE),
            }

    charts: dict[str, Any] = {"cumulative_returns": [], "drawdown": []}
    current_metrics: dict[str, float]
    optimized_metrics: dict[str, float]

    if not r_opt.empty and not r_cur.empty:
        common_idx = r_opt.index.intersection(r_cur.index)
        r_o = r_opt.reindex(common_idx).dropna()
        r_c = r_cur.reindex(common_idx).dropna()
        ix = r_o.index.intersection(r_c.index)
        r_o = r_o.reindex(ix)
        r_c = r_c.reindex(ix)

        ba: pd.Series | None = None
        if benchmark_for_charts:
            try:
                bmp = data_service.fetch_historical_prices(
                    benchmark_for_charts,
                    test_start,
                    test_end,
                    use_cache=True,
                    save_to_db=False,
                )
                if (
                    isinstance(bmp, pd.DataFrame)
                    and not bmp.empty
                    and "Date" in bmp.columns
                ):
                    bmp = bmp.set_index("Date")
                    bmp.index = pd.to_datetime(bmp.index, errors="coerce")
                    if bmp.index.tz is not None:
                        bmp.index = bmp.index.tz_localize(None)
                    ba = bmp["Adjusted_Close"].astype(float).pct_change().dropna()
                    ba = ba.reindex(ix).dropna()
                    ix = ix.intersection(ba.index)
                    r_o = r_o.reindex(ix)
                    r_c = r_c.reindex(ix)
                    ba = ba.reindex(ix)
            except Exception as exc:
                logger.warning("Benchmark (notebook test window): %s", exc)
                ba = None

        optimized_metrics = _metrics_from_returns(r_o, RISK_FREE)
        current_metrics = _metrics_from_returns(r_c, RISK_FREE)

        try:
            cur_cum = (1 + r_c).cumprod() - 1
            opt_cum = (1 + r_o).cumprod() - 1
            bench_cum = None
            if ba is not None and not ba.empty:
                bench_cum = (1 + ba).cumprod() - 1

            cum_rows: list[dict[str, Any]] = []
            for idx in ix:
                row: dict[str, Any] = {
                    "x": str(idx)[:10],
                    "current": float(cur_cum.loc[idx]) * 100.0,
                    "optimized": float(opt_cum.loc[idx]) * 100.0,
                }
                if (
                    bench_cum is not None
                    and idx in bench_cum.index
                    and pd.notna(bench_cum.loc[idx])
                ):
                    row["benchmark"] = float(bench_cum.loc[idx]) * 100.0
                cum_rows.append(row)
            charts["cumulative_returns"] = cum_rows

            init = 10000.0
            opt_vals = (1 + r_o).cumprod() * init
            cur_vals = (1 + r_c).cumprod() * init
            opt_dd = _drawdown_pct(opt_vals)
            cur_dd = _drawdown_pct(cur_vals)
            bench_dd_series = None
            if ba is not None and not ba.empty:
                bv = (1 + ba).cumprod() * init
                bench_dd_series = _drawdown_pct(bv)

            dd_rows: list[dict[str, Any]] = []
            for idx in ix:
                dd_rows.append(
                    {
                        "x": str(idx)[:10],
                        "optimized": (
                            float(opt_dd.loc[idx])
                            if idx in opt_dd.index and pd.notna(opt_dd.loc[idx])
                            else None
                        ),
                        "current": (
                            float(cur_dd.loc[idx])
                            if idx in cur_dd.index and pd.notna(cur_dd.loc[idx])
                            else None
                        ),
                        "benchmark": (
                            float(bench_dd_series.loc[idx])
                            if bench_dd_series is not None
                            and idx in bench_dd_series.index
                            and pd.notna(bench_dd_series.loc[idx])
                            else None
                        ),
                    }
                )
            charts["drawdown"] = dd_rows
        except Exception as exc:
            logger.warning("Notebook performance charts failed: %s", exc)
            warnings_list.append(f"Charts: {exc}")
    else:
        optimized_metrics = {
            "total_return": 0.0,
            "annualized_return": float(result.expected_return or 0.0),
            "volatility": float(result.volatility or 0.0),
            "sharpe_ratio": float(result.sharpe_ratio or 0.0),
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        current_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        warnings_list.append(
            "Notebook test window produced no aligned returns for comparison."
        )

    base["metrics"] = {"current": current_metrics, "optimized": optimized_metrics}
    if validation_metrics is not None:
        base["validation_metrics"] = validation_metrics
    base["interpretation_comparison"] = _interpret_comparison(
        optimized_metrics, current_metrics
    )

    allocation = []
    for t in result.tickers:
        cw = current_w.get(t, 0.0)
        ow = optimal_w.get(t, 0.0)
        allocation.append(
            {
                "ticker": t,
                "current_weight": cw,
                "optimal_weight": ow,
                "difference": ow - cw,
            }
        )
    base["allocation"] = allocation
    base["interpretation_allocation"] = _interpret_allocation(current_w, optimal_w)

    trades: list[dict[str, Any]] = []
    try:
        trades = optimization_service.generate_trade_list(portfolio_id, result)
    except Exception as exc:
        logger.warning("Trade list failed: %s", exc)
        warnings_list.append(f"Could not generate trade list: {exc}")
    base["trades"] = trades
    base["interpretation_trades"] = _interpret_trades(trades) if trades else ""

    base["charts"] = charts

    frontier_payload: dict[str, Any] | None = None
    frontier_analytics: dict[str, Any] = {}
    try:
        frontier_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            train_start,
            train_end,
            benchmark_ticker=benchmark_for_charts if benchmark_for_charts else None,
        )
    except Exception as exc:
        logger.warning("Analytics for frontier period (notebook) failed: %s", exc)

    if include_efficient_frontier:
        try:
            fd = optimization_service.generate_efficient_frontier(
                portfolio_id=portfolio_id,
                start_date=train_start,
                end_date=train_end,
                n_points=frontier_n_points,
                constraints=constraints,
            )
            frontier_payload = _build_frontier_payload_from_fd(
                fd,
                frontier_analytics=frontier_analytics,
                benchmark_for_charts=benchmark_for_charts,
                result=result,
                current_metrics=current_metrics,
            )
        except InsufficientDataError:
            frontier_payload = None
        except Exception as exc:
            logger.warning("Efficient frontier failed (notebook): %s", exc)
            warnings_list.append(f"Efficient frontier: {exc}")

    base["efficient_frontier"] = frontier_payload

    correlation_block: dict[str, Any] | None = None
    try:
        price_frames = []
        for ticker in result.tickers:
            if ticker == "CASH":
                continue
            prices = data_service.fetch_historical_prices(
                ticker, train_start, train_end, use_cache=True, save_to_db=False
            )
            if (
                isinstance(prices, pd.DataFrame)
                and not prices.empty
                and "Date" in prices.columns
            ):
                prices = prices[["Date", "Adjusted_Close"]].copy()
                prices["Ticker"] = ticker
                price_frames.append(prices)
        if len(price_frames) >= 2:
            combined = pd.concat(price_frames, ignore_index=True)
            price_data = combined.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="first",
            )
            rets = price_data.pct_change().dropna()
            if not rets.empty and rets.shape[1] >= 2:
                corr = rets.corr()
                labels = [str(c) for c in corr.columns]
                matrix = [
                    [float(corr.iloc[i, j]) for j in range(len(labels))]
                    for i in range(len(labels))
                ]
                correlation_block = {
                    "tickers": labels,
                    "matrix": matrix,
                    "interpretation": _interpret_corr_block(
                        corr, optimal_w, len(labels)
                    ),
                }
    except Exception as exc:
        logger.warning("Correlation block failed (notebook): %s", exc)

    base["correlation"] = correlation_block

    sensitivity_block: dict[str, Any] | None = None
    if include_sensitivity:
        try:
            if method in ("min_tracking_error", "max_alpha"):
                warnings_list.append(
                    "Sensitivity skipped: not supported for this method."
                )
            else:
                sensitivity_block = optimization_service.perform_sensitivity_analysis(
                    portfolio_id=portfolio_id,
                    method=method,
                    start_date=train_start,
                    end_date=train_end,
                    base_constraints=constraints,
                    analysis_type=sensitivity_analysis_type,
                    variation_range=0.1,
                    num_points=10,
                )
                sens_results = sensitivity_block.get("results") or []
                base["interpretation_sensitivity"] = _interpret_sensitivity_block(
                    sens_results
                )
        except (CalculationError, ValueError, InsufficientDataError) as exc:
            warnings_list.append(f"Sensitivity: {exc}")
        except Exception as exc:
            logger.warning("Sensitivity failed (notebook): %s", exc)
            warnings_list.append(f"Sensitivity: {exc}")

    base["sensitivity"] = sensitivity_block
    base["warnings"] = warnings_list
    return base


def build_optimization_full_bundle(
    optimization_service: OptimizationService,
    portfolio_service: PortfolioService,
    *,
    portfolio_id: str,
    method: str,
    start_date: date,
    end_date: date,
    constraints: dict[str, Any] | None = None,
    benchmark_ticker: str | None = None,
    method_params: dict[str, Any] | None = None,
    out_of_sample: bool = False,
    training_ratio: float = 0.3,
    benchmark_for_charts: str | None = None,
    include_efficient_frontier: bool = True,
    frontier_n_points: int = 150,
    include_sensitivity: bool = False,
    sensitivity_analysis_type: str = "returns",
    notebook_split: bool = False,
    notebook_train_fraction: float = 0.7,
) -> dict[str, Any]:
    """
    Run optimization and assemble charts, metrics, trades, frontier, correlation.
    """
    if notebook_split:
        return _build_notebook_split_bundle(
            optimization_service,
            portfolio_service,
            portfolio_id=portfolio_id,
            method=method,
            start_date=start_date,
            end_date=end_date,
            constraints=constraints,
            benchmark_ticker=benchmark_ticker,
            method_params=method_params,
            benchmark_for_charts=benchmark_for_charts,
            include_efficient_frontier=include_efficient_frontier,
            frontier_n_points=frontier_n_points,
            include_sensitivity=include_sensitivity,
            sensitivity_analysis_type=sensitivity_analysis_type,
            notebook_train_fraction=notebook_train_fraction,
        )

    data_service = optimization_service._data_service  # noqa: SLF001
    analytics_service = AnalyticsService()

    opt_start, opt_end = _optimization_period_bounds(
        start_date, end_date, out_of_sample, training_ratio
    )

    result = optimization_service.optimize_portfolio(
        portfolio_id=portfolio_id,
        method=method,
        start_date=start_date,
        end_date=end_date,
        constraints=constraints,
        benchmark_ticker=benchmark_ticker,
        method_params=method_params,
        out_of_sample=out_of_sample,
        training_ratio=training_ratio,
    )

    base: dict[str, Any] = {
        "optimization": result.to_dict(),
        "success": result.success,
        "message": result.message or "",
        "optimization_period": {
            "start": opt_start.isoformat(),
            "end": opt_end.isoformat(),
        },
        "validation_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "out_of_sample": out_of_sample,
        "training_ratio": training_ratio,
        "portfolio_snapshot": portfolio_snapshot_rows(
            portfolio_service, data_service, portfolio_id
        ),
        "constraints_applied": constraints or {},
        "method": method,
        "method_params": method_params or {},
    }

    if not result.success:
        return base

    warnings_list: list[str] = []
    min_ret = (constraints or {}).get("min_return")
    if min_ret is not None and result.expected_return is not None:
        if result.expected_return < min_ret:
            warnings_list.append(
                f"Expected return ({result.expected_return:.2%}) is below minimum ({min_ret:.2%}). "
                "Try relaxing constraints."
            )

    current_analytics: dict[str, Any] = {}
    try:
        current_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            start_date,
            end_date,
            benchmark_ticker=benchmark_for_charts if benchmark_for_charts else None,
        )
    except Exception as exc:
        logger.warning("Analytics for current portfolio failed: %s", exc)

    current_returns = current_analytics.get("portfolio_returns")
    if hasattr(current_returns, "empty") and current_returns.empty:
        current_returns = None

    optimized_returns, optimized_assets_returns = _build_optimized_returns(
        analytics_service, result, start_date, end_date, current_returns
    )
    if method in ("cvar_optimization", "mean_cvar"):
        sample_size = 0 if optimized_returns is None else len(optimized_returns)
        if 0 < sample_size < 500:
            warnings_list.append(
                f"CVaR-based methods are more stable with larger samples (recommended 500+, got {sample_size})."
            )

    if optimized_returns is not None and not optimized_returns.empty:
        # Fair comparison: compute current and optimized metrics on the same dates.
        if current_returns is not None and not current_returns.empty:
            common_idx = current_returns.index.intersection(optimized_returns.index)
            cur_cmp = current_returns.reindex(common_idx).dropna()
            opt_cmp = optimized_returns.reindex(common_idx).dropna()
            common_idx_2 = cur_cmp.index.intersection(opt_cmp.index)
            cur_cmp = cur_cmp.reindex(common_idx_2)
            opt_cmp = opt_cmp.reindex(common_idx_2)

            optimized_metrics = _metrics_from_returns(opt_cmp, RISK_FREE)
            current_metrics = _metrics_from_returns(cur_cmp, RISK_FREE)
        else:
            optimized_metrics = _metrics_from_returns(optimized_returns, RISK_FREE)
            current_metrics = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
            }
    else:
        optimized_metrics = {
            "total_return": 0.0,
            "annualized_return": float(result.expected_return or 0.0),
            "volatility": float(result.volatility or 0.0),
            "sharpe_ratio": float(result.sharpe_ratio or 0.0),
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        if current_returns is not None and not current_returns.empty:
            current_metrics = _metrics_from_returns(current_returns, RISK_FREE)
        else:
            current_metrics = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
            }

    base["metrics"] = {"current": current_metrics, "optimized": optimized_metrics}
    base["interpretation_comparison"] = _interpret_comparison(
        optimized_metrics, current_metrics
    )

    current_w = _current_weights_for_result(
        portfolio_service, data_service, portfolio_id, list(result.tickers)
    )
    optimal_w = result.get_weights_dict()
    allocation = []
    for t in result.tickers:
        cw = current_w.get(t, 0.0)
        ow = optimal_w.get(t, 0.0)
        allocation.append(
            {
                "ticker": t,
                "current_weight": cw,
                "optimal_weight": ow,
                "difference": ow - cw,
            }
        )
    base["allocation"] = allocation
    base["interpretation_allocation"] = _interpret_allocation(current_w, optimal_w)

    trades: list[dict[str, Any]] = []
    try:
        trades = optimization_service.generate_trade_list(portfolio_id, result)
    except Exception as exc:
        logger.warning("Trade list failed: %s", exc)
        warnings_list.append(f"Could not generate trade list: {exc}")
    base["trades"] = trades
    base["interpretation_trades"] = _interpret_trades(trades) if trades else ""

    # Charts: cumulative returns & drawdowns
    charts: dict[str, Any] = {"cumulative_returns": [], "drawdown": []}
    benchmark_returns = current_analytics.get("benchmark_returns")
    current_values = current_analytics.get("portfolio_values")

    try:
        if optimized_returns is not None and not optimized_returns.empty:
            opt_r = optimized_returns
            opt_vals = (1 + opt_r).cumprod()
            if current_values is not None and not current_values.empty:
                init = float(current_values.iloc[0])
                opt_vals = opt_vals * init
            else:
                opt_vals = opt_vals * 10000.0

            aligned_idx = (
                current_returns.index.intersection(opt_r.index)
                if current_returns is not None and not current_returns.empty
                else opt_r.index
            )
            cur_al = (
                current_returns.reindex(aligned_idx)
                if current_returns is not None
                else None
            )
            opt_al = opt_r.reindex(aligned_idx)
            bench_al = None
            if benchmark_returns is not None and not benchmark_returns.empty:
                bench_al = benchmark_returns.reindex(aligned_idx)

            cum_rows: list[dict[str, Any]] = []
            if cur_al is not None and not cur_al.empty:
                cur_cum = (1 + cur_al).cumprod() - 1
                opt_cum = (1 + opt_al).cumprod() - 1
                bench_cum = (
                    (1 + bench_al).cumprod() - 1
                    if bench_al is not None and not bench_al.empty
                    else None
                )
                for idx in cur_cum.index:
                    row: dict[str, Any] = {
                        "x": str(idx)[:10],
                        "current": float(cur_cum.loc[idx]) * 100.0,
                        "optimized": (
                            float(opt_cum.loc[idx]) * 100.0
                            if idx in opt_cum.index
                            else None
                        ),
                    }
                    if bench_cum is not None and idx in bench_cum.index:
                        row["benchmark"] = float(bench_cum.loc[idx]) * 100.0
                    cum_rows.append(row)
                charts["cumulative_returns"] = cum_rows
            elif opt_al is not None and not opt_al.empty:
                opt_cum = (1 + opt_al).cumprod() - 1
                bench_cum = (
                    (1 + bench_al).cumprod() - 1
                    if bench_al is not None and not bench_al.empty
                    else None
                )
                only_opt_rows: list[dict[str, Any]] = []
                for idx in opt_cum.index:
                    row_o: dict[str, Any] = {
                        "x": str(idx)[:10],
                        "current": None,
                        "optimized": float(opt_cum.loc[idx]) * 100.0,
                    }
                    if bench_cum is not None and idx in bench_cum.index:
                        row_o["benchmark"] = float(bench_cum.loc[idx]) * 100.0
                    only_opt_rows.append(row_o)
                charts["cumulative_returns"] = only_opt_rows

            # Drawdown
            dd_rows: list[dict[str, Any]] = []
            opt_dd = _drawdown_pct(opt_vals)
            cur_dd = (
                _drawdown_pct(current_values)
                if current_values is not None
                else pd.Series()
            )
            bench_dd_series = None
            if bench_al is not None and not bench_al.empty and benchmark_for_charts:
                bv = (1 + bench_al).cumprod()
                if current_values is not None and not current_values.empty:
                    bv = bv * float(current_values.iloc[0])
                bench_dd_series = _drawdown_pct(bv)

            all_idx = opt_dd.index
            if not cur_dd.empty:
                all_idx = all_idx.union(cur_dd.index)
            for idx in all_idx:
                dd_rows.append(
                    {
                        "x": str(idx)[:10],
                        "optimized": (
                            float(opt_dd.loc[idx])
                            if idx in opt_dd.index and pd.notna(opt_dd.loc[idx])
                            else None
                        ),
                        "current": (
                            float(cur_dd.loc[idx])
                            if not cur_dd.empty
                            and idx in cur_dd.index
                            and pd.notna(cur_dd.loc[idx])
                            else None
                        ),
                        "benchmark": (
                            float(bench_dd_series.loc[idx])
                            if bench_dd_series is not None
                            and idx in bench_dd_series.index
                            and pd.notna(bench_dd_series.loc[idx])
                            else None
                        ),
                    }
                )
            charts["drawdown"] = dd_rows
    except Exception as exc:
        logger.warning("Performance charts failed: %s", exc)
        warnings_list.append(f"Charts: {exc}")

    base["charts"] = charts

    # Efficient frontier
    frontier_payload: dict[str, Any] | None = None
    frontier_analytics: dict[str, Any] = {}
    try:
        frontier_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            opt_start,
            opt_end,
            benchmark_ticker=benchmark_for_charts if benchmark_for_charts else None,
        )
    except Exception as exc:
        logger.warning("Analytics for frontier period failed: %s", exc)

    if include_efficient_frontier:
        try:
            fd = optimization_service.generate_efficient_frontier(
                portfolio_id=portfolio_id,
                start_date=opt_start,
                end_date=opt_end,
                n_points=frontier_n_points,
                constraints=constraints,
            )
            frontier_payload = _build_frontier_payload_from_fd(
                fd,
                frontier_analytics=frontier_analytics,
                benchmark_for_charts=benchmark_for_charts,
                result=result,
                current_metrics=current_metrics,
            )
        except InsufficientDataError:
            if out_of_sample:
                try:
                    fd = optimization_service.generate_efficient_frontier(
                        portfolio_id=portfolio_id,
                        start_date=start_date,
                        end_date=end_date,
                        n_points=frontier_n_points,
                        constraints=constraints,
                    )
                    frontier_payload = _build_frontier_payload_from_fd(
                        fd,
                        frontier_analytics=frontier_analytics,
                        benchmark_for_charts=benchmark_for_charts,
                        result=result,
                        current_metrics=current_metrics,
                        fallback_validation_period=True,
                    )
                except Exception:
                    frontier_payload = None
            else:
                frontier_payload = None
        except Exception as exc:
            logger.warning("Efficient frontier failed: %s", exc)
            warnings_list.append(f"Efficient frontier: {exc}")

    base["efficient_frontier"] = frontier_payload

    # Correlation matrix (non-CASH tickers)
    correlation_block: dict[str, Any] | None = None
    try:
        price_frames = []
        for ticker in result.tickers:
            if ticker == "CASH":
                continue
            prices = data_service.fetch_historical_prices(
                ticker, start_date, end_date, use_cache=True, save_to_db=False
            )
            if (
                isinstance(prices, pd.DataFrame)
                and not prices.empty
                and "Date" in prices.columns
            ):
                prices = prices[["Date", "Adjusted_Close"]].copy()
                prices["Ticker"] = ticker
                price_frames.append(prices)
        if len(price_frames) >= 2:
            combined = pd.concat(price_frames, ignore_index=True)
            price_data = combined.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="first",
            )
            rets = price_data.pct_change().dropna()
            if not rets.empty and rets.shape[1] >= 2:
                corr = rets.corr()
                labels = [str(c) for c in corr.columns]
                matrix = [
                    [float(corr.iloc[i, j]) for j in range(len(labels))]
                    for i in range(len(labels))
                ]
                correlation_block = {
                    "tickers": labels,
                    "matrix": matrix,
                    "interpretation": _interpret_corr_block(
                        corr, optimal_w, len(labels)
                    ),
                }
    except Exception as exc:
        logger.warning("Correlation block failed: %s", exc)

    base["correlation"] = correlation_block

    sensitivity_block: dict[str, Any] | None = None
    if include_sensitivity:
        try:
            if method in ("min_tracking_error", "max_alpha"):
                warnings_list.append(
                    "Sensitivity skipped: not supported for this method."
                )
            else:
                sensitivity_block = optimization_service.perform_sensitivity_analysis(
                    portfolio_id=portfolio_id,
                    method=method,
                    start_date=start_date,
                    end_date=end_date,
                    base_constraints=constraints,
                    analysis_type=sensitivity_analysis_type,
                    variation_range=0.1,
                    num_points=10,
                )
                sens_results = sensitivity_block.get("results") or []
                base["interpretation_sensitivity"] = _interpret_sensitivity_block(
                    sens_results
                )
        except (CalculationError, ValueError, InsufficientDataError) as exc:
            warnings_list.append(f"Sensitivity: {exc}")
        except Exception as exc:
            logger.warning("Sensitivity failed: %s", exc)
            warnings_list.append(f"Sensitivity: {exc}")

    base["sensitivity"] = sensitivity_block
    base["warnings"] = warnings_list
    return base


def _interpret_corr_block(
    corr: pd.DataFrame, optimal_w: dict[str, float], num_assets: int
) -> str:
    lines = ["**Diversification Assessment:**"]
    mask = ~np.eye(len(corr), dtype=bool)
    values = corr.values[mask]
    values = values[~np.isnan(values)]
    if len(values) > 0:
        avg_corr = float(np.mean(values))
        if avg_corr < 0.3:
            lines.append(
                f"✓ Low average correlation ({avg_corr:.2f}) - Excellent diversification"
            )
        elif avg_corr < 0.5:
            lines.append(
                f"Moderate average correlation ({avg_corr:.2f}) - Good diversification"
            )
        else:
            lines.append(
                f"⚠ High average correlation ({avg_corr:.2f}) - Limited diversification"
            )
    mx = max(optimal_w.values()) if optimal_w else 0
    if mx > 0.5:
        lines.append(f"⚠ High concentration: Max weight is {mx:.1%}")
    elif mx > 0.3:
        lines.append(f"Moderate concentration: Max weight is {mx:.1%}")
    else:
        lines.append(f"✓ Well-diversified weights: Max weight is {mx:.1%}")
    if num_assets >= 10:
        lines.append(
            f"✓ Sufficient number of assets ({num_assets}) for diversification"
        )
    elif num_assets >= 5:
        lines.append(f"Moderate number of assets ({num_assets})")
    else:
        lines.append(f"⚠ Low number of assets ({num_assets})")
    return "\n".join(lines)


def _interpret_sensitivity_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["**Sensitivity Analysis:**"]
    df = pd.DataFrame(results)
    if df.empty:
        return ""
    variation_col = "variation"
    ticker_cols = [c for c in df.columns if c != variation_col]
    if not ticker_cols:
        return ""
    sens = []
    for t in ticker_cols:
        w = df[t].values
        sens.append({"ticker": t, "range": float(np.max(w) - np.min(w))})
    sens.sort(key=lambda x: x["range"], reverse=True)
    if sens:
        top = sens[0]
        lines.append(
            f"Most sensitive asset: {top['ticker']} (weight range: {top['range']:.1%})"
        )
    avg_range = float(np.mean([s["range"] for s in sens]))
    if avg_range < 0.05:
        lines.append(f"✓ Portfolio weights are stable (avg range: {avg_range:.1%})")
    elif avg_range < 0.15:
        lines.append(f"Moderate stability (avg range: {avg_range:.1%})")
    else:
        lines.append(f"⚠ High sensitivity (avg range: {avg_range:.1%})")
    return "\n".join(lines)
