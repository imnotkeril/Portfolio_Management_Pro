"""
JSON-oriented bundles for the Next.js Risk Analysis page (Streamlit parity).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.analytics_engine.chart_data import get_rolling_var_data
from core.analytics_engine.risk_metrics import calculate_cvar, calculate_var
from core.risk_engine.var_calculator import calculate_portfolio_var_covariance
from core.scenario_engine.historical_scenarios import get_all_scenarios
from services.portfolio_service import PortfolioService
from services.risk_service import RiskService

logger = logging.getLogger(__name__)


def _interpret_var_metrics(
    var_95: float,
    cvar_95: float,
    var_99: float | None,
    cvar_99: float | None,
) -> str:
    lines = ["**VaR & CVaR Analysis:**"]
    if abs(var_95) < 0.02:
        lines.append(f"VaR (95%): {var_95*100:.2f}% - Low daily risk exposure")
    elif abs(var_95) < 0.05:
        lines.append(f"VaR (95%): {var_95*100:.2f}% - Moderate daily risk exposure")
    else:
        lines.append(f"⚠ VaR (95%): {var_95*100:.2f}% - High daily risk exposure")
    if cvar_95 and var_95:
        excess = abs(cvar_95) - abs(var_95)
        if excess > 0.01:
            lines.append(
                f"CVaR (95%): {cvar_95*100:.2f}% - Tail risk is {excess*100:.2f}% higher than VaR"
            )
        else:
            lines.append(
                f"CVaR (95%): {cvar_95*100:.2f}% - Tail risk is similar to VaR"
            )
    if var_99 is not None:
        if abs(var_99) < 0.03:
            lines.append(f"VaR (99%): {var_99*100:.2f}% - Low extreme risk exposure")
        elif abs(var_99) < 0.08:
            lines.append(
                f"VaR (99%): {var_99*100:.2f}% - Moderate extreme risk exposure"
            )
        else:
            lines.append(f"⚠ VaR (99%): {var_99*100:.2f}% - High extreme risk exposure")
    if var_99 is not None and var_95:
        if abs(var_99) > abs(var_95) * 1.5:
            lines.append("\n⚠ Fat tails: large jump from 95% to 99% VaR")
        else:
            lines.append("\nModerate increase from 95% to 99% VaR")
    return "\n".join(lines)


def _interpret_var_methods(var_data: dict[str, Any], conf: float) -> str:
    methods = ["historical", "parametric", "cornish_fisher", "monte_carlo"]
    values = {}
    for m in methods:
        v = var_data.get(m)
        if v is not None and not isinstance(v, str):
            values[m] = float(v)
    if len(values) < 2:
        return ""
    lines = ["**VaR Methods Comparison:**"]
    min_m = min(values, key=values.get)
    max_m = max(values, key=values.get)
    spread = values[max_m] - values[min_m]
    if spread < 0.01:
        lines.append(f"Methods agree (spread {spread*100:.2f}%)")
    elif spread < 0.03:
        lines.append(f"Moderate spread ({spread*100:.2f}%) — some model uncertainty")
    else:
        lines.append(f"⚠ Large spread ({spread*100:.2f}%) — consider multiple methods")
    if "historical" in values and "parametric" in values:
        d = abs(values["historical"] - values["parametric"])
        if d > 0.02:
            lines.append(
                f"Historical vs Parametric differ by {d*100:.2f}% — returns may be non-normal"
            )
    if "cornish_fisher" in values:
        cf_val = values["cornish_fisher"]
        lines.append(
            f"Cornish-Fisher ({cf_val*100:.2f}%) accounts for skewness and kurtosis — "
            "may be more accurate if returns are non-normal"
        )
    return "\n".join(lines)


def _nearest_cov_confidence(conf: float) -> float:
    return min((0.90, 0.95, 0.99), key=lambda c: abs(c - conf))


def _return_histogram(
    returns: pd.Series, bins: int = 50
) -> tuple[list[dict[str, float]], list[float]]:
    arr = returns.dropna().values
    if len(arr) == 0:
        return [], []
    counts, edges = np.histogram(arr, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return [
        {"x": float(c), "count": float(n)} for c, n in zip(centers, counts)
    ], edges.tolist()


def _pct_lookup(pct: dict[Any, Any], k: float) -> float | None:
    for key in (k, float(k), int(k)):
        if key in pct:
            return float(pct[key])
    return None


def _interpret_rolling_var(rolling_stats: dict[str, float]) -> str:
    if not rolling_stats:
        return ""
    lines = ["**Rolling VaR Analysis:**"]
    avg = float(rolling_stats.get("avg", 0))
    median = float(rolling_stats.get("median", 0))
    min_var = float(rolling_stats.get("min", 0))
    max_var = float(rolling_stats.get("max", 0))
    lines.append(f"Average VaR: {avg*100:.2f}%")
    lines.append(f"Median VaR: {median*100:.2f}%")
    if max_var and min_var:
        var_range = max_var - min_var
        if var_range > 0.05:
            lines.append(
                f"⚠ High VaR volatility: range {min_var*100:.2f}%–{max_var*100:.2f}% "
                f"({var_range*100:.2f}% spread)"
            )
        elif var_range > 0.02:
            lines.append(
                f"Moderate VaR volatility: range {min_var*100:.2f}%–{max_var*100:.2f}%"
            )
        else:
            lines.append(
                f"Stable VaR: range {min_var*100:.2f}%–{max_var*100:.2f}% — consistent risk levels"
            )
    return "\n".join(lines)


def _interpret_portfolio_var_covariance(portfolio_var: float, confidence: float) -> str:
    lines = ["**Portfolio VaR (Covariance Method) Analysis:**"]
    lines.append(
        f"Portfolio VaR ({confidence*100:.0f}%): {abs(portfolio_var)*100:.2f}%"
    )
    if abs(portfolio_var) < 0.02:
        lines.append("Low portfolio risk relative to typical daily moves.")
    elif abs(portfolio_var) < 0.05:
        lines.append("Moderate portfolio risk — correlations included.")
    else:
        lines.append("⚠ High portfolio risk — monitor concentration and correlations.")
    lines.append(
        "\nThis method accounts for correlations between assets, giving a portfolio-level measure."
    )
    return "\n".join(lines)


def _interpret_var_decomposition_rows(
    decomp: list[dict[str, Any]], portfolio_var_pct: float
) -> str:
    if not decomp:
        return ""
    lines = ["**VaR Decomposition Analysis:**"]
    sorted_rows = sorted(
        decomp, key=lambda r: float(r.get("contribution_pct", 0)), reverse=True
    )
    top = sorted_rows[0]
    lines.append(
        f"Largest risk contributor: {top.get('asset')} "
        f"({float(top.get('weight_pct', 0)):.1f}% weight, "
        f"{float(top.get('contribution_pct', 0)):.1f}% of VaR)"
    )
    top3 = sum(float(r.get("contribution_pct", 0)) for r in sorted_rows[:3])
    if top3 > 70:
        lines.append(
            f"⚠ High concentration: top 3 assets contribute {top3:.1f}% of VaR"
        )
    elif top3 > 50:
        lines.append(
            f"Moderate concentration: top 3 assets contribute {top3:.1f}% of VaR"
        )
    else:
        lines.append(f"✓ Diversified: top 3 assets contribute {top3:.1f}% of VaR")
    mismatches = []
    for r in decomp:
        w = float(r.get("weight_pct", 0))
        c = float(r.get("contribution_pct", 0))
        if w > 0 and c / w > 1.5:
            mismatches.append((r.get("asset"), w, c, c / w))
    mismatches.sort(key=lambda x: -x[3])
    if mismatches:
        t = mismatches[0]
        lines.append(
            f"⚠ Risk/weight mismatch: {t[0]} — {t[1]:.1f}% weight but {t[2]:.1f}% VaR contribution "
            f"(ratio {t[3]:.1f}x)"
        )
    return "\n".join(lines)


def _interpret_monte_carlo_stats_full(
    stats: dict[str, Any], initial_value: float
) -> str:
    if not stats:
        return ""
    lines = ["**Monte Carlo Simulation Analysis:**"]
    mean = float(stats.get("mean", 0))
    median = float(stats.get("median", 0))
    std = float(stats.get("std", 0))
    min_val = float(stats.get("min", 0))
    max_val = float(stats.get("max", 0))
    if initial_value > 0:
        lines.append(
            f"Expected value: ${mean:,.2f} ({(mean / initial_value - 1) * 100:+.2f}% return)"
        )
        lines.append(
            f"Median value: ${median:,.2f} ({(median / initial_value - 1) * 100:+.2f}% return)"
        )
    if mean > median * 1.1:
        lines.append("Positive skew: mean above median — upside potential")
    elif mean < median * 0.9:
        lines.append("Negative skew: mean below median — downside dominates")
    else:
        lines.append("Symmetric: mean ≈ median")
    vol_pct = (std / initial_value) * 100 if initial_value > 0 else 0
    if vol_pct < 5:
        lines.append(f"Low volatility: {vol_pct:.1f}% std vs initial — stable outcomes")
    elif vol_pct < 15:
        lines.append(f"Moderate volatility: {vol_pct:.1f}% std vs initial")
    else:
        lines.append(f"⚠ High volatility: {vol_pct:.1f}% std — wide outcome range")
    if initial_value > 0:
        range_pct = ((max_val - min_val) / initial_value) * 100
        lines.append(
            f"Outcome range: ${min_val:,.2f} to ${max_val:,.2f} ({range_pct:.1f}% spread)"
        )
    return "\n".join(lines)


def _interpret_monte_carlo_percentiles_full(
    percentiles: dict[Any, Any], initial_value: float
) -> str:
    if not percentiles or initial_value <= 0:
        return ""
    lines = ["**Percentile Analysis:**"]
    p5 = _pct_lookup(percentiles, 5.0)
    p25 = _pct_lookup(percentiles, 25.0)
    _pct_lookup(percentiles, 50.0)
    p75 = _pct_lookup(percentiles, 75.0)
    p95 = _pct_lookup(percentiles, 95.0)
    if p5 is not None:
        lines.append(
            f"5th percentile (worst 5%): ${p5:,.2f} ({(p5 / initial_value - 1) * 100:+.2f}% return)"
        )
    if p95 is not None:
        lines.append(
            f"95th percentile (best 5%): ${p95:,.2f} ({(p95 / initial_value - 1) * 100:+.2f}% return)"
        )
    if p25 is not None and p75 is not None:
        iqr_pct = ((p75 - p25) / initial_value) * 100
        lines.append(f"Interquartile range (middle 50%): {iqr_pct:.1f}% spread")
    if p5 is not None and p95 is not None:
        downside = abs(initial_value - p5)
        upside = abs(p95 - initial_value)
        if downside > upside * 1.5:
            lines.append("⚠ Downside wider than upside — negative asymmetry")
        elif upside > downside * 1.5:
            lines.append("Upside wider than downside — positive asymmetry")
        else:
            lines.append("Balanced upside vs downside")
    return "\n".join(lines)


def _interpret_confidence_intervals_mc(finals: np.ndarray, initial_value: float) -> str:
    if finals is None or len(finals) == 0 or initial_value <= 0:
        return ""
    lines = ["**Final Value Distribution Analysis:**"]
    mean_val = float(np.mean(finals))
    median_val = float(np.median(finals))
    if mean_val > median_val * 1.05:
        lines.append("Right-skewed — more upside than downside in the bulk")
    elif mean_val < median_val * 0.95:
        lines.append("Left-skewed — more downside than upside in the bulk")
    else:
        lines.append("Roughly symmetric distribution")
    lines.append("**Confidence Intervals Analysis:**")
    for ci in (0.90, 0.95, 0.99):
        lower = float(np.percentile(finals, (1 - ci) / 2 * 100))
        upper = float(np.percentile(finals, (1 + ci) / 2 * 100))
        lr = (lower / initial_value - 1) * 100
        ur = (upper / initial_value - 1) * 100
        range_pct = ((upper - lower) / initial_value) * 100
        lines.append(
            f"{int(ci * 100)}% CI: ${lower:,.2f} to ${upper:,.2f} "
            f"({range_pct:.1f}% range, {lr:+.1f}% to {ur:+.1f}% return)"
        )
    ci_90_lower = float(np.percentile(finals, 5))
    ci_90_upper = float(np.percentile(finals, 95))
    ci_99_lower = float(np.percentile(finals, 0.5))
    ci_99_upper = float(np.percentile(finals, 99.5))
    tail_width_90 = (ci_90_upper - ci_90_lower) / initial_value * 100
    tail_width_99 = (ci_99_upper - ci_99_lower) / initial_value * 100
    if tail_width_90 < 10:
        lines.append(
            f"\nNarrow 90% range ({tail_width_90:.1f}%) — predictable outcomes"
        )
    elif tail_width_90 < 25:
        lines.append(f"\nModerate 90% range ({tail_width_90:.1f}%)")
    else:
        lines.append(f"\n⚠ Wide 90% range ({tail_width_90:.1f}%) — high uncertainty")
    tail_expansion = tail_width_99 / tail_width_90 if tail_width_90 > 0 else 0
    if tail_expansion > 1.5:
        lines.append(
            f"Significant tail expansion (99% vs 90% width ratio {tail_expansion:.1f}x) — fat tails possible"
        )
    elif tail_expansion > 1.2:
        lines.append(f"Moderate tail expansion (ratio {tail_expansion:.1f}x)")
    else:
        lines.append(f"Normal-ish tail behavior (ratio {tail_expansion:.1f}x)")
    return "\n".join(lines)


def _interpret_monte_carlo_var_comparison(
    historical_var: float, mc_var: float, confidence: float
) -> str:
    lines = ["**Historical vs Monte Carlo VaR Comparison:**"]
    lines.append(f"Historical VaR ({confidence*100:.0f}%): {historical_var*100:.2f}%")
    lines.append(f"Monte Carlo VaR ({confidence*100:.0f}%): {mc_var*100:.2f}%")
    diff = mc_var - historical_var
    diff_pct = abs(diff) / abs(historical_var) * 100 if historical_var != 0 else 0
    if abs(diff) < 0.01:
        lines.append(f"Methods agree (diff {diff*100:.2f}%)")
    elif diff_pct < 20:
        lines.append(f"Moderate difference {diff*100:.2f}% ({diff_pct:.1f}% relative)")
    elif diff > 0:
        lines.append(
            "⚠ MC VaR higher — forward model implies more tail risk than history"
        )
    else:
        lines.append(
            "MC VaR lower than historical — model implies less tail risk than history"
        )
    return "\n".join(lines)


def _interpret_extreme_rows(
    extreme_rows: list[dict[str, Any]], initial_value: float
) -> str:
    if not extreme_rows or initial_value <= 0:
        return ""
    lines = ["**Extreme Scenarios Analysis:**"]
    worst = next(
        (r for r in extreme_rows if "Worst 5%" in str(r.get("scenario", ""))), None
    )
    best = next(
        (r for r in extreme_rows if "Best 5%" in str(r.get("scenario", ""))), None
    )
    if worst:
        v = float(worst.get("value", 0))
        lines.append(
            f"Worst case (5% tail): ${v:,.2f} ({(v / initial_value - 1) * 100:+.2f}% return)"
        )
    if best:
        v = float(best.get("value", 0))
        lines.append(
            f"Best case (95%): ${v:,.2f} ({(v / initial_value - 1) * 100:+.2f}% return)"
        )
    return "\n".join(lines)


def _interpret_var_cvar_sim_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["**VaR & CVaR (from terminal returns):**"]
    r95 = next((r for r in rows if r.get("confidence") == "95%"), None)
    if r95:
        vp = float(r95.get("var_pct", 0))
        cp = float(r95.get("cvar_pct", 0))
        lines.append(f"95% VaR: {abs(vp):.2f}% — loss at 5th percentile of outcomes")
        lines.append(f"95% CVaR: {abs(cp):.2f}% — mean loss beyond that threshold")
        spread = abs(cp) - abs(vp)
        if spread > 2:
            lines.append(
                f"Large CVaR–VaR spread ({spread:.2f}%) — meaningful tail beyond VaR"
            )
        elif spread > 1:
            lines.append(f"Moderate tail beyond VaR ({spread:.2f}%)")
        else:
            lines.append(f"Limited tail beyond VaR ({spread:.2f}%)")
    r90 = next((r for r in rows if r.get("confidence") == "90%"), None)
    r99 = next((r for r in rows if r.get("confidence") == "99%"), None)
    if r90 is not None and r99 is not None:
        v90 = float(r90.get("var_pct", 0))
        v99 = float(r99.get("var_pct", 0))
        if abs(v90) > 1e-9:
            exp_ratio = abs(v99) / abs(v90)
            if exp_ratio > 2:
                lines.append(
                    f"⚠ Risk escalation: 99% VaR is {exp_ratio:.1f}x 90% VaR — extreme scenarios matter"
                )
            elif exp_ratio > 1.5:
                lines.append(
                    f"Moderate escalation: 99% vs 90% VaR ratio {exp_ratio:.1f}x"
                )
    return "\n".join(lines)


def _interpret_hist_vs_mc_chart(diffs: list[dict[str, float]]) -> str:
    if not diffs:
        return ""
    lines = ["**Historical vs Monte Carlo VaR (chart summary):**"]
    avg_diff = float(np.mean([d["diff"] for d in diffs]))
    if abs(avg_diff) < 0.01:
        lines.append("Methods track closely across confidence levels")
    elif avg_diff > 0:
        lines.append(
            f"MC VaR averages {avg_diff*100:.2f}% above historical — forward paths show more downside"
        )
    else:
        lines.append(
            f"MC VaR averages {abs(avg_diff)*100:.2f}% below historical — forward paths show less downside"
        )
    if len(diffs) >= 2:
        low = abs(diffs[0]["diff"])
        high = abs(diffs[-1]["diff"])
        if high > low * 1.5:
            lines.append(
                "Gap widens at higher confidence — model uncertainty in the far tail"
            )
    return "\n".join(lines)


def _interpret_scenario_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["**Scenario Analysis:**"]
    impacts = [float(r.get("portfolio_impact_pct", 0)) * 100 for r in results]
    worst = min(results, key=lambda x: float(x.get("portfolio_impact_pct", 0)))
    best = max(results, key=lambda x: float(x.get("portfolio_impact_pct", 0)))
    lines.append(
        f"Worst: {worst.get('scenario_name')} ({float(worst.get('portfolio_impact_pct', 0))*100:.2f}%)"
    )
    bv = float(best.get("portfolio_impact_pct", 0)) * 100
    if bv > 0:
        lines.append(f"Best: {best.get('scenario_name')} ({bv:.2f}% gain)")
    else:
        lines.append(f"Least negative: {best.get('scenario_name')} ({bv:.2f}%)")
    lines.append(f"Average impact: {float(np.mean(impacts)):.2f}%")
    wi = min(impacts)
    if wi < -30:
        lines.append(f"⚠ Severe downside: worst case {wi:.1f}%")
    elif wi < -15:
        lines.append(f"Moderate stress: worst case {wi:.1f}%")
    else:
        lines.append(f"Resilient book: worst case {wi:.1f}%")
    rec = [r.get("recovery_time_days") for r in results if r.get("recovery_time_days")]
    if rec:
        lines.append(
            f"Recovery: avg {float(np.mean(rec)):.0f} days, max {float(max(rec)):.0f} days"
        )
    return "\n".join(lines)


def _interpret_scenario_recovery(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["**Recovery Timeline Analysis:**"]
    rec_rows = []
    for r in results:
        d = r.get("recovery_time_days")
        if d and float(d) > 0:
            rec_rows.append(
                {
                    "name": r.get("scenario_name", ""),
                    "days": float(d),
                    "impact": float(r.get("portfolio_impact_pct", 0)) * 100,
                }
            )
    if not rec_rows:
        lines.append("Recovery data not available")
        return "\n".join(lines)
    fastest = min(rec_rows, key=lambda x: x["days"])
    slowest = max(rec_rows, key=lambda x: x["days"])
    lines.append(f"Fastest recovery: {fastest['name']} ({fastest['days']:.0f} days)")
    lines.append(f"Slowest recovery: {slowest['name']} ({slowest['days']:.0f} days)")
    lines.append(
        f"Average recovery: {float(np.mean([x['days'] for x in rec_rows])):.0f} days"
    )
    return "\n".join(lines)


def _interpret_position_impact_breakdown(
    position_impacts: dict[str, float], scenario_name: str
) -> str:
    if not position_impacts:
        return ""
    lines = [f"**Position Impact ({scenario_name}):**"]
    items = sorted(position_impacts.items(), key=lambda x: x[1])
    worst_ticker, worst_impact = items[0]
    lines.append(f"Worst: {worst_ticker} ({worst_impact*100:.2f}%)")
    best_ticker, best_impact = items[-1]
    if best_impact > 0:
        lines.append(f"Best: {best_ticker} ({best_impact*100:.2f}%)")
    losses = sum(1 for _, v in items if v < 0)
    gains = sum(1 for _, v in items if v > 0)
    if losses > gains:
        lines.append(f"Broadly negative: {losses} losers vs {gains} gainers")
    elif gains > losses:
        lines.append(f"Broadly positive: {gains} gainers vs {losses} losers")
    if len(items) > 1:
        abs_i = sorted([abs(v * 100) for _, v in items], reverse=True)
        t2 = sum(abs_i[:2])
        tot = sum(abs_i) or 1
        conc = t2 / tot * 100
        if conc > 70:
            lines.append(
                f"Concentrated: top 2 positions drive {conc:.1f}% of gross impact"
            )
    return "\n".join(lines)


def build_stress_historical_display_bundle(
    risk_service: RiskService,
    portfolio_service: PortfolioService,
    *,
    portfolio_id: str,
    scenario_keys: list[str],
) -> dict[str, Any]:
    scenarios = get_all_scenarios()
    results = risk_service.run_stress_test(portfolio_id, scenario_keys)

    recovery_series: list[dict[str, Any]] = []
    for r in results:
        name = r.get("scenario_name", "")
        impact_pct = float(r.get("portfolio_impact_pct", 0.0))
        rec_days = r.get("recovery_time_days")
        if rec_days is not None and float(rec_days) > 0:
            rd = int(float(rec_days))
            step = max(1, rd // 20)
            days = list(range(0, rd + 1, step))
            if days[-1] != rd:
                days.append(rd)
            path = [max(0.0, 1.0 + impact_pct * (1 - d / rd)) for d in days]
            recovery_series.append(
                {
                    "scenario_name": name,
                    "points": [
                        {"day": float(d), "pct": float(v) * 100}
                        for d, v in zip(days, path)
                    ],
                }
            )

    enhanced: list[dict[str, Any]] = []
    for r in results:
        sn = r.get("scenario_name")
        key = next(
            (k for k, sc in scenarios.items() if k in scenario_keys and sc.name == sn),
            None,
        )
        if not key:
            continue
        sc = scenarios[key]
        enhanced.append(
            {
                "scenario": sc.name,
                "period_start": str(sc.start_date),
                "period_end": str(sc.end_date),
                "duration_days": (sc.end_date - sc.start_date).days,
                "market_impact_pct": float(sc.market_impact_pct) * 100,
                "portfolio_impact_pct": float(r.get("portfolio_impact_pct", 0)) * 100,
                "recovery_days": r.get("recovery_time_days"),
            }
        )

    portfolio = portfolio_service.get_portfolio(portfolio_id)
    positions = portfolio.get_all_positions() if portfolio else []
    weight_by_ticker: dict[str, float] = {}
    for p in positions:
        weight_by_ticker[p.ticker] = float(p.weight_target or 0.0)
    tw = sum(weight_by_ticker.values())
    if tw > 0:
        weight_by_ticker = {k: v / tw for k, v in weight_by_ticker.items()}

    position_breakdowns: list[dict[str, Any]] = []
    for r in results:
        details = r.get("details") or {}
        pi = details.get("position_impacts") or {}
        if not isinstance(pi, dict) or not pi:
            continue
        rows = []
        for ticker, impact in pi.items():
            w = weight_by_ticker.get(ticker, 0.0) * 100
            imp = float(impact)
            rows.append(
                {
                    "ticker": ticker,
                    "weight_pct": w,
                    "impact_pct": imp * 100,
                    "kind": "Loss" if imp < 0 else "Gain",
                }
            )
        rows.sort(key=lambda x: x["impact_pct"])
        position_breakdowns.append(
            {
                "scenario_name": r.get("scenario_name"),
                "rows": rows,
                "interpretation": _interpret_position_impact_breakdown(
                    pi, str(r.get("scenario_name", ""))
                ),
            }
        )

    timeline: list[dict[str, Any]] = []
    for k in scenario_keys:
        if k not in scenarios:
            continue
        sc = scenarios[k]
        timeline.append(
            {
                "key": k,
                "name": sc.name,
                "start": str(sc.start_date),
                "end": str(sc.end_date),
                "duration_days": (sc.end_date - sc.start_date).days,
                "market_impact_pct": float(sc.market_impact_pct) * 100,
            }
        )
    timeline.sort(key=lambda x: x["start"])

    interp_timeline = ""
    if timeline:
        worst = min(timeline, key=lambda x: x["market_impact_pct"])
        longest = max(timeline, key=lambda x: x["duration_days"])
        interp_timeline = (
            f"**Historical Timeline Analysis:**\n"
            f"{len(timeline)} scenario(s). "
            f"Most severe market shock: {worst['name']} ({worst['market_impact_pct']:.1f}%). "
            f"Longest episode: {longest['name']} ({longest['duration_days']} days)."
        )

    return {
        "results": results,
        "interpretation_scenarios": _interpret_scenario_results(results),
        "recovery": {
            "series": recovery_series,
            "interpretation": _interpret_scenario_recovery(results),
        },
        "enhanced_comparison": enhanced,
        "position_breakdowns": position_breakdowns,
        "timeline": timeline,
        "interpretation_timeline": interp_timeline,
    }


def build_var_full_bundle(
    risk_service: RiskService,
    portfolio_service: PortfolioService,
    *,
    portfolio_id: str,
    start_date: date,
    end_date: date,
    confidence_level: float,
    time_horizon: int,
    rolling_window: int = 63,
    include_monte_carlo: bool = True,
    num_simulations: int = 10000,
) -> dict[str, Any]:
    base = risk_service.calculate_var_analysis(
        portfolio_id=portfolio_id,
        start_date=start_date,
        end_date=end_date,
        confidence_level=confidence_level,
        include_monte_carlo=include_monte_carlo,
        num_simulations=num_simulations,
        time_horizon=time_horizon,
    )
    var_data = base.get("var_results") or {}
    portfolio_returns = risk_service._get_portfolio_returns(  # noqa: SLF001
        portfolio_id, start_date, end_date
    )

    cvar_sel = float(calculate_cvar(portfolio_returns, confidence_level))
    var_95_hist = float(var_data.get("historical") or 0)
    cvar_95 = float(calculate_cvar(portfolio_returns, 0.95))
    var_99_hist = None
    cvar_99 = None
    if portfolio_returns is not None and not portfolio_returns.empty:
        var_99_hist = float(calculate_var(portfolio_returns, 0.99, "historical"))
        cvar_99 = float(calculate_cvar(portfolio_returns, 0.99))

    comparison_rows: list[dict[str, Any]] = []
    method_labels = {
        "historical": "Historical",
        "parametric": "Parametric",
        "cornish_fisher": "Cornish-Fisher",
        "monte_carlo": "Monte Carlo",
    }
    for key, label in method_labels.items():
        val = var_data.get(key)
        if val is not None and not isinstance(val, str):
            fv = float(val)
            comparison_rows.append(
                {
                    "method": label,
                    "var_decimal": fv,
                    "var_pct": fv * 100,
                }
            )
    comparison_rows.append(
        {
            "method": "CVaR (Expected Shortfall)",
            "var_decimal": cvar_sel,
            "var_pct": cvar_sel * 100,
        }
    )

    hist_bars, _ = _return_histogram(portfolio_returns, 50)
    var_hist = var_data.get("historical")
    var_hist_f = (
        float(var_hist)
        if var_hist is not None and not isinstance(var_hist, str)
        else None
    )

    sensitivity = []
    for cl in (0.90, 0.95, 0.99):
        sensitivity.append(
            {
                "confidence_pct": int(cl * 100),
                "var_pct": float(calculate_var(portfolio_returns, cl, "historical"))
                * 100,
                "cvar_pct": float(calculate_cvar(portfolio_returns, cl)) * 100,
            }
        )

    rolling_block: dict[str, Any] | None = None
    try:
        rv = get_rolling_var_data(
            portfolio_returns,
            None,
            window=rolling_window,
            confidence_level=confidence_level,
        )
        if rv:
            pv = rv["portfolio"]
            stats_r = {k: float(v) for k, v in rv.get("portfolio_stats", {}).items()}
            rolling_block = {
                "window": rolling_window,
                "stats": stats_r,
                "series": [
                    {"x": str(i)[:10], "y": float(v) * 100}
                    for i, v in pv.items()
                    if pd.notna(v)
                ],
                "interpretation": _interpret_rolling_var(stats_r),
            }
    except Exception as exc:
        logger.warning("Rolling VaR failed: %s", exc)

    covariance_block: dict[str, Any] | None = None
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
        positions = portfolio.get_all_positions() if portfolio else []
        tickers = [p.ticker for p in positions if p.ticker != "CASH"]
        if len(tickers) >= 2:
            ds = risk_service._data_service  # noqa: SLF001
            frames = []
            for t in tickers:
                px = ds.fetch_historical_prices(
                    t, start_date, end_date, use_cache=True, save_to_db=False
                )
                if isinstance(px, pd.DataFrame) and not px.empty:
                    px = px.copy()
                    px["Ticker"] = t
                    frames.append(px[["Date", "Adjusted_Close", "Ticker"]])
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                price_data = combined.pivot_table(
                    index="Date",
                    columns="Ticker",
                    values="Adjusted_Close",
                    aggfunc="first",
                )
                returns_df = price_data.pct_change().dropna()
                if not returns_df.empty:
                    first_row = price_data.iloc[0]
                    total_v = 0.0
                    w_list = []
                    for pos in positions:
                        if pos.ticker == "CASH" or pos.ticker not in returns_df.columns:
                            continue
                        v = float(pos.shares) * float(first_row.get(pos.ticker, 0) or 0)
                        total_v += v
                        w_list.append((pos.ticker, v))
                    if total_v > 0:
                        weights_map = {t: v / total_v for t, v in w_list}
                        weights = np.array(
                            [weights_map.get(c, 0.0) for c in returns_df.columns]
                        )
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                            cov_conf = _nearest_cov_confidence(confidence_level)
                            pvr = calculate_portfolio_var_covariance(
                                returns_df, weights, cov_conf, time_horizon
                            )
                            decomp = []
                            cols = list(returns_df.columns)
                            for i, ticker in enumerate(cols):
                                if ticker in pvr.get("component_var", {}):
                                    decomp.append(
                                        {
                                            "asset": ticker,
                                            "weight_pct": float(weights[i]) * 100,
                                            "component_var_pct": float(
                                                pvr["component_var"][ticker]
                                            )
                                            * 100,
                                            "contribution_pct": float(
                                                pvr["var_contribution_pct"][ticker]
                                            ),
                                            "marginal_var_pct": float(
                                                pvr["marginal_var"][ticker]
                                            )
                                            * 100,
                                        }
                                    )
                            pv_dec = float(pvr["portfolio_var"])
                            covariance_block = {
                                "portfolio_var_pct": pv_dec * 100,
                                "confidence_used": cov_conf,
                                "decomposition": decomp,
                                "interpretation_covariance": _interpret_portfolio_var_covariance(
                                    pv_dec, cov_conf
                                ),
                                "interpretation_decomposition": _interpret_var_decomposition_rows(
                                    decomp, pv_dec * 100
                                ),
                            }
    except Exception as exc:
        logger.warning("Covariance VaR block failed: %s", exc)

    return {
        **base,
        "key_metrics": {
            "var_95_historical_pct": var_95_hist * 100,
            "cvar_95_pct": cvar_95 * 100,
            "var_99_historical_pct": (
                var_99_hist * 100 if var_99_hist is not None else None
            ),
            "cvar_99_pct": cvar_99 * 100 if cvar_99 is not None else None,
            "var_selected_pct": float(var_data.get("historical") or 0) * 100,
            "cvar_selected_pct": cvar_sel * 100,
        },
        "interpretation_metrics": _interpret_var_metrics(
            var_95_hist, cvar_95, var_99_hist, cvar_99
        ),
        "comparison_table": comparison_rows,
        "interpretation_methods": _interpret_var_methods(var_data, confidence_level),
        "return_distribution": {
            "histogram": hist_bars,
            "var_historical": var_hist_f,
            "cvar": cvar_sel,
            "confidence_level": confidence_level,
        },
        "sensitivity_by_confidence": sensitivity,
        "rolling_var": rolling_block,
        "covariance_var": covariance_block,
    }


def build_monte_carlo_display_bundle(
    risk_service: RiskService,
    *,
    portfolio_id: str,
    start_date: date,
    end_date: date,
    time_horizon: int,
    num_simulations: int,
    initial_value: float,
    model: str,
    include_sample_paths: bool,
    max_paths: int = 40,
) -> dict[str, Any]:
    raw = risk_service.run_monte_carlo_simulation(
        portfolio_id=portfolio_id,
        start_date=start_date,
        end_date=end_date,
        time_horizon=time_horizon,
        num_simulations=num_simulations,
        initial_value=initial_value,
        model=model,
    )
    finals = np.array(raw.get("final_values") or [], dtype=float)
    hist: list[dict[str, float]] = []
    normal_curve: list[dict[str, float]] = []
    percentile_markers: list[dict[str, Any]] = []

    if len(finals) > 0:
        counts, edges = np.histogram(finals, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_w = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
        mean_f = float(np.mean(finals))
        std_f = float(np.std(finals))
        hist = []
        for c, n in zip(centers, counts):
            entry: dict[str, float] = {"x": float(c), "count": float(n)}
            if std_f > 0:
                entry["norm_y"] = float(
                    scipy_stats.norm.pdf(float(c), loc=mean_f, scale=std_f)
                    * len(finals)
                    * bin_w
                )
            else:
                entry["norm_y"] = 0.0
            hist.append(entry)
        if std_f > 0:
            x_norm = np.linspace(float(edges[0]), float(edges[-1]), 100)
            pdf_values = scipy_stats.norm.pdf(x_norm, loc=mean_f, scale=std_f)
            y_norm = pdf_values * len(finals) * bin_w
            normal_curve = [
                {"x": float(x), "y": float(y)} for x, y in zip(x_norm, y_norm)
            ]

        pct_raw = raw.get("percentiles") or {}
        for label, pk, side in [
            ("5%", 5.0, "lower"),
            ("10%", 10.0, "lower"),
            ("90%", 90.0, "upper"),
            ("95%", 95.0, "upper"),
            ("99%", 99.0, "upper"),
        ]:
            v = _pct_lookup(pct_raw, pk)
            if v is not None:
                percentile_markers.append({"label": label, "x": v, "side": side})
        if initial_value > 0:
            percentile_markers.append(
                {"label": "Initial", "x": float(initial_value), "side": "initial"}
            )

    stats = raw.get("statistics") or {}
    mean_s = float(stats.get("mean", 0))
    std_s = float(stats.get("std", 0))
    interp_stats = _interpret_monte_carlo_stats_full(stats, initial_value)
    percentiles = raw.get("percentiles") or {}
    interp_pct = _interpret_monte_carlo_percentiles_full(percentiles, initial_value)
    interp_distribution = (
        _interpret_confidence_intervals_mc(finals, initial_value) if len(finals) else ""
    )

    var_cvar_sim_rows: list[dict[str, Any]] = []
    hist_vs_mc_rows: list[dict[str, Any]] = []
    hist_vs_mc_diffs: list[dict[str, float]] = []
    interp_hist_mc = ""
    interp_hist_mc_chart = ""
    returns_sim = (
        finals / initial_value - 1.0
        if initial_value > 0 and len(finals)
        else np.array([])
    )

    if len(returns_sim) > 0:
        for cl in (0.90, 0.95, 0.99):
            var_val = float(np.percentile(returns_sim, (1 - cl) * 100))
            tail = returns_sim[returns_sim <= var_val]
            cvar_val = float(tail.mean()) if len(tail) else var_val
            var_cvar_sim_rows.append(
                {
                    "confidence": f"{int(cl * 100)}%",
                    "var_pct": var_val * 100,
                    "var_usd": var_val * initial_value,
                    "cvar_pct": cvar_val * 100,
                    "cvar_usd": cvar_val * initial_value,
                }
            )
        pr = risk_service._get_portfolio_returns(
            portfolio_id, start_date, end_date
        )  # noqa: SLF001
        if pr is not None and not pr.empty:
            for cl in (0.90, 0.95, 0.99):
                hist_var = float(calculate_var(pr, cl, "historical"))
                mc_var = float(np.percentile(returns_sim, (1 - cl) * 100))
                hist_vs_mc_rows.append(
                    {
                        "confidence": f"{int(cl * 100)}%",
                        "historical_var_pct": hist_var * 100,
                        "mc_var_pct": mc_var * 100,
                        "diff_pct": (mc_var - hist_var) * 100,
                    }
                )
                hist_vs_mc_diffs.append({"diff": mc_var - hist_var})
            hist95 = float(calculate_var(pr, 0.95, "historical"))
            mc95 = float(np.percentile(returns_sim, 5))
            interp_hist_mc = _interpret_monte_carlo_var_comparison(hist95, mc95, 0.95)
            interp_hist_mc_chart = _interpret_hist_vs_mc_chart(hist_vs_mc_diffs)
    interp_var_cvar_sim = _interpret_var_cvar_sim_table(var_cvar_sim_rows)

    extreme_rows: list[dict[str, Any]] = []
    if len(finals) > 0:

        def _ext_row(label: str, pct: float) -> dict[str, Any]:
            v = float(np.percentile(finals, pct))
            return {
                "scenario": label,
                "value": v,
                "return_usd": v - initial_value,
                "return_pct": (
                    ((v - initial_value) / initial_value) * 100
                    if initial_value > 0
                    else 0.0
                ),
            }

        extreme_rows = [
            _ext_row("Worst 5%", 5),
            _ext_row("Worst 1%", 1),
            _ext_row("Worst 0.1%", 0.1),
            _ext_row("Best 5%", 95),
            _ext_row("Best 1%", 99),
            _ext_row("Best 0.1%", 99.9),
        ]
    interp_extreme = _interpret_extreme_rows(extreme_rows, initial_value)

    paths_out: list[list[dict[str, float]]] | None = None
    path_envelope: dict[str, list[dict[str, float]]] | None = None
    interp_paths = ""
    if include_sample_paths and raw.get("simulated_paths"):
        sp = np.array(raw["simulated_paths"], dtype=float)
        if sp.ndim == 2 and sp.shape[0] > 0:
            if len(finals) == sp.shape[0]:
                max_path = sp.max(axis=0)
                min_path = sp.min(axis=0)
                p50 = float(np.percentile(finals, 50))
                mid_i = int(np.argmin(np.abs(finals - p50)))
                median_path = sp[mid_i]
                path_envelope = {
                    "max": [
                        {"day": float(d), "value": float(v)}
                        for d, v in enumerate(max_path.tolist())
                    ],
                    "min": [
                        {"day": float(d), "value": float(v)}
                        for d, v in enumerate(min_path.tolist())
                    ],
                    "median": [
                        {"day": float(d), "value": float(v)}
                        for d, v in enumerate(median_path.tolist())
                    ],
                }
                fv_paths = [float(sp[i, -1]) for i in range(sp.shape[0])]
                path_std = float(np.std(fv_paths))
                path_mean = float(np.mean(fv_paths))
                path_cv = path_std / path_mean if path_mean > 0 else 0.0
                pos_pct = (
                    sum(1 for x in fv_paths if x > initial_value) / len(fv_paths) * 100
                    if fv_paths
                    else 0.0
                )
                pl = ["**Simulation Paths Analysis:**"]
                pl.append(f"{sp.shape[0]} paths over {time_horizon} days")
                if path_cv > 0.3:
                    pl.append(f"High path divergence (CV {path_cv:.2f})")
                elif path_cv > 0.15:
                    pl.append(f"Moderate path divergence (CV {path_cv:.2f})")
                else:
                    pl.append(f"Low path divergence (CV {path_cv:.2f})")
                if pos_pct > 60:
                    pl.append(f"{pos_pct:.0f}% of paths end above initial value")
                elif pos_pct < 40:
                    pl.append(f"{100 - pos_pct:.0f}% end below initial value")
                else:
                    pl.append(f"Mixed outcomes ({pos_pct:.0f}% finish above start)")
                interp_paths = "\n".join(pl)

            n = min(max_paths, sp.shape[0])
            idx = np.linspace(0, sp.shape[0] - 1, n, dtype=int)
            paths_out = []
            for i in idx:
                path = sp[i]
                paths_out.append(
                    [
                        {"day": float(d), "value": float(v)}
                        for d, v in enumerate(path.tolist())
                    ]
                )

    return {
        **raw,
        "histogram": hist,
        "normal_curve": normal_curve,
        "percentile_markers": percentile_markers,
        "interpretation_statistics": interp_stats,
        "interpretation_percentiles": interp_pct,
        "interpretation_distribution": interp_distribution,
        "var_cvar_simulation": var_cvar_sim_rows,
        "interpretation_var_cvar_sim": interp_var_cvar_sim,
        "historical_vs_mc_var": hist_vs_mc_rows,
        "interpretation_historical_vs_mc": interp_hist_mc,
        "interpretation_historical_vs_mc_chart": interp_hist_mc_chart,
        "extreme_scenarios": extreme_rows,
        "interpretation_extreme": interp_extreme,
        "path_envelope": path_envelope,
        "interpretation_paths": interp_paths,
        "normal_overlay": {
            "mean": mean_s,
            "std": std_s,
            "x_min": float(finals.min()) if len(finals) else 0.0,
            "x_max": float(finals.max()) if len(finals) else 0.0,
        },
        "sample_paths": paths_out,
    }


def serialize_scenario_catalog(risk_service: RiskService) -> list[dict[str, Any]]:
    scenarios = risk_service.get_available_scenarios()
    out = []
    for key, sc in scenarios.items():
        out.append(
            {
                "key": key,
                "name": sc.name,
                "description": getattr(sc, "description", "") or "",
                "start_date": str(sc.start_date),
                "end_date": str(sc.end_date),
                "market_impact_pct": float(sc.market_impact_pct),
                "recovery_period_days": getattr(sc, "recovery_period_days", None),
            }
        )
    return sorted(out, key=lambda x: x["name"])
