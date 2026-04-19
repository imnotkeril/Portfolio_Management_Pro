/**
 * Aligns /opti "Markowitz.ipynb" preset with 01_Portfolio_Optimization_Markowitz.ipynb:
 * per-ticker upper_bounds, train split, benchmark ticker for charts (AOR).
 */

/** Upper caps from the notebook `upper_bounds` dict (long-only, min 0). */
export const MARKOWITZ_NOTEBOOK_UPPER_BOUNDS: Record<string, number> = {
  SPY: 0.35,
  QQQ: 0.3,
  IEF: 0.35,
  TLT: 0.3,
  GLD: 0.25,
  DBC: 0.2,
  VNQ: 0.2,
  "BTC-USD": 0.12,
};

/** Default cap for tickers not in the notebook universe (conservative). */
export const MARKOWITZ_FALLBACK_MAX_WEIGHT = 0.3;

/** Notebook benchmark for cumulative / comparison charts. */
export const NOTEBOOK_BENCHMARK_CHART_TICKER = "AOR";

export function markowitzUpperCapForTicker(ticker: string): number {
  const t = ticker.trim();
  if (MARKOWITZ_NOTEBOOK_UPPER_BOUNDS[t] != null) {
    return MARKOWITZ_NOTEBOOK_UPPER_BOUNDS[t];
  }
  if (t === "BTC" && MARKOWITZ_NOTEBOOK_UPPER_BOUNDS["BTC-USD"] != null) {
    return MARKOWITZ_NOTEBOOK_UPPER_BOUNDS["BTC-USD"];
  }
  return MARKOWITZ_FALLBACK_MAX_WEIGHT;
}

/**
 * API constraints: global long-only box + per-asset max from the notebook map (or fallback).
 */
export function buildMarkowitzNotebookConstraints(riskTickers: string[]): Record<string, unknown> {
  const weight_bounds: Record<string, { min: number; max: number }> = {};
  for (const ticker of riskTickers) {
    if (ticker === "CASH") continue;
    const maxW = markowitzUpperCapForTicker(ticker);
    weight_bounds[ticker] = { min: 0, max: maxW };
  }
  return {
    long_only: true,
    max_cash_weight: 1,
    min_weight: 0,
    max_weight: 1,
    weight_bounds,
  };
}

/** Sum of per-ticker max weights must be >= 1 for a fully invested long-only portfolio. */
export function markowitzCapsFeasibilityError(riskTickers: string[]): string | null {
  const risk = riskTickers.filter((t) => t !== "CASH");
  if (risk.length === 0) {
    return "Portfolio has no risk assets (excluding CASH).";
  }
  let sumMax = 0;
  for (const t of risk) {
    sumMax += markowitzUpperCapForTicker(t);
  }
  if (sumMax < 1 - 1e-6) {
    return (
      `Asset-level caps are infeasible for this portfolio: total max weight ${(sumMax * 100).toFixed(1)}% < 100%. ` +
      `Add more assets, relax caps, or disable the asset-level cap policy.`
    );
  }
  return null;
}

export type OptiWeightConstraintBuildOpts = {
  longOnly: boolean;
  maxCashPct: number;
  useMinW: boolean;
  minWPct: number;
  useMaxW: boolean;
  maxWPct: number;
  useMinReturn: boolean;
  minReturnPct: number;
  useNotebookPerTickerCaps: boolean;
  useDiversificationLambda: boolean;
  diversificationLambda: number;
  methodSupportsDiversificationLambda: boolean;
};

/** Builds API `constraints` from editable UI (workbench-style), optional Markowitz per-ticker map. */
export function buildOptiWeightConstraints(
  riskTickers: string[],
  opts: OptiWeightConstraintBuildOpts,
): Record<string, unknown> {
  const c: Record<string, unknown> = {};

  if (opts.useNotebookPerTickerCaps) {
    Object.assign(c, buildMarkowitzNotebookConstraints(riskTickers));
    c.long_only = opts.longOnly;
    c.max_cash_weight = opts.maxCashPct / 100;
    if (opts.useMinW) c.min_weight = opts.minWPct / 100;
    if (opts.useMaxW) c.max_weight = opts.maxWPct / 100;
  } else {
    c.long_only = opts.longOnly;
    c.max_cash_weight = opts.maxCashPct / 100;
    if (opts.useMinW) c.min_weight = opts.minWPct / 100;
    else c.min_weight = opts.longOnly ? 0 : -1;
    if (opts.useMaxW) c.max_weight = opts.maxWPct / 100;
    else c.max_weight = 1;
  }

  if (opts.useMinReturn) c.min_return = opts.minReturnPct / 100;
  if (opts.methodSupportsDiversificationLambda && opts.useDiversificationLambda) {
    c.diversification_lambda = opts.diversificationLambda;
  }

  return c;
}
