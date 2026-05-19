"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { percentDecimal } from "@/lib/format";
import type { Portfolio } from "@/lib/types";
import {
  AVAILABLE_OPTIMIZATION_METHODS,
  BENCHMARK_CHART_PRESETS,
  getMethodPreset,
  METHOD_DESCRIPTIONS,
  METHOD_NAMES,
  METHODS_NEEDING_BENCHMARK,
  OBJECTIVE_METHOD_MAPPING,
  TRAINING_WINDOW_OPTIONS,
  getAvailableObjectives,
  getDefaultObjective,
  validateWeightConstraints,
} from "@/app/optimization/config";
import { NOTEBOOK_METHOD_SERIES } from "@/app/optimization-studio/notebook-methods";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ─── types ─── */

type SnapshotRow = {
  ticker: string;
  shares: number;
  price: number;
  value: number;
  weight: number;
};

type FullBundle = {
  success: boolean;
  message?: string;
  optimization?: Record<string, unknown>;
  optimization_period?: { start: string; end: string };
  validation_period?: { start: string; end: string };
  out_of_sample?: boolean;
  training_ratio?: number;
  metrics?: {
    current: Record<string, number>;
    optimized: Record<string, number>;
  };
  interpretation_comparison?: string;
  allocation?: Array<{
    ticker: string;
    current_weight: number;
    optimal_weight: number;
    difference: number;
  }>;
  interpretation_allocation?: string;
  trades?: Array<{
    ticker: string;
    action: string;
    shares: number;
    value: number;
    current_weight?: number;
    optimal_weight?: number;
  }>;
  interpretation_trades?: string;
  charts?: {
    cumulative_returns: Array<{
      x: string;
      current?: number | null;
      optimized?: number | null;
      benchmark?: number | null;
    }>;
    drawdown: Array<{
      x: string;
      current?: number | null;
      optimized?: number | null;
      benchmark?: number | null;
    }>;
  };
  efficient_frontier?: {
    volatilities_pct: number[];
    returns_pct: number[];
    tangency_portfolio?: { volatility?: number; expected_return?: number } | null;
    min_variance_portfolio?: { volatility?: number; expected_return?: number } | null;
    optimized_point?: {
      volatility_pct?: number;
      return_pct?: number;
      sharpe?: number;
    };
    current_point?: { volatility_pct?: number; return_pct?: number };
    benchmark_point?: { volatility_pct?: number; return_pct?: number };
    fallback_validation_period?: boolean;
  } | null;
  correlation?: {
    tickers: string[];
    matrix: number[][];
    interpretation?: string;
  } | null;
  sensitivity?: {
    analysis_type?: string;
    results?: Record<string, unknown>[];
    base_weights?: Record<string, number>;
  } | null;
  interpretation_sensitivity?: string;
  warnings?: string[];
};

const C = {
  grid: "rgba(255,255,255,0.06)",
  text: "rgba(255,255,255,0.5)",
  primary: "#bf9ffb",
  secondary: "#7dc4e4",
  ok: "#74f174",
  danger: "#faa1a4",
  warn: "#ffd066",
};

function InterpretBox({ text }: { text: string }) {
  if (!text?.trim()) return null;
  const plain = text.replace(/\*\*(.*?)\*\*/g, "$1");
  return (
    <div className="mt-3 rounded-lg border border-white/10 bg-[var(--info-bg)] p-3 text-sm text-white/85 whitespace-pre-wrap">
      {plain}
    </div>
  );
}

function CmpMetricCard({
  label,
  portfolioValue,
  benchmarkValue,
  format,
  higherIsBetter,
}: {
  label: string;
  portfolioValue: number | null | undefined;
  benchmarkValue: number | null | undefined;
  format: "percent" | "ratio";
  higherIsBetter: boolean | null;
}) {
  const fmtV = (v: number | null | undefined) => {
    if (v == null || !Number.isFinite(v)) return "—";
    if (format === "percent") return percentDecimal(v, 2);
    return v.toFixed(3);
  };
  const pStr = fmtV(portfolioValue);
  const bStr = benchmarkValue != null ? fmtV(benchmarkValue) : null;
  let dotColor = "bg-white/30";
  if (portfolioValue != null && benchmarkValue != null && higherIsBetter !== null) {
    const better = higherIsBetter ? portfolioValue > benchmarkValue : portfolioValue < benchmarkValue;
    dotColor = better ? "bg-[var(--ok)]" : "bg-[var(--danger)]";
  }
  return (
    <div className="metric-card">
      <div className="text-xs text-white/40 mb-1">{label}</div>
      <div className="text-xl font-bold text-white">{pStr}</div>
      {bStr && (
        <div className="flex items-center gap-1.5 mt-1">
          <span className={`h-2 w-2 rounded-full ${dotColor}`} />
          <span className="text-xs text-white/40">Current: {bStr}</span>
        </div>
      )}
    </div>
  );
}

function corrColor(c: number): string {
  const t = Math.max(0, Math.min(1, (c + 1) / 2));
  const r = Math.round(40 + t * 200);
  const g = Math.round(90 + (1 - Math.abs(2 * t - 1)) * 120);
  const b = Math.round(220 - t * 180);
  return `rgb(${r},${g},${b})`;
}

function corrTextColor(c: number): string {
  return Math.abs(c) >= 0.55 ? "rgba(255,255,255,0.95)" : "rgba(17,24,39,0.95)";
}

function extractErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    const raw = err.message?.trim();
    if (!raw) return "Request failed.";
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (typeof parsed?.detail === "string" && parsed.detail.trim()) {
        return parsed.detail;
      }
    } catch {
      // Non-JSON error payload, use the original message.
    }
    return raw;
  }
  return "Request failed.";
}

export type OptimizationWorkbenchVariant = "standard" | "notebook";

export function OptimizationWorkbench({ variant }: { variant: OptimizationWorkbenchVariant }) {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [portfolioId, setPortfolioId] = useState("");
  const [snapshot, setSnapshot] = useState<SnapshotRow[]>([]);

  const defaultEnd = () => new Date().toISOString().slice(0, 10);
  const defaultStart = () => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().slice(0, 10);
  };
  const [startDate, setStartDate] = useState(defaultStart);
  const [endDate, setEndDate] = useState(defaultEnd);

  const [useOOS, setUseOOS] = useState(variant === "notebook");
  const [trainingWindowLabel, setTrainingWindowLabel] = useState("30% (Recommended)");

  const [benchmarkChart, setBenchmarkChart] = useState<string>("None");
  const [benchmarkOpt, setBenchmarkOpt] = useState("SPY");

  const [selectedMethod, setSelectedMethod] = useState<string>(() =>
    variant === "notebook" ? "mean_variance" : "",
  );

  const [longOnly, setLongOnly] = useState(true);
  const [maxCashPct, setMaxCashPct] = useState(10);
  const [useMinW, setUseMinW] = useState(false);
  const [minWPct, setMinWPct] = useState(0);
  const [useMaxW, setUseMaxW] = useState(false);
  const [maxWPct, setMaxWPct] = useState(100);
  const [useMinReturn, setUseMinReturn] = useState(false);
  const [minReturnPct, setMinReturnPct] = useState(3);
  const [useDivLambda, setUseDivLambda] = useState(false);
  const [divLambda, setDivLambda] = useState(1);

  const [objective, setObjective] = useState<string>("maximize_sharpe");
  const [cvarConfidence, setCvarConfidence] = useState(0.95);
  const [robustUr, setRobustUr] = useState(0.1);
  const [robustUc, setRobustUc] = useState(0.1);

  const [includeFrontier, setIncludeFrontier] = useState(true);
  const [includeSensitivity, setIncludeSensitivity] = useState(false);
  const [sensitivityType, setSensitivityType] = useState<"returns" | "covariance">("returns");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<FullBundle | null>(null);

  useEffect(() => {
    api
      .get<Portfolio[]>("/portfolios")
      .then((list) => {
        setPortfolios(list);
        setPortfolioId((prev) => prev || list[0]?.id || "");
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!portfolioId) return;
    api
      .get<SnapshotRow[]>(`/portfolios/${portfolioId}/optimization-snapshot`)
      .then(setSnapshot)
      .catch(() => setSnapshot([]));
  }, [portfolioId]);

  const selectedPortfolio = useMemo(
    () => portfolios.find((p) => p.id === portfolioId) ?? null,
    [portfolios, portfolioId],
  );

  const numAssets = useMemo(() => {
    if (!selectedPortfolio) return 0;
    return selectedPortfolio.positions.filter((p) => p.ticker !== "CASH").length;
  }, [selectedPortfolio]);

  const maxAllowedMinPct = useMemo(() => {
    if (numAssets <= 0) return 50;
    return Math.min(50, (1 / numAssets) * 100);
  }, [numAssets]);

  const constraintWarnings = useMemo(() => {
    const mw = useMinW ? minWPct / 100 : undefined;
    const xw = useMaxW ? maxWPct / 100 : undefined;
    return validateWeightConstraints(mw, xw, Math.max(numAssets, 1));
  }, [useMinW, minWPct, useMaxW, maxWPct, numAssets]);

  useEffect(() => {
    if (!useMinW) return;
    if (minWPct > maxAllowedMinPct) {
      setMinWPct(maxAllowedMinPct);
    }
  }, [useMinW, minWPct, maxAllowedMinPct]);

  useEffect(() => {
    if (!selectedMethod) return;
    const preset = getMethodPreset(selectedMethod);
    setLongOnly(preset.longOnly);
    setMaxCashPct(preset.maxCashPct);
    setUseMinW(preset.useMinW);
    setMinWPct(Math.min(preset.minWPct, maxAllowedMinPct));
    setUseMaxW(preset.useMaxW);
    setMaxWPct(preset.maxWPct);
    setUseMinReturn(preset.useMinReturn);
    setMinReturnPct(preset.minReturnPct);
    setUseDivLambda(preset.useDivLambda);
    setDivLambda(preset.divLambda);
    setUseOOS(preset.useOOS);
    setTrainingWindowLabel(preset.trainingWindowLabel);
    setIncludeFrontier(preset.includeFrontier);
    setIncludeSensitivity(preset.includeSensitivity);
    setSensitivityType(preset.sensitivityType);
    if (preset.cvarConfidence != null) setCvarConfidence(preset.cvarConfidence);
    if (preset.robustUr != null) setRobustUr(preset.robustUr);
    if (preset.robustUc != null) setRobustUc(preset.robustUc);
    if (preset.benchmarkOpt) setBenchmarkOpt(preset.benchmarkOpt);

    const avail = getAvailableObjectives(selectedMethod);
    const def = getDefaultObjective(selectedMethod);
    if (avail.length) {
      const recommended =
        (preset.objective && avail.includes(preset.objective) && preset.objective) ||
        (def && avail.includes(def) && def) ||
        avail[0];
      setObjective((prev) => (avail.includes(prev) ? prev : recommended));
    }
  }, [selectedMethod, maxAllowedMinPct]);

  const trainingRatio = TRAINING_WINDOW_OPTIONS[trainingWindowLabel]?.ratio ?? 0.3;

  const oosPreview = useMemo(() => {
    if (!useOOS) return null;
    const s = new Date(startDate);
    const e = new Date(endDate);
    const analysisDays = Math.max(0, Math.round((e.getTime() - s.getTime()) / 86400000));
    const trainingDays = Math.floor(analysisDays * trainingRatio);
    const trainStart = new Date(s);
    trainStart.setDate(trainStart.getDate() - trainingDays);
    return {
      trainingDays,
      analysisDays,
      trainStart: trainStart.toISOString().slice(0, 10),
      trainEnd: startDate,
    };
  }, [useOOS, startDate, endDate, trainingRatio]);

  const buildPayload = useCallback(() => {
    const constraints: Record<string, unknown> = {
      long_only: longOnly,
      max_cash_weight: maxCashPct / 100,
    };
    if (useMinW) constraints.min_weight = minWPct / 100;
    else constraints.min_weight = longOnly ? 0 : -1;
    if (useMaxW) constraints.max_weight = maxWPct / 100;
    else constraints.max_weight = 1;
    if (useMinReturn) constraints.min_return = minReturnPct / 100;
    if (useDivLambda) constraints.diversification_lambda = divLambda;

    const method_params: Record<string, unknown> = {};
    const availObj = getAvailableObjectives(selectedMethod);
    if (availObj.length && objective) method_params.objective = objective;
    if (selectedMethod === "cvar_optimization" || selectedMethod === "mean_cvar") {
      method_params.confidence_level = cvarConfidence;
    }
    if (selectedMethod === "robust") {
      method_params.uncertainty_radius_returns = robustUr;
      method_params.uncertainty_radius_cov = robustUc;
    }

    const needsBench = METHODS_NEEDING_BENCHMARK.has(selectedMethod);

    return {
      portfolio_id: portfolioId,
      method: selectedMethod,
      start_date: startDate,
      end_date: endDate,
      constraints,
      benchmark_ticker: needsBench ? benchmarkOpt.trim().toUpperCase() : null,
      method_params: Object.keys(method_params).length ? method_params : null,
      out_of_sample: useOOS,
      training_ratio: trainingRatio,
      benchmark_for_charts: benchmarkChart === "None" ? null : benchmarkChart,
      include_efficient_frontier: includeFrontier,
      frontier_n_points: 150,
      include_sensitivity: includeSensitivity,
      sensitivity_analysis_type: sensitivityType,
    };
  }, [
    portfolioId,
    selectedMethod,
    startDate,
    endDate,
    longOnly,
    maxCashPct,
    useMinW,
    minWPct,
    useMaxW,
    maxWPct,
    useMinReturn,
    minReturnPct,
    useDivLambda,
    divLambda,
    objective,
    cvarConfidence,
    robustUr,
    robustUc,
    benchmarkOpt,
    benchmarkChart,
    useOOS,
    trainingRatio,
    includeFrontier,
    includeSensitivity,
    sensitivityType,
  ]);

  async function runOptimize(e?: FormEvent) {
    e?.preventDefault();
    if (!portfolioId || !selectedMethod) {
      setError("Select a portfolio and optimization method.");
      return;
    }
    if (new Date(startDate) >= new Date(endDate)) {
      setError("Start date must be before end date.");
      return;
    }
    if (constraintWarnings.length > 0) {
      setError("Fix constraint warnings before running optimization.");
      return;
    }
    if (METHODS_NEEDING_BENCHMARK.has(selectedMethod) && !benchmarkOpt.trim()) {
      setError("Benchmark ticker is required for the selected method.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<FullBundle>("/optimization/full", buildPayload());
      setBundle(data);
    } catch (err) {
      setError(extractErrorMessage(err));
      setBundle(null);
    } finally {
      setLoading(false);
    }
  }

  const frontierCurve = useMemo(() => {
    const ff = bundle?.efficient_frontier;
    if (!ff?.volatilities_pct?.length) return [];
    const pts = ff.volatilities_pct
      .map((v, i) => ({
        x: Number(v),
        y: Number(ff.returns_pct[i] ?? 0),
      }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!pts.length) return [];

    // 1) Deduplicate very-close vol points by keeping the best return.
    const byVolBucket = new Map<number, { x: number; y: number }>();
    for (const p of pts) {
      const key = Math.round(p.x * 100) / 100; // 0.01 vol bucket
      const prev = byVolBucket.get(key);
      if (!prev || p.y > prev.y) byVolBucket.set(key, p);
    }

    const deduped = [...byVolBucket.values()].sort((a, b) => a.x - b.x);

    // 2) Keep only Pareto-efficient (upper envelope) points.
    const efficient: Array<{ x: number; y: number }> = [];
    let bestY = -Infinity;
    for (const p of deduped) {
      if (p.y > bestY + 1e-6) {
        efficient.push(p);
        bestY = p.y;
      }
    }
    return efficient.length >= 2 ? efficient : deduped;
  }, [bundle]);

  const frontierMarkers = useMemo(() => {
    const ff = bundle?.efficient_frontier;
    if (!ff) return [];
    const m: Array<{ name: string; x: number; y: number; fill: string }> = [];
    const tp = ff.tangency_portfolio;
    if (tp?.volatility != null && tp?.expected_return != null) {
      m.push({
        name: "Max Sharpe",
        x: tp.volatility * 100,
        y: tp.expected_return * 100,
        fill: "#ffd700",
      });
    }
    const mv = ff.min_variance_portfolio;
    if (mv?.volatility != null && mv?.expected_return != null) {
      m.push({
        name: "Min Vol",
        x: mv.volatility * 100,
        y: mv.expected_return * 100,
        fill: "#4CAF50",
      });
    }
    const op = ff.optimized_point;
    if (op?.volatility_pct != null && op?.return_pct != null) {
      m.push({ name: "Optimized", x: op.volatility_pct, y: op.return_pct, fill: C.ok });
    }
    const cp = ff.current_point;
    if (cp?.volatility_pct != null && cp?.return_pct != null) {
      m.push({ name: "Current", x: cp.volatility_pct, y: cp.return_pct, fill: C.primary });
    }
    const bp = ff.benchmark_point;
    if (bp?.volatility_pct != null && bp?.return_pct != null) {
      m.push({ name: "Benchmark", x: bp.volatility_pct, y: bp.return_pct, fill: C.secondary });
    }
    return m;
  }, [bundle]);

  const frontierDomain = useMemo(() => {
    const allX = frontierCurve.map((p) => p.x);
    const allY = frontierCurve.map((p) => p.y);
    if (!allX.length || !allY.length) return null;

    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);

    const xPad = Math.max(1, (maxX - minX) * 0.08);
    const yPad = Math.max(1, (maxY - minY) * 0.1);
    return {
      x: [Math.max(0, minX - xPad), maxX + xPad] as [number, number],
      y: [minY - yPad, maxY + yPad] as [number, number],
    };
  }, [frontierCurve]);

  const visibleFrontierMarkers = useMemo(() => {
    if (!frontierDomain) return frontierMarkers;
    const [xMin, xMax] = frontierDomain.x;
    const [yMin, yMax] = frontierDomain.y;
    const xPad = (xMax - xMin) * 0.08;
    const yPad = (yMax - yMin) * 0.08;
    return frontierMarkers.filter(
      (m) => m.x >= xMin - xPad && m.x <= xMax + xPad && m.y >= yMin - yPad && m.y <= yMax + yPad,
    );
  }, [frontierMarkers, frontierDomain]);

  const outsideFrontierMarkers = useMemo(() => {
    const visible = new Set(visibleFrontierMarkers.map((m) => m.name));
    return frontierMarkers.filter((m) => !visible.has(m.name));
  }, [frontierMarkers, visibleFrontierMarkers]);

  const curM = bundle?.metrics?.current;
  const optM = bundle?.metrics?.optimized;

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-16">
      <header>
        <h1 className="text-3xl font-semibold text-white">
          {variant === "notebook" ? "Optimization Studio" : "Portfolio Optimization"}
        </h1>
        <p className="mt-2 text-white/60">
          {variant === "notebook" ? (
            <>
              Production runbook aligned with the Opti notebook series (01–10): same API as the rest of the
              terminal, buy-and-hold backtest for fair comparison vs current, optional train / validation split.
            </>
          ) : (
            <>Optimize weights using the same methods and constraints as the Streamlit app.</>
          )}
        </p>
      </header>

      {!portfolios.length ? (
        <div className="panel p-6 text-amber-200">No portfolios found. Create one first.</div>
      ) : (
        <form className="space-y-6" onSubmit={runOptimize}>
          <section className="panel p-6 space-y-4">
            <label className="block text-sm text-white/50">Portfolio</label>
            <select
              className="input"
              value={portfolioId}
              onChange={(e) => setPortfolioId(e.target.value)}
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>

            <details className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
              <summary className="cursor-pointer text-sm font-medium text-white/80">
                Current Portfolio Information
              </summary>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-sm text-white/80">
                  <thead>
                    <tr className="border-b border-white/10 text-white/50">
                      <th className="py-2 pr-3">Ticker</th>
                      <th className="py-2 pr-3">Shares</th>
                      <th className="py-2 pr-3">Price</th>
                      <th className="py-2 pr-3">Value</th>
                      <th className="py-2">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.map((row) => (
                      <tr key={row.ticker} className="border-b border-white/5">
                        <td className="py-1.5 pr-3">{row.ticker}</td>
                        <td className="py-1.5 pr-3">{row.shares.toLocaleString()}</td>
                        <td className="py-1.5 pr-3">${row.price.toFixed(2)}</td>
                        <td className="py-1.5 pr-3">${row.value.toLocaleString()}</td>
                        <td className="py-1.5">{(row.weight * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!snapshot.length ? (
                  <p className="text-xs text-white/40 mt-2">No live price rows (check tickers / network).</p>
                ) : null}
              </div>
            </details>
          </section>

          <section className="panel p-6 space-y-4">
            <h2 className="text-lg font-medium text-white">Optimization Parameters</h2>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="block text-xs text-white/50 mb-1">Start date</label>
                <input
                  type="date"
                  className="input"
                  value={startDate}
                  max={defaultEnd()}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-xs text-white/50 mb-1">End date</label>
                <input
                  type="date"
                  className="input"
                  value={endDate}
                  min={startDate}
                  max={defaultEnd()}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>

            <div className="border-t border-white/10 pt-4">
              <h3 className="text-sm font-medium text-white/90 mb-2">Out-of-Sample Testing</h3>
              <label className="flex items-center gap-2 text-sm text-white/80">
                <input type="checkbox" checked={useOOS} onChange={(e) => setUseOOS(e.target.checked)} />
                Use out-of-sample testing
              </label>
              {useOOS ? (
                <div className="mt-3 space-y-2">
                  <select
                    className="input"
                    value={trainingWindowLabel}
                    onChange={(e) => setTrainingWindowLabel(e.target.value)}
                  >
                    {Object.keys(TRAINING_WINDOW_OPTIONS).map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-[var(--info)]">
                    {TRAINING_WINDOW_OPTIONS[trainingWindowLabel]?.description}
                  </p>
                  {oosPreview ? (
                    <div className="text-xs text-white/50 space-y-1">
                      <p>
                        Training: {oosPreview.trainStart} → {oosPreview.trainEnd} (
                        {oosPreview.trainingDays} days)
                      </p>
                      <p>
                        Validation: {startDate} → {endDate} ({oosPreview.analysisDays} days)
                      </p>
                    </div>
                  ) : null}
                  <p className="text-xs text-emerald-200/80">
                    Recommended for honest validation: train on prior window, evaluate on selected period.
                  </p>
                </div>
              ) : (
                <p className="text-xs text-white/40 mt-2">
                  Optimization uses {startDate} → {endDate}
                </p>
              )}
            </div>

            <div className="border-t border-white/10 pt-4">
              <p className="text-sm text-white/70 mb-2">Benchmark (optional, for charts)</p>
              <select
                className="input max-w-xs"
                value={benchmarkChart}
                onChange={(e) => setBenchmarkChart(e.target.value)}
              >
                {BENCHMARK_CHART_PRESETS.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>

            <div className="border-t border-white/10 pt-4">
              <label className="block text-xs text-white/50 mb-2">
                {variant === "notebook" ? "Method (notebook series 01–10)" : "Optimization method"}
              </label>
              {variant === "notebook" ? (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {NOTEBOOK_METHOD_SERIES.map((entry) => {
                    const active = selectedMethod === entry.method;
                    return (
                      <button
                        key={entry.method}
                        type="button"
                        onClick={() => setSelectedMethod(entry.method)}
                        className={`rounded-xl border p-3 text-left transition ${
                          active
                            ? "border-emerald-400/60 bg-emerald-400/10 ring-1 ring-emerald-400/30"
                            : "border-white/10 bg-white/[0.02] hover:border-white/20"
                        }`}
                      >
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="font-mono text-xs text-emerald-300/90">{entry.seriesId}</span>
                          <span className="truncate text-sm font-medium text-white">{entry.displayName}</span>
                        </div>
                        <p className="mt-1 truncate font-mono text-[10px] text-white/35" title={entry.notebookFile}>
                          {entry.notebookFile}
                        </p>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <select
                  className="input"
                  value={selectedMethod}
                  onChange={(e) => setSelectedMethod(e.target.value)}
                >
                  <option value="">-- Select Method --</option>
                  {AVAILABLE_OPTIMIZATION_METHODS.map((m) => (
                    <option key={m} value={m}>
                      {METHOD_NAMES[m] ?? m}
                    </option>
                  ))}
                </select>
              )}
              {selectedMethod && METHOD_DESCRIPTIONS[selectedMethod] ? (
                <details className="mt-2 rounded-lg border border-white/10 p-3">
                  <summary className="cursor-pointer text-sm text-[var(--accent)]">
                    ℹ️ {METHOD_DESCRIPTIONS[selectedMethod].short}
                  </summary>
                  <p className="mt-2 text-xs text-white/65 leading-relaxed">
                    {METHOD_DESCRIPTIONS[selectedMethod].long}
                  </p>
                </details>
              ) : null}
            </div>

            {METHODS_NEEDING_BENCHMARK.has(selectedMethod) ? (
              <div>
                <label className="block text-xs text-white/50 mb-1">
                  Benchmark ticker (required for this method)
                </label>
                <input
                  className="input max-w-xs"
                  value={benchmarkOpt}
                  onChange={(e) => setBenchmarkOpt(e.target.value.toUpperCase())}
                  placeholder="SPY"
                />
              </div>
            ) : null}

            {selectedMethod ? (
              <>
                <div className="border-t border-white/10 pt-4 space-y-4">
                  <h3 className="text-sm font-medium text-white">Constraints</h3>
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="flex items-center justify-between gap-2 text-sm text-white/80 metric-card">
                      Long only (no shorts)
                      <input
                        type="checkbox"
                        checked={longOnly}
                        onChange={(e) => setLongOnly(e.target.checked)}
                      />
                    </label>
                    <div>
                      <label className="text-xs text-white/50">
                        Max cash weight: {maxCashPct}%
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        className="w-full"
                        value={maxCashPct}
                        onChange={(e) => setMaxCashPct(Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="metric-card space-y-2">
                      <label className="flex items-center gap-2 text-sm text-white/80">
                        <input
                          type="checkbox"
                          checked={useMinW}
                          onChange={(e) => setUseMinW(e.target.checked)}
                        />
                        Minimum weight per asset
                      </label>
                      {useMinW ? (
                        <>
                          <input
                            type="range"
                            min={0}
                            max={maxAllowedMinPct}
                            step={0.5}
                            className="w-full"
                            value={minWPct}
                            onChange={(e) => setMinWPct(Number(e.target.value))}
                          />
                          <span className="text-xs text-white/50">{minWPct.toFixed(1)}%</span>
                        </>
                      ) : null}
                    </div>
                    <div className="metric-card space-y-2">
                      <label className="flex items-center gap-2 text-sm text-white/80">
                        <input
                          type="checkbox"
                          checked={useMaxW}
                          onChange={(e) => setUseMaxW(e.target.checked)}
                        />
                        Maximum weight per asset
                      </label>
                      {useMaxW ? (
                        <>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            className="w-full"
                            value={maxWPct}
                            onChange={(e) => setMaxWPct(Number(e.target.value))}
                          />
                          <span className="text-xs text-white/50">{maxWPct.toFixed(1)}%</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                  {constraintWarnings.map((w) => (
                    <div key={w} className="text-xs text-amber-200">
                      {w}
                    </div>
                  ))}
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="metric-card space-y-2">
                      <label className="flex items-center gap-2 text-sm text-white/80">
                        <input
                          type="checkbox"
                          checked={useMinReturn}
                          onChange={(e) => setUseMinReturn(e.target.checked)}
                        />
                        Minimum expected return (annualized)
                      </label>
                      {useMinReturn ? (
                        <>
                          <input
                            type="range"
                            min={0}
                            max={20}
                            step={0.5}
                            className="w-full"
                            value={minReturnPct}
                            onChange={(e) => setMinReturnPct(Number(e.target.value))}
                          />
                          <span className="text-xs text-white/50">{minReturnPct.toFixed(1)}%</span>
                        </>
                      ) : null}
                    </div>
                    <div className="metric-card space-y-2">
                      <label className="flex items-center gap-2 text-sm text-white/80">
                        <input
                          type="checkbox"
                          checked={useDivLambda}
                          onChange={(e) => setUseDivLambda(e.target.checked)}
                        />
                        Diversification penalty (λ)
                      </label>
                      {useDivLambda ? (
                        <>
                          <input
                            type="range"
                            min={0.1}
                            max={10}
                            step={0.1}
                            className="w-full"
                            value={divLambda}
                            onChange={(e) => setDivLambda(Number(e.target.value))}
                          />
                          <span className="text-xs text-white/50">{divLambda.toFixed(1)}</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                </div>

                <div className="border-t border-white/10 pt-4 space-y-3">
                  <h3 className="text-sm font-medium text-white">Objective function</h3>
                  {getAvailableObjectives(selectedMethod).length === 0 ? (
                    <p className="text-xs text-white/50">
                      This method uses a fixed objective (no selection).
                    </p>
                  ) : (
                    <select
                      className="input"
                      value={objective}
                      onChange={(e) => setObjective(e.target.value)}
                    >
                      {getAvailableObjectives(selectedMethod).map((k) => (
                        <option key={k} value={k}>
                          {OBJECTIVE_METHOD_MAPPING[k]?.display ?? k}
                        </option>
                      ))}
                    </select>
                  )}
                </div>

                {(selectedMethod === "cvar_optimization" || selectedMethod === "mean_cvar") && (
                  <div>
                    <label className="text-xs text-white/50">CVaR confidence</label>
                    <select
                      className="input max-w-xs mt-1"
                      value={cvarConfidence}
                      onChange={(e) => setCvarConfidence(Number(e.target.value))}
                    >
                      <option value={0.9}>90%</option>
                      <option value={0.95}>95%</option>
                      <option value={0.99}>99%</option>
                    </select>
                  </div>
                )}

                {selectedMethod === "robust" && (
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="text-xs text-white/50">Uncertainty radius (returns)</label>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        className="w-full"
                        value={robustUr}
                        onChange={(e) => setRobustUr(Number(e.target.value))}
                      />
                      <span className="text-xs text-white/50">{robustUr.toFixed(2)}</span>
                    </div>
                    <div>
                      <label className="text-xs text-white/50">Uncertainty radius (covariance)</label>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        className="w-full"
                        value={robustUc}
                        onChange={(e) => setRobustUc(Number(e.target.value))}
                      />
                      <span className="text-xs text-white/50">{robustUc.toFixed(2)}</span>
                    </div>
                  </div>
                )}

                {selectedMethod === "black_litterman" && (
                  <div className="rounded-lg border border-[var(--info)]/30 bg-[var(--info-bg)] p-3 text-xs text-white/70">
                    Black-Litterman uses market-implied equilibrium; custom views can be added later
                    (same note as Streamlit).
                  </div>
                )}

                <div className="border-t border-white/10 pt-4 space-y-2">
                  <label className="flex items-center gap-2 text-sm text-white/80">
                    <input
                      type="checkbox"
                      checked={includeFrontier}
                      onChange={(e) => setIncludeFrontier(e.target.checked)}
                    />
                    Include efficient frontier (slower)
                  </label>
                  <label className="flex items-center gap-2 text-sm text-white/80">
                    <input
                      type="checkbox"
                      checked={includeSensitivity}
                      onChange={(e) => setIncludeSensitivity(e.target.checked)}
                    />
                    Include sensitivity analysis (runs extra optimization — slower)
                  </label>
                  {includeSensitivity ? (
                    <select
                      className="input max-w-xs"
                      value={sensitivityType}
                      onChange={(e) =>
                        setSensitivityType(e.target.value as "returns" | "covariance")
                      }
                    >
                      <option value="returns">Return sensitivity</option>
                      <option value="covariance">Covariance sensitivity</option>
                    </select>
                  ) : null}
                </div>
              </>
            ) : null}

            {error ? <div className="text-sm text-[var(--danger)]">{error}</div> : null}

            <button
              type="submit"
              className="btn btn-primary w-full md:w-auto"
              disabled={loading || !selectedMethod}
            >
              {loading ? "Running…" : "Optimize Portfolio"}
            </button>
          </section>
        </form>
      )}

      {bundle ? (
        <section className="space-y-8">
          {!bundle.success ? (
            <div className="panel p-6 text-[var(--danger)]">
              Optimization failed: {bundle.message || "Unknown error"}
            </div>
          ) : (
            <>
              {bundle.warnings?.length ? (
                <div className="space-y-2">
                  {bundle.warnings.map((w) => (
                    <div key={w} className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-100">
                      {w}
                    </div>
                  ))}
                </div>
              ) : null}

              <div className="panel p-6">
                <h2 className="text-xl font-medium text-white">Optimization Results</h2>
                <p className="text-xs text-white/45 mt-1">
                  Validation / display period: {bundle.validation_period?.start} →{" "}
                  {bundle.validation_period?.end}
                </p>
                {bundle.out_of_sample ? (
                  <p className="text-xs text-white/45 mt-1">
                    Training window used for weights: {bundle.optimization_period?.start} →{" "}
                    {bundle.optimization_period?.end}
                  </p>
                ) : (
                  <p className="text-xs text-white/45 mt-1">
                    In-sample mode: weights and the efficient frontier are estimated on the same interval as above.
                    Performance charts apply those fixed weights across the display period (no separate train/test split
                    on the chart).
                  </p>
                )}
                {bundle.efficient_frontier?.fallback_validation_period ? (
                  <p className="text-xs text-amber-200/90 mt-2">
                    Efficient frontier fell back to validation period (insufficient training data).
                  </p>
                ) : null}
              </div>

              <div className="panel p-6 space-y-3">
                <h3 className="text-lg text-white">Comparison: Optimized vs Current</h3>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <CmpMetricCard
                    label="Total Return"
                    portfolioValue={optM?.total_return}
                    benchmarkValue={curM?.total_return}
                    format="percent"
                    higherIsBetter={true}
                  />
                  <CmpMetricCard
                    label="Sharpe Ratio"
                    portfolioValue={optM?.sharpe_ratio}
                    benchmarkValue={curM?.sharpe_ratio}
                    format="ratio"
                    higherIsBetter={true}
                  />
                  <CmpMetricCard
                    label="Volatility"
                    portfolioValue={optM?.volatility}
                    benchmarkValue={curM?.volatility}
                    format="percent"
                    higherIsBetter={false}
                  />
                  <CmpMetricCard
                    label="Max Drawdown"
                    portfolioValue={optM?.max_drawdown}
                    benchmarkValue={curM?.max_drawdown}
                    format="percent"
                    higherIsBetter={false}
                  />
                </div>
                <InterpretBox text={bundle.interpretation_comparison ?? ""} />
              </div>

              <div className="panel p-6 space-y-3">
                <h3 className="text-lg text-white">Current vs Optimal Allocation</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-white/85">
                    <thead>
                      <tr className="border-b border-white/10 text-white/50">
                        <th className="py-2 text-left">Ticker</th>
                        <th className="py-2 text-right">Current</th>
                        <th className="py-2 text-right">Optimal</th>
                        <th className="py-2 text-right">Diff</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bundle.allocation?.map((row) => (
                        <tr key={row.ticker} className="border-b border-white/5">
                          <td className="py-1.5">{row.ticker}</td>
                          <td className="py-1.5 text-right">{(row.current_weight * 100).toFixed(2)}%</td>
                          <td className="py-1.5 text-right">{(row.optimal_weight * 100).toFixed(2)}%</td>
                          <td className="py-1.5 text-right">
                            {(row.difference * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={bundle.allocation ?? []}
                      margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                      <XAxis dataKey="ticker" tick={{ fill: C.text, fontSize: 10 }} />
                      <YAxis
                        tick={{ fill: C.text, fontSize: 10 }}
                        tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                        domain={[0, 1]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1a1d25",
                          border: "1px solid rgba(255,255,255,0.1)",
                          fontSize: 12,
                        }}
                        formatter={(value) => {
                          if (value == null) return "";
                          const n = typeof value === "number" ? value : Number(value);
                          return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : "";
                        }}
                      />
                      <Legend />
                      <Bar
                        dataKey="current_weight"
                        name="Current"
                        fill={C.primary}
                        radius={[4, 4, 0, 0]}
                      />
                      <Bar
                        dataKey="optimal_weight"
                        name="Optimal"
                        fill={C.secondary}
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <InterpretBox text={bundle.interpretation_allocation ?? ""} />
              </div>

              <div className="panel p-6 space-y-3">
                <h3 className="text-lg text-white">Trade List</h3>
                {bundle.trades?.length ? (
                  <>
                    <table className="w-full text-sm text-white/85">
                      <thead>
                        <tr className="border-b border-white/10 text-white/50">
                          <th className="py-2 text-left">Ticker</th>
                          <th className="py-2">Action</th>
                          <th className="py-2 text-right">Shares</th>
                          <th className="py-2 text-right">Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bundle.trades.map((t) => (
                          <tr key={`${t.ticker}-${t.action}`} className="border-b border-white/5">
                            <td className="py-1.5">{t.ticker}</td>
                            <td className="py-1.5">{t.action}</td>
                            <td className="py-1.5 text-right">{t.shares.toLocaleString()}</td>
                            <td className="py-1.5 text-right">${t.value.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <InterpretBox text={bundle.interpretation_trades ?? ""} />
                  </>
                ) : (
                  <p className="text-sm text-white/50">No trades required.</p>
                )}
              </div>

              <div className="panel p-6 space-y-4">
                <h3 className="text-lg text-white">Performance Charts</h3>
                {bundle.out_of_sample ? (
                  <p className="text-xs text-white/45">
                    Series start at the validation period ({bundle.validation_period?.start} →{" "}
                    {bundle.validation_period?.end}); weights were fitted on the prior training window only.
                  </p>
                ) : null}
                {bundle.charts?.cumulative_returns?.length ? (
                  <>
                    <p className="text-xs text-white/50">Cumulative returns (%)</p>
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={bundle.charts.cumulative_returns}
                          margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                        >
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 9 }} minTickGap={24} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              background: "#1a1d25",
                              border: "1px solid rgba(255,255,255,0.1)",
                              fontSize: 11,
                            }}
                          />
                          <Legend />
                          {bundle.charts.cumulative_returns.some((d) => d.current != null) ? (
                            <Line type="monotone" dataKey="current" name="Current" stroke={C.primary} dot={false} strokeWidth={2} />
                          ) : null}
                          <Line
                            type="monotone"
                            dataKey="optimized"
                            name="Optimized"
                            stroke={C.ok}
                            dot={false}
                            strokeWidth={2}
                          />
                          {bundle.charts.cumulative_returns.some((d) => d.benchmark != null) ? (
                            <Line
                              type="monotone"
                              dataKey="benchmark"
                              name="Benchmark"
                              stroke={C.secondary}
                              dot={false}
                              strokeWidth={2}
                            />
                          ) : null}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-white/45">No cumulative return series.</p>
                )}

                {bundle.charts?.drawdown?.length ? (
                  <>
                    <p className="text-xs text-white/50 mt-4">Drawdown (%)</p>
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={bundle.charts.drawdown}
                          margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                        >
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 9 }} minTickGap={24} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              background: "#1a1d25",
                              border: "1px solid rgba(255,255,255,0.1)",
                              fontSize: 11,
                            }}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="optimized"
                            name="Optimized"
                            stroke={C.ok}
                            dot={false}
                            strokeWidth={2}
                          />
                          {bundle.charts.drawdown.some((d) => d.current != null) ? (
                            <Line
                              type="monotone"
                              dataKey="current"
                              name="Current"
                              stroke={C.primary}
                              dot={false}
                              strokeWidth={2}
                            />
                          ) : null}
                          {bundle.charts.drawdown.some((d) => d.benchmark != null) ? (
                            <Line
                              type="monotone"
                              dataKey="benchmark"
                              name="Benchmark"
                              stroke={C.secondary}
                              dot={false}
                              strokeWidth={2}
                            />
                          ) : null}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : null}
              </div>

              {bundle.efficient_frontier && frontierCurve.length > 0 ? (
                <div className="panel p-6 space-y-3">
                  <h3 className="text-lg text-white">Efficient Frontier</h3>
                  <div className="h-[520px] w-full rounded-xl border border-white/10 bg-white/[0.02] p-3">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={frontierCurve}
                        margin={{ top: 16, right: 18, bottom: 16, left: 10 }}
                      >
                        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                        <XAxis
                          type="number"
                          dataKey="x"
                          name="Vol %"
                          domain={frontierDomain?.x ?? ["dataMin", "dataMax"]}
                          tick={{ fill: C.text, fontSize: 10 }}
                          tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                          label={{ value: "Volatility (annualized, %)", fill: C.text, fontSize: 11, position: "bottom" }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          domain={frontierDomain?.y ?? ["dataMin", "dataMax"]}
                          tick={{ fill: C.text, fontSize: 10 }}
                          tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                          label={{
                            value: "Expected return (annualized, %)",
                            angle: -90,
                            position: "insideLeft",
                            fill: C.text,
                            fontSize: 11,
                          }}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={{
                            background: "#1a1d25",
                            border: "1px solid rgba(255,255,255,0.1)",
                            fontSize: 11,
                          }}
                          formatter={(value) => {
                            if (value == null) return "";
                            const n = typeof value === "number" ? value : Number(value);
                            return Number.isFinite(n) ? `${n.toFixed(2)}%` : "";
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="y"
                          stroke={C.primary}
                          strokeWidth={2.5}
                          dot={false}
                          name="Frontier"
                          isAnimationActive={false}
                        />
                        <Scatter data={visibleFrontierMarkers} dataKey="y" name="Key portfolios">
                          {visibleFrontierMarkers.map((_, i) => (
                            <Cell key={i} fill={visibleFrontierMarkers[i].fill} />
                          ))}
                        </Scatter>
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <ul className="text-xs text-white/50 space-y-1">
                    {frontierMarkers.map((mk) => (
                      <li key={mk.name}>
                        <span className="inline-block h-2 w-2 rounded-full mr-2 align-middle" style={{ background: mk.fill }} />
                        {mk.name}: vol {mk.x.toFixed(2)}%, ret {mk.y.toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                  {outsideFrontierMarkers.length ? (
                    <p className="text-xs text-amber-200/90">
                      Note: {outsideFrontierMarkers.map((m) => m.name).join(", ")} point(s) are outside
                      frontier scale and shown in legend only to keep frontier readable.
                    </p>
                  ) : null}
                </div>
              ) : null}

              {bundle.correlation?.matrix?.length ? (
                <div className="panel p-6 space-y-3">
                  <h3 className="text-lg text-white">Correlation</h3>
                  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                    <div className="mb-3 flex items-center gap-3 text-xs text-white/60">
                      <span>Low (-1)</span>
                      <div className="h-2 w-44 rounded-full bg-gradient-to-r from-blue-500 via-slate-200 to-red-500" />
                      <span>High (+1)</span>
                    </div>
                    <div className="overflow-x-auto">
                      <div className="min-w-full">
                        <table className="w-full table-fixed border-separate border-spacing-1 text-xs">
                      <thead>
                        <tr>
                          <th className="w-12 p-1" />
                          {bundle.correlation.tickers.map((t) => (
                            <th key={t} className="p-1.5 text-white/70 text-center">
                              {t}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {bundle.correlation.matrix.map((row, i) => (
                          <tr key={bundle.correlation!.tickers[i]}>
                            <td className="p-1.5 text-white/70 font-medium text-center">
                              {bundle.correlation!.tickers[i]}
                            </td>
                            {row.map((c, j) => (
                              <td
                                key={j}
                                className="h-9 rounded text-center font-semibold"
                                style={{ backgroundColor: corrColor(c), color: corrTextColor(c) }}
                                title={`${bundle.correlation!.tickers[i]} vs ${bundle.correlation!.tickers[j]}: ${c.toFixed(2)}`}
                              >
                                {c.toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                      </div>
                  </div>
                  </div>
                  <InterpretBox text={bundle.correlation.interpretation ?? ""} />
                </div>
              ) : null}

              {bundle.sensitivity?.results?.length ? (
                <div className="panel p-6 space-y-3">
                  <h3 className="text-lg text-white">
                    Sensitivity ({bundle.sensitivity.analysis_type})
                  </h3>
                  <div className="overflow-x-auto max-h-80">
                    <table className="w-full text-xs text-white/80">
                      <thead>
                        <tr className="border-b border-white/10">
                          {Object.keys(bundle.sensitivity.results[0]).map((k) => (
                            <th key={k} className="py-2 px-2 text-left text-white/50">
                              {k}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {bundle.sensitivity.results.map((row, i) => (
                          <tr key={i} className="border-b border-white/5">
                            {Object.keys(bundle.sensitivity!.results![0]).map((k) => {
                              const v = row[k];
                              return (
                                <td key={k} className="py-1 px-2">
                                  {typeof v === "number" ? v.toFixed(4) : String(v ?? "")}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <InterpretBox text={bundle.interpretation_sensitivity ?? ""} />
                </div>
              ) : null}

              <details className="panel p-6">
                <summary className="cursor-pointer text-sm text-white/70">Raw optimization payload</summary>
                <pre className="mt-3 overflow-auto text-xs text-white/50 max-h-64">
                  {JSON.stringify(bundle.optimization, null, 2)}
                </pre>
              </details>
            </>
          )}
        </section>
      ) : null}
    </div>
  );
}
