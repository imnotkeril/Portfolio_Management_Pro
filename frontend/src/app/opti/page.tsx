"use client";

import { FormEvent, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { percentDecimal } from "@/lib/format";
import {
  clampStartDate,
  defaultAnalysisStart,
  todayIso,
  useLedgerAnalysisDates,
} from "@/lib/portfolio-ledger-bounds";
import type { Portfolio } from "@/lib/types";
import {
  AVAILABLE_OPTIMIZATION_METHODS,
  BENCHMARK_CHART_PRESETS,
  getAvailableObjectives,
  getDefaultObjective,
  getMethodPreset,
  METHOD_DESCRIPTIONS,
  METHOD_NAMES,
  METHODS_NEEDING_BENCHMARK,
  validateWeightConstraints,
} from "@/app/optimization/config";
import {
  buildOptiWeightConstraints,
  markowitzCapsFeasibilityError,
  NOTEBOOK_BENCHMARK_CHART_TICKER,
} from "@/app/opti/config";
import {
  buildMethodParamsFromTuning,
  COVARIANCE_METHOD_OPTIONS,
  getInitialOptiMethodTuning,
  HRP_LINKAGE_OPTIONS,
  methodSupportsDiversificationLambda,
  methodUsesCovarianceParams,
  type OptiMethodTuning,
} from "@/app/opti/method-params";
import {
  Area,
  AreaChart,
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

const C = {
  grid: "rgba(255,255,255,0.06)",
  text: "rgba(255,255,255,0.5)",
  primary: "#bf9ffb",
  secondary: "#7dc4e4",
  ok: "#74f174",
  danger: "#faa1a4",
  frontier: "#c8b6ff",
  drawdownCurrent: "#f59e0b",
  drawdownOptimized: "#ef4444",
  drawdownBenchmark: "#fb7185",
};

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
      /* plain text */
    }
    return raw;
  }
  return "Request failed.";
}

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
      <div className="mb-1 text-xs text-white/40">{label}</div>
      <div className="text-xl font-bold text-white">{pStr}</div>
      {bStr ? (
        <div className="mt-1 flex items-center gap-1.5">
          <span className={`h-2 w-2 rounded-full ${dotColor}`} />
          <span className="text-xs text-white/40">Current: {bStr}</span>
        </div>
      ) : null}
    </div>
  );
}

function InsightCallout({
  tone,
  title,
  children,
}: {
  tone: "good" | "neutral" | "warn";
  title: string;
  children: string;
}) {
  const styles =
    tone === "good"
      ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-100"
      : tone === "warn"
        ? "border-amber-400/30 bg-amber-500/10 text-amber-100"
        : "border-white/15 bg-white/[0.04] text-white/85";
  return (
    <div className={`rounded-lg border p-3 text-sm ${styles}`}>
      <div className="mb-1 text-xs uppercase tracking-wide opacity-80">{title}</div>
      <div className="whitespace-pre-wrap">{children}</div>
    </div>
  );
}

function corrColor(c: number): string {
  const t = Math.max(0, Math.min(1, (c + 1) / 2));
  // Low correlation (-1) -> #74f174, neutral (0) -> light gray, high (+1) -> #ef5350.
  const low = { r: 116, g: 241, b: 116 }; // #74f174
  const mid = { r: 232, g: 232, b: 232 }; // neutral (no blue tint)
  const high = { r: 239, g: 83, b: 80 }; // #ef5350
  const lerp = (a: number, b: number, w: number) => Math.round(a + (b - a) * w);
  if (t <= 0.5) {
    const w = t / 0.5;
    return `rgb(${lerp(low.r, mid.r, w)},${lerp(low.g, mid.g, w)},${lerp(low.b, mid.b, w)})`;
  }
  const w = (t - 0.5) / 0.5;
  return `rgb(${lerp(mid.r, high.r, w)},${lerp(mid.g, high.g, w)},${lerp(mid.b, high.b, w)})`;
}

function corrTextColor(c: number): string {
  return Math.abs(c) >= 0.55 ? "rgba(255,255,255,0.95)" : "rgba(17,24,39,0.95)";
}

type CumRow = {
  x: string;
  current?: number | null;
  optimized?: number | null;
  benchmark?: number | null;
};

type OptiBundle = {
  success: boolean;
  message?: string;
  optimization?: Record<string, unknown>;
  notebook_split?: boolean;
  full_data_period?: { start: string; end: string };
  optimization_period?: { start: string; end: string };
  validation_period?: { start: string; end: string };
  test_period?: { start: string; end: string };
  notebook_train_fraction?: number;
  split_mode?: string;
  split_fractions?: { train: number; validation: number; test: number };
  charts?: { cumulative_returns: CumRow[]; drawdown?: CumRow[] };
  metrics?: {
    current: Record<string, number>;
    optimized: Record<string, number>;
  };
  validation_metrics?: {
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
  }>;
  interpretation_trades?: string;
  efficient_frontier?: {
    volatilities_pct: number[];
    returns_pct: number[];
    tangency_portfolio?: { volatility?: number; expected_return?: number } | null;
    min_variance_portfolio?: { volatility?: number; expected_return?: number } | null;
    optimized_point?: { volatility_pct?: number; return_pct?: number; sharpe?: number };
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

function LabeledRange({
  label,
  valuePct,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  valuePct: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className={disabled ? "opacity-45" : ""}>
      <div className="flex justify-between text-xs text-white/55">
        <span>{label}</span>
        <span className="tabular-nums text-white/80">{valuePct.toFixed(1)}%</span>
      </div>
      <input
        type="range"
        className="mt-1 h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#bf9ffb] disabled:cursor-not-allowed"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        value={valuePct}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function OptiNotebookPageContent() {
  const searchParams = useSearchParams();
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [portfolioId, setPortfolioId] = useState("");
  const [method, setMethod] = useState<string>(AVAILABLE_OPTIMIZATION_METHODS[0]);
  const [objective, setObjective] = useState<string | null>(
    getDefaultObjective(AVAILABLE_OPTIMIZATION_METHODS[0]),
  );
  const [methodTuning, setMethodTuning] = useState<OptiMethodTuning>(() =>
    getInitialOptiMethodTuning(AVAILABLE_OPTIMIZATION_METHODS[0]),
  );
  const [startDate, setStartDate] = useState(defaultAnalysisStart);
  const [endDate, setEndDate] = useState(todayIso);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [benchmarkOpt, setBenchmarkOpt] = useState("AOR");
  const [benchmarkChart, setBenchmarkChart] = useState<string>(NOTEBOOK_BENCHMARK_CHART_TICKER);
  const [longOnly, setLongOnly] = useState(true);
  const [maxCashPct, setMaxCashPct] = useState(10);
  const [useMinW, setUseMinW] = useState(true);
  const [minWPct, setMinWPct] = useState(1);
  const [useMaxW, setUseMaxW] = useState(true);
  const [maxWPct, setMaxWPct] = useState(30);
  const [useMinReturn, setUseMinReturn] = useState(false);
  const [minReturnPct, setMinReturnPct] = useState(3);
  const [useNotebookPerTickerCaps, setUseNotebookPerTickerCaps] = useState(false);
  const [useDivLambda, setUseDivLambda] = useState(false);
  const [divLambda, setDivLambda] = useState(0.5);
  const [includeFrontier, setIncludeFrontier] = useState(true);
  const [includeSensitivity, setIncludeSensitivity] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<OptiBundle | null>(null);

  useEffect(() => {
    const idFromUrl = searchParams.get("id");
    api
      .get<Portfolio[]>("/portfolios")
      .then((list) => {
        setPortfolios(list);
        if (idFromUrl && list.some((p) => p.id === idFromUrl)) {
          setPortfolioId(idFromUrl);
        } else {
          setPortfolioId((prev) => prev || (list[0]?.id ?? ""));
        }
      })
      .catch(() => setPortfolios([]));
  }, [searchParams]);

  const selectedPortfolio = useMemo(
    () => portfolios.find((p) => p.id === portfolioId) ?? null,
    [portfolios, portfolioId],
  );
  const ledgerFirstTx = useLedgerAnalysisDates(
    portfolioId,
    selectedPortfolio,
    setStartDate,
    setEndDate,
  );

  const numRiskAssets = useMemo(() => {
    if (!selectedPortfolio) return 0;
    return selectedPortfolio.positions.filter((x) => x.ticker !== "CASH").length;
  }, [selectedPortfolio]);

  const maxAllowedMinPct = useMemo(() => {
    if (numRiskAssets <= 0) return 50;
    return Math.min(50, (1 / numRiskAssets) * 100);
  }, [numRiskAssets]);

  const constraintWarnings = useMemo(() => {
    const mw = useMinW ? minWPct / 100 : undefined;
    const xw = useMaxW ? maxWPct / 100 : undefined;
    return validateWeightConstraints(mw, xw, Math.max(numRiskAssets, 1));
  }, [useMinW, minWPct, useMaxW, maxWPct, numRiskAssets]);

  useEffect(() => {
    if (!useMinW) return;
    if (minWPct > maxAllowedMinPct) setMinWPct(maxAllowedMinPct);
  }, [useMinW, minWPct, maxAllowedMinPct]);

  useEffect(() => {
    setMethodTuning(getInitialOptiMethodTuning(method));
    const preset = getMethodPreset(method);
    setLongOnly(preset.longOnly);
    setMaxCashPct(preset.maxCashPct);
    setUseMinW(preset.useMinW);
    setMinWPct(Math.min(preset.minWPct, maxAllowedMinPct));
    setUseMaxW(preset.useMaxW);
    setMaxWPct(preset.maxWPct);
    setUseMinReturn(preset.useMinReturn);
    setMinReturnPct(preset.minReturnPct);
    setUseNotebookPerTickerCaps(false);
    setUseDivLambda(preset.useDivLambda);
    setDivLambda(preset.divLambda);
    setIncludeFrontier(preset.includeFrontier);
    setIncludeSensitivity(preset.includeSensitivity);
    const avail = getAvailableObjectives(method);
    const def = getDefaultObjective(method);
    if (avail.length) {
      const recommended =
        (preset.objective && avail.includes(preset.objective) && preset.objective) ||
        (def && avail.includes(def) && def) ||
        avail[0];
      setObjective((prev) => (prev && avail.includes(prev) ? prev : recommended));
    } else {
      setObjective(null);
    }
  }, [method, maxAllowedMinPct]);

  const buildPayload = useCallback(() => {
    const sel = portfolios.find((p) => p.id === portfolioId);
    const riskTickers = sel?.positions.map((x) => x.ticker).filter((t) => t !== "CASH") ?? [];
    const constraints = buildOptiWeightConstraints(riskTickers, {
      longOnly,
      maxCashPct,
      useMinW,
      minWPct,
      useMaxW,
      maxWPct,
      useMinReturn,
      minReturnPct,
      useNotebookPerTickerCaps,
      useDiversificationLambda: useDivLambda,
      diversificationLambda: divLambda,
      methodSupportsDiversificationLambda: methodSupportsDiversificationLambda(method),
    });

    const method_params = buildMethodParamsFromTuning(method, methodTuning, objective);
    const needsBench = METHODS_NEEDING_BENCHMARK.has(method);

    return {
      portfolio_id: portfolioId,
      method,
      start_date: startDate,
      end_date: endDate,
      constraints,
      benchmark_ticker: needsBench ? benchmarkOpt.trim().toUpperCase() : null,
      method_params: Object.keys(method_params).length ? method_params : null,
      out_of_sample: false,
      training_ratio: trainFrac,
      benchmark_for_charts: benchmarkChart === "None" ? null : benchmarkChart,
      include_efficient_frontier: includeFrontier,
      frontier_n_points: Math.min(250, Math.max(20, Math.round(methodTuning.frontierNPoints))),
      include_sensitivity: includeSensitivity,
      sensitivity_analysis_type: "returns",
      notebook_split: true,
      notebook_train_fraction: trainFrac,
    };
  }, [
    portfolios,
    portfolioId,
    method,
    objective,
    methodTuning,
    longOnly,
    maxCashPct,
    useMinW,
    minWPct,
    useMaxW,
    maxWPct,
    useMinReturn,
    minReturnPct,
    useNotebookPerTickerCaps,
    useDivLambda,
    divLambda,
    includeFrontier,
    includeSensitivity,
    startDate,
    endDate,
    trainFrac,
    benchmarkOpt,
    benchmarkChart,
  ]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (!portfolioId) {
      setError("Select a portfolio.");
      return;
    }
    if (METHODS_NEEDING_BENCHMARK.has(method) && !benchmarkOpt.trim()) {
      setError("This method requires an optimization benchmark ticker.");
      return;
    }
    if (new Date(startDate) >= new Date(endDate)) {
      setError("Start date must be before end date.");
      return;
    }
    const sel = portfolios.find((p) => p.id === portfolioId);
    const nRisk = sel?.positions.filter((x) => x.ticker !== "CASH").length ?? 0;
    if (!useNotebookPerTickerCaps && useMinW && nRisk > 0) {
      const minW = minWPct / 100;
      if (minW * nRisk > 1 + 1e-6) {
        setError(
          `Minimum weight is infeasible: lower-bound sum ${(minW * nRisk * 100).toFixed(0)}% > 100%. Reduce minimum weight or disable it.`,
        );
        return;
      }
    }
    if (useNotebookPerTickerCaps) {
      const tickers = sel?.positions.map((x) => x.ticker) ?? [];
      const capErr = markowitzCapsFeasibilityError(tickers);
      if (capErr) {
        setError(capErr);
        return;
      }
    }
    if (constraintWarnings.length > 0) {
      setError(constraintWarnings[0] ?? "Review constraint settings.");
      return;
    }
    setLoading(true);
    try {
      const data = await api.post<OptiBundle>("/optimization/full", buildPayload());
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

    const byVolBucket = new Map<number, { x: number; y: number }>();
    for (const p of pts) {
      const key = Math.round(p.x * 100) / 100;
      const prev = byVolBucket.get(key);
      if (!prev || p.y > prev.y) byVolBucket.set(key, p);
    }

    const deduped = [...byVolBucket.values()].sort((a, b) => a.x - b.x);

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

  const availObj = getAvailableObjectives(method);
  const chartCum = bundle?.charts?.cumulative_returns ?? [];
  const chartDd = bundle?.charts?.drawdown ?? [];
  const curM = bundle?.metrics?.current;
  const optM = bundle?.metrics?.optimized;
  const comparisonInsights = useMemo<
    Array<{ tone: "good" | "neutral" | "warn"; title: string; text: string }>
  >(() => {
    if (!curM || !optM) return [];
    const sharpeDelta = (optM.sharpe_ratio ?? 0) - (curM.sharpe_ratio ?? 0);
    const retDelta = (optM.total_return ?? 0) - (curM.total_return ?? 0);
    const volDelta = (optM.volatility ?? 0) - (curM.volatility ?? 0);
    const ddDelta = (optM.max_drawdown ?? 0) - (curM.max_drawdown ?? 0);
    const score =
      (sharpeDelta > 0 ? 1 : 0) +
      (retDelta > 0 ? 1 : 0) +
      (volDelta < 0 ? 1 : 0) +
      (ddDelta > 0 ? 1 : 0);
    const overallTone: "good" | "neutral" | "warn" = score >= 3 ? "good" : score >= 2 ? "neutral" : "warn";
    return [
      {
        tone: overallTone,
        title: "Overall",
        text:
          score >= 3
            ? "Optimization improves most risk/return dimensions on the selected test window."
            : score >= 2
              ? "Optimization is mixed: some dimensions improve, others are flat or weaker."
              : "Optimization is weak on this window. Consider different constraints or method settings.",
      },
      {
        tone: sharpeDelta > 0 ? "good" : "warn",
        title: "Risk-adjusted return",
        text: `Sharpe ${(optM.sharpe_ratio ?? 0).toFixed(3)} vs ${(curM.sharpe_ratio ?? 0).toFixed(3)}.`,
      },
      {
        tone: volDelta < 0 && ddDelta > 0 ? "good" : "neutral",
        title: "Downside profile",
        text: `Volatility ${((optM.volatility ?? 0) * 100).toFixed(2)}% vs ${((curM.volatility ?? 0) * 100).toFixed(2)}%; max drawdown ${((optM.max_drawdown ?? 0) * 100).toFixed(2)}% vs ${((curM.max_drawdown ?? 0) * 100).toFixed(2)}%.`,
      },
    ];
  }, [curM, optM]);

  const frontierInsight = useMemo(() => {
    const op = bundle?.efficient_frontier?.optimized_point;
    const cp = bundle?.efficient_frontier?.current_point;
    if (!op || !cp || op.volatility_pct == null || op.return_pct == null || cp.volatility_pct == null || cp.return_pct == null) {
      return null;
    }
    const betterReturn = op.return_pct > cp.return_pct;
    const lowerRisk = op.volatility_pct < cp.volatility_pct;
    const tone: "good" | "neutral" | "warn" = betterReturn && lowerRisk ? "good" : betterReturn || lowerRisk ? "neutral" : "warn";
    return {
      tone,
      text: `Optimized point: ${op.return_pct.toFixed(2)}% return at ${op.volatility_pct.toFixed(2)}% vol. Current point: ${cp.return_pct.toFixed(2)}% at ${cp.volatility_pct.toFixed(2)}%.`,
    };
  }, [bundle?.efficient_frontier]);

  const correlationInsight = useMemo(() => {
    const block = bundle?.correlation;
    if (!block?.matrix?.length) return null;
    const vals: number[] = [];
    for (let i = 0; i < block.matrix.length; i += 1) {
      for (let j = i + 1; j < block.matrix[i].length; j += 1) {
        vals.push(block.matrix[i][j]);
      }
    }
    if (!vals.length) return null;
    const avg = vals.reduce((s, v) => s + v, 0) / vals.length;
    const max = Math.max(...vals);
    const tone: "good" | "neutral" | "warn" = avg < 0.25 ? "good" : avg < 0.5 ? "neutral" : "warn";
    return {
      tone,
      text: `Average pairwise correlation is ${avg.toFixed(2)} (max ${max.toFixed(2)}). Lower average correlation means better diversification resilience.`,
    };
  }, [bundle?.correlation]);

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-16">
      <section className="panel p-6">
        <h1 className="text-3xl text-white">Portfolio Optimization</h1>
        <p className="mt-2 max-w-3xl text-sm text-white/70">
          Configure optimization over your selected date window. Model fitting uses the training segment,
          while comparison charts and metrics are evaluated on the validation segment. Method defaults are
          auto-applied when you switch methods, and all values remain editable.
        </p>
      </section>

      <form className="panel space-y-4 p-6" onSubmit={onSubmit}>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="label">
            Portfolio
            <select
              className="input"
              value={portfolioId}
              onChange={(e) => setPortfolioId(e.target.value)}
            >
              <option value="">—</option>
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label className="label">
            Method
            <select className="input" value={method} onChange={(e) => setMethod(e.target.value)}>
              {AVAILABLE_OPTIMIZATION_METHODS.map((m) => (
                <option key={m} value={m}>
                  {METHOD_NAMES[m] ?? m}
                </option>
              ))}
            </select>
          </label>
          {METHOD_DESCRIPTIONS[method] ? (
            <div className="md:col-span-2 rounded-lg border border-white/10 bg-white/[0.02] p-4">
              <div className="text-xs font-semibold uppercase tracking-wide text-white/45">
                Method Description
              </div>
              <div className="mt-1 text-sm text-white/90">{METHOD_DESCRIPTIONS[method].short}</div>
              <p className="mt-2 text-xs leading-relaxed text-white/60">
                {METHOD_DESCRIPTIONS[method].long}
              </p>
            </div>
          ) : null}
          <div className="md:col-span-2 space-y-4 rounded-lg border border-white/10 bg-white/[0.03] p-4">
            <div>
              <h3 className="text-sm font-medium text-white">Constraints</h3>
              <p className="mt-1 text-xs text-white/45">
                Method defaults are auto-applied when you switch methods. You can then adjust any constraint;
                optimization runs with the current values.
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="label flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                  checked={longOnly}
                  onChange={(e) => setLongOnly(e.target.checked)}
                />
                <span>Long only (no shorting)</span>
              </label>
              <LabeledRange
                label="Max cash weight"
                valuePct={maxCashPct}
                min={0}
                max={100}
                step={1}
                onChange={setMaxCashPct}
              />
              <div className="space-y-2">
                <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                    checked={useMinW}
                    onChange={(e) => setUseMinW(e.target.checked)}
                  />
                  Minimum weight per asset
                </label>
                <LabeledRange
                  label="Min weight"
                  valuePct={minWPct}
                  min={0}
                  max={maxAllowedMinPct}
                  step={0.1}
                  disabled={!useMinW}
                  onChange={setMinWPct}
                />
              </div>
              <div className="space-y-2">
                <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                    checked={useMaxW}
                    onChange={(e) => setUseMaxW(e.target.checked)}
                  />
                  Maximum weight per asset
                </label>
                <LabeledRange
                  label="Max weight"
                  valuePct={maxWPct}
                  min={1}
                  max={100}
                  step={0.5}
                  disabled={!useMaxW}
                  onChange={setMaxWPct}
                />
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80 sm:col-span-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                  checked={useNotebookPerTickerCaps}
                  onChange={(e) => setUseNotebookPerTickerCaps(e.target.checked)}
                />
                <span>Use asset-level cap policy (ticker-specific maximum weights)</span>
              </label>
              <div className="space-y-2 sm:col-span-2">
                <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                    checked={useMinReturn}
                    onChange={(e) => setUseMinReturn(e.target.checked)}
                  />
                  Minimum expected return (annualized), %
                </label>
                <input
                  className="input max-w-[12rem]"
                  type="number"
                  step={0.1}
                  disabled={!useMinReturn}
                  value={minReturnPct}
                  onChange={(e) => setMinReturnPct(Number(e.target.value))}
                />
              </div>
              {methodSupportsDiversificationLambda(method) ? (
                <div className="space-y-2 sm:col-span-2">
                  <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                      checked={useDivLambda}
                      onChange={(e) => setUseDivLambda(e.target.checked)}
                    />
                    Diversification penalty (lambda)
                  </label>
                  <input
                    className="input max-w-[10rem]"
                    type="number"
                    min={0}
                    max={5}
                    step={0.05}
                    disabled={!useDivLambda}
                    value={divLambda}
                    onChange={(e) => setDivLambda(Number(e.target.value))}
                  />
                </div>
              ) : null}
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                checked={includeFrontier}
                onChange={(e) => setIncludeFrontier(e.target.checked)}
              />
              Include efficient frontier (slower)
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-white/80">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-white/20 accent-[#bf9ffb]"
                checked={includeSensitivity}
                onChange={(e) => setIncludeSensitivity(e.target.checked)}
              />
              Include sensitivity analysis (extra runs, slower)
            </label>
          </div>
          {availObj.length > 0 ? (
            <label className="label">
              Objective
              <select
                className="input"
                value={objective ?? ""}
                onChange={(e) => setObjective(e.target.value || null)}
              >
                {availObj.map((o) => (
                  <option key={o} value={o}>
                    {o.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <div className="md:col-span-2 space-y-3 rounded-lg border border-white/10 bg-white/[0.03] p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-medium text-white">Method Parameters</h3>
              <button
                type="button"
                className="rounded-md border border-white/15 px-2.5 py-1 text-xs text-white/80 transition hover:bg-white/10"
                onClick={() => setMethodTuning(getInitialOptiMethodTuning(method))}
              >
                Reset To Method Default
              </button>
            </div>
            <p className="text-xs text-white/45">
              Parameters are initialized from method defaults and remain fully editable before running.
            </p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {methodUsesCovarianceParams(method) ? (
                <>
                  <label className="label">
                    Covariance estimator
                    <select
                      className="input"
                      value={methodTuning.covarianceMethod}
                      onChange={(e) =>
                        setMethodTuning((t) => ({
                          ...t,
                          covarianceMethod: e.target.value as OptiMethodTuning["covarianceMethod"],
                        }))
                      }
                    >
                      {COVARIANCE_METHOD_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="label">
                    Shrinkage α (if shrink)
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={methodTuning.shrinkageAlpha}
                      onChange={(e) =>
                        setMethodTuning((t) => ({ ...t, shrinkageAlpha: Number(e.target.value) }))
                      }
                    />
                  </label>
                </>
              ) : null}
              {method === "black_litterman" ? (
                <label className="label">
                  Black–Litterman τ
                  <input
                    className="input"
                    type="number"
                    min={0.01}
                    max={0.5}
                    step={0.01}
                    value={methodTuning.blackLittermanTau}
                    onChange={(e) =>
                      setMethodTuning((t) => ({ ...t, blackLittermanTau: Number(e.target.value) }))
                    }
                  />
                </label>
              ) : null}
              {method === "hrp" ? (
                <label className="label">
                  HRP linkage
                  <select
                    className="input"
                    value={methodTuning.hrpLinkageMethod}
                    onChange={(e) =>
                      setMethodTuning((t) => ({
                        ...t,
                        hrpLinkageMethod: e.target.value as OptiMethodTuning["hrpLinkageMethod"],
                      }))
                    }
                  >
                    {HRP_LINKAGE_OPTIONS.map((l) => (
                      <option key={l} value={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {method === "cvar_optimization" || method === "mean_cvar" ? (
                <label className="label">
                  CVaR confidence
                  <select
                    className="input"
                    value={String(methodTuning.confidenceLevel)}
                    onChange={(e) =>
                      setMethodTuning((t) => ({ ...t, confidenceLevel: Number(e.target.value) }))
                    }
                  >
                    <option value="0.9">0.90</option>
                    <option value="0.95">0.95</option>
                    <option value="0.99">0.99</option>
                  </select>
                </label>
              ) : null}
              {method === "mean_cvar" ? (
                <>
                  <label className="label">
                    Mean-CVaR mode
                    <select
                      className="input"
                      value={methodTuning.meanCvarOptimizationMode}
                      onChange={(e) =>
                        setMethodTuning((t) => ({
                          ...t,
                          meanCvarOptimizationMode: e.target.value as "cvar_cap" | "penalty",
                        }))
                      }
                    >
                      <option value="cvar_cap">cvar_cap (cap on CVaR)</option>
                      <option value="penalty">penalty (legacy)</option>
                    </select>
                  </label>
                  <label className="label">
                    CVaR cap relax
                    <input
                      className="input"
                      type="number"
                      min={1.01}
                      max={2}
                      step={0.01}
                      value={methodTuning.meanCvarCapRelax}
                      onChange={(e) =>
                        setMethodTuning((t) => ({ ...t, meanCvarCapRelax: Number(e.target.value) }))
                      }
                    />
                  </label>
                  <label className="label">
                    Risk aversion (penalty mode)
                    <input
                      className="input"
                      type="number"
                      min={0.01}
                      max={20}
                      step={0.05}
                      value={methodTuning.meanCvarRiskAversion}
                      onChange={(e) =>
                        setMethodTuning((t) => ({ ...t, meanCvarRiskAversion: Number(e.target.value) }))
                      }
                    />
                  </label>
                </>
              ) : null}
              {method === "robust" ? (
                <>
                  <label className="label">
                    Uncertainty radius (returns)
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={methodTuning.uncertaintyRadiusReturns}
                      onChange={(e) =>
                        setMethodTuning((t) => ({
                          ...t,
                          uncertaintyRadiusReturns: Number(e.target.value),
                        }))
                      }
                    />
                  </label>
                  <label className="label">
                    Uncertainty radius (covariance)
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={methodTuning.uncertaintyRadiusCov}
                      onChange={(e) =>
                        setMethodTuning((t) => ({ ...t, uncertaintyRadiusCov: Number(e.target.value) }))
                      }
                    />
                  </label>
                  <label className="label flex flex-col gap-2 sm:col-span-2">
                    <span className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={methodTuning.useRobustAdvanced}
                        onChange={(e) =>
                          setMethodTuning((t) => ({ ...t, useRobustAdvanced: e.target.checked }))
                        }
                      />
                      <span>Advanced robust_kappa / robust_lambda (optional)</span>
                    </span>
                    {methodTuning.useRobustAdvanced ? (
                      <span className="grid gap-2 sm:grid-cols-2">
                        <input
                          className="input"
                          type="text"
                          inputMode="decimal"
                          placeholder="robust_kappa"
                          value={methodTuning.robustKappa}
                          onChange={(e) =>
                            setMethodTuning((t) => ({ ...t, robustKappa: e.target.value }))
                          }
                        />
                        <input
                          className="input"
                          type="text"
                          inputMode="decimal"
                          placeholder="robust_lambda"
                          value={methodTuning.robustLambda}
                          onChange={(e) =>
                            setMethodTuning((t) => ({ ...t, robustLambda: e.target.value }))
                          }
                        />
                      </span>
                    ) : null}
                  </label>
                </>
              ) : null}
              {method === "mean_variance" ? (
                <label className="label flex items-center gap-2 sm:col-span-2">
                  <input
                    type="checkbox"
                    checked={methodTuning.targetReturnAsFloor}
                    onChange={(e) =>
                      setMethodTuning((t) => ({ ...t, targetReturnAsFloor: e.target.checked }))
                    }
                  />
                  <span className="text-xs text-white/70">
                    target_return_as_floor (for target-return min-vol and frontier sweeps)
                  </span>
                </label>
              ) : null}
              <label className="label">
                Frontier points
                <input
                  className="input"
                  type="number"
                  min={20}
                  max={250}
                  step={10}
                  value={methodTuning.frontierNPoints}
                  onChange={(e) =>
                    setMethodTuning((t) => ({
                      ...t,
                      frontierNPoints: Number(e.target.value),
                    }))
                  }
                />
              </label>
            </div>
          </div>
          <label className="label">
            Start (full window)
            <input
              className="input"
              type="date"
              value={startDate}
              min={ledgerFirstTx ?? undefined}
              max={endDate}
              onChange={(e) =>
                setStartDate(clampStartDate(e.target.value, ledgerFirstTx ?? undefined))
              }
            />
          </label>
          <label className="label">
            End (full window)
            <input
              className="input"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>
          <label className="label">
            Train fraction (row split)
            <input
              className="input"
              type="number"
              min={0.5}
              max={0.95}
              step={0.05}
              value={trainFrac}
              onChange={(e) => setTrainFrac(Number(e.target.value))}
            />
          </label>
          <label className="label">
            Benchmark (chart)
            <select
              className="input"
              value={benchmarkChart}
              onChange={(e) => setBenchmarkChart(e.target.value)}
            >
              {BENCHMARK_CHART_PRESETS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </label>
          {METHODS_NEEDING_BENCHMARK.has(method) ? (
            <label className="label">
              Benchmark (optimization)
              <input
                className="input"
                value={benchmarkOpt}
                onChange={(e) => setBenchmarkOpt(e.target.value)}
                placeholder="AOR"
              />
            </label>
          ) : null}
        </div>
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? "Running…" : "Run Optimization"}
        </button>
        {error ? <p className="text-sm text-red-300">{error}</p> : null}
      </form>

      {bundle ? (
        <>
          {!bundle.success ? (
            <div className="panel p-6">
              <p className="text-amber-200">{bundle.message || "Optimization failed."}</p>
            </div>
          ) : null}

          {bundle.warnings?.length ? (
            <div className="space-y-2">
              {bundle.warnings.map((w) => (
                <div
                  key={w}
                  className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-100"
                >
                  {w}
                </div>
              ))}
            </div>
          ) : null}

          {bundle.success ? (
            <>
              <div className="panel p-6">
                <h2 className="text-xl font-medium text-white">Periods</h2>
                {bundle.full_data_period ? (
                  <p className="mt-2 text-xs text-white/45">
                    Full data: {bundle.full_data_period.start} → {bundle.full_data_period.end}
                  </p>
                ) : null}
                {bundle.optimization_period ? (
                  <p className="text-xs text-white/45">
                    Training (fit weights): {bundle.optimization_period.start} →{" "}
                    {bundle.optimization_period.end}
                  </p>
                ) : null}
                {bundle.validation_period ? (
                  <p className="text-xs text-violet-200/90">
                    Validation (selection window): {bundle.validation_period.start} →{" "}
                    {bundle.validation_period.end}
                  </p>
                ) : null}
                {bundle.test_period ? (
                  <p className="text-xs text-emerald-200/90">
                    Test (charts + final comparison): {bundle.test_period.start} →{" "}
                    {bundle.test_period.end}
                  </p>
                ) : null}
                {bundle.split_fractions ? (
                  <p className="text-xs text-white/45">
                    Split ratios: train {(bundle.split_fractions.train * 100).toFixed(1)}% /
                    validation {(bundle.split_fractions.validation * 100).toFixed(1)}% / test{" "}
                    {(bundle.split_fractions.test * 100).toFixed(1)}%
                  </p>
                ) : null}
                {bundle.efficient_frontier?.fallback_validation_period ? (
                  <p className="mt-2 text-xs text-amber-200/90">
                    Efficient frontier fell back to the wider validation period (insufficient training data).
                  </p>
                ) : null}
              </div>

              <div className="panel space-y-3 p-6">
                <h3 className="text-lg text-white">Optimized vs current (test window)</h3>
                <p className="text-xs text-white/45">
                  Cards show optimized portfolio vs your current allocation; both series are evaluated on the
                  test dates only.
                </p>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <CmpMetricCard
                    label="Total Return"
                    portfolioValue={optM?.total_return}
                    benchmarkValue={curM?.total_return}
                    format="percent"
                    higherIsBetter
                  />
                  <CmpMetricCard
                    label="Sharpe Ratio"
                    portfolioValue={optM?.sharpe_ratio}
                    benchmarkValue={curM?.sharpe_ratio}
                    format="ratio"
                    higherIsBetter
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
                    higherIsBetter
                  />
                </div>
                {comparisonInsights.length ? (
                  <div className="grid gap-2 md:grid-cols-3">
                    {comparisonInsights.map((item) => (
                      <InsightCallout key={item.title} tone={item.tone} title={item.title}>
                        {item.text}
                      </InsightCallout>
                    ))}
                  </div>
                ) : null}
                <InterpretBox text={bundle.interpretation_comparison ?? ""} />
              </div>

              <div className="panel space-y-3 p-6">
                <h3 className="text-lg text-white">Allocation</h3>
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
                          <td className="py-1.5 text-right">{(row.difference * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {bundle.allocation?.length ? (
                  <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={bundle.allocation}
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
                        <Bar dataKey="current_weight" name="Current" fill={C.primary} radius={[4, 4, 0, 0]} />
                        <Bar dataKey="optimal_weight" name="Optimal" fill={C.secondary} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : null}
                <InterpretBox text={bundle.interpretation_allocation ?? ""} />
              </div>

              <div className="panel space-y-3 p-6">
                <h3 className="text-lg text-white">Suggested trades</h3>
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
                        {bundle.trades.map((t, i) => (
                          <tr key={`${t.ticker}-${t.action}-${i}`} className="border-b border-white/5">
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

              <div className="panel space-y-4 p-6">
                <h3 className="text-lg text-white">Performance (test window)</h3>
                {chartCum.length ? (
                  <>
                    <p className="text-xs text-white/50">Cumulative return (%)</p>
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartCum} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 9 }} minTickGap={24} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                          <Tooltip
                            contentStyle={{
                              background: "#1a1d25",
                              border: "1px solid rgba(255,255,255,0.1)",
                              fontSize: 11,
                            }}
                          />
                          <Legend />
                          {chartCum.some((d) => d.current != null) ? (
                            <Line
                              type="monotone"
                              dataKey="current"
                              name="Current"
                              stroke={C.secondary}
                              dot={false}
                              strokeWidth={2}
                            />
                          ) : null}
                          <Line
                            type="monotone"
                            dataKey="optimized"
                            name="Optimized"
                            stroke={C.primary}
                            dot={false}
                            strokeWidth={2}
                          />
                          {chartCum.some((d) => d.benchmark != null) ? (
                            <Line
                              type="monotone"
                              dataKey="benchmark"
                              name="Benchmark"
                              stroke={C.ok}
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

                {chartDd.length ? (
                  <>
                    <p className="mt-4 text-xs text-white/50">Drawdown (%)</p>
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartDd} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                          <defs>
                            <linearGradient id="ddOpt" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={C.drawdownOptimized} stopOpacity={0.45} />
                              <stop offset="100%" stopColor={C.drawdownOptimized} stopOpacity={0.08} />
                            </linearGradient>
                            <linearGradient id="ddCur" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={C.drawdownCurrent} stopOpacity={0.4} />
                              <stop offset="100%" stopColor={C.drawdownCurrent} stopOpacity={0.08} />
                            </linearGradient>
                            <linearGradient id="ddBmk" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={C.drawdownBenchmark} stopOpacity={0.34} />
                              <stop offset="100%" stopColor={C.drawdownBenchmark} stopOpacity={0.06} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 9 }} minTickGap={24} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              background: "#1a1d25",
                              border: "1px solid rgba(255,255,255,0.1)",
                              fontSize: 11,
                            }}
                            labelStyle={{ color: "#d8dee9" }}
                            itemStyle={{ color: "#eef2ff" }}
                          />
                          <Legend wrapperStyle={{ color: "#d7ddf5", paddingTop: 8 }} />
                          <Area
                            type="monotone"
                            dataKey="optimized"
                            name="Optimized"
                            fill="url(#ddOpt)"
                            stroke={C.drawdownOptimized}
                            strokeWidth={2.2}
                            dot={false}
                            isAnimationActive={false}
                          />
                          {chartDd.some((d) => d.current != null) ? (
                            <Area
                              type="monotone"
                              dataKey="current"
                              name="Current"
                              fill="url(#ddCur)"
                              stroke={C.drawdownCurrent}
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                            />
                          ) : null}
                          {chartDd.some((d) => d.benchmark != null) ? (
                            <Area
                              type="monotone"
                              dataKey="benchmark"
                              name="Benchmark"
                              fill="url(#ddBmk)"
                              stroke={C.drawdownBenchmark}
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                            />
                          ) : null}
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : null}
              </div>

              {bundle.efficient_frontier && frontierCurve.length > 0 ? (
                <div className="panel space-y-3 p-6">
                  <h3 className="text-lg text-white">Efficient frontier (training window)</h3>
                  <p className="text-xs text-white/45">
                    Curve and key points are estimated on the same dates used to fit optimal weights.
                  </p>
                  <div className="h-[480px] w-full rounded-xl border border-white/10 bg-white/[0.02] p-3">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={frontierCurve}
                        margin={{ top: 44, right: 24, bottom: 34, left: 12 }}
                      >
                        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                        <XAxis
                          type="number"
                          dataKey="x"
                          name="Vol %"
                          domain={frontierDomain?.x ?? ["dataMin", "dataMax"]}
                          tick={{ fill: C.text, fontSize: 10 }}
                          tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                          label={{
                            value: "Volatility (annualized, %)",
                            fill: C.text,
                            fontSize: 11,
                            position: "bottom",
                          }}
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
                            background: "rgba(20, 24, 34, 0.96)",
                            border: "1px solid rgba(200,210,255,0.28)",
                            color: "#e7edff",
                            fontSize: 11,
                            borderRadius: 10,
                            boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
                          }}
                          labelStyle={{ color: "#d8e1ff", fontWeight: 600 }}
                          itemStyle={{ color: "#f3f6ff" }}
                          formatter={(value) => {
                            if (value == null) return "";
                            const n = typeof value === "number" ? value : Number(value);
                            return Number.isFinite(n) ? `${n.toFixed(2)}%` : "";
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="y"
                          stroke={C.frontier}
                          strokeWidth={2.8}
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
                  {frontierInsight ? (
                    <InsightCallout tone={frontierInsight.tone} title="Frontier interpretation">
                      {frontierInsight.text}
                    </InsightCallout>
                  ) : null}
                  <ul className="space-y-1 text-xs text-white/50">
                    {frontierMarkers.map((mk) => (
                      <li key={mk.name}>
                        <span
                          className="mr-2 inline-block h-2 w-2 rounded-full align-middle"
                          style={{ background: mk.fill }}
                        />
                        {mk.name}: vol {mk.x.toFixed(2)}%, ret {mk.y.toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                  {outsideFrontierMarkers.length ? (
                    <p className="text-xs text-amber-200/90">
                      Note: {outsideFrontierMarkers.map((m) => m.name).join(", ")} outside chart scale (see
                      list above).
                    </p>
                  ) : null}
                </div>
              ) : null}

              {bundle.correlation?.matrix?.length ? (
                <div className="panel space-y-3 p-6">
                  <h3 className="text-lg text-white">Correlation (training window)</h3>
                  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                    <div className="mb-3 flex items-center gap-3 text-xs text-white/60">
                      <span>Low (-1)</span>
                      <div className="h-2 w-44 rounded-full bg-gradient-to-r from-[#74f174] via-[#e8e8e8] to-[#ef5350]" />
                      <span>High (+1)</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full min-w-full table-fixed border-separate border-spacing-1 text-xs">
                        <thead>
                          <tr>
                            <th className="w-12 p-1" />
                            {bundle.correlation.tickers.map((t) => (
                              <th key={t} className="p-1.5 text-center text-white/70">
                                {t}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {bundle.correlation.matrix.map((row, i) => (
                            <tr key={bundle.correlation!.tickers[i]}>
                              <td className="p-1.5 text-center font-medium text-white/70">
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
                  {correlationInsight ? (
                    <InsightCallout tone={correlationInsight.tone} title="Diversification interpretation">
                      {correlationInsight.text}
                    </InsightCallout>
                  ) : null}
                  <InterpretBox text={bundle.correlation.interpretation ?? ""} />
                </div>
              ) : null}

              {bundle.sensitivity?.results?.length ? (
                <div className="panel space-y-3 p-6">
                  <h3 className="text-lg text-white">
                    Sensitivity ({bundle.sensitivity.analysis_type ?? "returns"})
                  </h3>
                  <div className="max-h-80 overflow-x-auto">
                    <table className="w-full text-xs text-white/80">
                      <thead>
                        <tr className="border-b border-white/10">
                          {Object.keys(bundle.sensitivity.results[0]).map((k) => (
                            <th key={k} className="px-2 py-2 text-left text-white/50">
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
                                <td key={k} className="px-2 py-1">
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
                <summary className="cursor-pointer text-sm text-white/70">
                  Raw numbers (metrics + optimization payload)
                </summary>
                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <pre className="max-h-64 overflow-auto text-xs text-white/50">
                    {JSON.stringify(bundle.metrics?.current, null, 2)}
                  </pre>
                  <pre className="max-h-64 overflow-auto text-xs text-white/50">
                    {JSON.stringify(bundle.metrics?.optimized, null, 2)}
                  </pre>
                </div>
                <pre className="mt-3 max-h-64 overflow-auto text-xs text-white/50">
                  {JSON.stringify(bundle.optimization, null, 2)}
                </pre>
              </details>
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export default function OptiNotebookPage() {
  return (
    <Suspense fallback={<p className="text-white/60">Loading…</p>}>
      <OptiNotebookPageContent />
    </Suspense>
  );
}
