"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { Portfolio } from "@/lib/types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type RiskTab = "var" | "montecarlo" | "historical" | "custom" | "chain";

type ScenarioMeta = {
  key: string;
  name: string;
  description: string;
  market_impact_pct: number;
  recovery_period_days: number | null;
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

function fmtTooltip(value: unknown) {
  if (value == null) return "";
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? String(n) : "";
}

function RiskPageContent() {
  const searchParams = useSearchParams();
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [portfolioId, setPortfolioId] = useState("");
  const [tab, setTab] = useState<RiskTab>("var");

  const defaultEnd = () => new Date().toISOString().slice(0, 10);
  const defaultStart = () => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().slice(0, 10);
  };
  const [startDate, setStartDate] = useState(defaultStart);
  const [endDate, setEndDate] = useState(defaultEnd);

  /* VaR */
  const [varConf, setVarConf] = useState(95);
  const [varHorizon, setVarHorizon] = useState(1);
  const [rollingWindow, setRollingWindow] = useState(63);
  const [varBundle, setVarBundle] = useState<Record<string, unknown> | null>(null);

  /* Monte Carlo */
  const [mcHorizon, setMcHorizon] = useState(30);
  const [mcSims, setMcSims] = useState(10000);
  const [mcInitial, setMcInitial] = useState(100_000);
  const [mcModel, setMcModel] = useState<"gbm" | "jump_diffusion">("gbm");
  const [mcPaths, setMcPaths] = useState(false);
  const [mcBundle, setMcBundle] = useState<Record<string, unknown> | null>(null);

  /* Historical */
  const [catalog, setCatalog] = useState<ScenarioMeta[]>([]);
  const [histSelected, setHistSelected] = useState<string[]>([]);
  const [histResult, setHistResult] = useState<Record<string, unknown> | null>(null);

  /* Custom */
  const [custName, setCustName] = useState("");
  const [custDesc, setCustDesc] = useState("");
  const [custMarketPct, setCustMarketPct] = useState(-20);
  const [custAssetsText, setCustAssetsText] = useState("");
  const [custResult, setCustResult] = useState<Record<string, unknown> | null>(null);

  /* Chain */
  const [chainName, setChainName] = useState("");
  const [chainDesc, setChainDesc] = useState("");
  const [chainKeys, setChainKeys] = useState<string[]>([]);
  const [chainPick, setChainPick] = useState("");
  const [chainResult, setChainResult] = useState<Record<string, unknown> | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const idFromUrl = searchParams.get("id");
    api
      .get<Portfolio[]>("/portfolios")
      .then((list) => {
        setPortfolios(list);
        if (idFromUrl && list.some((p) => p.id === idFromUrl)) {
          setPortfolioId(idFromUrl);
        } else {
          setPortfolioId((p) => p || list[0]?.id || "");
        }
      })
      .catch(() => {});
  }, [searchParams]);

  useEffect(() => {
    api
      .get<ScenarioMeta[]>("/risk/scenarios")
      .then(setCatalog)
      .catch(() => setCatalog([]));
  }, []);

  const parseAssetImpacts = useCallback((text: string): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const line of text.trim().split("\n")) {
      if (!line.includes(":")) continue;
      const [t, pct] = line.split(":");
      const ticker = t.trim().toUpperCase();
      const v = parseFloat(pct.replace("%", "").trim());
      if (ticker && !Number.isNaN(v)) out[ticker] = v / 100;
    }
    return out;
  }, []);

  const runVar = async () => {
    if (!portfolioId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Record<string, unknown>>("/risk/var/full", {
        portfolio_id: portfolioId,
        start_date: startDate,
        end_date: endDate,
        confidence_level: varConf / 100,
        time_horizon: varHorizon,
        rolling_window: rollingWindow,
        include_monte_carlo: true,
        num_simulations: 10000,
      });
      setVarBundle(data);
    } catch (e) {
      setError(String(e));
      setVarBundle(null);
    } finally {
      setLoading(false);
    }
  };

  const runMc = async () => {
    if (!portfolioId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Record<string, unknown>>("/risk/monte-carlo/full", {
        portfolio_id: portfolioId,
        start_date: startDate,
        end_date: endDate,
        time_horizon: mcHorizon,
        num_simulations: mcSims,
        initial_value: mcInitial,
        model: mcModel,
        include_sample_paths: mcPaths,
      });
      setMcBundle(data);
    } catch (e) {
      setError(String(e));
      setMcBundle(null);
    } finally {
      setLoading(false);
    }
  };

  const runHistorical = async () => {
    if (!portfolioId || !histSelected.length) {
      setError("Select at least one scenario.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Record<string, unknown>>("/risk/stress-tests/full", {
        portfolio_id: portfolioId,
        scenario_names: histSelected,
      });
      setHistResult(data);
    } catch (e) {
      setError(String(e));
      setHistResult(null);
    } finally {
      setLoading(false);
    }
  };

  const runCustom = async () => {
    if (!portfolioId || !custName.trim()) {
      setError("Enter scenario name.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Record<string, unknown>>("/risk/custom-scenario", {
        portfolio_id: portfolioId,
        name: custName.trim(),
        description: custDesc,
        market_impact_pct: custMarketPct / 100,
        asset_impacts: parseAssetImpacts(custAssetsText),
      });
      setCustResult(data);
    } catch (e) {
      setError(String(e));
      setCustResult(null);
    } finally {
      setLoading(false);
    }
  };

  const addChainKey = () => {
    if (!chainPick || chainKeys.includes(chainPick)) return;
    setChainKeys((k) => [...k, chainPick]);
    setChainPick("");
  };

  const runChain = async () => {
    if (!portfolioId || !chainName.trim() || !chainKeys.length) {
      setError("Chain name and ordered scenarios required.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Record<string, unknown>>("/risk/scenario-chain", {
        portfolio_id: portfolioId,
        name: chainName.trim(),
        description: chainDesc,
        scenario_keys: chainKeys,
      });
      setChainResult(data);
    } catch (e) {
      setError(String(e));
      setChainResult(null);
    } finally {
      setLoading(false);
    }
  };

  const keyMetrics = varBundle?.key_metrics as Record<string, number | null> | undefined;
  const comparisonTable = (varBundle?.comparison_table as Array<Record<string, unknown>>) ?? [];
  const retDist = varBundle?.return_distribution as
    | {
        histogram: Array<{ x: number; count: number }>;
        var_historical: number | null;
        cvar: number;
        confidence_level: number;
      }
    | undefined;
  const sens = (varBundle?.sensitivity_by_confidence as Array<Record<string, number>>) ?? [];
  const rolling = varBundle?.rolling_var as
    | {
        series: Array<{ x: string; y: number }>;
        stats: Record<string, number>;
        interpretation?: string;
      }
    | undefined;
  const cov = varBundle?.covariance_var as {
    portfolio_var_pct: number;
    decomposition: Array<Record<string, number>>;
    interpretation_covariance?: string;
    interpretation_decomposition?: string;
    confidence_used?: number;
  } | null;

  const mcHist =
    (mcBundle?.histogram as Array<{ x: number; count: number; norm_y?: number }>) ?? [];
  const mcPathsData = mcBundle?.sample_paths as Array<Array<{ day: number; value: number }>> | null;
  const mcStats = mcBundle?.statistics as Record<string, number> | undefined;
  const mcPct = mcBundle?.percentiles as Record<string, number> | undefined;
  const mcPctMarkers =
    (mcBundle?.percentile_markers as Array<{ label: string; x: number; side: string }>) ?? [];
  const mcVarCvarSim =
    (mcBundle?.var_cvar_simulation as Array<Record<string, number | string>>) ?? [];
  const mcHistVsMc =
    (mcBundle?.historical_vs_mc_var as Array<Record<string, number | string>>) ?? [];
  const mcExtreme =
    (mcBundle?.extreme_scenarios as Array<Record<string, number | string>>) ?? [];
  const mcPathEnvelope = mcBundle?.path_envelope as
    | {
        max: Array<{ day: number; value: number }>;
        min: Array<{ day: number; value: number }>;
        median: Array<{ day: number; value: number }>;
      }
    | null
    | undefined;

  const histRawList = (histResult?.results as unknown[]) ?? [];
  const histRecovery = histResult?.recovery as
    | { series: Array<{ scenario_name: string; points: Array<{ day: number; pct: number }> }>; interpretation?: string }
    | undefined;
  const histEnhanced = (histResult?.enhanced_comparison as Array<Record<string, unknown>>) ?? [];
  const histBreakdowns =
    (histResult?.position_breakdowns as Array<{
      scenario_name: string;
      rows: Array<{ ticker: string; weight_pct: number; impact_pct: number; kind: string }>;
      interpretation?: string;
    }>) ?? [];
  const histTimeline = (histResult?.timeline as Array<Record<string, unknown>>) ?? [];
  const histInterpScenarios = String(histResult?.interpretation_scenarios ?? "");
  const histInterpTimeline = String(histResult?.interpretation_timeline ?? "");

  const histRows = useMemo(() => {
    if (!histRawList.length) return [];
    return histRawList.map((r) => {
      const row = r as Record<string, unknown>;
      const w = row.worst_position as { ticker?: string; impact_pct?: number } | undefined;
      return {
        scenario: String(row.scenario_name ?? ""),
        impactPct: Number(row.portfolio_impact_pct ?? 0) * 100,
        impactVal: Number(row.portfolio_impact_value ?? 0),
        worst: w?.ticker ?? "—",
        worstPct: (w?.impact_pct ?? 0) * 100,
        recovery: row.recovery_time_days,
      };
    });
  }, [histRawList]);

  /** Tight Y-axis around paths + initial line (avoids 0..max with empty band). */
  const mcPathsYDomain = useMemo((): [number, number] | undefined => {
    const vals: number[] = [];
    const pushSeries = (pts: Array<{ value: number }> | undefined) => {
      if (!pts?.length) return;
      for (const p of pts) {
        const v = Number(p.value);
        if (Number.isFinite(v)) vals.push(v);
      }
    };
    pushSeries(mcPathEnvelope?.max);
    pushSeries(mcPathEnvelope?.min);
    pushSeries(mcPathEnvelope?.median);
    mcPathsData?.forEach((path) => pushSeries(path));
    if (!vals.length) return undefined;
    if (Number.isFinite(mcInitial)) vals.push(mcInitial);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const span = hi - lo || Math.max(Math.abs(hi) * 0.02, 1);
    const pad = Math.max(span * 0.04, Math.abs(hi) * 0.005, 250);
    return [lo - pad, hi + pad];
  }, [mcPathEnvelope, mcPathsData, mcInitial]);

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-16">
      <header>
        <h1 className="text-3xl font-semibold text-white">Risk Analysis</h1>
        <p className="mt-2 text-white/60">
          VaR, Monte Carlo, historical stress tests, custom shocks and scenario chains — aligned with
          the Streamlit risk page.
        </p>
      </header>

      {!portfolios.length ? (
        <div className="panel p-6 text-amber-200">No portfolios. Create one first.</div>
      ) : (
        <>
          <section className="panel p-6 space-y-4">
            <h2 className="text-lg text-white">Analysis period</h2>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="text-xs text-white/50">Start</label>
                <input
                  type="date"
                  className="input mt-1"
                  value={startDate}
                  max={defaultEnd()}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs text-white/50">End</label>
                <input
                  type="date"
                  className="input mt-1"
                  value={endDate}
                  min={startDate}
                  max={defaultEnd()}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>
            <div>
              <label className="text-xs text-white/50">Portfolio</label>
              <select
                className="input mt-1"
                value={portfolioId}
                onChange={(e) => setPortfolioId(e.target.value)}
              >
                {portfolios.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
          </section>

          <div className="flex flex-wrap gap-2">
            {(
              [
                ["var", "VaR Analysis"],
                ["montecarlo", "Monte Carlo"],
                ["historical", "Historical Scenarios"],
                ["custom", "Custom Scenario"],
                ["chain", "Scenario Chain"],
              ] as [RiskTab, string][]
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                className={`rounded-lg border px-4 py-2 text-sm transition ${
                  tab === k
                    ? "border-violet-400/70 bg-violet-500/10 text-white"
                    : "border-white/10 text-white/70 hover:border-white/20"
                }`}
                onClick={() => setTab(k)}
              >
                {label}
              </button>
            ))}
          </div>

          {error ? <div className="text-sm text-[var(--danger)]">{error}</div> : null}

          {tab === "var" && (
            <section className="panel p-6 space-y-6">
              <h2 className="text-xl text-white">Value at Risk (VaR)</h2>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-xs text-white/50">
                    Confidence level: {varConf}%
                  </label>
                  <input
                    type="range"
                    min={90}
                    max={99}
                    className="w-full"
                    value={varConf}
                    onChange={(e) => setVarConf(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-xs text-white/50">Time horizon (days)</label>
                  <input
                    type="number"
                    min={1}
                    max={30}
                    className="input mt-1"
                    value={varHorizon}
                    onChange={(e) => setVarHorizon(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-xs text-white/50">Rolling VaR window (days)</label>
                  <input
                    type="number"
                    min={30}
                    max={252}
                    step={1}
                    className="input mt-1"
                    value={rollingWindow}
                    onChange={(e) => setRollingWindow(Number(e.target.value))}
                  />
                </div>
              </div>
              <button
                type="button"
                className="btn btn-primary"
                disabled={loading}
                onClick={runVar}
              >
                {loading ? "Running…" : "Calculate VaR"}
              </button>

              {varBundle ? (
                <div className="space-y-8 border-t border-white/10 pt-6">
                  <div>
                    <h3 className="text-lg text-white mb-3">Key risk metrics</h3>
                    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                      {[
                        ["VaR (95%) hist.", keyMetrics?.var_95_historical_pct],
                        ["CVaR (95%)", keyMetrics?.cvar_95_pct],
                        ["VaR (99%) hist.", keyMetrics?.var_99_historical_pct],
                        ["CVaR (99%)", keyMetrics?.cvar_99_pct],
                      ].map(([label, v]) => (
                        <div key={String(label)} className="metric-card">
                          <div className="text-xs text-white/45">{label}</div>
                          <div className="text-xl font-semibold text-white mt-1">
                            {v == null || !Number.isFinite(Number(v)) ? "—" : `${Number(v).toFixed(2)}%`}
                          </div>
                        </div>
                      ))}
                    </div>
                    <InterpretBox text={String(varBundle.interpretation_metrics ?? "")} />
                  </div>

                  {comparisonTable.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">VaR methods comparison</h3>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={comparisonTable}
                            margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                          >
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="method" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                              formatter={(value) => fmtTooltip(value)}
                            />
                            <Bar dataKey="var_pct" name="VaR %" fill={C.danger} radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(varBundle.interpretation_methods ?? "")} />
                    </div>
                  ) : null}

                  {retDist?.histogram?.length ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Return distribution & VaR</h3>
                      <p className="text-xs text-white/45 mb-2">
                        Histogram of daily returns; lines at historical VaR and CVaR (
                        {(Number(retDist.confidence_level) * 100).toFixed(0)}% confidence).
                      </p>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={retDist.histogram}
                            margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                          >
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis
                              dataKey="x"
                              tick={{ fill: C.text, fontSize: 9 }}
                              tickFormatter={(v) => Number(v).toFixed(3)}
                            />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            <Bar dataKey="count" fill={C.primary} opacity={0.75} />
                            {retDist.var_historical != null ? (
                              <ReferenceLine
                                x={retDist.var_historical}
                                stroke={C.danger}
                                strokeDasharray="4 4"
                                label={{ value: "VaR", fill: C.danger, fontSize: 10 }}
                              />
                            ) : null}
                            <ReferenceLine
                              x={retDist.cvar}
                              stroke={C.warn}
                              strokeDasharray="4 4"
                              label={{ value: "CVaR", fill: C.warn, fontSize: 10 }}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ) : null}

                  {sens.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">VaR / CVaR by confidence</h3>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={sens}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="confidence_pct" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="var_pct" name="VaR %" stroke={C.danger} dot />
                            <Line type="monotone" dataKey="cvar_pct" name="CVaR %" stroke={C.warn} dot />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ) : null}

                  {rolling?.series?.length ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Rolling VaR</h3>
                      <div className="grid gap-2 sm:grid-cols-4 mb-2">
                        {Object.entries(rolling.stats ?? {}).map(([k, v]) => (
                          <div key={k} className="metric-card text-center">
                            <div className="text-xs text-white/45">{k}</div>
                            <div className="text-sm text-white">{(Number(v) * 100).toFixed(2)}%</div>
                          </div>
                        ))}
                      </div>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={rolling.series}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 9 }} minTickGap={20} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            <Line type="monotone" dataKey="y" name="Rolling VaR %" stroke={C.primary} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(rolling.interpretation ?? "")} />
                    </div>
                  ) : null}

                  {cov?.decomposition?.length ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Covariance VaR & decomposition</h3>
                      <p className="text-sm text-white/60 mb-2">
                        Portfolio VaR (covariance):{" "}
                        <span className="text-white font-medium">
                          {cov.portfolio_var_pct.toFixed(2)}%
                        </span>
                        . Matrix method uses nearest of 90% / 95% / 99% — here:{" "}
                        <span className="text-white/80">
                          {cov.confidence_used != null
                            ? `${(Number(cov.confidence_used) * 100).toFixed(0)}%`
                            : "—"}
                        </span>
                        .
                      </p>
                      <InterpretBox text={String(cov.interpretation_covariance ?? "")} />
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm text-white/85">
                          <thead>
                            <tr className="border-b border-white/10 text-white/50">
                              <th className="py-2 text-left">Asset</th>
                              <th className="py-2 text-right">Weight %</th>
                              <th className="py-2 text-right">Component VaR %</th>
                              <th className="py-2 text-right">Contribution %</th>
                              <th className="py-2 text-right">Marginal VaR %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {cov.decomposition.map((row) => (
                              <tr key={String(row.asset)} className="border-b border-white/5">
                                <td className="py-1">{String(row.asset)}</td>
                                <td className="py-1 text-right">{Number(row.weight_pct).toFixed(2)}</td>
                                <td className="py-1 text-right">
                                  {Number(row.component_var_pct).toFixed(2)}
                                </td>
                                <td className="py-1 text-right">
                                  {Number(row.contribution_pct).toFixed(2)}
                                </td>
                                <td className="py-1 text-right">
                                  {Number(row.marginal_var_pct).toFixed(2)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="h-64 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={cov.decomposition}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="asset" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            <Bar dataKey="contribution_pct" fill={C.primary} name="Contribution %" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(cov.interpretation_decomposition ?? "")} />
                    </div>
                  ) : null}

                  <details className="text-xs text-white/50">
                    <summary className="cursor-pointer">Raw var_results</summary>
                    <pre className="mt-2 max-h-48 overflow-auto">
                      {JSON.stringify(varBundle.var_results, null, 2)}
                    </pre>
                  </details>
                </div>
              ) : null}
            </section>
          )}

          {tab === "montecarlo" && (
            <section className="panel p-6 space-y-6">
              <h2 className="text-xl text-white">Monte Carlo simulation</h2>
              <p className="text-sm text-white/55">
                Simulates forward paths from historical return patterns. Horizon = trading days ahead.
              </p>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-xs text-white/50">Horizon (days forward)</label>
                  <input
                    type="number"
                    min={1}
                    max={252}
                    className="input mt-1"
                    value={mcHorizon}
                    onChange={(e) => setMcHorizon(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-xs text-white/50">Simulations</label>
                  <select
                    className="input mt-1"
                    value={mcSims}
                    onChange={(e) => setMcSims(Number(e.target.value))}
                  >
                    {[1000, 5000, 10000, 50000].map((n) => (
                      <option key={n} value={n}>
                        {n.toLocaleString()}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-white/50">Initial portfolio value ($)</label>
                  <input
                    type="number"
                    min={1}
                    step={1000}
                    className="input mt-1"
                    value={mcInitial}
                    onChange={(e) => setMcInitial(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-xs text-white/50">Model</label>
                  <select
                    className="input mt-1"
                    value={mcModel}
                    onChange={(e) => setMcModel(e.target.value as typeof mcModel)}
                  >
                    <option value="gbm">GBM</option>
                    <option value="jump_diffusion">Jump diffusion</option>
                  </select>
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm text-white/80">
                <input type="checkbox" checked={mcPaths} onChange={(e) => setMcPaths(e.target.checked)} />
                Include sample paths (subset — slower payload)
              </label>
              <button type="button" className="btn btn-primary" disabled={loading} onClick={runMc}>
                {loading ? "Running…" : "Run simulation"}
              </button>

              {mcBundle ? (
                <div className="space-y-8 border-t border-white/10 pt-6">
                  {mcStats ? (
                    <div className="grid gap-3 sm:grid-cols-3">
                      {(["mean", "median", "std", "min", "max"] as const).map((k) => (
                        <div key={k} className="metric-card">
                          <div className="text-xs text-white/45">{k}</div>
                          <div className="text-lg text-white">
                            ${Number(mcStats[k] ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <InterpretBox text={String(mcBundle.interpretation_statistics ?? "")} />
                  {mcPct ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Percentiles</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <tbody>
                            {Object.entries(mcPct).map(([p, v]) => (
                              <tr key={p} className="border-b border-white/5">
                                <td className="py-1 text-white/50">{p}</td>
                                <td className="py-1 text-right text-white">
                                  ${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_percentiles ?? "")} />
                    </div>
                  ) : null}
                  {mcHist.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Final value distribution</h3>
                      <p className="text-xs text-white/45 mb-2">
                        Bars: simulated outcomes; orange line: normal overlay; vertical lines: percentiles
                        and initial value.
                      </p>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={mcHist}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis
                              dataKey="x"
                              tick={{ fill: C.text, fontSize: 9 }}
                              tickFormatter={(v) => `${(Number(v) / 1000).toFixed(0)}k`}
                            />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            <Legend />
                            <Bar dataKey="count" name="Frequency" fill={C.primary} opacity={0.75} />
                            <Line
                              type="monotone"
                              dataKey="norm_y"
                              name="Normal (scaled)"
                              stroke={C.warn}
                              dot={false}
                              strokeWidth={2}
                            />
                            {mcPctMarkers.map((m) => (
                              <ReferenceLine
                                key={`${m.label}-${m.x}`}
                                x={m.x}
                                stroke={
                                  m.side === "initial"
                                    ? "#ffffff"
                                    : m.side === "lower"
                                      ? C.danger
                                      : C.ok
                                }
                                strokeDasharray="4 4"
                                label={{ value: m.label, fill: C.text, fontSize: 9 }}
                              />
                            ))}
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_distribution ?? "")} />
                    </div>
                  ) : null}
                  {mcVarCvarSim.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">VaR &amp; CVaR from simulations</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10 text-white/50">
                              <th className="py-2 text-left">Confidence</th>
                              <th className="py-2 text-right">VaR %</th>
                              <th className="py-2 text-right">VaR $</th>
                              <th className="py-2 text-right">CVaR %</th>
                              <th className="py-2 text-right">CVaR $</th>
                            </tr>
                          </thead>
                          <tbody>
                            {mcVarCvarSim.map((r, i) => (
                              <tr key={i} className="border-b border-white/5">
                                <td className="py-1">{String(r.confidence)}</td>
                                <td className="py-1 text-right">{Number(r.var_pct).toFixed(2)}%</td>
                                <td className="py-1 text-right">
                                  ${Number(r.var_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                </td>
                                <td className="py-1 text-right">{Number(r.cvar_pct).toFixed(2)}%</td>
                                <td className="py-1 text-right">
                                  ${Number(r.cvar_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_var_cvar_sim ?? "")} />
                    </div>
                  ) : null}
                  {mcHistVsMc.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Historical vs Monte Carlo VaR</h3>
                      <div className="overflow-x-auto mb-3">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10 text-white/50">
                              <th className="py-2 text-left">Confidence</th>
                              <th className="py-2 text-right">Historical %</th>
                              <th className="py-2 text-right">MC %</th>
                              <th className="py-2 text-right">Diff (MC−H) %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {mcHistVsMc.map((r, i) => (
                              <tr key={i} className="border-b border-white/5">
                                <td className="py-1">{String(r.confidence)}</td>
                                <td className="py-1 text-right">
                                  {Number(r.historical_var_pct).toFixed(2)}%
                                </td>
                                <td className="py-1 text-right">{Number(r.mc_var_pct).toFixed(2)}%</td>
                                <td className="py-1 text-right">{Number(r.diff_pct).toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={mcHistVsMc}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="confidence" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip formatter={(value) => fmtTooltip(value)} />
                            <Legend />
                            <Bar dataKey="historical_var_pct" name="Historical %" fill={C.danger} />
                            <Bar dataKey="mc_var_pct" name="MC %" fill={C.warn} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_historical_vs_mc ?? "")} />
                      <InterpretBox text={String(mcBundle.interpretation_historical_vs_mc_chart ?? "")} />
                    </div>
                  ) : null}
                  {mcExtreme.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Extreme scenarios</h3>
                      <div className="overflow-x-auto mb-3">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10 text-white/50">
                              <th className="py-2 text-left">Scenario</th>
                              <th className="py-2 text-right">Value</th>
                              <th className="py-2 text-right">P/L $</th>
                              <th className="py-2 text-right">Return %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {mcExtreme.map((r, i) => (
                              <tr key={i} className="border-b border-white/5">
                                <td className="py-1">{String(r.scenario)}</td>
                                <td className="py-1 text-right">
                                  ${Number(r.value).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                </td>
                                <td className="py-1 text-right">
                                  ${Number(r.return_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                </td>
                                <td className="py-1 text-right">{Number(r.return_pct).toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={mcExtreme}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="scenario" tick={{ fill: C.text, fontSize: 9 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip formatter={(value) => fmtTooltip(value)} />
                            <Bar dataKey="value" name="Terminal value">
                              {mcExtreme.map((e, i) => (
                                <Cell
                                  key={i}
                                  fill={String(e.scenario).includes("Worst") ? C.danger : C.ok}
                                />
                              ))}
                            </Bar>
                            <ReferenceLine
                              y={mcInitial}
                              stroke="#ffffff"
                              strokeDasharray="5 5"
                              label={{ value: "Initial", fill: "#fff", fontSize: 9 }}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_extreme ?? "")} />
                    </div>
                  ) : null}
                  {mcPathEnvelope || mcPathsData?.length ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">
                        Simulation paths{mcPathEnvelope ? " (envelope + sample)" : ""}
                      </h3>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="day" type="number" tick={{ fill: C.text, fontSize: 9 }} />
                            <YAxis
                              domain={mcPathsYDomain ?? ["auto", "auto"]}
                              allowDataOverflow={false}
                              tick={{ fill: C.text, fontSize: 10 }}
                              tickFormatter={(v) =>
                                Number(v) >= 1e6 ? `${(Number(v) / 1e6).toFixed(1)}M` : `${(Number(v) / 1000).toFixed(0)}k`
                              }
                            />
                            <Tooltip
                              contentStyle={{
                                background: "#1a1d25",
                                border: "1px solid rgba(255,255,255,0.1)",
                                fontSize: 11,
                              }}
                            />
                            {mcPathEnvelope?.max ? (
                              <Line
                                data={mcPathEnvelope.max}
                                type="monotone"
                                dataKey="value"
                                name="Max path"
                                stroke={C.ok}
                                dot={false}
                                strokeWidth={2}
                                isAnimationActive={false}
                              />
                            ) : null}
                            {mcPathEnvelope?.min ? (
                              <Line
                                data={mcPathEnvelope.min}
                                type="monotone"
                                dataKey="value"
                                name="Min path"
                                stroke={C.danger}
                                dot={false}
                                strokeWidth={2}
                                isAnimationActive={false}
                              />
                            ) : null}
                            {mcPathEnvelope?.median ? (
                              <Line
                                data={mcPathEnvelope.median}
                                type="monotone"
                                dataKey="value"
                                name="Median path"
                                stroke={C.secondary}
                                dot={false}
                                strokeWidth={2}
                                isAnimationActive={false}
                              />
                            ) : null}
                            {mcPathsData?.map((path, i) => (
                              <Line
                                key={i}
                                data={path}
                                type="monotone"
                                dataKey="value"
                                stroke={i % 2 ? "rgba(125,196,228,0.25)" : "rgba(191,159,251,0.25)"}
                                dot={false}
                                strokeWidth={1}
                                legendType="none"
                                isAnimationActive={false}
                              />
                            ))}
                            <ReferenceLine
                              y={mcInitial}
                              stroke="#ffffff"
                              strokeDasharray="4 4"
                              label={{ value: "Initial", fill: "#fff", fontSize: 9 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex flex-wrap gap-x-6 gap-y-2 justify-center text-xs text-white/70 mt-2">
                        {mcPathEnvelope?.max ? (
                          <span className="inline-flex items-center gap-2">
                            <span className="h-0.5 w-7 shrink-0 rounded-full" style={{ background: C.ok }} />
                            Max path
                          </span>
                        ) : null}
                        {mcPathEnvelope?.median ? (
                          <span className="inline-flex items-center gap-2">
                            <span className="h-0.5 w-7 shrink-0 rounded-full" style={{ background: C.secondary }} />
                            Median path
                          </span>
                        ) : null}
                        {mcPathEnvelope?.min ? (
                          <span className="inline-flex items-center gap-2">
                            <span className="h-0.5 w-7 shrink-0 rounded-full" style={{ background: C.danger }} />
                            Min path
                          </span>
                        ) : null}
                        {mcPathsData?.length ? (
                          <span className="inline-flex items-center gap-2">
                            <span
                              className="h-0.5 w-7 shrink-0 rounded-full"
                              style={{
                                background:
                                  "linear-gradient(90deg, rgba(191,159,251,0.7), rgba(125,196,228,0.7))",
                              }}
                            />
                            Sample paths (subset)
                          </span>
                        ) : null}
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-0 w-7 border-t border-dashed border-white/80 shrink-0"
                            aria-hidden
                          />
                          Initial
                        </span>
                      </div>
                      <InterpretBox text={String(mcBundle.interpretation_paths ?? "")} />
                    </div>
                  ) : null}
                </div>
              ) : null}
            </section>
          )}

          {tab === "historical" && (
            <section className="panel p-6 space-y-4">
              <h2 className="text-xl text-white">Historical scenarios</h2>
              <p className="text-sm text-white/55">Multi-select scenarios to stress the current portfolio.</p>
              <div className="max-h-56 overflow-y-auto rounded-lg border border-white/10 p-3 space-y-2">
                {catalog.map((s) => (
                  <label key={s.key} className="flex gap-2 text-sm text-white/80 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={histSelected.includes(s.key)}
                      onChange={(e) => {
                        if (e.target.checked) setHistSelected((x) => [...x, s.key]);
                        else setHistSelected((x) => x.filter((k) => k !== s.key));
                      }}
                    />
                    <span>
                      <span className="text-white">{s.name}</span>{" "}
                      <span className="text-white/45">
                        ({(s.market_impact_pct * 100).toFixed(1)}% market)
                      </span>
                    </span>
                  </label>
                ))}
              </div>
              <button type="button" className="btn btn-primary" disabled={loading} onClick={runHistorical}>
                Run scenarios
              </button>
              {histRows.length > 0 ? (
                <>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-white/50">
                          <th className="py-2 text-left">Scenario</th>
                          <th className="py-2 text-right">Impact %</th>
                          <th className="py-2 text-right">Impact $</th>
                          <th className="py-2">Worst</th>
                          <th className="py-2 text-right">Recovery days</th>
                        </tr>
                      </thead>
                      <tbody>
                        {histRows.map((r) => (
                          <tr key={r.scenario} className="border-b border-white/5">
                            <td className="py-1.5">{r.scenario}</td>
                            <td className="py-1.5 text-right">{r.impactPct.toFixed(2)}%</td>
                            <td className="py-1.5 text-right">${r.impactVal.toLocaleString()}</td>
                            <td className="py-1.5">
                              {r.worst} ({r.worstPct.toFixed(2)}%)
                            </td>
                            <td className="py-1.5 text-right">{String(r.recovery ?? "—")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={histRows}>
                        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                        <XAxis dataKey="scenario" tick={{ fill: C.text, fontSize: 9 }} />
                        <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{
                            background: "#1a1d25",
                            border: "1px solid rgba(255,255,255,0.1)",
                            fontSize: 11,
                          }}
                        />
                        <Bar dataKey="impactPct" name="Impact %">
                          {histRows.map((e, i) => (
                            <Cell
                              key={i}
                              fill={e.impactPct < 0 ? C.danger : C.ok}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <InterpretBox text={histInterpScenarios} />
                  {histRecovery?.series && histRecovery.series.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Portfolio recovery timeline</h3>
                      <p className="text-xs text-white/45 mb-2">
                        Simplified linear recovery to baseline (100%) using scenario recovery periods.
                      </p>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="day" type="number" allowDuplicatedCategory={false} tick={{ fill: C.text, fontSize: 9 }} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} domain={[0, "auto"]} label={{ value: "% of initial", angle: -90, position: "insideLeft", fill: C.text, fontSize: 10 }} />
                            <Tooltip />
                            <Legend />
                            <ReferenceLine y={100} stroke="#666" strokeDasharray="4 4" label={{ value: "100%", fill: C.text, fontSize: 9 }} />
                            {histRecovery.series.map((s, idx) => (
                              <Line
                                key={s.scenario_name}
                                data={s.points}
                                type="monotone"
                                dataKey="pct"
                                name={s.scenario_name}
                                stroke={[C.primary, C.secondary, C.ok, C.warn, "#9aa3b8"][idx % 5]}
                                dot={false}
                                strokeWidth={2}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={String(histRecovery.interpretation ?? "")} />
                    </div>
                  ) : null}
                  {histEnhanced.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Enhanced scenario comparison</h3>
                      <div className="overflow-x-auto mb-3">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10 text-white/50">
                              <th className="py-2 text-left">Scenario</th>
                              <th className="py-2 text-left">Period</th>
                              <th className="py-2 text-right">Duration (d)</th>
                              <th className="py-2 text-right">Market %</th>
                              <th className="py-2 text-right">Portfolio %</th>
                              <th className="py-2 text-right">Recovery (d)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {histEnhanced.map((r, i) => (
                              <tr key={i} className="border-b border-white/5">
                                <td className="py-1">{String(r.scenario)}</td>
                                <td className="py-1 text-white/60 text-xs">
                                  {String(r.period_start)} → {String(r.period_end)}
                                </td>
                                <td className="py-1 text-right">{Number(r.duration_days)}</td>
                                <td className="py-1 text-right">{Number(r.market_impact_pct).toFixed(1)}%</td>
                                <td className="py-1 text-right">{Number(r.portfolio_impact_pct).toFixed(2)}%</td>
                                <td className="py-1 text-right">{String(r.recovery_days ?? "—")}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={histEnhanced}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="scenario" tick={{ fill: C.text, fontSize: 9 }} />
                            <YAxis yAxisId="left" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis yAxisId="right" orientation="right" tick={{ fill: C.text, fontSize: 10 }} />
                            <Tooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="portfolio_impact_pct" name="Portfolio impact %" fill={C.danger} />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="duration_days"
                              name="Duration (days)"
                              stroke={C.warn}
                              dot
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ) : null}
                  {histBreakdowns.length > 0 ? (
                    <div className="space-y-8">
                      <h3 className="text-lg text-white">Position impact breakdown</h3>
                      {histBreakdowns.map((block) => (
                        <div key={String(block.scenario_name)} className="border-t border-white/10 pt-4 space-y-3">
                          <h4 className="text-white font-medium">{block.scenario_name}</h4>
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                              <thead>
                                <tr className="border-b border-white/10 text-white/50">
                                  <th className="py-2 text-left">Ticker</th>
                                  <th className="py-2 text-right">Weight %</th>
                                  <th className="py-2 text-right">Impact %</th>
                                  <th className="py-2 text-left">Type</th>
                                </tr>
                              </thead>
                              <tbody>
                                {block.rows.map((row) => (
                                  <tr key={row.ticker} className="border-b border-white/5">
                                    <td className="py-1">{row.ticker}</td>
                                    <td className="py-1 text-right">{row.weight_pct.toFixed(2)}%</td>
                                    <td className="py-1 text-right">{row.impact_pct.toFixed(2)}%</td>
                                    <td className="py-1">{row.kind}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <div className="h-56">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={block.rows}>
                                <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                                <XAxis dataKey="ticker" tick={{ fill: C.text, fontSize: 9 }} />
                                <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                                <Tooltip />
                                <Bar dataKey="impact_pct" name="Impact %">
                                  {block.rows.map((row, i) => (
                                    <Cell key={i} fill={row.impact_pct < 0 ? C.danger : C.ok} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                          <InterpretBox text={String(block.interpretation ?? "")} />
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {histTimeline.length > 0 ? (
                    <div>
                      <h3 className="text-lg text-white mb-2">Historical timeline</h3>
                      <div
                        className="h-auto min-h-[200px]"
                        style={{ height: Math.min(520, 120 + histTimeline.length * 44) }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart layout="vertical" data={histTimeline} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis type="number" tick={{ fill: C.text, fontSize: 10 }} />
                            <YAxis type="category" dataKey="name" width={140} tick={{ fill: C.text, fontSize: 9 }} />
                            <Tooltip />
                            <Bar dataKey="duration_days" name="Duration (days)" radius={[0, 4, 4, 0]}>
                              {histTimeline.map((t, i) => (
                                <Cell
                                  key={i}
                                  fill={
                                    Number(t.market_impact_pct) < -20
                                      ? C.danger
                                      : Number(t.market_impact_pct) < -10
                                        ? C.warn
                                        : C.secondary
                                  }
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="h-64 mt-6">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={histTimeline}>
                            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                            <XAxis dataKey="name" tick={{ fill: C.text, fontSize: 9 }} interval={0} angle={-25} textAnchor="end" height={70} />
                            <YAxis tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Market impact %", angle: -90, position: "insideLeft", fill: C.text, fontSize: 10 }} />
                            <Tooltip />
                            <ReferenceLine y={0} stroke="#888" />
                            <Bar dataKey="market_impact_pct" name="Market shock %">
                              {histTimeline.map((t, i) => (
                                <Cell
                                  key={i}
                                  fill={
                                    Number(t.market_impact_pct) < -20
                                      ? C.danger
                                      : Number(t.market_impact_pct) < -10
                                        ? C.warn
                                        : C.secondary
                                  }
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <InterpretBox text={histInterpTimeline} />
                    </div>
                  ) : null}
                </>
              ) : null}
            </section>
          )}

          {tab === "custom" && (
            <section className="panel p-6 space-y-4">
              <h2 className="text-xl text-white">Custom scenario</h2>
              <input
                className="input"
                placeholder="Scenario name"
                value={custName}
                onChange={(e) => setCustName(e.target.value)}
              />
              <textarea
                className="input"
                placeholder="Description"
                value={custDesc}
                onChange={(e) => setCustDesc(e.target.value)}
              />
              <div>
                <label className="text-xs text-white/50">Market impact %</label>
                <input
                  type="range"
                  min={-100}
                  max={100}
                  className="w-full"
                  value={custMarketPct}
                  onChange={(e) => setCustMarketPct(Number(e.target.value))}
                />
                <span className="text-sm text-white/70">{custMarketPct}%</span>
              </div>
              <div>
                <label className="text-xs text-white/50">
                  Asset shocks (one per line: TICKER:-15 or TICKER:-15%)
                </label>
                <textarea
                  className="input mt-1 font-mono text-sm"
                  rows={5}
                  value={custAssetsText}
                  onChange={(e) => setCustAssetsText(e.target.value)}
                  placeholder={"AAPL:-30\nMSFT:-25"}
                />
              </div>
              <button type="button" className="btn btn-primary" disabled={loading} onClick={runCustom}>
                Create &amp; run
              </button>
              {custResult ? (
                <div className="grid gap-3 sm:grid-cols-2 border-t border-white/10 pt-4">
                  <div className="metric-card">
                    <div className="text-xs text-white/45">Portfolio impact</div>
                    <div className="text-xl text-white">
                      {(Number(custResult.portfolio_impact_pct) * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="text-xs text-white/45">Impact value</div>
                    <div className="text-xl text-white">
                      ${Number(custResult.portfolio_impact_value).toLocaleString()}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="text-xs text-white/45">Worst position</div>
                    <div className="text-lg text-white">
                      {(custResult.worst_position as { ticker?: string })?.ticker}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="text-xs text-white/45">Best position</div>
                    <div className="text-lg text-white">
                      {(custResult.best_position as { ticker?: string })?.ticker}
                    </div>
                  </div>
                </div>
              ) : null}
            </section>
          )}

          {tab === "chain" && (
            <section className="panel p-6 space-y-4">
              <h2 className="text-xl text-white">Scenario chain</h2>
              <p className="text-sm text-white/55">Apply historical scenarios sequentially (order matters).</p>
              <input
                className="input"
                placeholder="Chain name"
                value={chainName}
                onChange={(e) => setChainName(e.target.value)}
              />
              <textarea
                className="input"
                placeholder="Description"
                value={chainDesc}
                onChange={(e) => setChainDesc(e.target.value)}
              />
              <div className="flex flex-wrap gap-2 items-end">
                <select
                  className="input max-w-md"
                  value={chainPick}
                  onChange={(e) => setChainPick(e.target.value)}
                >
                  <option value="">Add scenario…</option>
                  {catalog.map((s) => (
                    <option key={s.key} value={s.key}>
                      {s.name}
                    </option>
                  ))}
                </select>
                <button type="button" className="btn" onClick={addChainKey}>
                  Add
                </button>
              </div>
              <ol className="list-decimal list-inside text-sm text-white/80 space-y-1">
                {chainKeys.map((k) => (
                  <li key={k} className="flex justify-between gap-2 items-center">
                    <span>{catalog.find((c) => c.key === k)?.name ?? k}</span>
                    <button
                      type="button"
                      className="text-xs text-rose-300"
                      onClick={() => setChainKeys((x) => x.filter((i) => i !== k))}
                    >
                      remove
                    </button>
                  </li>
                ))}
              </ol>
              <button type="button" className="btn btn-primary" disabled={loading} onClick={runChain}>
                Run chain
              </button>
              {chainResult ? (
                <div className="space-y-4 border-t border-white/10 pt-4">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="metric-card">
                      <div className="text-xs text-white/45">Cumulative impact</div>
                      <div className="text-xl text-white">
                        {(Number(chainResult.cumulative_impact_pct) * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="text-xs text-white/45">Final value</div>
                      <div className="text-xl text-white">
                        ${Number(chainResult.final_portfolio_value).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  {Array.isArray(chainResult.scenario_results) ? (
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-white/50">
                          <th className="py-2 text-left">Scenario</th>
                          <th className="py-2 text-right">Step impact %</th>
                          <th className="py-2 text-right">Cumulative %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(chainResult.scenario_results as Array<Record<string, unknown>>).map(
                          (r, i) => (
                            <tr key={i} className="border-b border-white/5">
                              <td className="py-1">{String(r.scenario_name)}</td>
                              <td className="py-1 text-right">
                                {(Number(r.impact_pct) * 100).toFixed(2)}%
                              </td>
                              <td className="py-1 text-right">
                                {(Number(r.cumulative_impact_pct) * 100).toFixed(2)}%
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              ) : null}
            </section>
          )}
        </>
      )}
    </div>
  );
}

export default function RiskPage() {
  return (
    <Suspense fallback={<p className="text-white/60">Loading…</p>}>
      <RiskPageContent />
    </Suspense>
  );
}
