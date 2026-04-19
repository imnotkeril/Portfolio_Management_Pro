"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { Portfolio } from "@/lib/types";
import {
  Bar,
  BarChart,
  CartesianGrid,
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
import {
  CHART_PALETTE,
  defaultParamsForMethod,
  HORIZON_PRESETS,
  METHOD_LABELS,
  METHOD_TABS,
  METHODS_BY_TAB,
  MODEL_BLURBS,
  TRAINING_WINDOW_RATIOS,
} from "./config";

type ForecastDict = Record<string, Record<string, unknown>>;

const C = {
  grid: "rgba(255,255,255,0.06)",
  text: "rgba(255,255,255,0.5)",
  primary: "#bf9ffb",
};

function fmtTooltip(v: unknown) {
  if (v == null) return "";
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : "";
}

function InterpretBox({ text }: { text: string }) {
  if (!text?.trim()) return null;
  const plain = text.replace(/\*\*(.*?)\*\*/g, "$1");
  return (
    <div className="mt-2 rounded-lg border border-white/10 bg-[var(--info-bg)] p-3 text-sm text-white/85 whitespace-pre-wrap">
      {plain}
    </div>
  );
}

export default function ForecastingPage() {
  const [forecastType, setForecastType] = useState<"Single Asset" | "Portfolio">("Single Asset");
  const [ticker, setTicker] = useState("AAPL");
  const [portfolioId, setPortfolioId] = useState("");
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);

  const defaultEnd = () => new Date().toISOString().slice(0, 10);
  const defaultStart = () => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().slice(0, 10);
  };
  const [startDate, setStartDate] = useState(defaultStart);
  const [endDate, setEndDate] = useState(defaultEnd);

  const [outOfSample, setOutOfSample] = useState(true);
  const [trainingWindowLabel, setTrainingWindowLabel] = useState("30% (Recommended)");
  const [horizonPreset, setHorizonPreset] = useState("1 Month (21 days)");
  const [customHorizon, setCustomHorizon] = useState(21);

  const [methodTab, setMethodTab] = useState<string>("classical");
  const [selectedMethods, setSelectedMethods] = useState<Record<string, boolean>>({});
  const [paramsByMethod, setParamsByMethod] = useState<Record<string, Record<string, unknown>>>({});
  const [createEnsemble, setCreateEnsemble] = useState(false);

  const [bundle, setBundle] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resultsTab, setResultsTab] = useState<"comparison" | "individual" | "quality" | "detailed">(
    "comparison",
  );
  const [individualMethod, setIndividualMethod] = useState<string>("");

  useEffect(() => {
    api
      .get<Portfolio[]>("/portfolios")
      .then((list) => {
        setPortfolios(list);
        setPortfolioId((p) => p || list[0]?.id || "");
        const tickers = new Set<string>();
        list.forEach((port) =>
          port.positions.forEach((pos) => {
            if (pos.ticker !== "CASH") tickers.add(pos.ticker);
          }),
        );
        if (tickers.size) setTicker(Array.from(tickers).sort()[0]);
      })
      .catch(() => {});
  }, []);

  const horizonDays = useMemo(() => {
    const v = HORIZON_PRESETS[horizonPreset];
    return v == null ? customHorizon : v;
  }, [horizonPreset, customHorizon]);

  const trainingRatio = TRAINING_WINDOW_RATIOS[trainingWindowLabel]?.ratio ?? 0.3;

  const toggleMethod = useCallback((m: string, on: boolean) => {
    setSelectedMethods((prev) => ({ ...prev, [m]: on }));
    if (on) {
      setParamsByMethod((prev) => ({
        ...prev,
        [m]: prev[m] ?? defaultParamsForMethod(m),
      }));
    }
  }, []);

  const updateParam = useCallback((method: string, key: string, value: unknown) => {
    setParamsByMethod((prev) => ({
      ...prev,
      [method]: { ...(prev[method] ?? defaultParamsForMethod(method)), [key]: value },
    }));
  }, []);

  const methodsToRun = useMemo(
    () => Object.entries(selectedMethods).filter(([, v]) => v).map(([k]) => k),
    [selectedMethods],
  );

  const forecasts = (bundle?.forecasts as ForecastDict) ?? {};
  const successfulKeys = useMemo(
    () => Object.entries(forecasts).filter(([, v]) => v?.success).map(([k]) => k),
    [forecasts],
  );

  useEffect(() => {
    if (individualMethod && !successfulKeys.includes(individualMethod)) {
      setIndividualMethod(successfulKeys[0] ?? "");
    } else if (!individualMethod && successfulKeys.length) {
      setIndividualMethod(successfulKeys[0]);
    }
  }, [successfulKeys, individualMethod]);

  async function runBatch() {
    if (!methodsToRun.length) {
      setError("Select at least one method.");
      return;
    }
    if (forecastType === "Single Asset" && !ticker.trim()) {
      setError("Select a ticker.");
      return;
    }
    if (forecastType === "Portfolio" && !portfolioId) {
      setError("Select a portfolio.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const method_params: Record<string, Record<string, unknown>> = {};
      for (const m of methodsToRun) {
        method_params[m] = paramsByMethod[m] ?? defaultParamsForMethod(m);
      }
      const payload = {
        scope: forecastType === "Single Asset" ? "asset" : "portfolio",
        ticker: forecastType === "Single Asset" ? ticker.trim().toUpperCase() : undefined,
        portfolio_id: forecastType === "Portfolio" ? portfolioId : undefined,
        start_date: startDate,
        end_date: endDate,
        horizon: horizonDays,
        methods: methodsToRun,
        method_params,
        out_of_sample: outOfSample,
        training_ratio: trainingRatio,
        create_ensemble: createEnsemble,
      };
      const data = await api.post<Record<string, unknown>>("/forecasting/batch", payload);
      setBundle(data);
      setResultsTab("comparison");
    } catch (e) {
      setError(String(e));
      setBundle(null);
    } finally {
      setLoading(false);
    }
  }

  const comparisonChart = (bundle?.comparison_chart as Array<Record<string, unknown>>) ?? [];
  const meta = bundle?.meta as Record<string, unknown> | undefined;
  const historical = (bundle?.historical as Array<{ date: string; value: number }>) ?? [];

  const comparisonInterpretation = useMemo(() => {
    if (!successfulKeys.length || !comparisonChart.length) return "";
    const last = comparisonChart[comparisonChart.length - 1];
    const parts = [`**Forecast comparison:** ${successfulKeys.length} method(s) with data on chart.`];
    const hist = last?.historical;
    if (typeof hist === "number" && Number.isFinite(hist)) {
      parts.push(`Last historical point: ${hist.toFixed(2)}.`);
    }
    for (const k of successfulKeys.slice(0, 5)) {
      const v = last?.[k];
      if (typeof v === "number" && Number.isFinite(v)) {
        parts.push(`${METHOD_LABELS[k] ?? k} terminal (on chart): ${v.toFixed(2)}`);
      }
    }
    return parts.join("\n");
  }, [successfulKeys, comparisonChart]);

  const individualFc = individualMethod ? forecasts[individualMethod] : null;
  const individualRows = useMemo(() => {
    if (!individualFc?.success) return [];
    const dates = (individualFc.forecast_dates as string[]) ?? [];
    const vals = (individualFc.forecast_values as number[]) ?? [];
    const upper = (individualFc.confidence_intervals as Record<string, number[]>)?.upper_95 ?? [];
    const lower = (individualFc.confidence_intervals as Record<string, number[]>)?.lower_95 ?? [];
    return dates.map((d, i) => ({
      date: String(d).slice(0, 10),
      forecast: vals[i],
      upper: upper[i],
      lower: lower[i],
    }));
  }, [individualFc]);

  const mergedIndividual = useMemo(() => {
    const m = new Map<string, Record<string, string | number | undefined>>();
    for (const h of historical) {
      m.set(h.date, { date: h.date, historical: h.value });
    }
    for (const row of individualRows) {
      const ex = m.get(row.date) ?? { date: row.date };
      ex.forecast = row.forecast;
      ex.upper = row.upper;
      ex.lower = row.lower;
      m.set(row.date, ex);
    }
    return Array.from(m.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [historical, individualRows]);

  const qualityRows = useMemo(() => {
    return successfulKeys.map((k) => {
      const fd = forecasts[k];
      const vm = fd?.validation_metrics as Record<string, number> | undefined;
      return {
        method: METHOD_LABELS[k] ?? k,
        key: k,
        mape: vm?.mape,
        rmse: vm?.rmse,
        mae: vm?.mae,
        dir: vm?.direction_accuracy,
        r2: vm?.r_squared,
        final: fd?.final_value,
        chg: fd?.change_pct,
      };
    });
  }, [successfulKeys, forecasts]);

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-16">
      <header>
        <h1 className="text-3xl font-semibold text-white">Price &amp; Returns Forecasting</h1>
        <p className="mt-2 text-white/60">
          Multi-method forecasts with out-of-sample validation — aligned with the Streamlit forecasting
          page (classical, ML, deep learning, Prophet, ensemble).
        </p>
      </header>

      {!portfolios.length && forecastType === "Single Asset" ? (
        <div className="panel p-6 text-amber-200">Create a portfolio with positions to pick tickers.</div>
      ) : null}

      <section className="panel p-6 space-y-6">
        <h2 className="text-lg text-white">Asset selection</h2>
        <div className="flex flex-wrap gap-2">
          {(["Single Asset", "Portfolio"] as const).map((k) => (
            <button
              key={k}
              type="button"
              className={`rounded-lg border px-4 py-2 text-sm ${
                forecastType === k
                  ? "border-violet-400/70 bg-violet-500/10 text-white"
                  : "border-white/10 text-white/70"
              }`}
              onClick={() => setForecastType(k)}
            >
              {k}
            </button>
          ))}
        </div>
        {forecastType === "Single Asset" ? (
          <div>
            <label className="text-xs text-white/50">Ticker</label>
            <input className="input mt-1" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
          </div>
        ) : (
          <div>
            <label className="text-xs text-white/50">Portfolio</label>
            <select className="input mt-1" value={portfolioId} onChange={(e) => setPortfolioId(e.target.value)}>
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </section>

      <section className="panel p-6 space-y-4">
        <h2 className="text-lg text-white">Forecasting parameters</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="text-xs text-white/50">Analysis / validation start</label>
            <input
              type="date"
              className="input mt-1"
              value={startDate}
              max={endDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs text-white/50">Analysis / validation end</label>
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

        <div className="border-t border-white/10 pt-4 space-y-3">
          <h3 className="text-white font-medium">Out-of-sample testing</h3>
          <label className="flex items-center gap-2 text-sm text-white/80">
            <input type="checkbox" checked={outOfSample} onChange={(e) => setOutOfSample(e.target.checked)} />
            Train on period before validation window (recommended)
          </label>
          {outOfSample ? (
            <>
              <div>
                <label className="text-xs text-white/50">Training window (share of validation span)</label>
                <select
                  className="input mt-1"
                  value={trainingWindowLabel}
                  onChange={(e) => setTrainingWindowLabel(e.target.value)}
                >
                  {Object.keys(TRAINING_WINDOW_RATIOS).map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </div>
              <p className="text-xs text-white/45">{TRAINING_WINDOW_RATIOS[trainingWindowLabel]?.description}</p>
              {meta == null && (
                <p className="text-xs text-white/50">
                  After running, training range is returned in{" "}
                  <code className="text-white/70">meta.training_start</code>.
                </p>
              )}
            </>
          ) : null}
        </div>

        <div className="border-t border-white/10 pt-4 space-y-3">
          <h3 className="text-white font-medium">Forecast horizon</h3>
          <select className="input" value={horizonPreset} onChange={(e) => setHorizonPreset(e.target.value)}>
            {Object.keys(HORIZON_PRESETS).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          {HORIZON_PRESETS[horizonPreset] == null ? (
            <input
              type="number"
              min={1}
              max={1000}
              className="input"
              value={customHorizon}
              onChange={(e) => setCustomHorizon(Number(e.target.value))}
            />
          ) : null}
        </div>
      </section>

      <section className="panel p-6 space-y-4">
        <h2 className="text-lg text-white">Forecasting methods</h2>
        <div className="flex flex-wrap gap-2">
          {METHOD_TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`rounded-lg border px-3 py-1.5 text-sm ${
                methodTab === t.id
                  ? "border-violet-400/70 bg-violet-500/10 text-white"
                  : "border-white/10 text-white/70"
              }`}
              onClick={() => setMethodTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {methodTab === "dl" ? (
          <p className="text-sm text-amber-200/90">
            Deep learning models need more data and CPU/GPU time — same as Streamlit warning.
          </p>
        ) : null}

        <div className="space-y-4">
          {(METHODS_BY_TAB[methodTab] ?? []).map((m) => (
            <div key={m} className="rounded-lg border border-white/10 p-4 space-y-2">
              <label className="flex items-center gap-2 text-white">
                <input
                  type="checkbox"
                  checked={!!selectedMethods[m]}
                  onChange={(e) => toggleMethod(m, e.target.checked)}
                />
                {METHOD_LABELS[m] ?? m}
              </label>
              <p className="text-xs text-white/50 pl-6">{MODEL_BLURBS[m]}</p>
              {selectedMethods[m] ? <MethodParamFields method={m} params={paramsByMethod[m] ?? {}} onChange={updateParam} /> : null}
            </div>
          ))}
        </div>

        <div className="border-t border-white/10 pt-4 space-y-2">
          <label className="flex items-center gap-2 text-sm text-white/85">
            <input type="checkbox" checked={createEnsemble} onChange={(e) => setCreateEnsemble(e.target.checked)} />
            Optimized ensemble (weighted by validation MAPE, needs ≥2 successful models)
          </label>
          <p className="text-xs text-white/45">{MODEL_BLURBS.ensemble}</p>
        </div>

        <button type="button" className="btn btn-primary w-full md:w-auto" disabled={loading} onClick={runBatch}>
          {loading ? "Running forecasts…" : "Run forecasts"}
        </button>
        {error ? <div className="text-sm text-[var(--danger)]">{error}</div> : null}
      </section>

      {bundle ? (
        <section className="panel p-6 space-y-6">
          <h2 className="text-xl text-white">Forecast results</h2>
          {meta ? (
            <div className="text-xs text-white/50 space-y-1">
              {meta.out_of_sample ? (
                <>
                  <div>
                    Validation: {String(meta.validation_start)} → {String(meta.validation_end)}
                  </div>
                  <div>Training starts: {String(meta.training_start ?? "—")}</div>
                </>
              ) : null}
              <div>Forecast horizon (days ahead after validation): {String(meta.horizon)}</div>
              <div>Forecast end (target): {String(meta.forecast_end ?? "—")}</div>
            </div>
          ) : null}

          <div className="flex flex-wrap gap-2">
            {(
              [
                ["comparison", "Forecasts comparison"],
                ["individual", "Individual method"],
                ["quality", "Forecast quality"],
                ["detailed", "Detailed analysis"],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                className={`rounded-lg border px-3 py-1.5 text-sm ${
                  resultsTab === k
                    ? "border-violet-400/70 bg-violet-500/10 text-white"
                    : "border-white/10 text-white/70"
                }`}
                onClick={() => setResultsTab(k)}
              >
                {label}
              </button>
            ))}
          </div>

          {resultsTab === "comparison" && comparisonChart.length > 0 ? (
            <div>
              <h3 className="text-lg text-white mb-2">All forecasts vs history</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparisonChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: C.text, fontSize: 9 }} minTickGap={24} />
                    <YAxis tick={{ fill: C.text, fontSize: 10 }} domain={["auto", "auto"]} />
                    <Tooltip formatter={(v) => fmtTooltip(v)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="historical" name="Historical" stroke="#ffffff" dot={false} strokeWidth={2} connectNulls />
                    {successfulKeys.map((k, i) => (
                      <Line
                        key={k}
                        type="monotone"
                        dataKey={k}
                        name={METHOD_LABELS[k] ?? k}
                        stroke={CHART_PALETTE[i % CHART_PALETTE.length]}
                        dot={false}
                        strokeWidth={1.5}
                        connectNulls
                      />
                    ))}
                    {outOfSample && meta?.validation_start ? (
                      <ReferenceLine x={String(meta.validation_start).slice(0, 10)} stroke="#888" strokeDasharray="4 4" />
                    ) : null}
                    {outOfSample && meta?.validation_end ? (
                      <ReferenceLine x={String(meta.validation_end).slice(0, 10)} stroke="#888" strokeDasharray="4 4" />
                    ) : null}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <InterpretBox text={comparisonInterpretation} />
            </div>
          ) : null}

          {resultsTab === "comparison" && successfulKeys.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-white/85">
                <thead>
                  <tr className="border-b border-white/10 text-white/50">
                    <th className="py-2 text-left">Method</th>
                    <th className="py-2 text-right">Final value</th>
                    <th className="py-2 text-right">Change %</th>
                    <th className="py-2 text-right">MAPE</th>
                    <th className="py-2 text-right">RMSE</th>
                  </tr>
                </thead>
                <tbody>
                  {successfulKeys.map((k) => {
                    const fd = forecasts[k];
                    const vm = fd?.validation_metrics as Record<string, number> | undefined;
                    return (
                      <tr key={k} className="border-b border-white/5">
                        <td className="py-1">{METHOD_LABELS[k] ?? k}</td>
                        <td className="py-1 text-right">
                          {Number(fd?.final_value).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-1 text-right">{Number(fd?.change_pct).toFixed(2)}%</td>
                        <td className="py-1 text-right">
                          {vm?.mape != null && Number.isFinite(vm.mape) ? `${vm.mape.toFixed(2)}%` : "—"}
                        </td>
                        <td className="py-1 text-right">
                          {vm?.rmse != null && Number.isFinite(vm.rmse) ? vm.rmse.toFixed(4) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}

          {resultsTab === "individual" && individualFc?.success ? (
            <div className="space-y-6">
              <div>
                <label className="text-xs text-white/50">Method</label>
                <select
                  className="input mt-1"
                  value={individualMethod}
                  onChange={(e) => setIndividualMethod(e.target.value)}
                >
                  {successfulKeys.map((k) => (
                    <option key={k} value={k}>
                      {METHOD_LABELS[k] ?? k}
                    </option>
                  ))}
                </select>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={mergedIndividual} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: C.text, fontSize: 9 }} minTickGap={20} />
                    <YAxis tick={{ fill: C.text, fontSize: 10 }} domain={["auto", "auto"]} />
                    <Tooltip formatter={(v) => fmtTooltip(v)} />
                    <Legend />
                    <Line type="monotone" dataKey="historical" name="Historical" stroke="#fff" dot={false} connectNulls />
                    <Line type="monotone" dataKey="forecast" name="Forecast" stroke={C.primary} dot={false} connectNulls />
                    <Line
                      type="monotone"
                      dataKey="upper"
                      name="Upper 95%"
                      stroke="#faa1a4"
                      dot={false}
                      strokeDasharray="4 4"
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="lower"
                      name="Lower 95%"
                      stroke="#74f174"
                      dot={false}
                      strokeDasharray="4 4"
                      connectNulls
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-white/45">
                Confidence bands when the backend provides{" "}
                <code className="text-white/60">upper_95</code> / <code className="text-white/60">lower_95</code>.
              </p>
            </div>
          ) : null}

          {resultsTab === "quality" ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-white/50">
                    <th className="py-2 text-left">Method</th>
                    <th className="py-2 text-right">MAPE</th>
                    <th className="py-2 text-right">RMSE</th>
                    <th className="py-2 text-right">MAE</th>
                    <th className="py-2 text-right">Dir. acc.</th>
                    <th className="py-2 text-right">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {qualityRows.map((r) => (
                    <tr key={r.key} className="border-b border-white/5">
                      <td className="py-1 text-white/90">{r.method}</td>
                      <td className="py-1 text-right">{r.mape != null && Number.isFinite(r.mape) ? `${r.mape.toFixed(2)}%` : "—"}</td>
                      <td className="py-1 text-right">{r.rmse != null && Number.isFinite(r.rmse) ? r.rmse.toFixed(4) : "—"}</td>
                      <td className="py-1 text-right">{r.mae != null && Number.isFinite(r.mae) ? r.mae.toFixed(4) : "—"}</td>
                      <td className="py-1 text-right">{r.dir != null && Number.isFinite(r.dir) ? `${(r.dir * 100).toFixed(1)}%` : "—"}</td>
                      <td className="py-1 text-right">{r.r2 != null && Number.isFinite(r.r2) ? r.r2.toFixed(3) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {resultsTab === "detailed" && successfulKeys.length ? (
            <div className="space-y-6">
              <div>
                <label className="text-xs text-white/50">Method</label>
                <select
                  className="input mt-1"
                  value={individualMethod}
                  onChange={(e) => setIndividualMethod(e.target.value)}
                >
                  {successfulKeys.map((k) => (
                    <option key={k} value={k}>
                      {METHOD_LABELS[k] ?? k}
                    </option>
                  ))}
                </select>
              </div>
              {individualMethod && forecasts[individualMethod] ? (
                <>
                  <pre className="max-h-64 overflow-auto rounded-lg border border-white/10 bg-black/30 p-3 text-xs text-white/80">
                    {JSON.stringify(forecasts[individualMethod].model_info ?? {}, null, 2)}
                  </pre>
                  {Array.isArray(forecasts[individualMethod].residuals) ? (
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={(forecasts[individualMethod].residuals as number[]).map((y, i) => ({
                            i,
                            r: y,
                          }))}
                        >
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="i" tick={{ fill: C.text, fontSize: 9 }} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} />
                          <Tooltip formatter={(v) => fmtTooltip(v)} />
                          <Bar dataKey="r" fill={C.primary} opacity={0.75} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="text-sm text-white/50">No residual vector for this method.</p>
                  )}
                </>
              ) : null}
            </div>
          ) : null}

          <details className="text-xs text-white/45">
            <summary className="cursor-pointer">Raw batch JSON</summary>
            <pre className="mt-2 max-h-48 overflow-auto">{JSON.stringify(bundle, null, 2)}</pre>
          </details>
        </section>
      ) : null}
    </div>
  );
}

function MethodParamFields({
  method,
  params,
  onChange,
}: {
  method: string;
  params: Record<string, unknown>;
  onChange: (m: string, key: string, v: unknown) => void;
}) {
  if (method === "arima") {
    const auto = params.auto !== false;
    return (
      <div className="pl-6 space-y-2 text-sm">
        <label className="flex items-center gap-2 text-white/80">
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => onChange(method, "auto", e.target.checked)}
          />
          Auto ARIMA
        </label>
        {!auto ? (
          <div className="grid gap-2 sm:grid-cols-3">
            {(["p", "d", "q"] as const).map((k) => (
              <label key={k} className="text-xs text-white/50">
                {k.toUpperCase()}
                <input
                  type="number"
                  className="input mt-1"
                  min={0}
                  max={5}
                  value={Number(params[k] ?? 1)}
                  onChange={(e) => onChange(method, k, Number(e.target.value))}
                />
              </label>
            ))}
          </div>
        ) : null}
      </div>
    );
  }
  if (method === "garch") {
    return (
      <div className="pl-6 grid gap-2 sm:grid-cols-2 text-sm">
        <label className="text-xs text-white/50">
          p
          <input
            type="number"
            className="input mt-1"
            min={1}
            max={3}
            value={Number(params.p ?? 1)}
            onChange={(e) => onChange(method, "p", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50">
          q
          <input
            type="number"
            className="input mt-1"
            min={1}
            max={3}
            value={Number(params.q ?? 1)}
            onChange={(e) => onChange(method, "q", Number(e.target.value))}
          />
        </label>
      </div>
    );
  }
  if (method === "arima_garch") {
    const auto = params.auto_arima !== false;
    return (
      <div className="pl-6 space-y-2 text-sm">
        <label className="flex items-center gap-2 text-white/80">
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => onChange(method, "auto_arima", e.target.checked)}
          />
          Auto ARIMA
        </label>
        {!auto ? (
          <div className="grid gap-2 sm:grid-cols-3">
            {(["arima_p", "arima_d", "arima_q"] as const).map((k) => (
              <label key={k} className="text-xs text-white/50">
                {k}
                <input
                  type="number"
                  className="input mt-1"
                  min={0}
                  max={5}
                  value={Number(params[k] ?? 1)}
                  onChange={(e) => onChange(method, k, Number(e.target.value))}
                />
              </label>
            ))}
          </div>
        ) : null}
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="text-xs text-white/50">
            garch_p
            <input
              type="number"
              className="input mt-1"
              min={1}
              max={3}
              value={Number(params.garch_p ?? 1)}
              onChange={(e) => onChange(method, "garch_p", Number(e.target.value))}
            />
          </label>
          <label className="text-xs text-white/50">
            garch_q
            <input
              type="number"
              className="input mt-1"
              min={1}
              max={3}
              value={Number(params.garch_q ?? 1)}
              onChange={(e) => onChange(method, "garch_q", Number(e.target.value))}
            />
          </label>
        </div>
      </div>
    );
  }
  if (method === "xgboost") {
    return (
      <div className="pl-6 grid gap-2 sm:grid-cols-2 text-sm">
        <label className="text-xs text-white/50">
          Trees
          <input
            type="number"
            className="input mt-1"
            min={50}
            max={500}
            value={Number(params.n_estimators ?? 100)}
            onChange={(e) => onChange(method, "n_estimators", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50">
          Max depth
          <input
            type="number"
            className="input mt-1"
            min={3}
            max={10}
            value={Number(params.max_depth ?? 6)}
            onChange={(e) => onChange(method, "max_depth", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50">
          Learning rate
          <input
            type="number"
            className="input mt-1"
            step={0.01}
            min={0.01}
            max={0.3}
            value={Number(params.learning_rate ?? 0.1)}
            onChange={(e) => onChange(method, "learning_rate", Number(e.target.value))}
          />
        </label>
        <label className="flex items-center gap-2 text-white/80 mt-6">
          <input
            type="checkbox"
            checked={!!params.use_technical_features}
            onChange={(e) => onChange(method, "use_technical_features", e.target.checked)}
          />
          Technical indicators
        </label>
      </div>
    );
  }
  if (method === "random_forest") {
    return (
      <div className="pl-6 grid gap-2 sm:grid-cols-2 text-sm">
        <label className="text-xs text-white/50">
          Trees
          <input
            type="number"
            className="input mt-1"
            min={50}
            max={300}
            value={Number(params.n_estimators ?? 100)}
            onChange={(e) => onChange(method, "n_estimators", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50">
          Max depth
          <input
            type="number"
            className="input mt-1"
            min={5}
            max={20}
            value={Number(params.max_depth ?? 10)}
            onChange={(e) => onChange(method, "max_depth", Number(e.target.value))}
          />
        </label>
      </div>
    );
  }
  if (method === "svm") {
    return (
      <div className="pl-6 grid gap-2 sm:grid-cols-2 text-sm">
        <label className="text-xs text-white/50">
          C
          <input
            type="number"
            className="input mt-1"
            step={0.1}
            min={0.1}
            max={100}
            value={Number(params.C ?? 1)}
            onChange={(e) => onChange(method, "C", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50">
          Epsilon
          <input
            type="number"
            className="input mt-1"
            step={0.001}
            min={0.001}
            max={0.1}
            value={Number(params.epsilon ?? 0.01)}
            onChange={(e) => onChange(method, "epsilon", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-white/50 sm:col-span-2">
          Kernel
          <select
            className="input mt-1"
            value={String(params.kernel ?? "rbf")}
            onChange={(e) => onChange(method, "kernel", e.target.value)}
          >
            {["rbf", "linear", "poly", "sigmoid"].map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
      </div>
    );
  }
  if (method === "prophet") {
    return (
      <div className="pl-6 grid gap-2 sm:grid-cols-2 text-sm">
        <label className="text-xs text-white/50">
          Growth
          <select
            className="input mt-1"
            value={String(params.growth ?? "linear")}
            onChange={(e) => onChange(method, "growth", e.target.value)}
          >
            <option value="linear">linear</option>
            <option value="logistic">logistic</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-white/80 mt-6">
          <input
            type="checkbox"
            checked={params.seasonality !== false}
            onChange={(e) => onChange(method, "seasonality", e.target.checked)}
          />
          Seasonality
        </label>
        <label className="flex items-center gap-2 text-white/80">
          <input
            type="checkbox"
            checked={!!params.holidays}
            onChange={(e) => onChange(method, "holidays", e.target.checked)}
          />
          US holidays
        </label>
      </div>
    );
  }
  return <p className="pl-6 text-xs text-white/45">Default engine parameters (tune in Python if needed).</p>;
}
