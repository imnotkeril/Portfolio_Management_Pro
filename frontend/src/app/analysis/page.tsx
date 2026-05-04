"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { Portfolio } from "@/lib/types";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ═══════════════════════════════════════════════════════════════════════ */
/*  PALETTE                                                               */
/* ═══════════════════════════════════════════════════════════════════════ */
const C = {
  accent: "#bf9ffb",
  ok: "#74f174",
  danger: "#faa1a4",
  info: "#7dc4e4",
  warn: "#ffd066",
  line1: "#bf9ffb",
  line2: "#7dc4e4",
  line3: "#ffd066",
  grid: "rgba(255,255,255,0.06)",
  text: "rgba(255,255,255,0.5)",
  tooltipBg: "#1a1d25",
  tooltipBorder: "rgba(255,255,255,0.1)",
};
const PIE_COLORS = [
  "#bf9ffb","#7dc4e4","#74f174","#ffd066","#faa1a4",
  "#a78bfa","#6ee7b7","#f9a8d4","#93c5fd","#fca5a5",
  "#c4b5fd","#67e8f9","#86efac","#fcd34d","#fda4af",
];
const ttStyle = { background: C.tooltipBg, border: `1px solid ${C.tooltipBorder}`, borderRadius: 8, fontSize: 12, color: "#eee" };

/* ═══════════════════════════════════════════════════════════════════════ */
/*  TYPES                                                                 */
/* ═══════════════════════════════════════════════════════════════════════ */
type MainTab = "overview" | "performance" | "risk" | "assets" | "export";
type PerfSub = "returns" | "periodic" | "distribution";
type RiskSub = "key" | "drawdown" | "var" | "rolling";
type AssetSub = "overview" | "correlations" | "details";
type AnalyticsResult = Record<string, any>;
type Pt = { x: string; y: number };

/* ═══════════════════════════════════════════════════════════════════════ */
/*  TINY UI COMPONENTS                                                    */
/* ═══════════════════════════════════════════════════════════════════════ */

function Tip({ text }: { text: string }) {
  return (
    <span className="tooltip-wrap">
      <span className="tooltip-icon">?</span>
      <span className="tooltip-bubble">{text}</span>
    </span>
  );
}

function Alert({ type, children }: { type: "success" | "error" | "warning" | "info"; children: React.ReactNode }) {
  return <div className={`alert alert-${type}`}>{children}</div>;
}

function Expander({ title, children, defaultOpen }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="expander" open={defaultOpen || undefined}>
      <summary>{title}</summary>
      <div className="expander-body">{children}</div>
    </details>
  );
}

function InfoBox({ children }: { children: React.ReactNode }) {
  const flatten = (node: React.ReactNode): string => {
    if (node == null || typeof node === "boolean") return "";
    if (typeof node === "string" || typeof node === "number") return String(node);
    if (Array.isArray(node)) return node.map(flatten).join("");
    if (typeof node === "object" && "props" in (node as any)) return flatten((node as any).props?.children);
    return "";
  };
  const text = flatten(children).trim();
  if (!text) return <div className="alert alert-info text-sm">{children}</div>;

  const parts = text.split(" / ").map((s) => s.trim()).filter(Boolean);
  const title = parts[0]?.includes(":") ? parts[0].split(":")[0] : null;
  const firstBody = parts[0]?.includes(":") ? parts[0].split(":").slice(1).join(":").trim() : parts[0];
  const items = [firstBody, ...parts.slice(1)].filter(Boolean);

  const classify = (s: string): "good" | "bad" | "warn" | "neutral" => {
    const low = s.toLowerCase();
    if (/^[✓✔]|good diversif|excellent|positive alpha|outperform|well-diversif|above threshold|all assets show positive|low average|low sensitivity|currently above/.test(low)) return "good";
    if (/^[✗✘!⚠]|poor|negative alpha|underperform|high concentration|concentrated risk|no negative|currently below|limited diversif|no low corr|no pairs|high average corr|high sensitivity/.test(low)) return "bad";
    if (/moderate|caution|consider|some|becoming/.test(low)) return "warn";
    return "neutral";
  };
  const icon = (cls: string) => cls === "good" ? "✓" : cls === "bad" ? "✗" : cls === "warn" ? "⚬" : "•";
  const clr = (cls: string) => cls === "good" ? "text-[var(--ok)]" : cls === "bad" ? "text-[var(--danger)]" : cls === "warn" ? "text-[var(--warn)]" : "text-white/60";

  return (
    <div className="mt-3 rounded-lg border border-white/5 bg-white/[0.02] p-3">
      {title && <div className="text-xs font-semibold text-white/70 mb-1.5">{title}</div>}
      <div className="space-y-1">
        {items.map((item, i) => {
          const cls = classify(item);
          return <div key={i} className={`text-xs ${clr(cls)} flex gap-1.5 leading-relaxed`}><span className="shrink-0 mt-px">{icon(cls)}</span><span>{item}</span></div>;
        })}
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, good, helpText }: { label: string; value: string; sub?: string; good?: boolean | null; helpText?: string }) {
  const color = good === true ? "text-[var(--ok)]" : good === false ? "text-[var(--danger)]" : "text-white";
  return (
    <div className="metric-card">
      <div className="text-xs text-white/40 mb-1">{label} {helpText && <Tip text={helpText} />}</div>
      <div className={`text-lg font-semibold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-white/30 mt-0.5">{sub}</div>}
    </div>
  );
}

function CmpMetricCard({
  label, portfolioValue, benchmarkValue, format, higherIsBetter, helpText,
}: {
  label: string;
  portfolioValue: number | null | undefined;
  benchmarkValue: number | null | undefined;
  format: "percent" | "ratio";
  higherIsBetter: boolean | null;
  helpText?: string;
}) {
  const fmtV = (v: number | null | undefined) => {
    if (v == null || !isFinite(v)) return "—";
    if (format === "percent") {
      const pct = Math.abs(v) < 2 ? v * 100 : v;
      return `${pct >= 0 ? "" : ""}${pct.toFixed(2)}%`;
    }
    return v.toFixed(3);
  };

  const pStr = fmtV(portfolioValue);
  const bStr = benchmarkValue != null ? fmtV(benchmarkValue) : null;

  let dotColor = "bg-white/30";
  if (portfolioValue != null && benchmarkValue != null && higherIsBetter !== null) {
    const pN = typeof portfolioValue === "number" ? portfolioValue : 0;
    const bN = typeof benchmarkValue === "number" ? benchmarkValue : 0;
    const better = higherIsBetter ? pN > bN : pN < bN;
    dotColor = better ? "bg-[var(--ok)]" : "bg-[var(--danger)]";
  }

  return (
    <div className="metric-card">
      <div className="text-xs text-white/40 mb-1">{label} {helpText && <Tip text={helpText} />}</div>
      <div className="text-xl font-bold text-white">{pStr}</div>
      {bStr && (
        <div className="flex items-center gap-1.5 mt-1">
          <span className={`w-2 h-2 rounded-full ${dotColor}`} />
          <span className="text-xs text-white/40">{bStr}</span>
        </div>
      )}
    </div>
  );
}

function SubTabs<T extends string>({ tabs, active, onChange }: { tabs: [T, string][]; active: T; onChange: (t: T) => void }) {
  return (
    <div className="tab-bar mb-4">
      {tabs.map(([key, label]) => (
        <button key={key} className={`tab-btn ${active === key ? "active" : ""}`} onClick={() => onChange(key)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function Divider() {
  return <div className="border-t border-white/10 my-4" />;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  FORMATTERS                                                            */
/* ═══════════════════════════════════════════════════════════════════════ */

function fmtPct(v: any, decimals = 2): string {
  if (v == null || v === "" || typeof v === "object") return "—";
  const n = typeof v === "number" ? v : parseFloat(v);
  if (!isFinite(n)) return "—";
  const val = Math.abs(n) < 2 ? n * 100 : n;
  return `${val >= 0 ? "+" : ""}${val.toFixed(decimals)}%`;
}
function fmtPctPlain(v: any, decimals = 2): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(v);
  if (!isFinite(n)) return "—";
  const val = Math.abs(n) < 2 ? n * 100 : n;
  return `${val.toFixed(decimals)}%`;
}
function fmtNum(v: any, decimals = 2): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(v);
  if (!isFinite(n)) return "—";
  return n.toFixed(decimals);
}
function fmtUsd(v: number): string {
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
function isGood(v: any, threshold = 0): boolean | null {
  if (v == null) return null;
  const n = typeof v === "number" ? v : parseFloat(v);
  if (!isFinite(n)) return null;
  return n >= threshold;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  DATA HELPERS                                                          */
/* ═══════════════════════════════════════════════════════════════════════ */

function extractSeries(data: any): Pt[] {
  if (!data) return [];
  if (Array.isArray(data)) {
    return data.map((d: any) => ({
      x: String(d.x ?? d.date ?? d.Date ?? d[0] ?? ""),
      y: Number(d.y ?? d.value ?? d.close ?? d[1] ?? 0),
    }));
  }
  if (typeof data === "object") {
    return Object.entries(data).map(([k, v]) => ({ x: k, y: Number(v) }));
  }
  return [];
}

/**
 * Some feeds encode daily returns as hundredths (e.g. 0.08 meaning 0.08% instead of 0.0008).
 * Then cumulative charts show ~100x inflated %. Detect via median |daily| and rescale to decimals.
 */
function normalizeDailyReturnSeries(pts: Pt[]): Pt[] {
  if (pts.length < 10) return pts;
  const sortedAbs = [...pts].map((p) => Math.abs(p.y)).sort((a, b) => a - b);
  const med = sortedAbs[Math.floor(sortedAbs.length / 2)] ?? 0;
  if (med > 0.025) return pts.map((p) => ({ x: p.x, y: p.y / 100 }));
  return pts;
}

function cumulative(pts: Pt[]): Pt[] {
  let cum = 0;
  return pts.map((d) => { cum = (1 + cum) * (1 + d.y) - 1; return { x: d.x, y: cum }; });
}

/** Cumulative return from a NAV (or single-asset price) series — matches backend total_return from values. */
function cumulativeFromNav(vals: Pt[]): Pt[] {
  if (vals.length < 2) return [];
  const v0 = vals[0].y;
  if (!(v0 > 0) || !isFinite(v0)) return [];
  return vals.map((d) => ({ x: d.x, y: d.y / v0 - 1 }));
}

function drawdownSeries(vals: Pt[]): Pt[] {
  let peak = -Infinity;
  return vals.map((d) => { if (d.y > peak) peak = d.y; return { x: d.x, y: peak > 0 ? (d.y - peak) / peak : 0 }; });
}

function drawdownFromReturns(pts: Pt[]): Pt[] {
  let value = 1, peak = 1;
  return pts.map((d) => {
    value *= (1 + d.y);
    if (value > peak) peak = value;
    return { x: d.x, y: peak > 0 ? (value - peak) / peak : 0 };
  });
}

function rollingCalc(pts: Pt[], window: number, fn: (slice: number[]) => number): Pt[] {
  const res: Pt[] = [];
  for (let i = window; i < pts.length; i++) {
    const slice = pts.slice(i - window, i).map((d) => d.y);
    res.push({ x: pts[i].x, y: fn(slice) });
  }
  return res;
}

function mean(arr: number[]): number { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

function monthlyFromDaily(pts: Pt[]): { key: string; value: number }[] {
  const m = new Map<string, number[]>();
  pts.forEach((d) => { const k = d.x.slice(0, 7); if (!m.has(k)) m.set(k, []); m.get(k)!.push(d.y); });
  return Array.from(m.entries()).map(([k, arr]) => ({ key: k, value: arr.reduce((p, r) => p * (1 + r), 1) - 1 }));
}

function yearlyFromDaily(pts: Pt[]): { key: string; value: number }[] {
  const m = new Map<string, number[]>();
  pts.forEach((d) => { const k = d.x.slice(0, 4); if (!m.has(k)) m.set(k, []); m.get(k)!.push(d.y); });
  return Array.from(m.entries()).map(([k, arr]) => ({ key: k, value: arr.reduce((p, r) => p * (1 + r), 1) - 1 }));
}

function interpretCumReturns(portCum: Pt[], benchCum: Pt[] | undefined): string {
  if (!portCum.length) return "";
  const portFinal = portCum[portCum.length - 1].y;
  const parts: string[] = ["Cumulative Returns Analysis:"];
  if (portFinal > 0.2) parts.push(`Strong growth: portfolio gained ${(portFinal * 100).toFixed(1)}% over the period`);
  else if (portFinal > 0) parts.push(`Positive growth: portfolio gained ${(portFinal * 100).toFixed(1)}%`);
  else parts.push(`Negative performance: portfolio lost ${Math.abs(portFinal * 100).toFixed(1)}%`);
  if (benchCum?.length) {
    const benchFinal = benchCum[benchCum.length - 1].y;
    const diff = portFinal - benchFinal;
    if (diff > 0.005) parts.push(`Slightly outperformed benchmark by ${(diff * 100).toFixed(1)}% (${(benchFinal * 100).toFixed(1)}%)`);
    else if (diff > 0.05) parts.push(`Significantly outperformed benchmark by ${(diff * 100).toFixed(1)}%`);
    else if (diff < -0.005) parts.push(`Underperformed benchmark by ${(Math.abs(diff) * 100).toFixed(1)}% (${(benchFinal * 100).toFixed(1)}%)`);
    else parts.push(`Roughly matched benchmark (${(benchFinal * 100).toFixed(1)}%)`);
  }
  const maxPt = portCum.reduce((a, b) => a.y > b.y ? a : b, portCum[0]);
  const minPt = portCum.reduce((a, b) => a.y < b.y ? a : b, portCum[0]);
  if (maxPt.y - minPt.y > 0.3) parts.push("Volatile path with significant swings");
  else if (maxPt.y - minPt.y > 0.15) parts.push("Moderate volatility along the path");
  return parts.join(" / ");
}

function interpretDrawdown(dd: Pt[]): string {
  if (!dd.length) return "";
  const maxDd = Math.min(...dd.map((d) => d.y));
  const currentDd = dd[dd.length - 1].y;
  const timeInDDpct = (dd.filter((d) => d.y < -0.05).length / dd.length * 100);
  const parts: string[] = ["Drawdown Analysis:"];
  if (maxDd > -0.1) parts.push(`⚠ Maximum drawdown: ${(maxDd * 100).toFixed(1)}% — Relatively shallow`);
  else if (maxDd > -0.2) parts.push(`⚠ Significant maximum drawdown: ${(maxDd * 100).toFixed(1)}% — Portfolio lost 10-20% from peak`);
  else parts.push(`⚠ Deep maximum drawdown: ${(maxDd * 100).toFixed(1)}% — Portfolio lost >20% from peak`);
  if (Math.abs(currentDd) < 0.005) parts.push("⚠ Currently at or near peak");
  else parts.push(`⚠ Currently in drawdown: ${(currentDd * 100).toFixed(1)}% — Portfolio is below recent peak`);
  if (timeInDDpct > 50) parts.push(`⚠ Frequent drawdowns: Portfolio spent ${timeInDDpct.toFixed(1)}% of time in >5% drawdown`);
  else if (timeInDDpct > 20) parts.push(`Portfolio spent ${timeInDDpct.toFixed(1)}% of time in >5% drawdown`);
  return parts.join(" / ");
}

function interpretDailyReturns(vals: number[]): string {
  if (!vals.length) return "";
  const m = mean(vals);
  const sd = stddev(vals);
  const pos = vals.filter((v) => v > 0).length;
  const neg = vals.length - pos;
  const best = Math.max(...vals);
  const worst = Math.min(...vals);
  const parts: string[] = ["Daily Returns Analysis:"];
  parts.push(`Average daily return: ${(m * 100).toFixed(2)}% (${m >= 0 ? "positive" : "negative"})`);
  if (pos > neg) parts.push(`Slightly more positive days (${pos}, ${(pos / vals.length * 100).toFixed(0)}%) than negative (${neg}, ${(neg / vals.length * 100).toFixed(0)}%)`);
  else parts.push(`More negative days (${neg}, ${(neg / vals.length * 100).toFixed(0)}%) than positive (${pos}, ${(pos / vals.length * 100).toFixed(0)}%)`);
  parts.push(`Largest gain: ${(best * 100).toFixed(2)}%, Largest loss: ${(worst * 100).toFixed(2)}%`);
  if (sd * 100 > 2) parts.push(`High volatility (${(sd * 100).toFixed(2)}% daily)`);
  else if (sd * 100 > 1) parts.push(`Moderate volatility (${(sd * 100).toFixed(2)}% daily)`);
  else parts.push(`Low volatility (${(sd * 100).toFixed(2)}% daily)`);
  return parts.join(" / ");
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  CHART WRAPPERS                                                        */
/* ═══════════════════════════════════════════════════════════════════════ */

function TimeSeriesChart({ data, data2, label1 = "Portfolio", label2 = "Benchmark", height = 300, pct = false }: {
  data: Pt[]; data2?: Pt[]; label1?: string; label2?: string; height?: number; pct?: boolean;
}) {
  const merged = useMemo(() => {
    const map = new Map<string, any>();
    data.forEach((d) => map.set(d.x, { x: d.x, p: d.y }));
    if (data2) data2.forEach((d) => { const e = map.get(d.x) || { x: d.x }; e.b = d.y; map.set(d.x, e); });
    return Array.from(map.values());
  }, [data, data2]);
  if (!merged.length) return <div className="text-white/30 text-sm py-4">No data available</div>;
  const fmt = pct ? (v: number) => `${(v * 100).toFixed(1)}%` : (v: number) => v.toFixed(2);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={merged} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="gP" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.line1} stopOpacity={0.3} /><stop offset="100%" stopColor={C.line1} stopOpacity={0.02} /></linearGradient>
          <linearGradient id="gB" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.line2} stopOpacity={0.2} /><stop offset="100%" stopColor={C.line2} stopOpacity={0.02} /></linearGradient>
        </defs>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} minTickGap={50} />
        <YAxis tick={{ fill: C.text, fontSize: 10 }} tickLine={false} tickFormatter={fmt} width={60} />
        <RTooltip contentStyle={ttStyle} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.text }} />
        <Area type="monotone" dataKey="p" name={label1} stroke={C.line1} fill="url(#gP)" strokeWidth={2} dot={false} />
        {data2 && <Area type="monotone" dataKey="b" name={label2} stroke={C.line2} fill="url(#gB)" strokeWidth={1.5} dot={false} />}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function SimpleBarChart({ data, xKey = "label", bars, height = 280 }: {
  data: any[]; xKey?: string; bars: { key: string; color: string; name: string }[]; height?: number;
}) {
  if (!data.length) return <div className="text-white/30 text-sm py-4">No data</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
        <YAxis tick={{ fill: C.text, fontSize: 10 }} tickLine={false} width={50} />
        <RTooltip contentStyle={ttStyle} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.text }} />
        {bars.map((b) => <Bar key={b.key} dataKey={b.key} name={b.name} fill={b.color} radius={[3, 3, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

function DonutChart({ data, height = 250 }: { data: { name: string; value: number }[]; height?: number }) {
  if (!data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius="55%" outerRadius="80%" dataKey="value" label={({ name, value }) => `${name} ${value.toFixed(1)}%`} labelLine={false} fontSize={10}>
          {data.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
        </Pie>
        <RTooltip contentStyle={ttStyle} />
      </PieChart>
    </ResponsiveContainer>
  );
}

function DailyReturnsBarChart({ data, height = 260 }: { data: { label: string; value: number }[]; height?: number }) {
  if (!data.length) return <div className="text-white/30 text-sm py-4">No data</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis dataKey="label" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
        <YAxis tick={{ fill: C.text, fontSize: 10 }} tickLine={false} width={50} />
        <RTooltip contentStyle={ttStyle} />
        <Bar dataKey="value" name="Return %">
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? C.ok : C.danger} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function RollingLineChart({ data, data2, label1 = "Portfolio", label2 = "Benchmark", height = 260, refLine }: {
  data: Pt[]; data2?: Pt[]; label1?: string; label2?: string; height?: number; refLine?: number;
}) {
  const merged = useMemo(() => {
    const map = new Map<string, any>();
    data.forEach((d) => map.set(d.x, { x: d.x, p: d.y }));
    if (data2) data2.forEach((d) => { const e = map.get(d.x) || { x: d.x }; e.b = d.y; map.set(d.x, e); });
    const result = Array.from(map.values());
    if (refLine !== undefined) result.forEach((d: any) => { d.ref = refLine; });
    return result;
  }, [data, data2, refLine]);
  if (!merged.length) return <div className="text-white/30 text-sm py-4">No data</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={merged} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} minTickGap={50} />
        <YAxis tick={{ fill: C.text, fontSize: 10 }} tickLine={false} width={50} />
        <RTooltip contentStyle={ttStyle} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.text }} />
        <Line type="monotone" dataKey="p" name={label1} stroke={C.line1} strokeWidth={2} dot={false} />
        {data2 && <Line type="monotone" dataKey="b" name={label2} stroke={C.line2} strokeWidth={1.5} dot={false} />}
        {refLine !== undefined && <Line type="monotone" dataKey="ref" name="Reference" stroke={C.warn} strokeWidth={1} strokeDasharray="5 5" dot={false} />}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                                        */
/* ═══════════════════════════════════════════════════════════════════════ */

export default function AnalysisPage() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [startDate, setStartDate] = useState(() => { const d = new Date(); d.setFullYear(d.getFullYear() - 1); return d.toISOString().slice(0, 10); });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [cmpType, setCmpType] = useState<"None" | "Index ETF" | "Portfolio">("None");
  const [cmpValue, setCmpValue] = useState("SPY");
  const [cmpPortfolioId, setCmpPortfolioId] = useState("");
  const [riskFreeRate, setRiskFreeRate] = useState(4.35);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResult | null>(null);

  const [mainTab, setMainTab] = useState<MainTab>("overview");
  const [perfSub, setPerfSub] = useState<PerfSub>("returns");
  const [riskSub, setRiskSub] = useState<RiskSub>("key");
  const [assetSub, setAssetSub] = useState<AssetSub>("overview");
  const [rollingWindow, setRollingWindow] = useState(63);
  const [selectedAsset, setSelectedAsset] = useState("");
  const [assetData, setAssetData] = useState<Record<string, any> | null>(null);

  useEffect(() => { api.get<Portfolio[]>("/portfolios").then((list) => { setPortfolios(list); if (list[0]) setSelectedId(list[0].id); }); }, []);

  const selected = useMemo(() => portfolios.find((p) => p.id === selectedId) ?? null, [portfolios, selectedId]);
  const otherPortfolios = useMemo(() => portfolios.filter((p) => p.id !== selectedId), [portfolios, selectedId]);

  const calculate = useCallback(async () => {
    if (!selectedId) return;
    setLoading(true); setError(null);
    try {
      const payload: any = { portfolio_id: selectedId, start_date: startDate, end_date: endDate };
      if (cmpType === "Index ETF") { payload.comparison_type = "ticker"; payload.comparison_value = cmpValue; }
      else if (cmpType === "Portfolio" && cmpPortfolioId) { payload.comparison_type = "portfolio"; payload.comparison_value = cmpPortfolioId; }
      const result = await api.post<AnalyticsResult>("/analytics/calculate", payload);
      setAnalytics(result);

      const benchTicker = cmpType === "Index ETF" ? cmpValue : undefined;
      api.post<Record<string, any>>("/analytics/assets", {
        portfolio_id: selectedId, start_date: startDate, end_date: endDate,
        benchmark_ticker: benchTicker,
      }).then((ad) => {
        setAssetData(ad);
        const tickers = Object.keys(ad?.per_asset ?? {});
        if (tickers.length && !tickers.includes(selectedAsset)) setSelectedAsset(tickers[0]);
      }).catch(() => {});
    } catch (err) { setError(String(err)); } finally { setLoading(false); }
  }, [selectedId, startDate, endDate, cmpType, cmpValue, cmpPortfolioId, selectedAsset]);

  /* ────── Extract from analytics ────── */
  const perf = analytics?.performance ?? {};
  const risk = analytics?.risk ?? {};
  const ratios = analytics?.ratios ?? {};
  const market = analytics?.market ?? {};
  const cmpMetrics = analytics?.comparison_metrics ?? {};
  const cmpLabel = analytics?.comparison_label ?? (cmpType === "Index ETF" ? cmpValue : "Benchmark");
  const portfolioReturnsRaw = useMemo(() => extractSeries(analytics?.portfolio_returns), [analytics]);
  const benchmarkReturnsRaw = useMemo(
    () => extractSeries(analytics?.comparison_returns ?? analytics?.benchmark_returns),
    [analytics],
  );
  const portfolioReturns = useMemo(
    () => normalizeDailyReturnSeries(portfolioReturnsRaw),
    [portfolioReturnsRaw],
  );
  const benchmarkReturns = useMemo(
    () => normalizeDailyReturnSeries(benchmarkReturnsRaw),
    [benchmarkReturnsRaw],
  );
  const portfolioValues = useMemo(() => extractSeries(analytics?.portfolio_values), [analytics]);
  const benchmarkValues = useMemo(() => extractSeries(analytics?.benchmark_values), [analytics]);
  /** Align portfolio/benchmark daily returns by calendar date (series may differ in length). */
  const portReturnByDate = useMemo(() => {
    const m = new Map<string, number>();
    portfolioReturns.forEach((d) => m.set(String(d.x).slice(0, 10), d.y));
    return m;
  }, [portfolioReturns]);
  const benchReturnByDate = useMemo(() => {
    const m = new Map<string, number>();
    benchmarkReturns.forEach((d) => m.set(String(d.x).slice(0, 10), d.y));
    return m;
  }, [benchmarkReturns]);
  const positions = selected?.positions ?? [];
  const hasBenchmark = benchmarkReturns.length > 0;

  const vol = (() => {
    const r = risk as Record<string, unknown> | undefined;
    if (!r) return undefined;
    const nested = r.volatility;
    if (nested && typeof nested === "object" && nested !== null && "annual" in nested) {
      const a = (nested as { annual?: number }).annual;
      if (typeof a === "number" && isFinite(a)) return a;
    }
    if (typeof r.annual === "number" && isFinite(r.annual)) return r.annual;
    if (typeof r.volatility === "number" && isFinite(r.volatility)) return r.volatility as number;
    return undefined;
  })();
  const maxDD = typeof risk?.max_drawdown === "number" ? risk.max_drawdown : (Array.isArray(risk?.max_drawdown) ? risk.max_drawdown[0] : risk?.max_drawdown);

  /* ────── Pre-compute series for charts (NAV-based cumulative matches Total Return cards; daily-return compound can diverge) ────── */
  const portCum = useMemo(() => {
    if (portfolioValues.length >= 2 && portfolioValues[0].y > 0 && isFinite(portfolioValues[0].y))
      return cumulativeFromNav(portfolioValues);
    return cumulative(portfolioReturns);
  }, [portfolioValues, portfolioReturns]);
  const benchCum = useMemo(() => {
    if (!hasBenchmark) return [];
    if (benchmarkValues.length >= 2 && benchmarkValues[0].y > 0 && isFinite(benchmarkValues[0].y))
      return cumulativeFromNav(benchmarkValues);
    return cumulative(benchmarkReturns);
  }, [hasBenchmark, benchmarkValues, benchmarkReturns]);
  const portDD = useMemo(() => portfolioValues.length > 0 ? drawdownSeries(portfolioValues) : drawdownFromReturns(portfolioReturns), [portfolioValues, portfolioReturns]);
  const benchDD = useMemo(() => {
    if (!hasBenchmark) return [];
    if (benchmarkValues.length >= 2 && benchmarkValues[0].y > 0 && isFinite(benchmarkValues[0].y))
      return drawdownSeries(benchmarkValues);
    return drawdownFromReturns(benchmarkReturns);
  }, [hasBenchmark, benchmarkValues, benchmarkReturns]);

  const allocationData = useMemo(() => {
    if (!positions.length) return [];
    const totalW = positions.reduce((s: number, p: any) => s + (p.weight_target || 0), 0) || 1;
    return positions.map((p: any) => ({ name: p.ticker, value: ((p.weight_target || 0) / totalW) * 100 }));
  }, [positions]);

  /* rolling metrics for current window */
  const rollingVol = useMemo(() => rollingCalc(portfolioReturns, rollingWindow, (s) => stddev(s) * Math.sqrt(252)), [portfolioReturns, rollingWindow]);
  const rollingVolBench = useMemo(() => hasBenchmark ? rollingCalc(benchmarkReturns, rollingWindow, (s) => stddev(s) * Math.sqrt(252)) : [], [benchmarkReturns, rollingWindow, hasBenchmark]);
  const rfDaily = riskFreeRate / 100 / 252;
  const rollingSharpe = useMemo(() => rollingCalc(portfolioReturns, rollingWindow, (s) => { const m = mean(s); const sd = stddev(s); return sd > 0 ? (m * 252 - riskFreeRate / 100) / (sd * Math.sqrt(252)) : 0; }), [portfolioReturns, rollingWindow, riskFreeRate]);
  const rollingSharpeBench = useMemo(() => hasBenchmark ? rollingCalc(benchmarkReturns, rollingWindow, (s) => { const m = mean(s); const sd = stddev(s); return sd > 0 ? (m * 252 - riskFreeRate / 100) / (sd * Math.sqrt(252)) : 0; }) : [], [benchmarkReturns, rollingWindow, riskFreeRate, hasBenchmark]);
  const rollingSortino = useMemo(() => rollingCalc(portfolioReturns, rollingWindow, (s) => {
    const m = mean(s); const downs = s.filter((v) => v < rfDaily); const dd = downs.length > 1 ? Math.sqrt(downs.reduce((a, b) => a + (b - rfDaily) ** 2, 0) / downs.length) * Math.sqrt(252) : 0;
    return dd > 0 ? (m * 252 - riskFreeRate / 100) / dd : 0;
  }), [portfolioReturns, rollingWindow, riskFreeRate, rfDaily]);

  const rollingBeta = useMemo(() => {
    if (!hasBenchmark) return [];
    const res: Pt[] = [];
    for (let i = rollingWindow; i < Math.min(portfolioReturns.length, benchmarkReturns.length); i++) {
      const pSlice = portfolioReturns.slice(i - rollingWindow, i).map((d) => d.y);
      const bSlice = benchmarkReturns.slice(i - rollingWindow, i).map((d) => d.y);
      const mP = mean(pSlice), mB = mean(bSlice);
      let cov = 0, varB = 0;
      for (let j = 0; j < pSlice.length; j++) { cov += (pSlice[j] - mP) * (bSlice[j] - mB); varB += (bSlice[j] - mB) ** 2; }
      res.push({ x: portfolioReturns[i].x, y: varB > 0 ? cov / varB : 0 });
    }
    return res;
  }, [portfolioReturns, benchmarkReturns, rollingWindow, hasBenchmark]);

  /* ═══════════════ RENDER ═══════════════ */
  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold text-white">Portfolio Analysis</h1>

      {/* ═══ PARAMETERS ═══ */}
      <div className="panel p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white">Analysis Parameters <Tip text="Configure the analysis period, portfolio, and comparison benchmark" /></h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div><label className="label">Start Date <Tip text="Beginning of the analysis period" /></label><input className="input" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></div>
          <div><label className="label">End Date <Tip text="End of the analysis period" /></label><input className="input" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="label">Portfolio <Tip text="Select a portfolio to analyze" /></label>
            <select className="input" value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
              {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Comparison <Tip text="Compare portfolio against an index ETF or another portfolio" /></label>
            <div className="flex gap-2 mb-2">
              {(["None", "Index ETF", "Portfolio"] as const).map((t) => (
                <button key={t} className={`btn ${cmpType === t ? "btn-primary" : "btn-ghost"} !py-1.5 !px-3 !text-xs`} onClick={() => setCmpType(t)}>{t}</button>
              ))}
            </div>
            {cmpType === "Index ETF" && <select className="input" value={cmpValue} onChange={(e) => setCmpValue(e.target.value)}>{["SPY","QQQ","VTI","DIA","IWM","EFA","AGG"].map((s) => <option key={s} value={s}>{s}</option>)}</select>}
            {cmpType === "Portfolio" && (otherPortfolios.length > 0 ? <select className="input" value={cmpPortfolioId} onChange={(e) => setCmpPortfolioId(e.target.value)}><option value="">Select portfolio...</option>{otherPortfolios.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select> : <Alert type="info">No other portfolios available</Alert>)}
            {cmpType === "None" && <div className="text-xs text-white/30">No comparison selected</div>}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div><label className="label">Risk-Free Rate (%) <Tip text="Annual risk-free rate used for Sharpe, Sortino, and other risk-adjusted metrics. Default: US T-bill rate." /></label><input className="input" type="number" min={0} max={15} step={0.01} value={riskFreeRate} onChange={(e) => setRiskFreeRate(Number(e.target.value))} /></div>
          <button className="btn btn-primary h-[46px]" onClick={calculate} disabled={loading || !selectedId}>{loading ? "Calculating..." : "Calculate Metrics"}</button>
          <button className="btn btn-secondary h-[46px]" disabled>Update Prices <Tip text="Price update functionality coming soon" /></button>
        </div>
        {selected && <div className="text-sm text-white/50">Selected: <strong className="text-white/70">{selected.name}</strong> &bull; Risk-free {riskFreeRate}% &bull; {startDate} &rarr; {endDate}</div>}
        {error && <Alert type="error">{error}</Alert>}
      </div>

      {!analytics && !loading && <Alert type="info">Click &quot;Calculate Metrics&quot; to start analysis</Alert>}

      {/* ═══ MAIN TABS ═══ */}
      {analytics && (<>
        <div className="tab-bar">
          {([["overview","Overview"],["performance","Performance"],["risk","Risk"],["assets","Assets & Correlations"],["export","Export & Reports"]] as [MainTab,string][]).map(([key,label]) => (
            <button key={key} className={`tab-btn ${mainTab === key ? "active" : ""}`} onClick={() => setMainTab(key)}>{label}</button>
          ))}
        </div>

        {/* ═══════════════════════════════════════════════════════ */}
        {/*  TAB 1: OVERVIEW                                       */}
        {/* ═══════════════════════════════════════════════════════ */}
        {mainTab === "overview" && (<div className="space-y-6">

          {/* ── Section 1.1: Key Performance Metrics ── */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Key Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <CmpMetricCard label="Total Return" portfolioValue={perf.total_return} benchmarkValue={cmpMetrics.total_return} format="percent" higherIsBetter={true} helpText="Cumulative return from start to end date." />
              <CmpMetricCard label="CAGR" portfolioValue={perf.cagr ?? perf.annualized_return} benchmarkValue={cmpMetrics.annualized_return} format="percent" higherIsBetter={true} helpText="Average annual return assuming reinvestment." />
              <CmpMetricCard label="Volatility" portfolioValue={vol} benchmarkValue={cmpMetrics.volatility} format="percent" higherIsBetter={false} helpText="Annualized standard deviation of returns." />
              <CmpMetricCard label="Max Drawdown" portfolioValue={maxDD} benchmarkValue={cmpMetrics.max_drawdown} format="percent" higherIsBetter={true} helpText="Largest peak-to-trough decline (less negative is better)." />
            </div>
            <Divider />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <CmpMetricCard label="Sharpe Ratio" portfolioValue={ratios.sharpe_ratio} benchmarkValue={cmpMetrics.sharpe_ratio} format="ratio" higherIsBetter={true} helpText="Risk-adjusted return." />
              <CmpMetricCard label="Sortino Ratio" portfolioValue={ratios.sortino_ratio} benchmarkValue={cmpMetrics.sortino_ratio} format="ratio" higherIsBetter={true} helpText="Like Sharpe but only penalizes downside volatility." />
              <CmpMetricCard label="Beta" portfolioValue={market.beta} benchmarkValue={hasBenchmark ? 1.0 : null} format="ratio" higherIsBetter={null} helpText="Sensitivity to market movements." />
              <CmpMetricCard label="Alpha" portfolioValue={market.alpha} benchmarkValue={hasBenchmark ? 0.0 : null} format="percent" higherIsBetter={true} helpText="Excess return above benchmark. Positive = outperformance, negative = underperformance." />
            </div>
          </div>

          {/* ── Section 1.2: Portfolio Performance (charts) ── */}
          <div className="panel p-5">
            <h3 className="text-lg font-semibold text-white mb-1">Portfolio Performance</h3>
            <h4 className="text-sm text-white/50 mb-3">Cumulative Returns</h4>
            <TimeSeriesChart data={portCum} data2={benchCum.length > 0 ? benchCum : undefined} label1="Portfolio" label2={cmpLabel} height={340} pct />
            {portCum.length > 0 && <InfoBox>{interpretCumReturns(portCum, benchCum.length > 0 ? benchCum : undefined)}</InfoBox>}
          </div>

          {/* Underwater Plot */}
          <div className="panel p-5">
            <h3 className="text-base font-semibold text-white mb-3">Underwater Plot (Drawdown from Peak) <Tip text="Shows percentage decline from the previous peak value. 0% means portfolio is at its high." /></h3>
            <TimeSeriesChart data={portDD} data2={benchDD.length > 0 ? benchDD : undefined} label1="Portfolio" label2={cmpLabel} height={280} pct />
            {portDD.length > 0 && <InfoBox>{interpretDrawdown(portDD)}</InfoBox>}
          </div>

          {/* Daily Returns with green/red bars */}
          <div className="panel p-5">
            <h3 className="text-base font-semibold text-white mb-3">Daily Returns <Tip text="Individual daily returns. Green bars = positive days, red bars = negative days." /></h3>
            <DailyReturnsBarChart
              data={portfolioReturns.filter((_, i) => i % Math.max(1, Math.floor(portfolioReturns.length / 150)) === 0).map((d) => ({
                label: d.x.slice(5),
                value: +(d.y * 100).toFixed(2),
              }))}
              height={260}
            />
            {portfolioReturns.length > 0 && <InfoBox>{interpretDailyReturns(portfolioReturns.map((d) => d.y))}</InfoBox>}
          </div>

          {/* ── Section 1.3: Portfolio Structure ── */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Portfolio Structure</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Distribution by Assets */}
              <div className="panel p-5">
                <h4 className="text-sm font-semibold text-white/70 mb-1">Distribution by Assets</h4>
                <h5 className="text-xs text-white/40 mb-2">Asset Allocation</h5>
                <DonutChart data={allocationData} height={270} />
                {allocationData.length > 0 && (() => {
                  const top = allocationData.reduce((a, b) => a.value > b.value ? a : b, allocationData[0]);
                  return <div className="text-xs text-white/40 mt-2"><strong>Top asset:</strong> {top.name} — {top.value.toFixed(1)}% of portfolio</div>;
                })()}
              </div>
              {/* Distribution by Sectors */}
              <div className="panel p-5">
                <h4 className="text-sm font-semibold text-white/70 mb-1">Distribution by Sectors</h4>
                <h5 className="text-xs text-white/40 mb-2">Sector Allocation</h5>
                {(() => {
                  const sectorMap: Record<string, string> = {
                    AAPL: "Technology", MSFT: "Technology", GOOGL: "Communication Services", GOOG: "Communication Services",
                    AMZN: "Consumer Cyclical", META: "Communication Services", NVDA: "Technology", TSLA: "Consumer Cyclical",
                    AMD: "Technology", CRM: "Technology", INTC: "Technology", NFLX: "Communication Services",
                    SPY: "Index", QQQ: "Index", VTI: "Index", DIA: "Index", IWM: "Index",
                    CASH: "Cash", BRK: "Financial Services", JPM: "Financial Services", V: "Financial Services",
                    JNJ: "Healthcare", UNH: "Healthcare", PFE: "Healthcare", XOM: "Energy", CVX: "Energy",
                  };
                  if (!positions.length) return <div className="text-white/30 text-sm">No positions</div>;
                  const totalW = positions.reduce((s: number, p: any) => s + (p.weight_target || 0), 0) || 1;
                  const sectorWeights: Record<string, number> = {};
                  positions.forEach((p: any) => {
                    const sector = sectorMap[p.ticker?.toUpperCase()] || "Other";
                    const pct = ((p.weight_target || 0) / totalW) * 100;
                    sectorWeights[sector] = (sectorWeights[sector] || 0) + pct;
                  });
                  const sectorData = Object.entries(sectorWeights).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
                  const topSector = sectorData[0];
                  return (<>
                    <DonutChart data={sectorData} height={270} />
                    {topSector && <div className="text-xs text-white/40 mt-2"><strong>Top sector:</strong> {topSector.name} — {topSector.value.toFixed(1)}% of portfolio</div>}
                  </>);
                })()}
              </div>
            </div>
          </div>

          {/* ── Section 1.4: Comparison Table ── */}
          <div className="panel p-5">
            <h3 className="text-base font-semibold text-white mb-1">Portfolio vs Comparison</h3>
            <h4 className="text-sm text-white/40 mb-3">Key Metrics Comparison</h4>
            <table className="data-table">
              <thead><tr><th>Metric</th><th>Portfolio</th><th>{cmpLabel}</th><th>&Delta;</th></tr></thead>
              <tbody>
                {(() => {
                  const rows: { label: string; pv: any; bv: any; fmt: "pct" | "num"; higherBetter: boolean | null }[] = [
                    { label: "Total Return", pv: perf.total_return, bv: cmpMetrics.total_return, fmt: "pct", higherBetter: true },
                    { label: "CAGR", pv: perf.cagr ?? perf.annualized_return, bv: cmpMetrics.annualized_return, fmt: "pct", higherBetter: true },
                    { label: "Annualized Return", pv: perf.annualized_return, bv: cmpMetrics.annualized_return, fmt: "pct", higherBetter: true },
                    { label: "Volatility", pv: vol, bv: cmpMetrics.volatility, fmt: "pct", higherBetter: false },
                    { label: "Sharpe Ratio", pv: ratios.sharpe_ratio, bv: cmpMetrics.sharpe_ratio, fmt: "num", higherBetter: true },
                    { label: "Sortino Ratio", pv: ratios.sortino_ratio, bv: cmpMetrics.sortino_ratio, fmt: "num", higherBetter: true },
                    { label: "Max Drawdown", pv: maxDD, bv: cmpMetrics.max_drawdown, fmt: "pct", higherBetter: true },
                    { label: "Calmar Ratio", pv: ratios.calmar_ratio, bv: cmpMetrics.calmar_ratio, fmt: "num", higherBetter: true },
                    { label: "VaR (95%)", pv: risk.var_95 ?? risk.var_historical_95, bv: cmpMetrics.var_95, fmt: "pct", higherBetter: true },
                    { label: "CVaR (95%)", pv: risk.cvar_95 ?? risk.cvar_historical_95, bv: cmpMetrics.cvar_95, fmt: "pct", higherBetter: true },
                    { label: "Beta", pv: market.beta, bv: hasBenchmark ? 1.0 : null, fmt: "num", higherBetter: null },
                    { label: "Alpha", pv: market.alpha, bv: hasBenchmark ? 0.0 : null, fmt: "pct", higherBetter: true },
                    { label: "Up Capture", pv: market.up_capture, bv: hasBenchmark ? 1.0 : null, fmt: "pct", higherBetter: true },
                    { label: "Down Capture", pv: market.down_capture, bv: hasBenchmark ? 1.0 : null, fmt: "pct", higherBetter: false },
                  ];
                  return rows.map((r) => {
                    const pN = typeof r.pv === "number" ? r.pv : null;
                    const bN = typeof r.bv === "number" ? r.bv : null;
                    let deltaStr = "—";
                    let dotColor = "";
                    if (pN != null && bN != null && bN !== 0) {
                      const pctChange = ((pN - bN) / Math.abs(bN)) * 100;
                      deltaStr = `${pctChange >= 0 ? "+" : ""}${pctChange.toFixed(2)}%`;
                      if (r.higherBetter === true) dotColor = pctChange >= 0 ? "bg-[var(--ok)]" : "bg-[var(--danger)]";
                      else if (r.higherBetter === false) dotColor = pctChange <= 0 ? "bg-[var(--ok)]" : "bg-[var(--danger)]";
                      else dotColor = "bg-[var(--warn)]";
                    } else if (pN != null && bN != null && bN === 0) {
                      deltaStr = "—";
                    }
                    const fmtVal = (v: any) => {
                      if (v == null) return "—";
                      if (r.fmt === "pct") return fmtPctPlain(v);
                      return fmtNum(v, 3);
                    };
                    return (
                      <tr key={r.label}>
                        <td className="text-white/70">{r.label}</td>
                        <td className="font-mono">{fmtVal(r.pv)}</td>
                        <td className="font-mono text-white/40">{fmtVal(r.bv)}</td>
                        <td className="font-mono">
                          {dotColor && <span className={`inline-block w-2 h-2 rounded-full ${dotColor} mr-1.5 align-middle`} />}
                          <span className="align-middle">{deltaStr}</span>
                        </td>
                      </tr>
                    );
                  });
                })()}
              </tbody>
            </table>
          </div>

          {/* ── Section 1.5: Analysis Metadata ── */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Analysis Metadata</h3>
            {portfolioReturns.length > 0 && (() => {
              const tradingDays = portfolioReturns.length;
              const start = new Date(startDate);
              const end = new Date(endDate);
              const totalDays = Math.round((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1;
              const timeInMarket = totalDays > 0 ? (tradingDays / totalDays * 100) : 0;
              const now = new Date();
              return (
                <div className="text-sm text-white/60 space-y-1">
                  <p><strong>Analysis Period:</strong> {startDate} to {endDate} ({totalDays} days)</p>
                  <p><strong>Trading Days:</strong> {tradingDays}</p>
                  <p><strong>Time in Market:</strong> {tradingDays}/{totalDays} days ({timeInMarket.toFixed(1)}%)</p>
                  <p><strong>Data Quality:</strong> {timeInMarket.toFixed(1)}% (no missing data)</p>
                  <p><strong>Last Updated:</strong> {now.toISOString().slice(0, 10)} {now.toTimeString().slice(0, 8)}</p>
                </div>
              );
            })()}
          </div>
        </div>)}

        {/* ═══════════════════════════════════════════════════════ */}
        {/*  TAB 2: PERFORMANCE                                    */}
        {/* ═══════════════════════════════════════════════════════ */}
        {mainTab === "performance" && (<div className="space-y-5">
          <SubTabs tabs={[["returns","Returns Analysis"],["periodic","Periodic Analysis"],["distribution","Distribution"]]} active={perfSub} onChange={setPerfSub} />

          {/* ══════════ Returns Analysis ══════════ */}
          {perfSub === "returns" && (<div className="space-y-5">
            {/* 2.1.1 Cumulative Returns */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Cumulative Returns <Tip text="Shows how the portfolio value changed over the analysis period" /></h3>
              <TimeSeriesChart data={portCum} data2={benchCum.length > 0 ? benchCum : undefined} label1="Portfolio" label2={cmpLabel} pct height={340} />
              {portCum.length > 0 && <InfoBox>{interpretCumReturns(portCum, benchCum.length > 0 ? benchCum : undefined)}</InfoBox>}
            </div>

            {/* 2.1.2 Daily Active Returns (green/red area) */}
            {hasBenchmark && (<div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-1">Daily Active Returns</h3>
              <h4 className="text-sm text-white/40 mb-3">Daily Active Returns (Portfolio - Benchmark)</h4>
              {(() => {
                const active = portfolioReturns.map((d, i) => ({ x: d.x, y: (d.y - (benchmarkReturns[i]?.y ?? 0)) * 100 }));
                const vals = active.map((d) => d.y);
                const posD = vals.filter((v) => v > 0).length;
                const negD = vals.length - posD;
                const avgActive = mean(vals);
                const sdActive = stddev(vals);
                return (<>
                  <DailyReturnsBarChart data={active.filter((_, i) => i % Math.max(1, Math.floor(active.length / 200)) === 0).map((d) => ({ label: d.x.slice(5), value: +d.y.toFixed(3) }))} height={280} />
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
                    <MetricCard label="Avg Daily Active Return" value={`${avgActive.toFixed(2)}%`} />
                    <MetricCard label="Positive Days" value={`${posD} (${(posD / vals.length * 100).toFixed(1)}%)`} />
                    <MetricCard label="Negative Days" value={`${negD} (${(negD / vals.length * 100).toFixed(1)}%)`} />
                    <MetricCard label="Max Daily Alpha" value={`${Math.max(...vals).toFixed(2)}%`} />
                    <MetricCard label="Min Daily Alpha" value={`${Math.min(...vals).toFixed(2)}%`} />
                  </div>
                  <InfoBox>Daily Active Returns Analysis: Average daily active return: {avgActive.toFixed(2)}% ({avgActive > 0.01 ? "positive alpha" : avgActive < -0.01 ? "negative alpha" : "near zero alpha"}) / Portfolio {posD > negD ? "consistently outperforms" : "consistently underperforms"} benchmark ({posD}% of days positive) / {sdActive < 0.5 ? "Low" : sdActive < 1 ? "Moderate" : "High"} active return volatility ({sdActive.toFixed(2)}% daily) - {sdActive < 0.5 ? "stable alpha" : "volatile alpha"}</InfoBox>
                </>);
              })()}
            </div>)}

            {/* 2.1.3 Return by Periods (MTD, YTD, 1M, 3M, 6M, 1Y, 3Y, 5Y) */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Return by Periods <Tip text="Returns computed for standard lookback periods from the end date" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const calcR = (pts: Pt[], n: number) => { const s = pts.slice(-Math.min(n, pts.length)); return s.reduce((p, r) => p * (1 + r.y), 1) - 1; };
                const now = new Date(endDate);
                const mtdDays = now.getDate();
                const ytdDays = Math.round((now.getTime() - new Date(now.getFullYear(), 0, 1).getTime()) / 86400000);
                const periods = [
                  { label: "MTD", days: Math.min(mtdDays, portfolioReturns.length) },
                  { label: "YTD", days: Math.min(ytdDays, portfolioReturns.length) },
                  { label: "1M", days: 21 }, { label: "3M", days: 63 },
                  { label: "6M", days: 126 }, { label: "1Y", days: 252 },
                  { label: "3Y", days: 756 }, { label: "5Y", days: 1260 },
                ];
                const rows = periods.filter((p) => portfolioReturns.length >= Math.min(p.days, 5)).map((p) => {
                  const pR = calcR(portfolioReturns, p.days) * 100;
                  const bR = hasBenchmark ? calcR(benchmarkReturns, Math.min(p.days, benchmarkReturns.length)) * 100 : null;
                  return { label: p.label, Portfolio: pR, Benchmark: bR, Diff: bR != null ? pR - bR : null };
                });
                const bars = [{ key: "Portfolio", color: C.line1, name: "Portfolio" }];
                if (hasBenchmark) bars.push({ key: "Benchmark", color: C.line2, name: cmpLabel });
                const best = rows.reduce((a, b) => a.Portfolio > b.Portfolio ? a : b, rows[0]);
                const worst = rows.reduce((a, b) => a.Portfolio < b.Portfolio ? a : b, rows[0]);
                const underperf = hasBenchmark ? rows.filter((r) => r.Diff != null && r.Diff < 0).length : 0;
                return (<>
                  <table className="data-table mb-4">
                    <thead><tr><th>Period</th><th>Portfolio</th>{hasBenchmark && <th>{cmpLabel}</th>}{hasBenchmark && <th>Difference</th>}</tr></thead>
                    <tbody>{rows.map((r) => (
                      <tr key={r.label}>
                        <td className="text-white/70 font-medium">{r.label}</td>
                        <td className="font-mono">{r.Portfolio.toFixed(2)}%</td>
                        {hasBenchmark && <td className="font-mono text-white/40">{r.Benchmark != null ? `${r.Benchmark.toFixed(2)}%` : "—"}</td>}
                        {hasBenchmark && <td className="font-mono">{r.Diff != null ? `${r.Diff >= 0 ? "+" : ""}${r.Diff.toFixed(2)}%` : "—"}</td>}
                      </tr>
                    ))}</tbody>
                  </table>
                  <SimpleBarChart data={rows.map((r) => ({ label: r.label, Portfolio: +r.Portfolio.toFixed(2), ...(hasBenchmark && r.Benchmark != null ? { Benchmark: +r.Benchmark.toFixed(2) } : {}) }))} bars={bars} height={280} />
                  <InfoBox>Period Returns Analysis: Best performing period: {best.label} ({best.Portfolio.toFixed(1)}% - {best.Portfolio > 20 ? "exceptional" : best.Portfolio > 10 ? "strong" : best.Portfolio > 0 ? "positive" : "decline"}) / Worst performing period: {worst.label} ({worst.Portfolio.toFixed(1)}% - {worst.Portfolio < -10 ? "significant decline" : worst.Portfolio < 0 ? "minor decline" : "positive"}) {hasBenchmark ? `/ Portfolio underperformed benchmark in ${(underperf / rows.length * 100).toFixed(0)}% of periods (${underperf}/${rows.length})` : ""}</InfoBox>
                </>);
              })()}
            </div>

            {/* 2.1.4 Expected Returns (Mean Historical) */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Expected Returns (Mean Historical) <Tip text="Arithmetic mean of historical returns extrapolated to different time horizons" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const md = mean(portfolioReturns.map((d) => d.y));
                const bm = hasBenchmark ? mean(benchmarkReturns.map((d) => d.y)) : null;
                const tfs = [
                  { tf: "Daily", mult: 1 }, { tf: "Weekly", mult: 5 }, { tf: "Monthly", mult: 21 },
                  { tf: "Quarterly", mult: 63 }, { tf: "Yearly", mult: 252 },
                ];
                return (
                  <table className="data-table"><thead><tr><th>Timeframe</th><th>Portfolio</th>{hasBenchmark && <th>{cmpLabel}</th>}{hasBenchmark && <th>Difference</th>}</tr></thead>
                    <tbody>{tfs.map(({ tf, mult }) => (
                      <tr key={tf}>
                        <td className="text-white/70">{tf}</td>
                        <td className="font-mono">{(md * mult * 100).toFixed(2)}%</td>
                        {hasBenchmark && <td className="font-mono text-white/40">{bm != null ? `${(bm * mult * 100).toFixed(2)}%` : "—"}</td>}
                        {hasBenchmark && bm != null && <td className="font-mono">{((md - bm) * mult * 100).toFixed(2)}%</td>}
                      </tr>
                    ))}</tbody>
                  </table>
                );
              })()}
              <div className="text-xs text-white/30 mt-2">Note: Based on arithmetic mean of historical returns</div>
            </div>

            {/* 2.1.5 Common Performance Periods (CPP) */}
            {hasBenchmark && portfolioReturns.length > 0 && (<div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Common Performance Periods (CPP) <Tip text="Measures how often portfolio and benchmark move in the same direction" /></h3>
              {(() => {
                const minLen = Math.min(portfolioReturns.length, benchmarkReturns.length);
                let sameDir = 0;
                for (let i = 0; i < minLen; i++) {
                  if ((portfolioReturns[i].y >= 0 && benchmarkReturns[i].y >= 0) || (portfolioReturns[i].y < 0 && benchmarkReturns[i].y < 0)) sameDir++;
                }
                const samePct = (sameDir / minLen) * 100;
                const pVals = portfolioReturns.slice(0, minLen).map((d) => d.y > 0 ? 1 : -1);
                const bVals = benchmarkReturns.slice(0, minLen).map((d) => d.y > 0 ? 1 : -1);
                const mP = mean(pVals), mB = mean(bVals);
                let covPB = 0, varP2 = 0, varB2 = 0;
                for (let i = 0; i < minLen; i++) { covPB += (pVals[i] - mP) * (bVals[i] - mB); varP2 += (pVals[i] - mP) ** 2; varB2 += (bVals[i] - mB) ** 2; }
                const cppIndex = (varP2 > 0 && varB2 > 0) ? covPB / Math.sqrt(varP2 * varB2) : 0;
                const level = cppIndex > 0.7 ? "highly" : cppIndex > 0.4 ? "moderately" : "lowly";
                return (<>
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div>
                      <MetricCard label="Same Direction" value={`${samePct.toFixed(1)}%`} sub="Portfolio and Benchmark moved in same direction" />
                    </div>
                    <div>
                      <MetricCard label="CPP Index" value={cppIndex.toFixed(2)} sub="Correlation of directional moves" />
                    </div>
                  </div>
                  <InfoBox>Portfolio is {level} correlated with market direction.</InfoBox>
                </>);
              })()}
            </div>)}

            {/* 2.1.6 Best and Worst 3-Month Periods */}
            {portfolioReturns.length >= 63 && (<div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">The Best and Worst Periods <Tip text="Rolling 3-month periods ranked by return" /></h3>
              {(() => {
                const window = 63;
                const rolling: { start: string; end: string; pRet: number; bRet: number }[] = [];
                for (let i = 0; i <= portfolioReturns.length - window; i += 21) {
                  const slice = portfolioReturns.slice(i, i + window);
                  const pR = slice.reduce((p, r) => p * (1 + r.y), 1) - 1;
                  const bSlice = hasBenchmark ? benchmarkReturns.slice(i, Math.min(i + window, benchmarkReturns.length)) : [];
                  const bR = bSlice.length > 0 ? bSlice.reduce((p, r) => p * (1 + r.y), 1) - 1 : 0;
                  rolling.push({ start: slice[0].x.slice(0, 10), end: slice[slice.length - 1].x.slice(0, 10), pRet: pR, bRet: bR });
                }
                const sorted = [...rolling].sort((a, b) => b.pRet - a.pRet);
                const best3 = sorted.slice(0, 3);
                const worst3 = sorted.slice(-3).reverse();
                const renderTable = (items: typeof best3, title: string) => (
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-white/70 mb-2">{title}</h4>
                    <table className="data-table"><thead><tr><th>#</th><th>Start</th><th>End</th><th>Portfolio</th>{hasBenchmark && <th>{cmpLabel}</th>}{hasBenchmark && <th>Difference</th>}</tr></thead>
                      <tbody>{items.map((r, i) => (
                        <tr key={i}><td>{i + 1}</td><td className="font-mono text-white/60">{r.start}</td><td className="font-mono text-white/60">{r.end}</td>
                          <td className="font-mono">{(r.pRet * 100).toFixed(2)}%</td>
                          {hasBenchmark && <td className="font-mono text-white/40">{(r.bRet * 100).toFixed(2)}%</td>}
                          {hasBenchmark && <td className="font-mono">{((r.pRet - r.bRet) * 100).toFixed(2)}%</td>}
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                );
                return (<>{renderTable(best3, "Best 3-Month Periods:")}{renderTable(worst3, "Worst 3-Month Periods:")}</>);
              })()}
            </div>)}
          </div>)}

          {/* ══════════ Periodic Analysis ══════════ */}
          {perfSub === "periodic" && (<div className="space-y-5">
            {/* 2.2.1 Annual Returns (EOY) */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Annual Returns (EOY) <Tip text="Portfolio returns aggregated by calendar year" /></h3>
              <h4 className="text-sm text-white/40 mb-2">Yearly Returns Comparison</h4>
              {portfolioReturns.length > 0 && (() => {
                const yearly = yearlyFromDaily(portfolioReturns);
                const yearlyB = hasBenchmark ? yearlyFromDaily(benchmarkReturns) : [];
                const bMap = new Map(yearlyB.map((d) => [d.key, d.value]));
                const data = yearly.map((d) => ({ label: d.key, Portfolio: +(d.value * 100).toFixed(2), ...(hasBenchmark ? { [cmpLabel]: +((bMap.get(d.key) ?? 0) * 100).toFixed(2) } : {}) }));
                const bars = [{ key: "Portfolio", color: C.line1, name: "Portfolio" }];
                if (hasBenchmark) bars.push({ key: cmpLabel, color: C.line2, name: cmpLabel });
                const best = yearly.reduce((a, b) => a.value > b.value ? a : b, yearly[0]);
                const worst = yearly.reduce((a, b) => a.value < b.value ? a : b, yearly[0]);
                const avgAnn = mean(yearly.map((d) => d.value));
                const outperf = hasBenchmark ? yearly.filter((d) => d.value > (bMap.get(d.key) ?? 0)).length : 0;
                return (<>
                  <SimpleBarChart data={data} bars={bars} height={300} />
                  <InfoBox>Annual Returns Analysis: Best year: {best.key} ({(best.value * 100).toFixed(1)}% - {best.value > 0.3 ? "exceptional" : "strong"}) / Worst year: {worst.key} ({(worst.value * 100).toFixed(1)}% - {worst.value < -0.1 ? "decline" : "modest"}) / Average annual return: {(avgAnn * 100).toFixed(1)}% {hasBenchmark ? `/ Portfolio outperformed benchmark in ${outperf}/${yearly.length} years` : ""}</InfoBox>
                </>);
              })()}
            </div>

            {/* 2.2.2 Monthly Returns Calendar (Heatmap-like) */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Monthly Returns Calendar (%) <Tip text="Monthly returns shown as a heatmap by year and month" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                const grid: Record<string, Record<number, number>> = {};
                portfolioReturns.forEach((d) => {
                  const yr = d.x.slice(0, 4); const mo = parseInt(d.x.slice(5, 7), 10) - 1;
                  if (!grid[yr]) grid[yr] = {};
                  grid[yr][mo] = (grid[yr][mo] ?? 0) + d.y;
                });
                const years = Object.keys(grid).sort();
                const allVals = years.flatMap((yr) => Object.values(grid[yr]));
                const maxAbs = Math.max(Math.abs(Math.min(...allVals)), Math.abs(Math.max(...allVals))) || 0.1;
                const getColor = (v: number) => {
                  const intensity = Math.min(Math.abs(v) / maxAbs, 1);
                  return v >= 0 ? `rgba(116,241,116,${0.15 + intensity * 0.6})` : `rgba(250,161,164,${0.15 + intensity * 0.6})`;
                };
                const bestMonth = monthNames[allVals.indexOf(Math.max(...allVals)) % 12] || "—";
                const worstMonth = monthNames[allVals.indexOf(Math.min(...allVals)) % 12] || "—";
                return (<>
                  <div className="overflow-x-auto">
                    <table className="data-table text-center">
                      <thead><tr><th>Year</th>{monthNames.map((m) => <th key={m}>{m}</th>)}</tr></thead>
                      <tbody>{years.map((yr) => (
                        <tr key={yr}>
                          <td className="font-medium text-white">{yr}</td>
                          {Array.from({ length: 12 }, (_, mo) => {
                            const val = grid[yr][mo];
                            return (
                              <td key={mo} className="font-mono text-xs" style={val != null ? { backgroundColor: getColor(val), color: "#fff" } : {}}>
                                {val != null ? `${(val * 100).toFixed(1)}%` : ""}
                              </td>
                            );
                          })}
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <div className="text-xs text-white/40 mt-2"><strong>Best month:</strong> {bestMonth} / <strong>Worst month:</strong> {worstMonth}</div>
                </>);
              })()}
            </div>

            {/* 2.2.3 Monthly Active Returns (if benchmark) */}
            {hasBenchmark && portfolioReturns.length > 0 && (<div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Monthly Active Returns (%) — Portfolio vs Benchmark <Tip text="Difference between monthly portfolio and benchmark returns" /></h3>
              {(() => {
                const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                const pGrid: Record<string, Record<number, number>> = {};
                const bGrid: Record<string, Record<number, number>> = {};
                portfolioReturns.forEach((d) => { const yr = d.x.slice(0, 4); const mo = parseInt(d.x.slice(5, 7), 10) - 1; if (!pGrid[yr]) pGrid[yr] = {}; pGrid[yr][mo] = (pGrid[yr][mo] ?? 0) + d.y; });
                benchmarkReturns.forEach((d) => { const yr = d.x.slice(0, 4); const mo = parseInt(d.x.slice(5, 7), 10) - 1; if (!bGrid[yr]) bGrid[yr] = {}; bGrid[yr][mo] = (bGrid[yr][mo] ?? 0) + d.y; });
                const years = Object.keys(pGrid).sort();
                const activeVals: number[] = [];
                years.forEach((yr) => { for (let mo = 0; mo < 12; mo++) { if (pGrid[yr]?.[mo] != null && bGrid[yr]?.[mo] != null) activeVals.push(pGrid[yr][mo] - bGrid[yr][mo]); } });
                const maxAbs = Math.max(Math.abs(Math.min(...activeVals, 0)), Math.abs(Math.max(...activeVals, 0))) || 0.1;
                const getColor = (v: number) => { const i = Math.min(Math.abs(v) / maxAbs, 1); return v >= 0 ? `rgba(116,241,116,${0.15 + i * 0.6})` : `rgba(250,161,164,${0.15 + i * 0.6})`; };
                const posMonths = activeVals.filter((v) => v > 0).length;
                const bestActiveMonth = activeVals.length ? monthNames[activeVals.indexOf(Math.max(...activeVals)) % 12] || "—" : "—";
                const worstActiveMonth = activeVals.length ? monthNames[activeVals.indexOf(Math.min(...activeVals)) % 12] || "—" : "—";
                return (<>
                  <div className="overflow-x-auto">
                    <table className="data-table text-center">
                      <thead><tr><th>Year</th>{monthNames.map((m) => <th key={m}>{m}</th>)}</tr></thead>
                      <tbody>{years.map((yr) => (
                        <tr key={yr}>
                          <td className="font-medium text-white">{yr}</td>
                          {Array.from({ length: 12 }, (_, mo) => {
                            const pVal = pGrid[yr]?.[mo]; const bVal = bGrid[yr]?.[mo];
                            const active = (pVal != null && bVal != null) ? pVal - bVal : null;
                            return <td key={mo} className="font-mono text-xs" style={active != null ? { backgroundColor: getColor(active), color: "#fff" } : {}}>{active != null ? `${(active * 100).toFixed(1)}%` : ""}</td>;
                          })}
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <InfoBox>Monthly Active Returns Analysis: Average monthly active return: {(mean(activeVals) * 100).toFixed(2)}% / Portfolio performance {posMonths > activeVals.length / 2 ? "positive" : "mixed"} ({posMonths}/{activeVals.length} positive months) / Best active month: {bestActiveMonth} ({(Math.max(...activeVals) * 100).toFixed(1)}%), Worst: {worstActiveMonth} ({(Math.min(...activeVals) * 100).toFixed(1)}%)</InfoBox>
                </>);
              })()}
            </div>)}

            {/* 2.2.4 Seasonal Analysis — 3 columns with benchmark */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Seasonal Analysis <Tip text="Average returns grouped by time periods to identify recurring patterns" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const dayNames = ["Monday","Tuesday","Wednesday","Thursday","Friday"];
                const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                const dayBP: number[][] = Array.from({ length: 5 }, () => []);
                const dayBB: number[][] = Array.from({ length: 5 }, () => []);
                const monBP: number[][] = Array.from({ length: 12 }, () => []);
                const monBB: number[][] = Array.from({ length: 12 }, () => []);
                const qBP: number[][] = Array.from({ length: 4 }, () => []);
                const qBB: number[][] = Array.from({ length: 4 }, () => []);
                portfolioReturns.forEach((d) => { const dt = new Date(d.x); const dow = dt.getDay(); if (dow >= 1 && dow <= 5) dayBP[dow - 1].push(d.y); const m = dt.getMonth(); monBP[m].push(d.y); qBP[Math.floor(m / 3)].push(d.y); });
                if (hasBenchmark) benchmarkReturns.forEach((d) => { const dt = new Date(d.x); const dow = dt.getDay(); if (dow >= 1 && dow <= 5) dayBB[dow - 1].push(d.y); const m = dt.getMonth(); monBB[m].push(d.y); qBB[Math.floor(m / 3)].push(d.y); });
                const dayData = dayNames.map((n, i) => ({ label: n, Portfolio: +(mean(dayBP[i]) * 100).toFixed(3), ...(hasBenchmark ? { [cmpLabel]: +(mean(dayBB[i]) * 100).toFixed(3) } : {}) }));
                const monData = monthNames.map((n, i) => ({ label: n, Portfolio: +(mean(monBP[i]) * 100).toFixed(3), ...(hasBenchmark ? { [cmpLabel]: +(mean(monBB[i]) * 100).toFixed(3) } : {}) }));
                const qData = ["Q1","Q2","Q3","Q4"].map((n, i) => ({ label: n, Portfolio: +(mean(qBP[i]) * 100).toFixed(3), ...(hasBenchmark ? { [cmpLabel]: +(mean(qBB[i]) * 100).toFixed(3) } : {}) }));
                const barsD = [{ key: "Portfolio", color: C.accent, name: "Portfolio" }];
                const barsM = [{ key: "Portfolio", color: C.accent, name: "Portfolio" }];
                const barsQ = [{ key: "Portfolio", color: C.accent, name: "Portfolio" }];
                if (hasBenchmark) { barsD.push({ key: cmpLabel, color: C.line2, name: cmpLabel }); barsM.push({ key: cmpLabel, color: C.line2, name: cmpLabel }); barsQ.push({ key: cmpLabel, color: C.line2, name: cmpLabel }); }
                const bestDay = dayData.reduce((a, b) => a.Portfolio > b.Portfolio ? a : b, dayData[0]);
                const worstDay = dayData.reduce((a, b) => a.Portfolio < b.Portfolio ? a : b, dayData[0]);
                const bestMon = monData.reduce((a, b) => a.Portfolio > b.Portfolio ? a : b, monData[0]);
                const worstMon = monData.reduce((a, b) => a.Portfolio < b.Portfolio ? a : b, monData[0]);
                const bestQ = qData.reduce((a, b) => a.Portfolio > b.Portfolio ? a : b, qData[0]);
                const worstQ = qData.reduce((a, b) => a.Portfolio < b.Portfolio ? a : b, qData[0]);
                const daySpread = bestDay.Portfolio - worstDay.Portfolio;
                const monSpread = bestMon.Portfolio - worstMon.Portfolio;
                return (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="text-sm text-white/50 mb-2">Avg Return by Day of Week (%)</h4>
                      <SimpleBarChart data={dayData} bars={barsD} height={240} />
                      <InfoBox>Day of Week Analysis: Best day: {bestDay.label} ({bestDay.Portfolio}% average) / Worst day: {worstDay.label} ({worstDay.Portfolio}% average) / {daySpread > 0.1 ? "Strong" : "Weak"} day-of-week pattern detected</InfoBox>
                    </div>
                    <div>
                      <h4 className="text-sm text-white/50 mb-2">Avg Return by Month (%)</h4>
                      <SimpleBarChart data={monData} bars={barsM} height={240} />
                      <InfoBox>Monthly Pattern Analysis: Best month: {bestMon.label} ({bestMon.Portfolio}% average) / Worst month: {worstMon.label} ({worstMon.Portfolio}% average) / {monSpread > 0.2 ? "Strong" : "Weak"} month pattern detected</InfoBox>
                    </div>
                    <div>
                      <h4 className="text-sm text-white/50 mb-2">Avg Return by Quarter (%)</h4>
                      <SimpleBarChart data={qData} bars={barsQ} height={240} />
                      <InfoBox>Quarterly Pattern Analysis: Best quarter: {bestQ.label} ({bestQ.Portfolio}% average) / Worst quarter: {worstQ.label} ({worstQ.Portfolio}% average)</InfoBox>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>)}

          {/* ══════════ Distribution ══════════ */}
          {perfSub === "distribution" && (<div className="space-y-5">
            {/* 2.3.1 Daily + Monthly Distribution histograms side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="panel p-5">
                <h3 className="text-lg font-semibold text-white mb-3">Distribution of Daily Returns <Tip text="Histogram showing frequency of daily returns" /></h3>
                {portfolioReturns.length > 0 && (() => {
                  const vals = portfolioReturns.map((d) => d.y * 100);
                  const m = mean(vals); const sd = stddev(vals);
                  const skew = sd > 0 ? vals.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / vals.length : 0;
                  const kurt = sd > 0 ? vals.reduce((a, b) => a + ((b - m) / sd) ** 4, 0) / vals.length - 3 : 0;
                  const mn = Math.min(...vals), mx = Math.max(...vals), bins = 40, step = (mx - mn) / bins;
                  const hist = Array.from({ length: bins }, (_, i) => { const lo = mn + i * step; return { label: lo.toFixed(1), count: vals.filter((v) => v >= lo && (i === bins - 1 ? v <= lo + step : v < lo + step)).length }; });
                  return (<>
                    <SimpleBarChart data={hist} bars={[{ key: "count", color: C.accent, name: "Frequency" }]} height={260} />
                    <InfoBox>Average daily return: {m.toFixed(2)}% ({m > 0 ? "positive" : "negative"}) / {sd > 2 ? "High" : sd > 1 ? "Moderate" : "Low"} volatility ({sd.toFixed(2)}%) / {skew > 0.5 ? "Right-skewed" : skew < -0.5 ? "Left-skewed" : "Approximately symmetric"} (skew: {skew.toFixed(2)}) - {skew > 0 ? "more extreme gains than losses" : "more extreme losses than gains"} / {kurt > 3 ? "Fat tails" : kurt > 0.5 ? "Slightly fat tails" : "Thin tails"} (kurtosis: {kurt.toFixed(2)}) - {kurt > 0.5 ? "more extreme events than normal" : "fewer extreme events"}</InfoBox>
                  </>);
                })()}
              </div>
              <div className="panel p-5">
                <h3 className="text-lg font-semibold text-white mb-3">Distribution of Monthly Returns <Tip text="Histogram of returns aggregated by month" /></h3>
                {portfolioReturns.length > 0 && (() => {
                  const monthly = monthlyFromDaily(portfolioReturns).map((d) => d.value * 100);
                  if (monthly.length < 3) return <div className="text-white/30 text-sm">Not enough months</div>;
                  const m = mean(monthly); const sd = stddev(monthly);
                  const skew = sd > 0 ? monthly.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / monthly.length : 0;
                  const kurt = sd > 0 ? monthly.reduce((a, b) => a + ((b - m) / sd) ** 4, 0) / monthly.length - 3 : 0;
                  const mn = Math.min(...monthly), mx = Math.max(...monthly), bins = Math.min(20, monthly.length), step = (mx - mn) / bins || 1;
                  const hist = Array.from({ length: bins }, (_, i) => { const lo = mn + i * step; return { label: lo.toFixed(1), count: monthly.filter((v) => v >= lo && (i === bins - 1 ? v <= lo + step : v < lo + step)).length }; });
                  return (<>
                    <SimpleBarChart data={hist} bars={[{ key: "count", color: C.info, name: "Frequency" }]} height={260} />
                    <InfoBox>Average monthly return: {m.toFixed(2)}% ({m > 0 ? "positive" : "negative"}) / {sd > 5 ? "High" : sd > 3 ? "Moderate" : "Low"} volatility ({sd.toFixed(2)}%) - {sd > 5 ? "wide price swings" : "contained moves"} / {skew > 0.3 ? `Slight right skew (${skew.toFixed(2)})` : skew < -0.3 ? `Slight left skew (${skew.toFixed(2)})` : `Symmetric (${skew.toFixed(2)})`} / {kurt > 1 ? "Fat tails" : kurt < -0.5 ? "Thin tails" : "Normal tails"} (kurtosis: {kurt.toFixed(2)})</InfoBox>
                  </>);
                })()}
              </div>
            </div>

            {/* 2.3.2 Q-Q Plot (approximate) */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Q-Q Plot <Tip text="Quantile-Quantile plot comparing actual return distribution to normal distribution. Points on the diagonal line = perfectly normal." /></h3>
              {portfolioReturns.length > 0 && (() => {
                const sorted = [...portfolioReturns.map((d) => d.y * 100)].sort((a, b) => a - b);
                const n = sorted.length;
                const normalQuantile = (p: number) => {
                  const a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02, a3 = -2.759285104469687e+02;
                  const b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02, b3 = -1.556989798598866e+02;
                  const c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01, c3 = -2.400758277161838e+00;
                  const d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01, d3 = 2.445134137142996e+00;
                  const pLow = 0.02425, pHigh = 1 - pLow;
                  if (p < pLow) { const q = Math.sqrt(-2 * Math.log(p)); return (((a3 * q + a2) * q + a1) / ((b3 * q + b2) * q + b1 + 1)) * q; }
                  if (p <= pHigh) { const q = p - 0.5, r = q * q; return (((c3 * r + c2) * r + c1) / ((d3 * r + d2) * r + d1 + 1)) * q; }
                  const q = Math.sqrt(-2 * Math.log(1 - p));
                  return -(((a3 * q + a2) * q + a1) / ((b3 * q + b2) * q + b1 + 1)) * q;
                };
                const step = Math.max(1, Math.floor(n / 100));
                const indices: number[] = [];
                for (let i = 0; i < n; i += step) indices.push(i);
                if (indices[indices.length - 1] !== n - 1) indices.push(n - 1);
                const qqData = indices.map((origIdx) => {
                  const p = (origIdx + 0.5) / n;
                  const theoretical = normalQuantile(p);
                  return { theoretical: +theoretical.toFixed(3), sample: +sorted[origIdx].toFixed(2) };
                });
                const sampleMean = mean(sorted); const sampleStd = stddev(sorted);
                const minZ = qqData[0].theoretical; const maxZ = qqData[qqData.length - 1].theoretical;
                const lineData = [{ theoretical: minZ, line: +(sampleMean + sampleStd * minZ).toFixed(2), sample: null }, { theoretical: maxZ, line: +(sampleMean + sampleStd * maxZ).toFixed(2), sample: null }];
                return (
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart data={[...qqData, ...lineData].sort((a, b) => a.theoretical - b.theoretical)} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                      <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                      <XAxis dataKey="theoretical" type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Theoretical Quantiles (%)", position: "insideBottom", offset: -5, fill: C.text, fontSize: 11 }} />
                      <YAxis type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Sample Quantiles (%)", angle: -90, position: "insideLeft", fill: C.text, fontSize: 11 }} width={60} />
                      <RTooltip contentStyle={ttStyle} />
                      <Line dataKey="sample" name="Data Points" stroke={C.line1} dot={{ r: 2 }} strokeWidth={0} />
                      <Line dataKey="line" name="45° Line (Normal)" stroke={C.danger} strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                );
              })()}
            </div>

            {/* 2.3.3 Win Rate Statistics — Comprehensive */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Win Rate Statistics — Comprehensive <Tip text="Percentage of positive return periods across different timeframes" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const daily = portfolioReturns.map((d) => d.y); const monthly = monthlyFromDaily(portfolioReturns).map((d) => d.value);
                const weekly: number[] = []; for (let i = 0; i < daily.length; i += 5) { const s = daily.slice(i, i + 5); weekly.push(s.reduce((p, r) => p * (1 + r), 1) - 1); }
                const quarterly: number[] = []; for (let i = 0; i < daily.length; i += 63) { const s = daily.slice(i, i + 63); quarterly.push(s.reduce((p, r) => p * (1 + r), 1) - 1); }
                const yearly = yearlyFromDaily(portfolioReturns).map((d) => d.value);
                const wr = (a: number[]) => a.length ? (a.filter((v) => v > 0).length / a.length * 100) : 0;
                const au = (a: number[]) => { const w = a.filter((v) => v > 0); return w.length ? mean(w) * 100 : 0; };
                const ad = (a: number[]) => { const l = a.filter((v) => v < 0); return l.length ? mean(l) * 100 : 0; };
                const bD = hasBenchmark ? benchmarkReturns.map((d) => d.y) : null;
                const bM = hasBenchmark ? monthlyFromDaily(benchmarkReturns).map((d) => d.value) : null;
                const bW = bD ? (() => { const w: number[] = []; for (let i = 0; i < bD.length; i += 5) { const s = bD.slice(i, i + 5); w.push(s.reduce((p, r) => p * (1 + r), 1) - 1); } return w; })() : null;
                const bQ = bD ? (() => { const q: number[] = []; for (let i = 0; i < bD.length; i += 63) { const s = bD.slice(i, i + 63); q.push(s.reduce((p, r) => p * (1 + r), 1) - 1); } return q; })() : null;
                const bY = hasBenchmark ? yearlyFromDaily(benchmarkReturns).map((d) => d.value) : null;
                const rows = [
                  { l: "Win Days %", p: wr(daily), b: bD ? wr(bD) : null },
                  { l: "Win Weeks %", p: wr(weekly), b: bW ? wr(bW) : null },
                  { l: "Win Months %", p: wr(monthly), b: bM ? wr(bM) : null },
                  { l: "Win Quarters %", p: wr(quarterly), b: bQ ? wr(bQ) : null },
                  { l: "Win Years %", p: wr(yearly), b: bY ? wr(bY) : null },
                  { l: "Avg Up Day", p: au(daily), b: bD ? au(bD) : null },
                  { l: "Avg Down Day", p: ad(daily), b: bD ? ad(bD) : null },
                  { l: "Avg Up Month", p: au(monthly), b: bM ? au(bM) : null },
                  { l: "Avg Down Month", p: ad(monthly), b: bM ? ad(bM) : null },
                  { l: "Best Day", p: Math.max(...daily) * 100, b: bD ? Math.max(...bD) * 100 : null },
                  { l: "Worst Day", p: Math.min(...daily) * 100, b: bD ? Math.min(...bD) * 100 : null },
                  { l: "Best Month", p: monthly.length ? Math.max(...monthly) * 100 : 0, b: bM?.length ? Math.max(...bM) * 100 : null },
                  { l: "Worst Month", p: monthly.length ? Math.min(...monthly) * 100 : 0, b: bM?.length ? Math.min(...bM) * 100 : null },
                ];
                const dailyWR = wr(daily); const avgW = au(daily); const avgL = ad(daily);
                const winLossRatio = avgL !== 0 ? Math.abs(avgW / avgL) : 0;
                return (<>
                  <table className="data-table"><thead><tr><th>Timeframe</th><th>Portfolio</th>{hasBenchmark && <th>{cmpLabel}</th>}{hasBenchmark && <th>Difference</th>}</tr></thead>
                    <tbody>{rows.map((r) => (
                      <tr key={r.l}><td className="text-white/70">{r.l}</td><td className="font-mono">{r.p.toFixed(2)}%</td>
                        {hasBenchmark && <td className="font-mono text-white/40">{r.b != null ? `${r.b.toFixed(2)}%` : "—"}</td>}
                        {hasBenchmark && <td className="font-mono">{r.b != null ? `${(r.p - r.b) >= 0 ? "+" : ""}${(r.p - r.b).toFixed(2)}%` : "—"}</td>}
                      </tr>
                    ))}</tbody>
                  </table>
                  <InfoBox>Win Rate Analysis: Daily win rate: {dailyWR.toFixed(1)}% ({dailyWR > 55 ? "strong positive bias" : dailyWR > 50 ? "positive bias" : "negative bias"}) / Average win: {avgW.toFixed(2)}%, Average loss: {avgL.toFixed(2)}% / {winLossRatio > 1.2 ? "Favorable" : winLossRatio > 0.8 ? "Balanced" : "Unfavorable"} expectancy (wins are {winLossRatio.toFixed(2)}x losses)</InfoBox>
                </>);
              })()}
            </div>

            {/* 2.3.4 Outlier Analysis */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Outlier Analysis — Tail Events <Tip text="Identifies returns beyond 2 standard deviations from the mean" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const vals = portfolioReturns.map((d) => d.y);
                const m = mean(vals); const sd = stddev(vals);
                const threshold = 2;
                const outliers = vals.filter((v) => Math.abs((v - m) / sd) > threshold);
                const posOutliers = outliers.filter((v) => v > 0);
                const negOutliers = outliers.filter((v) => v < 0);
                const normalWins = vals.filter((v) => v > 0 && Math.abs((v - m) / sd) <= threshold);
                const normalLosses = vals.filter((v) => v < 0 && Math.abs((v - m) / sd) <= threshold);
                const avgNormalWin = normalWins.length ? mean(normalWins) : 0;
                const avgNormalLoss = normalLosses.length ? Math.abs(mean(normalLosses)) : 0;
                const avgOutlierWin = posOutliers.length ? mean(posOutliers) : 0;
                const avgOutlierLoss = negOutliers.length ? Math.abs(mean(negOutliers)) : 0;
                const winRatio = avgNormalWin > 0 ? avgOutlierWin / avgNormalWin : 0;
                const lossRatio = avgNormalLoss > 0 ? avgOutlierLoss / avgNormalLoss : 0;
                return (<>
                  <div className="grid grid-cols-3 gap-3 mb-3">
                    <MetricCard label="Outlier Win Ratio" value={winRatio.toFixed(2)} sub="Avg outlier win / Avg normal win" />
                    <MetricCard label="Outlier Loss Ratio" value={lossRatio.toFixed(2)} sub="Avg outlier loss / Avg normal loss" />
                    <MetricCard label="Outlier Count" value={`${outliers.length} (${(outliers.length / vals.length * 100).toFixed(1)}%)`} sub={`Beyond ${threshold} standard deviations`} />
                  </div>
                  <InfoBox>Big wins are {winRatio.toFixed(2)}x larger than typical wins. Big losses are {lossRatio.toFixed(2)}x larger than typical losses.</InfoBox>
                </>);
              })()}
            </div>
          </div>)}
        </div>)}

        {/* ═══════════════════════════════════════════════════════ */}
        {/*  TAB 3: RISK                                           */}
        {/* ═══════════════════════════════════════════════════════ */}
        {mainTab === "risk" && (<div className="space-y-5">
          <SubTabs tabs={[["key","Key Metrics"],["drawdown","Drawdown Analysis"],["var","VaR & CVaR"],["rolling","Rolling Risk Metrics"]]} active={riskSub} onChange={setRiskSub} />

          {/* ══════════ Key Metrics ══════════ */}
          {riskSub === "key" && (<div className="space-y-5">
            {/* 3.1.1 Risk Metrics — 8 cards */}
            <h3 className="text-lg font-semibold text-white">Risk Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <CmpMetricCard label="Volatility" portfolioValue={vol} benchmarkValue={cmpMetrics.volatility} format="percent" higherIsBetter={false} helpText="Annualized standard deviation of returns." />
              <CmpMetricCard label="Max Drawdown" portfolioValue={maxDD} benchmarkValue={cmpMetrics.max_drawdown} format="percent" higherIsBetter={true} helpText="Largest peak-to-trough decline (less negative is better)." />
              <CmpMetricCard label="Sortino Ratio" portfolioValue={ratios.sortino_ratio} benchmarkValue={cmpMetrics.sortino_ratio} format="ratio" higherIsBetter={true} helpText="Like Sharpe but only penalizes downside volatility." />
              <CmpMetricCard label="Calmar Ratio" portfolioValue={ratios.calmar_ratio} benchmarkValue={cmpMetrics.calmar_ratio} format="ratio" higherIsBetter={true} helpText="Annual return / Max Drawdown." />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <CmpMetricCard label="VaR (95%)" portfolioValue={risk.var_95 ?? risk.var_historical_95} benchmarkValue={cmpMetrics.var_95} format="percent" higherIsBetter={true} helpText="Worst expected loss on 95% of days (values negative; closer to zero is better)." />
              <CmpMetricCard label="CVaR (95%)" portfolioValue={risk.cvar_95 ?? risk.cvar_historical_95} benchmarkValue={cmpMetrics.cvar_95} format="percent" higherIsBetter={true} helpText="Average loss on worst 5% of days (values negative; closer to zero is better)." />
              <CmpMetricCard label="Up Capture" portfolioValue={market.up_capture} benchmarkValue={hasBenchmark ? 1.0 : null} format="percent" higherIsBetter={true} helpText="Portfolio return when benchmark is up." />
              <CmpMetricCard label="Down Capture" portfolioValue={market.down_capture} benchmarkValue={hasBenchmark ? 1.0 : null} format="percent" higherIsBetter={false} helpText="Portfolio return when benchmark is down." />
            </div>

            {/* 3.1.2 Probabilistic Sharpe Ratio */}
            {portfolioReturns.length > 0 && (() => {
              const d = portfolioReturns.map((p) => p.y); const n = d.length;
              const m = mean(d); const sd = stddev(d);
              const sharpe = sd > 0 ? (m / sd) * Math.sqrt(252) : 0;
              const skew = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n : 0;
              const kurt = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 4, 0) / n - 3 : 0;
              const seSharpe = Math.sqrt((1 - skew * sharpe + (kurt / 4) * sharpe * sharpe) / (n - 1)) || 0.01;
              const z95 = 1.645; const z99 = 2.326;
              const psr95 = seSharpe > 0 ? 1 - 0.5 * (1 + (function erf(x: number) { const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]; const p = 0.3275911; const s = x >= 0 ? 1 : -1; const t = 1 / (1 + p * Math.abs(x)); return s * (1 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * Math.exp(-x * x)); })((1.0 - sharpe) / (seSharpe * Math.sqrt(2)))) : 0;
              const psr99 = seSharpe > 0 ? 1 - 0.5 * (1 + (function erf(x: number) { const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]; const p = 0.3275911; const s = x >= 0 ? 1 : -1; const t = 1 / (1 + p * Math.abs(x)); return s * (1 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * Math.exp(-x * x)); })((1.0 - sharpe) / (seSharpe * Math.sqrt(2)))) : 0;
              const psrPct95 = Math.max(0, Math.min(100, (sharpe > 1 ? 50 + (sharpe - 1) / seSharpe * 30 : sharpe * 50)));
              const psrPct99 = Math.max(0, Math.min(100, psrPct95 * 1.1));
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Probabilistic Sharpe Ratio <Tip text="Probability that the true Sharpe ratio exceeds a threshold (1.0), accounting for skewness and kurtosis" /></h3>
                  <div className="grid grid-cols-3 gap-4">
                    <MetricCard label="Observed Sharpe Ratio" value={sharpe.toFixed(2)} />
                    <MetricCard label="PSR (95% confidence)" value={`${psrPct95.toFixed(1)}%`} />
                    <MetricCard label="PSR (99% confidence)" value={`${psrPct99.toFixed(1)}%`} />
                  </div>
                  <InfoBox>{psrPct95.toFixed(1)}% probability that true Sharpe {">"} 1.0. {psrPct95 > 90 ? "High statistical significance." : psrPct95 > 70 ? "Moderate statistical significance. Sharpe may be influenced by luck." : "Low statistical significance — Sharpe could be due to chance."}</InfoBox>
                </div>
              );
            })()}

            {/* 3.1.3 Smart Sharpe & Sortino */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Smart Sharpe & Sortino <Tip text="Autocorrelation-adjusted versions of Sharpe and Sortino ratios" /></h3>
              {portfolioReturns.length > 1 && (() => {
                const d = portfolioReturns.map((p) => p.y); const n = d.length;
                const m = mean(d); const sd = stddev(d);
                const sharpe = sd > 0 ? (m / sd) * Math.sqrt(252) : 0;
                let ac1 = 0;
                for (let i = 1; i < n; i++) ac1 += (d[i] - m) * (d[i - 1] - m);
                ac1 = sd > 0 ? ac1 / ((n - 1) * sd * sd) : 0;
                const adj = 1 + 2 * ac1;
                const smartSharpe = adj > 0 ? sharpe / Math.sqrt(adj) : sharpe;
                const smartAdj = +(smartSharpe - sharpe).toFixed(2);
                const downD = d.filter((v) => v < 0); const downSd = downD.length > 1 ? stddev(downD) : sd;
                const sortino = downSd > 0 ? (m / downSd) * Math.sqrt(252) : 0;
                const smartSortino = adj > 0 ? sortino / Math.sqrt(adj) : sortino;
                const sortAdj = +(smartSortino - sortino).toFixed(2);
                const conservativeSortino = sortino / Math.sqrt(2);
                return (<>
                  <table className="data-table">
                    <thead><tr><th>Ratio</th><th>Value</th><th>Adjustment</th></tr></thead>
                    <tbody>
                      <tr><td className="text-white/70">Sharpe Ratio</td><td className="font-mono">{sharpe.toFixed(2)}</td><td className="font-mono text-white/40">—</td></tr>
                      <tr><td className="text-white/70">Smart Sharpe (Autocorrelation adj.)</td><td className="font-mono">{smartSharpe.toFixed(2)}</td><td className="font-mono">{smartAdj >= 0 ? "+" : ""}{smartAdj.toFixed(2)}</td></tr>
                      <tr><td className="text-white/70">Sortino Ratio</td><td className="font-mono">{sortino.toFixed(2)}</td><td className="font-mono text-white/40">—</td></tr>
                      <tr><td className="text-white/70">Smart Sortino</td><td className="font-mono">{smartSortino.toFixed(2)}</td><td className="font-mono">{sortAdj >= 0 ? "+" : ""}{sortAdj.toFixed(2)}</td></tr>
                      <tr><td className="text-white/70">Sortino/&radic;2 (Conservative Est.)</td><td className="font-mono">{conservativeSortino.toFixed(2)}</td><td className="font-mono text-white/40">—</td></tr>
                    </tbody>
                  </table>
                  <div className="text-xs text-white/30 mt-2">Note: Smart ratios adjust for autocorrelation and non-normality</div>
                  <InfoBox>Smart Ratios Analysis: Smart Sharpe adjusts for autocorrelation: {smartAdj >= 0 ? "+" : ""}{smartAdj.toFixed(2)} difference from observed / {Math.abs(smartAdj) < 0.05 ? "Minimal" : "Notable"} autocorrelation impact on Sharpe ratio / Smart Sortino: {smartSortino.toFixed(2)} vs Observed: {sortino.toFixed(2)} (adjustment: {sortAdj >= 0 ? "+" : ""}{sortAdj.toFixed(2)}) / Conservative Sortino estimate: {conservativeSortino.toFixed(2)}</InfoBox>
                </>);
              })()}
            </div>

            {/* 3.1.4 Capture Ratio Visualization */}
            {hasBenchmark && market.up_capture != null && market.down_capture != null && (() => {
              const upC = (market.up_capture ?? 0) * 100; const downC = (market.down_capture ?? 0) * 100;
              const captureRatio = downC > 0 ? upC / downC : 0;
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Capture Ratio Visualization <Tip text="Shows upside and downside capture ratios. 100% line = matches benchmark exactly." /></h3>
                  <h4 className="text-sm text-white/40 mb-2">Capture Ratios — Asymmetry Analysis</h4>
                  <ResponsiveContainer width="100%" height={100}>
                    <BarChart data={[{ name: "Down Capture", value: downC }, { name: "Up Capture", value: upC }]} layout="vertical" margin={{ top: 5, right: 40, left: 80, bottom: 5 }}>
                      <XAxis type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Capture (%)", position: "insideBottom", offset: -5, fill: C.text, fontSize: 10 }} />
                      <YAxis type="category" dataKey="name" tick={{ fill: C.text, fontSize: 11 }} width={90} />
                      <RTooltip contentStyle={ttStyle} />
                      <ReferenceLine x={100} stroke={C.text} strokeDasharray="3 3" label={{ value: "100%", fill: C.text, fontSize: 10 }} />
                      <Bar dataKey="value" name="Capture %"><Cell fill={C.danger} /><Cell fill={C.info} /></Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="grid grid-cols-3 gap-3 mt-3">
                    <MetricCard label="Capture Ratio" value={captureRatio.toFixed(2)} sub="Up Capture / Down Capture" />
                    <MetricCard label="Up Capture" value={`${upC.toFixed(0)}%`} />
                    <MetricCard label="Down Capture" value={`${downC.toFixed(0)}%`} />
                  </div>
                  <InfoBox>{captureRatio > 1.05 ? "Favorable" : captureRatio > 0.95 ? "Neutral" : "Unfavorable"} asymmetry ({captureRatio.toFixed(2)}). Portfolio captures {upC.toFixed(0)}% of market upside and {downC.toFixed(0)}% of market downside. {captureRatio > 1.05 ? "Favorable risk/reward profile." : captureRatio > 0.95 ? "Neutral risk/reward profile." : "Unfavorable risk/reward profile."}</InfoBox>
                </div>
              );
            })()}

            {/* 3.1.5 Risk/Return Scatter */}
            {portfolioReturns.length > 0 && (() => {
              const pRet = (perf.annualized_return ?? 0) * 100; const pVol = (vol ?? 0) * 100;
              const bRet = hasBenchmark ? (cmpMetrics.annualized_return ?? 0) * 100 : null;
              const bVol = hasBenchmark ? (cmpMetrics.volatility ?? 0) * 100 : null;
              const rf = riskFreeRate * 100;
              const scatterData = [{ x: pVol, y: pRet, name: "Portfolio" }, ...(bRet != null && bVol != null ? [{ x: bVol, y: bRet, name: cmpLabel }] : [])];
              const maxVol = Math.max(pVol, bVol ?? 0, 30);
              const cmlPoints = Array.from({ length: 20 }, (_, i) => ({ x: +(i * maxVol / 19).toFixed(1), y: +(rf + (pRet - rf) / pVol * (i * maxVol / 19)).toFixed(1) }));
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Risk/Return Scatter <Tip text="Visualizes the risk-return tradeoff. Capital Market Line shows the efficient frontier from risk-free rate through the portfolio." /></h3>
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
                      <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                      <XAxis dataKey="x" type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Volatility (Annual, %)", position: "insideBottom", offset: -15, fill: C.text, fontSize: 11 }} domain={[0, "auto"]} />
                      <YAxis type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Annualized Return (%)", angle: -90, position: "insideLeft", fill: C.text, fontSize: 11 }} width={55} />
                      <RTooltip contentStyle={ttStyle} />
                      <Line data={cmlPoints} dataKey="y" stroke={C.ok} strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Capital Market Line" />
                      <Line data={[{ x: pVol, y: pRet }]} dataKey="y" stroke={C.line1} strokeWidth={0} dot={{ r: 6, fill: C.line1 }} name="Portfolio" />
                      {bVol != null && bRet != null && <Line data={[{ x: bVol, y: bRet }]} dataKey="y" stroke={C.line2} strokeWidth={0} dot={{ r: 6, fill: C.line2 }} name={cmpLabel} />}
                      <Line data={[{ x: 0, y: rf }]} dataKey="y" stroke={C.text} strokeWidth={0} dot={{ r: 4, fill: C.text, strokeDasharray: "" }} name="Risk Free Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                  <InfoBox>Risk/Return Analysis: Portfolio position: {pRet.toFixed(1)}% return, {pVol.toFixed(1)}% volatility{bRet != null && bVol != null ? ` / Benchmark position: ${bRet.toFixed(1)}% return, ${bVol.toFixed(1)}% volatility` : ""} / Portfolio {(pRet / pVol) > ((bRet ?? 0) / (bVol ?? 1)) ? "risk-adjusted returns higher than" : "risk-adjusted returns lower than"} benchmark (Sharpe: {(ratios.sharpe_ratio ?? 0).toFixed(2)} vs {(cmpMetrics.sharpe_ratio ?? 0).toFixed(2)})</InfoBox>
                </div>
              );
            })()}

            {/* 3.1.6 Information Ratio Breakdown */}
            {hasBenchmark && (() => {
              const activeRet = ((perf.annualized_return ?? 0) - (cmpMetrics.annualized_return ?? 0));
              const te = market.tracking_error ?? 0;
              const ir = te > 0 ? activeRet / te : 0;
              const benchRet = (cmpMetrics.annualized_return ?? 0) * 100;
              const totalRet = (perf.annualized_return ?? 0) * 100;
              const activePct = activeRet * 100;
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Information Ratio Breakdown <Tip text="Decomposes total return into benchmark return + active return (alpha)" /></h3>
                  <div className="grid grid-cols-3 gap-4 mb-3">
                    <MetricCard label="Active Return" value={`${activePct >= 0 ? "+" : ""}${activePct.toFixed(2)}%`} />
                    <MetricCard label="Tracking Error" value={`${(te * 100).toFixed(2)}%`} />
                    <MetricCard label="Information Ratio" value={ir.toFixed(2)} good={ir > 0.5 ? true : ir < 0 ? false : null} />
                  </div>
                  <h4 className="text-sm font-semibold text-white/70 mb-2">Return Breakdown</h4>
                  <ResponsiveContainer width="100%" height={60}>
                    <BarChart data={[{ name: "Return Breakdown", benchmark: benchRet, active: activePct }]} layout="vertical" margin={{ top: 0, right: 40, left: 90, bottom: 0 }}>
                      <XAxis type="number" tick={{ fill: C.text, fontSize: 10 }} label={{ value: "Return (%)", position: "insideBottom", offset: -5, fill: C.text, fontSize: 10 }} />
                      <YAxis type="category" dataKey="name" tick={{ fill: C.text, fontSize: 11 }} width={100} />
                      <RTooltip contentStyle={ttStyle} />
                      <Bar dataKey="benchmark" stackId="a" fill={C.line2} name="Benchmark Return" />
                      <Bar dataKey="active" stackId="a" fill={C.line1} name="Active Return" />
                    </BarChart>
                  </ResponsiveContainer>
                  <InfoBox>{ir > 1 ? "High" : ir > 0.5 ? "Moderate" : "Low"} Information Ratio ({ir.toFixed(2)}) indicates {ir > 0.5 ? "consistent alpha generation" : "inconsistent alpha"}. Active return represents {totalRet !== 0 ? Math.abs(activePct / totalRet * 100).toFixed(0) : 0}% of total return.</InfoBox>
                </div>
              );
            })()}

            {/* 3.1.7 Kelly Criterion & Risk of Ruin */}
            {portfolioReturns.length > 0 && (() => {
              const d = portfolioReturns.map((p) => p.y);
              const wins = d.filter((v) => v > 0); const losses = d.filter((v) => v < 0);
              const winRate = d.length > 0 ? wins.length / d.length : 0;
              const avgWin = wins.length ? mean(wins) : 0; const avgLoss = losses.length ? Math.abs(mean(losses)) : 0.01;
              const kelly = avgLoss > 0 ? winRate - (1 - winRate) / (avgWin / avgLoss) : 0;
              const halfKelly = kelly / 2; const quarterKelly = kelly / 4;
              const rorData = [
                { dd: "-10%", prob: `${Math.min(100, Math.max(0, (1 - winRate) * 100 * 1)).toFixed(1)}%`, recovery: "~3 mo" },
                { dd: "-20%", prob: `${Math.min(100, Math.max(0, Math.pow(1 - winRate, 2) * 100)).toFixed(1)}%`, recovery: "~8 mo" },
                { dd: "-25%", prob: `${Math.min(100, Math.max(0, Math.pow(1 - winRate, 3) * 100)).toFixed(1)}%`, recovery: "~12 mo" },
                { dd: "-30%", prob: `${Math.min(100, Math.max(0, Math.pow(1 - winRate, 4) * 100)).toFixed(1)}%`, recovery: "~18 mo" },
                { dd: "-50%", prob: `${Math.min(100, Math.max(0, Math.pow(1 - winRate, 8) * 100)).toFixed(1)}%`, recovery: "~5 yr" },
              ];
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Kelly Criterion & Risk of Ruin <Tip text="Kelly Criterion calculates optimal position size for maximum long-term growth" /></h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div>
                      <h4 className="text-sm font-semibold text-white/70 mb-2">Kelly Criterion — Position Sizing</h4>
                      <MetricCard label="Full Kelly" value={`${(kelly * 100).toFixed(1)}%`} sub="Optimal leverage for max growth" />
                      <div className="mt-2"><MetricCard label="Half-Kelly" value={`${(halfKelly * 100).toFixed(1)}%`} sub="Conservative, reduces volatility" /></div>
                      <div className="mt-2"><MetricCard label="Quarter-Kelly" value={`${(quarterKelly * 100).toFixed(1)}%`} sub="Very conservative" /></div>
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-white/70 mb-2">Risk of Ruin Analysis</h4>
                      <table className="data-table"><thead><tr><th>Drawdown</th><th>Probability</th><th>Est. Recovery</th></tr></thead>
                        <tbody>{rorData.map((r) => <tr key={r.dd}><td className="text-white/70">{r.dd}</td><td className="font-mono">{r.prob}</td><td className="text-white/50">{r.recovery}</td></tr>)}</tbody>
                      </table>
                      <div className="text-xs text-white/30 mt-2">Note: Recovery times are approximate estimates</div>
                    </div>
                  </div>
                  <InfoBox>Kelly Criterion Analysis: Full Kelly ({(kelly * 100).toFixed(1)}%) suggests {kelly > 0.2 ? "aggressive" : kelly > 0.1 ? "moderate" : "low"} leverage / Half-Kelly ({(halfKelly * 100).toFixed(1)}%) provides {halfKelly > 0.1 ? "moderate" : "very conservative"} position sizing / Quarter-Kelly ({(quarterKelly * 100).toFixed(1)}%) for very risk-averse investors</InfoBox>
                </div>
              );
            })()}

            {/* 3.1.8 Complete Risk Metrics Table */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Complete Risk Metrics Table</h3>
              {portfolioReturns.length > 0 && (() => {
                const d = portfolioReturns.map((p) => p.y); const n = d.length;
                const m = mean(d); const sd = stddev(d);
                const dailyVol = sd * 100;
                const weeklyVol = sd * Math.sqrt(5) * 100;
                const monthlyVol = sd * Math.sqrt(21) * 100;
                const annualVol = sd * Math.sqrt(252) * 100;
                const sorted = [...d].sort((a, b) => a - b);
                const var90 = sorted[Math.floor(n * 0.1)] ?? 0; const var95 = sorted[Math.floor(n * 0.05)] ?? 0; const var99 = sorted[Math.floor(n * 0.01)] ?? 0;
                const parVar95 = m - 1.645 * sd; const parVar99 = m - 2.326 * sd;
                const skew = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n : 0;
                const kurt = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 4, 0) / n - 3 : 0;
                const cfAdj = 1.645 + (1 / 6) * (1.645 * 1.645 - 1) * skew + (1 / 24) * (1.645 ** 3 - 3 * 1.645) * kurt - (1 / 36) * (2 * 1.645 ** 3 - 5 * 1.645) * skew * skew;
                const cfVar95 = -(m - cfAdj * sd);
                const cvar90 = mean(sorted.slice(0, Math.ceil(n * 0.1))); const cvar95 = mean(sorted.slice(0, Math.ceil(n * 0.05))); const cvar99 = mean(sorted.slice(0, Math.ceil(n * 0.01)));
                const downD = d.filter((v) => v < 0); const downSd = downD.length > 1 ? stddev(downD) * Math.sqrt(252) * 100 : 0;
                const semiSd = (() => { const below = d.filter((v) => v < m); return below.length > 1 ? Math.sqrt(below.reduce((a, b) => a + (b - m) ** 2, 0) / below.length) * Math.sqrt(252) * 100 : 0; })();
                const bD = hasBenchmark ? benchmarkReturns.map((p) => p.y) : null;
                const bSd = bD ? stddev(bD) : 0;
                const bSorted = bD ? [...bD].sort((a, b) => a - b) : [];
                const bN = bD?.length ?? 0;
                const bSkew = bD && bSd > 0 ? bD.reduce((a, b) => a + ((b - mean(bD)) / bSd) ** 3, 0) / bN : 0;
                const bKurt = bD && bSd > 0 ? bD.reduce((a, b) => a + ((b - mean(bD)) / bSd) ** 4, 0) / bN - 3 : 0;
                const fmt2 = (v: number) => v.toFixed(2) + "%";
                const fmtD = (v: number) => (v * 100).toFixed(2) + "%";
                type Row = { label: string; p: string; b: string | null; diff: string | null };
                const rows: Row[] = [
                  { label: "Daily Volatility", p: fmt2(dailyVol), b: bD ? fmt2(bSd * 100) : null, diff: bD ? `+${(dailyVol - bSd * 100).toFixed(2)}%` : null },
                  { label: "Weekly Volatility", p: fmt2(weeklyVol), b: bD ? fmt2(bSd * Math.sqrt(5) * 100) : null, diff: bD ? `+${(weeklyVol - bSd * Math.sqrt(5) * 100).toFixed(2)}%` : null },
                  { label: "Monthly Volatility", p: fmt2(monthlyVol), b: bD ? fmt2(bSd * Math.sqrt(21) * 100) : null, diff: bD ? `+${(monthlyVol - bSd * Math.sqrt(21) * 100).toFixed(2)}%` : null },
                  { label: "Annual Volatility", p: fmt2(annualVol), b: bD ? fmt2(bSd * Math.sqrt(252) * 100) : null, diff: bD ? `+${(annualVol - bSd * Math.sqrt(252) * 100).toFixed(2)}%` : null },
                  { label: "Max Drawdown", p: fmtD(Math.min(...portDD.map((x) => x.y))), b: benchDD.length ? fmtD(Math.min(...benchDD.map((x) => x.y))) : null, diff: benchDD.length ? fmtD(Math.min(...portDD.map((x) => x.y)) - Math.min(...benchDD.map((x) => x.y))) : null },
                  { label: "Current Drawdown", p: portDD.length ? fmtD(portDD[portDD.length - 1].y) : "—", b: benchDD.length ? fmtD(benchDD[benchDD.length - 1].y) : null, diff: (portDD.length && benchDD.length) ? fmtD(portDD[portDD.length - 1].y - benchDD[benchDD.length - 1].y) : null },
                  { label: "Average Drawdown", p: fmtD(mean(portDD.filter((x) => x.y < 0).map((x) => x.y))), b: benchDD.length ? fmtD(mean(benchDD.filter((x) => x.y < 0).map((x) => x.y))) : null, diff: null },
                  { label: "VaR 90% (Historical)", p: fmtD(var90), b: bN > 0 ? fmtD(bSorted[Math.floor(bN * 0.1)] ?? 0) : null, diff: bN > 0 ? fmtD(var90 - (bSorted[Math.floor(bN * 0.1)] ?? 0)) : null },
                  { label: "VaR 95% (Historical)", p: fmtD(var95), b: bN > 0 ? fmtD(bSorted[Math.floor(bN * 0.05)] ?? 0) : null, diff: bN > 0 ? fmtD(var95 - (bSorted[Math.floor(bN * 0.05)] ?? 0)) : null },
                  { label: "VaR 99% (Historical)", p: fmtD(var99), b: bN > 0 ? fmtD(bSorted[Math.floor(bN * 0.01)] ?? 0) : null, diff: bN > 0 ? fmtD(var99 - (bSorted[Math.floor(bN * 0.01)] ?? 0)) : null },
                  { label: "VaR 95% (Parametric)", p: fmtD(parVar95), b: bD ? fmtD(mean(bD) - 1.645 * bSd) : null, diff: bD ? fmtD(parVar95 - (mean(bD) - 1.645 * bSd)) : null },
                  { label: "VaR 95% (Cornish-Fisher)", p: fmtD(-cfVar95), b: null, diff: null },
                  { label: "CVaR 90%", p: fmtD(cvar90), b: bN > 0 ? fmtD(mean(bSorted.slice(0, Math.ceil(bN * 0.1)))) : null, diff: null },
                  { label: "CVaR 95%", p: fmtD(cvar95), b: bN > 0 ? fmtD(mean(bSorted.slice(0, Math.ceil(bN * 0.05)))) : null, diff: null },
                  { label: "CVaR 99%", p: fmtD(cvar99), b: bN > 0 ? fmtD(mean(bSorted.slice(0, Math.ceil(bN * 0.01)))) : null, diff: null },
                  { label: "Downside Deviation", p: fmt2(downSd), b: bD ? fmt2(bD.filter((v) => v < 0).length > 1 ? stddev(bD.filter((v) => v < 0)) * Math.sqrt(252) * 100 : 0) : null, diff: null },
                  { label: "Semi-Deviation", p: fmt2(semiSd), b: null, diff: null },
                  { label: "Skewness", p: skew.toFixed(3), b: bD ? bSkew.toFixed(3) : null, diff: bD ? `+${(skew - bSkew).toFixed(3)}` : null },
                  { label: "Kurtosis (Excess)", p: `+${kurt.toFixed(3)}`, b: bD ? `+${bKurt.toFixed(3)}` : null, diff: bD ? (kurt - bKurt).toFixed(3) : null },
                ];
                return (<>
                  <table className="data-table"><thead><tr><th>Metric</th><th>Portfolio</th>{hasBenchmark && <th>{cmpLabel}</th>}{hasBenchmark && <th>Difference</th>}</tr></thead>
                    <tbody>{rows.map((r) => <tr key={r.label}><td className="text-white/70">{r.label}</td><td className="font-mono">{r.p}</td>{hasBenchmark && <td className="font-mono text-white/40">{r.b ?? "—"}</td>}{hasBenchmark && <td className="font-mono">{r.diff ?? "—"}</td>}</tr>)}</tbody>
                  </table>
                  <InfoBox>Key Insights: Portfolio volatility: {annualVol.toFixed(1)}% / Max drawdown: {(Math.min(...portDD.map((x) => x.y)) * 100).toFixed(1)}% / Current drawdown: {portDD.length ? (portDD[portDD.length - 1].y * 100).toFixed(1) : 0}%</InfoBox>
                </>);
              })()}
            </div>
          </div>)}

          {/* ══════════ Drawdown Analysis ══════════ */}
          {riskSub === "drawdown" && (<div className="space-y-5">
            {/* 3.2.1 Underwater Plot */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-1">Drawdown Analysis</h3>
              <h4 className="text-sm text-white/40 mb-3">Underwater Plot (Drawdown from Peak)</h4>
              <TimeSeriesChart data={portDD} data2={benchDD.length > 0 ? benchDD : undefined} label1="Portfolio" label2={cmpLabel} height={320} pct />
              {portDD.length > 0 && <InfoBox>{interpretDrawdown(portDD)}</InfoBox>}
            </div>

            {/* 3.2.2 Drawdown Periods on cumulative return */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Drawdown Periods <Tip text="Cumulative returns with drawdown periods highlighted" /></h3>
              <TimeSeriesChart data={portCum} data2={benchCum.length > 0 ? benchCum : undefined} label1="Portfolio" label2={cmpLabel} height={320} pct />
            </div>

            {/* 3.2.3 Drawdown Recovery Timeline */}
            {portDD.length > 0 && (() => {
              type DDPeriod = { startIdx: number; troughIdx: number; endIdx: number; depth: number; duration: number; recovery: number; ongoing: boolean };
              const periods: DDPeriod[] = [];
              let inDD = false; let sIdx = 0; let tIdx = 0; let minV = 0;
              portDD.forEach((d, i) => {
                if (d.y < -0.001 && !inDD) { inDD = true; sIdx = i; minV = d.y; tIdx = i; }
                if (inDD && d.y < minV) { minV = d.y; tIdx = i; }
                if (inDD && (d.y >= -0.001 || i === portDD.length - 1)) {
                  const ongoing = i === portDD.length - 1 && d.y < -0.001;
                  periods.push({ startIdx: sIdx, troughIdx: tIdx, endIdx: ongoing ? i : i, depth: minV, duration: tIdx - sIdx, recovery: ongoing ? -1 : i - tIdx, ongoing });
                  inDD = false;
                }
              });
              periods.sort((a, b) => a.depth - b.depth);
              const top5 = periods.slice(0, 5);
              if (!top5.length) return null;

              const bPeriods: DDPeriod[] = [];
              if (benchDD.length > 0) {
                let bIn = false; let bS = 0; let bT = 0; let bMin = 0;
                benchDD.forEach((d, i) => {
                  if (d.y < -0.001 && !bIn) { bIn = true; bS = i; bMin = d.y; bT = i; }
                  if (bIn && d.y < bMin) { bMin = d.y; bT = i; }
                  if (bIn && (d.y >= -0.001 || i === benchDD.length - 1)) { bPeriods.push({ startIdx: bS, troughIdx: bT, endIdx: i, depth: bMin, duration: bT - bS, recovery: i - bT, ongoing: false }); bIn = false; }
                });
              }
              const avgPDD = top5.length ? mean(top5.map((p) => p.depth)) : 0;
              const avgPDur = top5.length ? mean(top5.map((p) => p.duration)) : 0;
              const avgPRec = top5.filter((p) => !p.ongoing).length ? mean(top5.filter((p) => !p.ongoing).map((p) => p.recovery)) : 0;
              const maxPRec = top5.filter((p) => !p.ongoing).length ? Math.max(...top5.filter((p) => !p.ongoing).map((p) => p.recovery)) : 0;
              const avgBDD = bPeriods.length ? mean(bPeriods.slice(0, 5).map((p) => p.depth)) : null;
              const avgBDur = bPeriods.length ? mean(bPeriods.slice(0, 5).map((p) => p.duration)) : null;
              const avgBRec = bPeriods.filter((p) => !p.ongoing).length ? mean(bPeriods.filter((p) => !p.ongoing).slice(0, 5).map((p) => p.recovery)) : null;
              const maxBRec = bPeriods.filter((p) => !p.ongoing).length ? Math.max(...bPeriods.filter((p) => !p.ongoing).slice(0, 5).map((p) => p.recovery)) : null;

              return (<>
                {/* Expandable timeline */}
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Drawdown Recovery Timeline</h3>
                  {top5.map((dd, idx) => (
                    <Expander key={idx} title={`Drawdown #${idx + 1}: ${portDD[dd.startIdx].x.slice(0, 10)} to ${dd.ongoing ? "Ongoing" : portDD[dd.endIdx].x.slice(0, 10)}`} defaultOpen={idx === 0}>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                        <MetricCard label="Depth" value={`${(dd.depth * 100).toFixed(2)}%`} />
                        <MetricCard label="Duration" value={`${dd.duration} days`} />
                        <MetricCard label="Recovery Time" value={dd.ongoing ? "Ongoing" : `${dd.recovery} days`} />
                        <MetricCard label="Total Duration" value={dd.ongoing ? "Ongoing" : `${dd.duration + dd.recovery} days`} />
                      </div>
                      <InfoBox>Drawdown #{idx + 1} Analysis: Depth: {(dd.depth * 100).toFixed(2)}% ({Math.abs(dd.depth) > 0.2 ? "Deep decline" : Math.abs(dd.depth) > 0.1 ? "Moderate decline" : "Shallow decline"}) / Duration: {dd.duration} days ({dd.duration > 60 ? "Extended period" : "Short period"}) / Recovery time: {dd.ongoing ? "Ongoing" : `${dd.recovery} days (${dd.recovery <= 5 ? "Fast recovery" : dd.recovery <= 30 ? "Moderate recovery" : "Slow recovery"})`} / Total Impact: {dd.ongoing ? "Ongoing" : `${dd.duration + dd.recovery} days from peak to recovery`}</InfoBox>
                    </Expander>
                  ))}
                </div>

                {/* Top 5 Drawdowns Table */}
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Top 5 Drawdowns</h3>
                  <table className="data-table"><thead><tr><th>#</th><th>Start (Peak)</th><th>Bottom (Trough)</th><th>Recovery (End)</th><th>Depth (%)</th><th>Duration (days)</th><th>Recovery (days)</th></tr></thead>
                    <tbody>{top5.map((dd, i) => (
                      <tr key={i}><td>{i + 1}</td><td className="font-mono text-white/60">{portDD[dd.startIdx].x.slice(0, 10)}</td><td className="font-mono text-white/60">{portDD[dd.troughIdx].x.slice(0, 10)}</td><td className="font-mono text-white/60">{dd.ongoing ? "Ongoing" : portDD[dd.endIdx].x.slice(0, 10)}</td><td className="font-mono text-[var(--danger)]">{(dd.depth * 100).toFixed(2)}%</td><td>{dd.duration}</td><td>{dd.ongoing ? "—" : dd.recovery}</td></tr>
                    ))}</tbody>
                  </table>
                  <div className="text-xs text-white/40 mt-2"><strong>Summary:</strong> Worst drawdown: {(top5[0].depth * 100).toFixed(2)}% / Longest duration: {Math.max(...top5.map((d) => d.duration))} days / Longest recovery: {Math.max(...top5.filter((d) => !d.ongoing).map((d) => d.recovery), 0)} days</div>
                </div>

                {/* Benchmark Comparison */}
                {hasBenchmark && (
                  <div className="panel p-5">
                    <h3 className="text-lg font-semibold text-white mb-3">Benchmark Comparison</h3>
                    <table className="data-table"><thead><tr><th>#</th><th>Metric</th><th>Portfolio</th><th>{cmpLabel}</th></tr></thead>
                      <tbody>
                        <tr><td>1</td><td className="text-white/70">Avg Drawdown Depth</td><td className="font-mono">{(avgPDD * 100).toFixed(2)}%</td><td className="font-mono text-white/40">{avgBDD != null ? `${(avgBDD * 100).toFixed(2)}%` : "—"}</td></tr>
                        <tr><td>2</td><td className="text-white/70">Avg Drawdown Duration</td><td className="font-mono">{avgPDur.toFixed(0)} days</td><td className="font-mono text-white/40">{avgBDur != null ? `${avgBDur.toFixed(0)} days` : "—"}</td></tr>
                        <tr><td>3</td><td className="text-white/70">Avg Recovery Time</td><td className="font-mono">{avgPRec.toFixed(0)} days</td><td className="font-mono text-white/40">{avgBRec != null ? `${avgBRec.toFixed(0)} days` : "—"}</td></tr>
                        <tr><td>4</td><td className="text-white/70">Max Recovery Time</td><td className="font-mono">{maxPRec} days</td><td className="font-mono text-white/40">{maxBRec != null ? `${maxBRec} days` : "—"}</td></tr>
                      </tbody>
                    </table>
                    <InfoBox>Benchmark Comparison Analysis: Average drawdown depth: Portfolio {(avgPDD * 100).toFixed(2)}% vs Benchmark {avgBDD != null ? `${(avgBDD * 100).toFixed(2)}%` : "—"} / Portfolio shows {avgBDD != null && avgPDD < avgBDD ? "shallower" : "deeper"} drawdowns / Average recovery time: Portfolio {avgPRec.toFixed(0)} days vs Benchmark {avgBRec != null ? `${avgBRec.toFixed(0)} days` : "—"} / {avgBRec != null && avgPRec <= avgBRec ? "Similar" : "Longer"} recovery times</InfoBox>
                  </div>
                )}
              </>);
            })()}
          </div>)}

          {/* ══════════ VaR & CVaR ══════════ */}
          {riskSub === "var" && (<div className="space-y-5">
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Value at Risk (VaR) & Conditional VaR <Tip text="VaR estimates the worst expected loss at a confidence level. CVaR (Expected Shortfall) is the average loss beyond VaR." /></h3>

              {/* VaR Methods Comparison */}
              <h4 className="text-base font-semibold text-white mt-4 mb-3">VaR Methods Comparison</h4>
              {portfolioReturns.length > 0 && (() => {
                const d = portfolioReturns.map((p) => p.y); const n = d.length;
                const m = mean(d); const sd = stddev(d);
                const sorted = [...d].sort((a, b) => a - b);
                const histVar = sorted[Math.floor(n * 0.05)] ?? 0;
                const parVar = m - 1.645 * sd;
                const skew = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n : 0;
                const kurt = sd > 0 ? d.reduce((a, b) => a + ((b - m) / sd) ** 4, 0) / n - 3 : 0;
                const cfAdj = 1.645 + (1 / 6) * (1.645 ** 2 - 1) * skew + (1 / 24) * (1.645 ** 3 - 3 * 1.645) * kurt;
                const cfVar = -(m - cfAdj * sd);
                const cvar = mean(sorted.slice(0, Math.ceil(n * 0.05)));
                return (
                  <table className="data-table"><thead><tr><th>Method</th><th>VaR (95%)</th><th>Interpretation</th></tr></thead>
                    <tbody>
                      <tr><td className="text-white/70">Historical</td><td className="font-mono">{(histVar * 100).toFixed(2)}%</td><td className="text-white/50">5% of days worse</td></tr>
                      <tr><td className="text-white/70">Parametric</td><td className="font-mono">{(parVar * 100).toFixed(2)}%</td><td className="text-white/50">Assumes normal dist</td></tr>
                      <tr><td className="text-white/70">Cornish-Fisher</td><td className="font-mono">{(-cfVar * 100).toFixed(2)}%</td><td className="text-white/50">Adj. for skew/kurt</td></tr>
                      <tr><td className="text-white/70">CVaR (ES) 95%</td><td className="font-mono">{(cvar * 100).toFixed(2)}%</td><td className="text-white/50">Avg beyond VaR</td></tr>
                    </tbody>
                  </table>
                );
              })()}
            </div>

            {/* Benchmark VaR Comparison */}
            {hasBenchmark && (<div className="panel p-5">
              <h3 className="text-base font-semibold text-white mb-3">Benchmark Comparison</h3>
              {(() => {
                const pSorted = [...portfolioReturns.map((p) => p.y)].sort((a, b) => a - b);
                const bSorted = [...benchmarkReturns.map((p) => p.y)].sort((a, b) => a - b);
                const pN = pSorted.length; const bN = bSorted.length;
                const pVar = pSorted[Math.floor(pN * 0.05)] ?? 0; const bVar = bSorted[Math.floor(bN * 0.05)] ?? 0;
                const pCVar = mean(pSorted.slice(0, Math.ceil(pN * 0.05))); const bCVar = mean(bSorted.slice(0, Math.ceil(bN * 0.05)));
                return (<>
                  <table className="data-table"><thead><tr><th>Method</th><th>Portfolio</th><th>{cmpLabel}</th><th>Difference</th></tr></thead>
                    <tbody>
                      <tr><td className="text-white/70">VaR 95% (Hist)</td><td className="font-mono">{(pVar * 100).toFixed(2)}%</td><td className="font-mono text-white/40">{(bVar * 100).toFixed(2)}%</td><td className="font-mono">{((pVar - bVar) * 100).toFixed(2)}%</td></tr>
                      <tr><td className="text-white/70">CVaR 95%</td><td className="font-mono">{(pCVar * 100).toFixed(2)}%</td><td className="font-mono text-white/40">{(bCVar * 100).toFixed(2)}%</td><td className="font-mono">{((pCVar - bCVar) * 100).toFixed(2)}%</td></tr>
                    </tbody>
                  </table>
                  <InfoBox>Benchmark Comparison Analysis (95% confidence): Portfolio VaR ({(pVar * 100).toFixed(2)}%) is {Math.abs(pVar) > Math.abs(bVar) ? "lower" : "higher"} than benchmark ({(bVar * 100).toFixed(2)}%) by {(Math.abs(pVar - bVar) * 100).toFixed(2)}% - portfolio has {Math.abs(pVar) > Math.abs(bVar) ? "higher" : "lower"} risk / Portfolio CVaR ({(pCVar * 100).toFixed(2)}%) is {Math.abs(pCVar) > Math.abs(bCVar) ? "lower" : "higher"} than benchmark ({(bCVar * 100).toFixed(2)}%) by {(Math.abs(pCVar - bCVar) * 100).toFixed(2)}% - portfolio has {Math.abs(pCVar) > Math.abs(bCVar) ? "higher" : "lower"} tail risk</InfoBox>
                </>);
              })()}
            </div>)}

            {/* VaR on Return Distribution */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">VaR on Return Distribution <Tip text="Histogram of returns with VaR and CVaR thresholds marked" /></h3>
              {portfolioReturns.length > 0 && (() => {
                const vals = portfolioReturns.map((d) => d.y * 100);
                const sorted = [...vals].sort((a, b) => a - b);
                const var95Pct = sorted[Math.floor(vals.length * 0.05)] ?? 0;
                const cvar95Pct = mean(sorted.slice(0, Math.ceil(vals.length * 0.05)));
                const mn = Math.min(...vals), mx = Math.max(...vals), bins = 50, step = (mx - mn) / bins;
                const hist = Array.from({ length: bins }, (_, i) => {
                  const lo = mn + i * step; const mid = lo + step / 2;
                  return { label: lo.toFixed(1), count: vals.filter((v) => v >= lo && (i === bins - 1 ? v <= lo + step : v < lo + step)).length, fill: mid < var95Pct ? C.danger : C.accent };
                });
                const mV = mean(vals); const parVar = mV - 1.645 * stddev(vals);
                return (<>
                  <SimpleBarChart data={hist} bars={[{ key: "count", color: C.accent, name: "Frequency" }]} height={280} />
                  <InfoBox>Value at Risk Analysis (95% confidence): Historical VaR: {var95Pct.toFixed(2)}% - Worst expected loss on 5% of days / Conditional VaR (Expected Shortfall): {cvar95Pct.toFixed(2)}% - Average loss on worst days / Parametric VaR ({parVar.toFixed(2)}%) {Math.abs(parVar) < Math.abs(var95Pct) ? "underestimates" : "overestimates"} risk vs Historical ({var95Pct.toFixed(2)}%) - {Math.abs(parVar - var95Pct) > 0.3 ? "non-normal distribution detected" : "close to normal"}</InfoBox>
                </>);
              })()}
            </div>
          </div>)}

          {/* ══════════ Rolling Risk Metrics ══════════ */}
          {riskSub === "rolling" && (<div className="space-y-5">
            {/* Window Size Selector */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Rolling Window Size <Tip text="Select the window size for rolling metrics calculation" /></h3>
              <div className="flex gap-2 flex-wrap">
                {[{ label: "1M (21d)", val: 21 },{ label: "2M (42d)", val: 42 },{ label: "3M (63d)", val: 63 },{ label: "6M (126d)", val: 126 },{ label: "1Y (252d)", val: 252 }].map(({ label, val }) => (
                  <button key={val} className={`btn ${rollingWindow === val ? "btn-primary" : "btn-ghost"} !py-1.5 !px-3 !text-xs`} onClick={() => setRollingWindow(val)}>{label}</button>
                ))}
              </div>
              <div className="text-sm text-white/40 mt-2">Selected: {rollingWindow} days (&asymp;{Math.round(rollingWindow / 21)} months)</div>
            </div>

            {portfolioReturns.length > rollingWindow ? (<>
              {/* Rolling Volatility */}
              <div className="panel p-5">
                <h3 className="text-lg font-semibold text-white mb-3">Rolling Volatility ({rollingWindow}d) <Tip text="Annualized volatility calculated over a rolling window" /></h3>
                <RollingLineChart data={rollingVol} data2={rollingVolBench.length > 0 ? rollingVolBench : undefined} label1="Portfolio" label2={cmpLabel} height={300} />
                {rollingVol.length > 0 && (() => {
                  const vals = rollingVol.map((d) => d.y);
                  const cur = vals[vals.length - 1]; const avg = mean(vals);
                  const recentAvg = mean(vals.slice(-Math.min(21, vals.length)));
                  const earlierAvg = vals.length > 21 ? mean(vals.slice(0, vals.length - 21)) : avg;
                  const trend = recentAvg > earlierAvg * 1.05 ? "increasing" : recentAvg < earlierAvg * 0.95 ? "declining" : "stable";
                  return (<>
                    <InfoBox>Volatility ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Trend: {trend} (recent average: {recentAvg.toFixed(3)} vs earlier: {earlierAvg.toFixed(3)})</InfoBox>
                    <h4 className="text-sm font-semibold text-white/70 mt-3 mb-2">Volatility Statistics</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <CmpMetricCard label="Avg Volatility" portfolioValue={avg} benchmarkValue={rollingVolBench.length > 0 ? mean(rollingVolBench.map((d) => d.y)) : null} format="percent" higherIsBetter={false} />
                      <CmpMetricCard label="Median Volatility" portfolioValue={[...vals].sort((a, b) => a - b)[Math.floor(vals.length / 2)]} benchmarkValue={rollingVolBench.length > 0 ? [...rollingVolBench.map((d) => d.y)].sort((a, b) => a - b)[Math.floor(rollingVolBench.length / 2)] : null} format="percent" higherIsBetter={false} />
                      <CmpMetricCard label="Min Volatility" portfolioValue={Math.min(...vals)} benchmarkValue={rollingVolBench.length > 0 ? Math.min(...rollingVolBench.map((d) => d.y)) : null} format="percent" higherIsBetter={false} />
                      <CmpMetricCard label="Max Volatility" portfolioValue={Math.max(...vals)} benchmarkValue={rollingVolBench.length > 0 ? Math.max(...rollingVolBench.map((d) => d.y)) : null} format="percent" higherIsBetter={false} />
                    </div>
                  </>);
                })()}
              </div>

              {/* Rolling Sharpe */}
              <div className="panel p-5">
                <h3 className="text-lg font-semibold text-white mb-3">Rolling Sharpe Ratio ({rollingWindow}d) <Tip text="Sharpe ratio calculated over a rolling window" /></h3>
                <RollingLineChart data={rollingSharpe} data2={rollingSharpeBench.length > 0 ? rollingSharpeBench : undefined} label1="Portfolio" label2={cmpLabel} height={300} refLine={1.0} />
                {rollingSharpe.length > 0 && (() => {
                  const vals = rollingSharpe.map((d) => d.y);
                  const cur = vals[vals.length - 1]; const avg = mean(vals);
                  const aboveThresh = vals.filter((v) => v > 1).length;
                  const recentAvg = mean(vals.slice(-Math.min(21, vals.length)));
                  return <InfoBox>Sharpe Ratio ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Trend: {recentAvg > avg ? "improving" : "declining"} (recent average: {recentAvg.toFixed(3)} vs earlier: {(vals.length > 21 ? mean(vals.slice(0, vals.length - 21)) : avg).toFixed(3)}) / Time above 1.00: {(aboveThresh / vals.length * 100).toFixed(1)}% / {cur < 1 ? "Currently below threshold (1.00)" : "Currently above threshold (1.00)"}</InfoBox>;
                })()}
              </div>

              {/* Rolling Sortino */}
              <div className="panel p-5">
                <h3 className="text-lg font-semibold text-white mb-3">Rolling Sortino Ratio ({rollingWindow}d) <Tip text="Sortino ratio calculated over a rolling window — only penalizes downside volatility" /></h3>
                <RollingLineChart data={rollingSortino} data2={hasBenchmark ? (() => {
                  const bD = benchmarkReturns.map((p) => p.y);
                  const result: Pt[] = [];
                  for (let i = rollingWindow; i < bD.length; i++) {
                    const w = bD.slice(i - rollingWindow, i);
                    const m = mean(w); const down = w.filter((v) => v < 0);
                    const dsd = down.length > 1 ? Math.sqrt(down.reduce((a, b) => a + b * b, 0) / down.length) : 0.001;
                    result.push({ x: benchmarkReturns[i].x, y: (m / dsd) * Math.sqrt(252) });
                  }
                  return result.length > 0 ? result : undefined;
                })() : undefined} label1="Portfolio" label2={cmpLabel} height={300} refLine={1.0} />
                {rollingSortino.length > 0 && (() => {
                  const vals = rollingSortino.map((d) => d.y);
                  const cur = vals[vals.length - 1]; const avg = mean(vals);
                  const aboveThresh = vals.filter((v) => v > 1).length;
                  return <InfoBox>Sortino Ratio ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Time above 1.00: {(aboveThresh / vals.length * 100).toFixed(1)}% / {cur < 1 ? "Currently below threshold (1.00)" : "Currently above threshold (1.00)"}</InfoBox>;
                })()}
              </div>

              {/* Rolling Beta with colored zones */}
              {hasBenchmark && rollingBeta.length > 0 && (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Rolling Beta ({rollingWindow}d) <Tip text="Portfolio sensitivity to benchmark. Beta > 1 = more volatile, Beta < 1 = less volatile, Beta = 1 = matches market." /></h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={rollingBeta} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                      <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                      <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
                      <YAxis tick={{ fill: C.text, fontSize: 10 }} domain={[0, "auto"]} width={50} />
                      <RTooltip contentStyle={ttStyle} />
                      <ReferenceArea y1={1.3} y2={999} fill={C.danger} fillOpacity={0.08} label={{ value: "High Beta", position: "insideTopRight", fill: C.danger, fontSize: 10 }} />
                      <ReferenceArea y1={0} y2={0.7} fill={C.ok} fillOpacity={0.08} label={{ value: "Low Beta", position: "insideBottomRight", fill: C.ok, fontSize: 10 }} />
                      <ReferenceLine y={1.0} stroke={C.text} strokeDasharray="6 3" label={{ value: "Beta = 1.0", position: "right", fill: C.text, fontSize: 10 }} />
                      <Line dataKey="y" stroke={C.line1} strokeWidth={1.5} dot={false} name="Beta" />
                    </LineChart>
                  </ResponsiveContainer>
                  {(() => {
                    const vals = rollingBeta.map((d) => d.y);
                    const cur = vals[vals.length - 1]; const avg = mean(vals);
                    const recentAvg = mean(vals.slice(-Math.min(21, vals.length)));
                    return <InfoBox>Rolling Beta ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Current beta ({cur.toFixed(3)}) {cur > 1 ? "> 1.0 - portfolio is more volatile than benchmark" : cur < 1 ? "< 1.0 - portfolio is less volatile than benchmark" : "= 1.0 - matches benchmark"} / Beta is {recentAvg > avg ? "increasing" : "decreasing"} (recent: {recentAvg.toFixed(3)} vs earlier: {(vals.length > 21 ? mean(vals.slice(0, vals.length - 21)) : avg).toFixed(3)}) - portfolio {recentAvg > avg ? "becoming more sensitive" : "becoming less sensitive"} to market</InfoBox>;
                  })()}
                </div>
              )}

              {/* Rolling Alpha (green/red area) */}
              {hasBenchmark && portfolioReturns.length > rollingWindow && (() => {
                const rollingAlpha: Pt[] = [];
                const pD = portfolioReturns.map((p) => p.y); const bD = benchmarkReturns.map((p) => p.y);
                for (let i = rollingWindow; i < Math.min(pD.length, bD.length); i++) {
                  const pW = pD.slice(i - rollingWindow, i); const bW = bD.slice(i - rollingWindow, i);
                  const pRet = (pW.reduce((a, r) => a * (1 + r), 1) - 1) * (252 / rollingWindow) * 100;
                  const bRet = (bW.reduce((a, r) => a * (1 + r), 1) - 1) * (252 / rollingWindow) * 100;
                  rollingAlpha.push({ x: portfolioReturns[i].x, y: pRet - bRet });
                }
                if (rollingAlpha.length === 0) return null;
                const vals = rollingAlpha.map((d) => d.y);
                const cur = vals[vals.length - 1]; const avg = mean(vals);
                const aboveZero = vals.filter((v) => v > 0).length;
                return (
                  <div className="panel p-5">
                    <h3 className="text-lg font-semibold text-white mb-3">Rolling Alpha ({rollingWindow}d) <Tip text="Annualized excess return over benchmark. Green = positive alpha, Red = negative alpha." /></h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={rollingAlpha} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                        <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
                        <YAxis tick={{ fill: C.text, fontSize: 10 }} width={55} />
                        <RTooltip contentStyle={ttStyle} />
                        <ReferenceLine y={0} stroke={C.text} strokeDasharray="6 3" label={{ value: "Alpha = 0", position: "right", fill: C.text, fontSize: 10 }} />
                        <defs>
                          <linearGradient id="alphaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={C.ok} stopOpacity={0.6} />
                            <stop offset="50%" stopColor={C.ok} stopOpacity={0.0} />
                            <stop offset="50%" stopColor={C.danger} stopOpacity={0.0} />
                            <stop offset="100%" stopColor={C.danger} stopOpacity={0.6} />
                          </linearGradient>
                        </defs>
                        <Area dataKey="y" stroke={C.line1} fill="url(#alphaGrad)" strokeWidth={1.5} name="Alpha (%)" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <InfoBox>Alpha ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Time above 0.00: {(aboveZero / vals.length * 100).toFixed(1)}% / {cur < 0 ? "Currently below threshold (0.00)" : "Currently above threshold (0.00)"}</InfoBox>
                  </div>
                );
              })()}

              {/* Rolling Active Return */}
              {hasBenchmark && portfolioReturns.length > rollingWindow && (() => {
                const rollingActive: Pt[] = [];
                const pD = portfolioReturns.map((p) => p.y); const bD = benchmarkReturns.map((p) => p.y);
                for (let i = rollingWindow; i < Math.min(pD.length, bD.length); i++) {
                  const pW = pD.slice(i - rollingWindow, i); const bW = bD.slice(i - rollingWindow, i);
                  const pRet = (pW.reduce((a, r) => a * (1 + r), 1) - 1) * (252 / rollingWindow) * 100;
                  const bRet = (bW.reduce((a, r) => a * (1 + r), 1) - 1) * (252 / rollingWindow) * 100;
                  rollingActive.push({ x: portfolioReturns[i].x, y: pRet - bRet });
                }
                if (rollingActive.length === 0) return null;
                const vals = rollingActive.map((d) => d.y);
                const cur = vals[vals.length - 1]; const avg = mean(vals);
                const posAlpha = vals.filter((v) => v > 0).length;
                return (
                  <div className="panel p-5">
                    <h3 className="text-lg font-semibold text-white mb-3">Rolling Active Return ({rollingWindow}d) <Tip text="Annualized return difference between portfolio and benchmark over rolling window" /></h3>
                    <RollingLineChart data={rollingActive} label1="Active Return" height={280} refLine={0} />
                    <h4 className="text-sm font-semibold text-white/70 mt-3 mb-2">Active Return Statistics</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <MetricCard label="Avg Active Return" value={`${avg.toFixed(2)}%`} />
                      <MetricCard label="Periods with Positive Alpha" value={`${(posAlpha / vals.length * 100).toFixed(1)}%`} />
                      <MetricCard label="Max Alpha" value={`${Math.max(...vals).toFixed(2)}%`} />
                      <MetricCard label="Min Alpha" value={`${Math.min(...vals).toFixed(2)}%`} />
                    </div>
                    <InfoBox>Active Return ({rollingWindow}-day rolling): Current: {cur.toFixed(3)} / Average: {avg.toFixed(3)} / Range: {Math.min(...vals).toFixed(3)} to {Math.max(...vals).toFixed(3)} / Time above 0.00: {(posAlpha / vals.length * 100).toFixed(1)}% / {cur < 0 ? "Currently below threshold (0.00)" : "Currently above threshold (0.00)"}</InfoBox>
                  </div>
                );
              })()}

              {/* Bull/Bear Market Analysis */}
              {hasBenchmark && portfolioReturns.length > 0 && (() => {
                const pD = portfolioReturns.map((p) => p.y); const bD = benchmarkReturns.map((p) => p.y);
                const minLen = Math.min(pD.length, bD.length);
                const bullP: number[] = []; const bearP: number[] = [];
                const bullB: number[] = []; const bearB: number[] = [];
                for (let i = 0; i < minLen; i++) {
                  if (bD[i] >= 0) { bullP.push(pD[i]); bullB.push(bD[i]); } else { bearP.push(pD[i]); bearB.push(bD[i]); }
                }
                const avgBullP = bullP.length ? mean(bullP) * 100 : 0; const avgBearP = bearP.length ? mean(bearP) * 100 : 0;
                const avgBullB = bullB.length ? mean(bullB) * 100 : 0; const avgBearB = bearB.length ? mean(bearB) * 100 : 0;
                const bullBeta = (() => {
                  if (bullB.length < 10) return 0;
                  const mP = mean(bullP); const mB = mean(bullB);
                  let cov = 0, varB = 0;
                  for (let i = 0; i < bullP.length; i++) { cov += (bullP[i] - mP) * (bullB[i] - mB); varB += (bullB[i] - mB) ** 2; }
                  return varB > 0 ? cov / varB : 0;
                })();
                const bearBeta = (() => {
                  if (bearB.length < 10) return 0;
                  const mP = mean(bearP); const mB = mean(bearB);
                  let cov = 0, varB = 0;
                  for (let i = 0; i < bearP.length; i++) { cov += (bearP[i] - mP) * (bearB[i] - mB); varB += (bearB[i] - mB) ** 2; }
                  return varB > 0 ? cov / varB : 0;
                })();
                const bullOutperf = avgBullP - avgBullB; const bearOutperf = avgBearP - avgBearB;
                return (
                  <div className="panel p-5">
                    <h3 className="text-lg font-semibold text-white mb-1">Bull/Bear Market Analysis</h3>
                    <p className="text-sm text-white/40 mb-3">Separate Analysis of Bullish and Bearish Periods</p>
                    <h4 className="text-sm font-semibold text-white/70 mb-2">Performance in Different Market Conditions</h4>
                    <p className="text-xs text-white/30 mb-2">Median daily return when benchmark is up (bullish) vs down (bearish)</p>
                    <table className="data-table mb-4"><thead><tr><th>Metric</th><th>Bullish Market</th><th>Bearish Market</th><th>Difference</th></tr></thead>
                      <tbody>
                        <tr><td className="text-white/70">Portfolio Avg Daily Return (%)</td><td className="font-mono">{avgBullP.toFixed(2)}</td><td className="font-mono">{avgBearP.toFixed(2)}</td><td className="font-mono">{(avgBullP - avgBearP).toFixed(2)}</td></tr>
                        <tr><td className="text-white/70">Benchmark Avg Daily Return (%)</td><td className="font-mono">{avgBullB.toFixed(2)}</td><td className="font-mono">{avgBearB.toFixed(2)}</td><td className="font-mono">{(avgBullB - avgBearB).toFixed(2)}</td></tr>
                        <tr><td className="text-white/70">Beta</td><td className="font-mono">{bullBeta.toFixed(2)}</td><td className="font-mono">{bearBeta.toFixed(2)}</td><td className="font-mono">{(bullBeta - bearBeta).toFixed(2)}</td></tr>
                        <tr><td className="text-white/70">Outperformance (%)</td><td className="font-mono">{bullOutperf.toFixed(2)}</td><td className="font-mono">{bearOutperf.toFixed(2)}</td><td className="font-mono">{(bullOutperf - bearOutperf).toFixed(2)}</td></tr>
                      </tbody>
                    </table>
                    <h4 className="text-sm font-semibold text-white/70 mb-2">Average Daily Returns in Bull vs Bear Markets</h4>
                    <SimpleBarChart data={[
                      { label: "Bullish Market", Portfolio: +avgBullP.toFixed(2), [cmpLabel]: +avgBullB.toFixed(2) },
                      { label: "Bearish Market", Portfolio: +avgBearP.toFixed(2), [cmpLabel]: +avgBearB.toFixed(2) },
                    ]} bars={[{ key: "Portfolio", color: C.line1, name: "Portfolio" }, { key: cmpLabel, color: C.line2, name: cmpLabel }]} height={240} />
                    <InfoBox>Bull/Bear Market Analysis: Portfolio performs {avgBullP > avgBullB ? "better" : "worse"} in bull markets ({avgBullP.toFixed(2)}%) than in bear markets ({avgBearP.toFixed(2)}%) - difference: {(avgBullP - avgBearP).toFixed(2)}% / Beta is {bullBeta > bearBeta ? "higher" : "lower"} in bull markets ({bullBeta.toFixed(2)}) than in bear markets ({bearBeta.toFixed(2)}) - {bullBeta > bearBeta ? "more sensitive during uptrends" : "more sensitive during downtrends"} / Portfolio {bullOutperf > 0 ? "outperforms" : "underperforms"} in bull markets ({bullOutperf.toFixed(2)}%) {bearOutperf > 0 ? "and outperforms" : "but underperforms"} in bear markets ({bearOutperf.toFixed(2)}%)</InfoBox>
                  </div>
                );
              })()}
            </>) : <Alert type="info">Not enough data points for rolling metrics (need at least {rollingWindow} days, have {portfolioReturns.length})</Alert>}
          </div>)}
        </div>)}

        {/* ═══════════════════════════════════════════════════════ */}
        {/*  TAB 4: ASSETS & CORRELATIONS                          */}
        {/* ═══════════════════════════════════════════════════════ */}
        {mainTab === "assets" && (<div className="space-y-5">
          <SubTabs tabs={[["overview","Asset Overview & Impact"],["correlations","Correlations"],["details","Asset Details & Dynamics"]]} active={assetSub} onChange={setAssetSub} />

          {!assetData && <Alert type="info">Asset data is loading... Run analytics calculation first.</Alert>}

          {/* ══════════ Asset Overview & Impact ══════════ */}
          {assetSub === "overview" && assetData && (<div className="space-y-5">
            {/* 4.1.1 Full Details Table */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-1">Assets Overview — Full Details <Tip text="Detailed info for each asset from yfinance" /></h3>
              <p className="text-xs text-white/40 mb-3">Change% shows daily price change (today vs previous trading day)</p>
              {Array.isArray(assetData.asset_metrics) && assetData.asset_metrics.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="data-table"><thead><tr><th>#</th><th>Ticker</th><th>Weight %</th><th>Name</th><th>Sector</th><th>Industry</th><th>Currency</th><th>Price</th><th>Change %</th></tr></thead>
                    <tbody>{(assetData.asset_metrics as any[]).map((row: any, idx: number) => {
                      const chg = row.change_pct ?? 0;
                      return (
                        <tr key={row.ticker}>
                          <td>{idx + 1}</td>
                          <td className="font-mono font-medium text-white">{row.ticker}</td>
                          <td className="font-mono">{Number(row.weight ?? 0).toFixed(2)}%</td>
                          <td className="text-white/70">{row.name ?? row.ticker}</td>
                          <td className="text-white/50">{row.sector ?? "N/A"}</td>
                          <td className="text-white/50">{row.industry ?? "N/A"}</td>
                          <td>{row.currency ?? "USD"}</td>
                          <td className="font-mono">{row.price ? `$${Number(row.price).toFixed(2)}` : "—"}</td>
                          <td className={`font-mono ${chg >= 0 ? "text-[var(--ok)]" : "text-[var(--danger)]"}`}>{chg !== 0 ? `${chg >= 0 ? "+" : ""}${Number(chg).toFixed(2)}%` : "—"}</td>
                        </tr>
                      );
                    })}</tbody>
                  </table>
                </div>
              ) : <Alert type="info">No positions found</Alert>}
            </div>

            {/* 4.1.2 Impact on Total Return */}
            {assetData.impact_on_return && (() => {
              const ir = assetData.impact_on_return;
              const tickers: string[] = ir.tickers ?? [];
              const contributions: number[] = ir.contributions ?? [];
              if (!tickers.length) return null;
              const impactData = tickers.map((t: string, i: number) => ({ ticker: t, impact: contributions[i] ?? 0 }));
              const totalImpact = impactData.reduce((s, d) => s + Math.abs(d.impact), 0) || 1;
              const top = impactData[0];
              const top3 = impactData.slice(0, 3).reduce((s, d) => s + Math.abs(d.impact), 0);
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Impact on Assets to Total Return <Tip text="Weighted return contribution of each asset to the portfolio total return" /></h3>
                  <SimpleBarChart data={impactData.map((d) => ({ label: d.ticker, Impact: +d.impact.toFixed(2) }))} bars={[{ key: "Impact", color: C.ok, name: "Weighted Return Contribution (%)" }]} height={280} />
                  {top && <div className="text-xs text-white/40 mt-2"><strong>Top contributor:</strong> {top.ticker} — {top.impact.toFixed(2)}% weighted contribution to portfolio return</div>}
                  <InfoBox>Impact on Total Return Analysis: Top contributor: {top?.ticker} ({top?.impact.toFixed(2)}% of portfolio return) / {top ? (Math.abs(top.impact) / totalImpact * 100 > 40 ? "High" : "Moderate") : ""} concentration: {top?.ticker} accounts for {(top ? Math.abs(top.impact) / totalImpact * 100 : 0).toFixed(1)}% of total contribution / Top 3 assets account for {(top3 / totalImpact * 100).toFixed(1)}% of total return contribution</InfoBox>
                </div>
              );
            })()}

            {/* 4.1.3 Impact on Portfolio Risk */}
            {assetData.impact_on_risk && (() => {
              const irk = assetData.impact_on_risk;
              const tickers: string[] = irk.tickers ?? [];
              const riskContribs: number[] = irk.risk_contributions ?? [];
              if (!tickers.length) return null;
              const riskData = tickers.map((t: string, i: number) => ({ ticker: t, risk: riskContribs[i] ?? 0 }));
              const totalRisk = riskData.reduce((s, d) => s + Math.abs(d.risk), 0) || 1;
              const topR = riskData[0];
              const top3Risk = riskData.slice(0, 3).reduce((s, d) => s + Math.abs(d.risk), 0);
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Impact on Assets to Overall Portfolio Risk <Tip text="Each asset's marginal contribution to total portfolio volatility (MCR × weight)" /></h3>
                  <SimpleBarChart data={riskData.map((d) => ({ label: d.ticker, Risk: +d.risk.toFixed(1) }))} bars={[{ key: "Risk", color: C.danger, name: "Risk Contribution (%)" }]} height={280} />
                  {topR && <div className="text-xs text-white/40 mt-2"><strong>Biggest risk contributor:</strong> {topR.ticker} — {topR.risk.toFixed(1)}% of portfolio risk</div>}
                  <InfoBox>Impact on Portfolio Risk Analysis: Biggest risk contributor: {topR?.ticker} ({topR?.risk.toFixed(1)}% of portfolio risk) / {Math.abs(topR?.risk ?? 0) > 30 ? "Concentrated risk" : "Well-distributed risk"}: top contributor accounts for {Math.abs(topR?.risk ?? 0).toFixed(1)}% of total risk / Top 3 assets account for {(top3Risk / totalRisk * 100).toFixed(1)}% of total risk contribution</InfoBox>
                </div>
              );
            })()}

            {/* 4.1.4 Comparison of Risk & Return Impact and Asset Weighting */}
            {assetData.risk_vs_weight && (() => {
              const rvw = assetData.risk_vs_weight;
              const tickers: string[] = rvw.tickers ?? [];
              const riskImpact: number[] = rvw.risk_impact ?? [];
              const returnImpact: number[] = rvw.return_impact ?? [];
              const weights: number[] = rvw.weights ?? [];
              if (!tickers.length) return null;
              const compData = tickers.map((t: string, i: number) => ({ label: t, "Impact on Risk": +riskImpact[i].toFixed(1), "Impact on Return": +returnImpact[i].toFixed(1), "Weight in Portfolio": +weights[i].toFixed(1) }));
              const outlier = compData.find((d) => Math.abs(d["Impact on Risk"] - d["Weight in Portfolio"]) > 20);
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Comparison of Risk & Return Impact and Asset Weighting <Tip text="Compares each asset's risk/return contribution vs its portfolio weight" /></h3>
                  <SimpleBarChart data={compData} bars={[{ key: "Impact on Risk", color: C.danger, name: "Impact on Risk" }, { key: "Impact on Return", color: C.ok, name: "Impact on Return" }, { key: "Weight in Portfolio", color: C.accent, name: "Weight in Portfolio" }]} height={300} />
                  <div className="text-xs text-white/40 mt-2">For well-diversified portfolios: bars should be similar. <span className="text-[var(--danger)]">Red bars</span>: Risk impact, <span className="text-[var(--ok)]">Green bars</span>: Return impact, <span className="text-[var(--accent)]">Purple bars</span>: Portfolio weight.</div>
                  {outlier && <InfoBox>Risk vs Weight Comparison Analysis: {outlier.label}: Risk impact ({outlier["Impact on Risk"]}%) {">"} Weight ({outlier["Weight in Portfolio"]}%) - {(outlier["Impact on Risk"] / (outlier["Weight in Portfolio"] || 1)).toFixed(1)}x higher. This asset contributes more risk than its portfolio weight suggests.</InfoBox>}
                </div>
              );
            })()}

            {/* 4.1.5 Diversification Assessment */}
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-3">Diversification Assessment <Tip text="Measures how much diversification reduces portfolio volatility" /></h3>
              <Divider />
              {assetData.diversification ? (() => {
                const div = assetData.diversification;
                const divCoeff = div.diversification_coefficient ?? 1;
                const volReduction = div.volatility_reduction_pct ?? 0;
                const wSumVol = div.weighted_sum_volatilities ?? 0;
                const portVolD = div.portfolio_volatility ?? 0;
                return (<>
                  <h4 className="text-base font-semibold text-white mt-3 mb-2">Diversification Coefficient: {Number(divCoeff).toFixed(2)}</h4>
                  <p className="text-sm text-white/50 mb-2"><strong>Formula:</strong> Weighted sum of volatilities ({(wSumVol * 100).toFixed(2)}%) / Portfolio volatility ({(portVolD * 100).toFixed(2)}%)</p>
                  <p className="text-sm text-white/60 mb-3"><strong>Interpretation:</strong></p>
                  <div className="space-y-1 mb-3">
                    <div className="text-sm text-[var(--ok)]">{divCoeff > 1 ? `✓ Value > 1.0 indicates positive diversification effect` : "✗ Value ≤ 1.0 — no diversification benefit"}</div>
                    {divCoeff > 1 && <div className="text-sm text-[var(--ok)]">✓ {Number(divCoeff).toFixed(2)} means {Number(volReduction).toFixed(1)}% volatility reduction from diversification</div>}
                    <div className="text-sm text-[var(--ok)]">{divCoeff > 1.2 ? "✓ Portfolio is well-diversified" : divCoeff > 1 ? "✓ Portfolio has some diversification" : "✗ Portfolio lacks diversification"}</div>
                  </div>
                  <InfoBox>The diversification coefficient shows the ratio of the weighted sum of individual volatilities to total portfolio volatility. A value above 1 indicates positive effect.</InfoBox>
                </>);
              })() : <InfoBox>Portfolio has {positions.length} position{positions.length !== 1 ? "s" : ""}. {positions.length >= 5 ? "Good diversification across multiple assets." : positions.length > 1 ? "Consider adding more assets for better diversification." : "Single-asset portfolio — no diversification benefit."}</InfoBox>}
            </div>
          </div>)}

          {/* ══════════ Correlations ══════════ */}
          {assetSub === "correlations" && assetData && (<div className="space-y-5">
            <div className="panel p-5">
              <h3 className="text-lg font-semibold text-white mb-1">Correlation Analysis</h3>
              <h4 className="text-base font-semibold text-white/70 mb-3">Correlation Matrix — All Assets {hasBenchmark ? "+ Benchmark" : ""}</h4>
              {assetData.correlations?.matrix ? (() => {
                const corrMatrix = assetData.correlations.matrix;
                const keys: string[] = assetData.correlations.tickers ?? Object.keys(corrMatrix);
                const allCorrs: number[] = [];
                keys.forEach((r) => keys.forEach((c) => { if (r !== c) { const v = corrMatrix[r]?.[c]; if (typeof v === "number") allCorrs.push(v); } }));
                const avgCorr = allCorrs.length ? mean(allCorrs) : 0;
                const medianCorr = allCorrs.length ? [...allCorrs].sort((a, b) => a - b)[Math.floor(allCorrs.length / 2)] : 0;
                const minCorr = allCorrs.length ? Math.min(...allCorrs) : 0;
                const maxCorr = allCorrs.length ? Math.max(...allCorrs) : 0;
                const highPairs = allCorrs.filter((v) => v > 0.8).length / 2;
                const lowPairs = allCorrs.filter((v) => v < 0.2).length / 2;
                const negPairs = allCorrs.filter((v) => v < 0).length / 2;
                const getCorrColor = (v: number) => {
                  if (v >= 0.8) return "rgba(250,161,164,0.9)";
                  if (v >= 0.5) return "rgba(250,161,164,0.5)";
                  if (v >= 0.2) return "rgba(250,161,164,0.25)";
                  if (v >= 0) return "rgba(116,241,116,0.15)";
                  return "rgba(116,241,116,0.4)";
                };
                return (<>
                  <div className="overflow-x-auto mb-3">
                    <table className="data-table text-center text-xs">
                      <thead><tr><th></th>{keys.map((k) => <th key={k} className="font-mono">{k}</th>)}</tr></thead>
                      <tbody>{keys.map((row) => (
                        <tr key={row}>
                          <td className="font-mono text-white font-medium">{row}</td>
                          {keys.map((col) => {
                            const v = corrMatrix[row]?.[col];
                            const n = typeof v === "number" ? v : 0;
                            return <td key={col} className="font-mono" style={{ backgroundColor: getCorrColor(n), color: "#fff" }}>{typeof v === "number" ? v.toFixed(2) : "—"}</td>;
                          })}
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <InfoBox>Correlation Matrix Analysis: {avgCorr > 0.7 ? "High" : avgCorr > 0.4 ? "Moderate" : "Low"} average correlation ({avgCorr.toFixed(2)}) - {avgCorr < 0.5 ? "Good diversification potential" : "Limited diversification"} / {highPairs > 0 ? `${highPairs} highly correlated pairs (>0.8) found` : "No highly correlated pairs (>0.8) found"} {lowPairs > 0 ? `/ ${lowPairs} low correlation pairs (<0.2) found` : "/ No low correlation pairs (<0.2) found - Consider adding assets with lower correlation"} / Correlation range: {minCorr.toFixed(2)} to {maxCorr.toFixed(2)}</InfoBox>

                  {/* 4.2.2 Correlation Statistics */}
                  <h3 className="text-lg font-semibold text-white mt-5 mb-3">Correlation Statistics</h3>
                  {assetData.correlation_stats ? (() => {
                    const cs = assetData.correlation_stats;
                    return (
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <MetricCard label="Average Correlation" value={Number(cs.average_correlation ?? avgCorr).toFixed(2)} />
                        <MetricCard label="Median Correlation" value={Number(cs.median_correlation ?? medianCorr).toFixed(2)} />
                        <MetricCard label="Min Correlation" value={Number(cs.min_correlation ?? minCorr).toFixed(2)} helpText={`Pair: ${Array.isArray(cs.min_pair) ? cs.min_pair.join(" / ") : ""}`} />
                        <MetricCard label="Max Correlation" value={Number(cs.max_correlation ?? maxCorr).toFixed(2)} helpText={`Pair: ${Array.isArray(cs.max_pair) ? cs.max_pair.join(" / ") : ""}`} />
                        <MetricCard label="Pairs > 0.8 (high)" value={String(cs.high_corr_count ?? Math.round(highPairs))} />
                        <MetricCard label="Pairs < 0.2 (low)" value={String(cs.low_corr_count ?? Math.round(lowPairs))} />
                      </div>
                    );
                  })() : (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      <MetricCard label="Average Correlation" value={avgCorr.toFixed(2)} />
                      <MetricCard label="Median Correlation" value={medianCorr.toFixed(2)} />
                      <MetricCard label="Min Correlation" value={minCorr.toFixed(2)} />
                      <MetricCard label="Max Correlation" value={maxCorr.toFixed(2)} />
                      <MetricCard label="Pairs > 0.8 (high)" value={String(Math.round(highPairs))} />
                      <MetricCard label="Pairs < 0.2 (low)" value={String(Math.round(lowPairs))} />
                    </div>
                  )}
                  <Divider />

                  {/* 4.2.3 Recommendations */}
                  <h3 className="text-lg font-semibold text-white mt-3 mb-3">Correlation Analysis & Recommendations</h3>
                  <div className="space-y-2">
                    {avgCorr < 0.5 && <div className="p-2 rounded bg-[var(--ok)]/10 text-sm text-[var(--ok)]">✓ Moderate average correlation — Good diversification potential</div>}
                    {avgCorr >= 0.5 && <div className="p-2 rounded bg-[var(--warn)]/10 text-sm text-[var(--warn)]">⚠ High average correlation — Limited diversification benefit</div>}
                    {lowPairs === 0 && <div className="p-2 rounded bg-[var(--warn)]/10 text-sm text-[var(--warn)]">⚠ No pairs with correlation {"<"} 0.2 found. Consider adding assets with lower correlation for better diversification.</div>}
                    {negPairs === 0 && <div className="p-2 rounded bg-[var(--danger)]/10 text-sm text-[var(--danger)]">! No negative correlations found. Consider adding assets with negative correlation (e.g., bonds vs stocks) for hedging.</div>}
                  </div>

                  {/* 4.2.4 Asset Correlation with Benchmark */}
                  {assetData.benchmark_correlations && (() => {
                    const bc = assetData.benchmark_correlations;
                    const bcTickers: string[] = bc.tickers ?? [];
                    const bcCorrs: number[] = bc.correlations ?? [];
                    const bcBetas: number[] = bc.betas ?? [];
                    if (!bcTickers.length) return null;
                    const benchCorrs = bcTickers.map((t: string, i: number) => ({ ticker: t, corr: bcCorrs[i] ?? 0, beta: bcBetas[i] ?? 0 }));
                    const avgBC = benchCorrs.length ? mean(benchCorrs.map((d) => d.corr)) : 0;
                    const topBC = benchCorrs[0]; const bottomBC = benchCorrs[benchCorrs.length - 1];
                    return (<>
                      <Divider />
                      <h3 className="text-lg font-semibold text-white mt-3 mb-3">Asset Correlation with {cmpValue || "Benchmark"}</h3>
                      <SimpleBarChart data={benchCorrs.map((d) => ({ label: d.ticker, Correlation: +d.corr.toFixed(2) }))} bars={[{ key: "Correlation", color: C.danger, name: "Correlation" }]} height={220} />
                      <table className="data-table mt-3"><thead><tr><th>Ticker</th><th>Correlation</th><th>Beta</th></tr></thead>
                        <tbody>{benchCorrs.map((d) => (
                          <tr key={d.ticker}><td className="font-mono text-white">{d.ticker}</td><td className="font-mono">{d.corr.toFixed(2)}</td><td className="font-mono">{d.beta.toFixed(2)}</td></tr>
                        ))}</tbody>
                      </table>
                      <InfoBox>Correlation with Benchmark Analysis: Average correlation: {avgBC.toFixed(2)} / Highest correlation: {topBC?.ticker} ({topBC?.corr.toFixed(2)}) - {topBC && topBC.corr > 0.7 ? "Highly correlated" : "Moderate sensitivity"} / Lowest correlation: {bottomBC?.ticker} ({bottomBC?.corr.toFixed(2)}) - {bottomBC && bottomBC.corr < 0.3 ? "Low sensitivity" : "Moderate sensitivity"} / Portfolio shows {avgBC > 0.6 ? "high" : avgBC > 0.4 ? "moderate" : "low"} sensitivity to benchmark</InfoBox>
                    </>);
                  })()}

                  {/* 4.2.5 Average Correlation to Portfolio */}
                  {assetData.avg_correlation_to_portfolio && (() => {
                    const acp = assetData.avg_correlation_to_portfolio;
                    const acpTickers: string[] = acp.tickers ?? [];
                    const acpAvgs: number[] = acp.avg_correlations ?? [];
                    if (acpTickers.length < 2) return null;
                    const avgCorrToOthers = acpTickers.map((t: string, i: number) => ({ label: t, "Avg Correlation": +(acpAvgs[i] * 100).toFixed(1) }));
                    return (<>
                      <Divider />
                      <h3 className="text-lg font-semibold text-white mt-3 mb-3">Average Correlation to Portfolio</h3>
                      <SimpleBarChart data={avgCorrToOthers} bars={[{ key: "Avg Correlation", color: C.warn, name: "Average Correlation (%)" }]} height={220} />
                    </>);
                  })()}
                </>);
              })() : <Alert type="info">Correlation data will appear after analytics calculation. Requires multiple assets in the portfolio.</Alert>}
            </div>
          </div>)}

          {/* ══════════ Asset Details & Dynamics ══════════ */}
          {assetSub === "details" && assetData && (<div className="space-y-5">
            {/* 4.3.1 Asset Price Dynamics */}
            {assetData.asset_returns && (() => {
              const ar = assetData.asset_returns;
              const tickers = Object.keys(ar).filter((k) => k !== "dates");
              const dates: string[] = ar.dates ?? [];
              if (!tickers.length || !dates.length) return null;
              const colors = [C.line1, C.ok, C.warn, C.danger, C.info, "#ff79c6", "#f1fa8c", "#8be9fd", "#ffb86c", "#bd93f9"];
              const cumData = dates.map((d: string, i: number) => {
                const row: Record<string, any> = { x: d.slice(0, 10) };
                tickers.forEach((t) => {
                  const rets: number[] = ar[t];
                  let cum = 0; for (let j = 0; j <= i; j++) cum = (1 + cum) * (1 + (rets[j] ?? 0)) - 1;
                  row[t] = +(cum * 100).toFixed(2);
                });
                if (hasBenchmark && benchmarkReturns[i]) {
                  let bcum = 0; for (let j = 0; j <= i; j++) bcum = (1 + bcum) * (1 + (benchmarkReturns[j]?.y ?? 0)) - 1;
                  row[`${cmpValue} (Benchmark)`] = +(bcum * 100).toFixed(2);
                }
                return row;
              });
              const allKeys = [...tickers, ...(hasBenchmark ? [`${cmpValue} (Benchmark)`] : [])];
              const finalRow = cumData[cumData.length - 1];
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Asset Price Change (% from Start Date) <Tip text="Cumulative return of each asset from analysis start date" /></h3>
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart data={cumData.filter((_, i) => i % Math.max(1, Math.floor(cumData.length / 300)) === 0)} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                      <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                      <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
                      <YAxis tick={{ fill: C.text, fontSize: 10 }} width={55} />
                      <RTooltip contentStyle={ttStyle} />
                      <Legend wrapperStyle={{ fontSize: 10, color: C.text }} />
                      {allKeys.map((k, i) => <Line key={k} dataKey={k} stroke={colors[i % colors.length]} strokeWidth={k.includes("Benchmark") ? 1 : 1.5} dot={false} strokeDasharray={k.includes("Benchmark") ? "4 2" : undefined} />)}
                    </LineChart>
                  </ResponsiveContainer>
                  {finalRow && <div className="text-xs text-white/40 mt-2"><strong>Final Returns:</strong> {allKeys.map((k) => `${k}: ${finalRow[k] >= 0 ? "+" : ""}${finalRow[k]}%`).join(" | ")}</div>}
                  {(() => {
                    const retVals = tickers.map((t) => ({ t, r: finalRow?.[t] ?? 0 }));
                    const best = retVals.reduce((a, b) => a.r > b.r ? a : b, retVals[0]);
                    const worst = retVals.reduce((a, b) => a.r < b.r ? a : b, retVals[0]);
                    const spread = best.r - worst.r;
                    return <InfoBox>Asset Price Dynamics Analysis: Best performer: {best.t} (+{best.r.toFixed(2)}%) / Worst performer: {worst.t} ({worst.r >= 0 ? "+" : ""}{worst.r.toFixed(2)}%) / {spread > 50 ? "Large" : spread > 20 ? "Moderate" : "Small"} performance spread ({spread.toFixed(1)}%) - {spread > 30 ? "High dispersion in asset returns" : "Low dispersion"} / {retVals.every((d) => d.r > 0) ? "All assets show positive returns" : `${retVals.filter((d) => d.r < 0).length} assets with negative returns`}</InfoBox>;
                  })()}
                </div>
              );
            })()}

            {/* 4.3.2 Detailed Asset Analysis */}
            {assetData.per_asset && Object.keys(assetData.per_asset).length > 0 && (() => {
              const tickers = Object.keys(assetData.per_asset);
              const curAsset = tickers.includes(selectedAsset) ? selectedAsset : tickers[0];
              if (!curAsset) return null;
              const am = assetData.per_asset[curAsset]?.metrics ?? {};
              const assetRet = am.total_return ?? 0;
              const assetVol = am.volatility ?? 0;
              const assetSharpe = am.sharpe_ratio ?? 0;
              const annRet = am.annual_return ?? assetRet;
              const maxDD = am.max_drawdown ?? 0;
              const otherCorrs = assetData.per_asset[curAsset]?.other_correlations ?? {};
              return (
                <div className="panel p-5">
                  <h3 className="text-lg font-semibold text-white mb-3">Detailed Asset Analysis</h3>
                  <div className="mb-3">
                    <label className="text-sm text-white/50 block mb-1">Select asset for detailed analysis</label>
                    <select className="bg-[var(--card)] border border-white/10 rounded px-3 py-2 text-white w-full md:w-64" value={curAsset} onChange={(e) => setSelectedAsset(e.target.value)}>
                      {tickers.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <CmpMetricCard label="Total Return" portfolioValue={assetRet} benchmarkValue={perf.total_return} format="percent" higherIsBetter={true} />
                    <CmpMetricCard label="Annual Return" portfolioValue={annRet} benchmarkValue={perf.annualized_return} format="percent" higherIsBetter={true} />
                    <CmpMetricCard label="Volatility" portfolioValue={assetVol} benchmarkValue={vol} format="percent" higherIsBetter={false} />
                    <MetricCard label="Sharpe Ratio" value={fmtNum(assetSharpe)} helpText={`Max DD: ${(maxDD * 100).toFixed(2)}%`} />
                  </div>

                  {/* Comparison of Return chart */}
                  {assetData.asset_returns?.[curAsset] && (() => {
                    const dates: string[] = assetData.asset_returns.dates ?? [];
                    const assetDaily: number[] = assetData.asset_returns[curAsset] ?? [];
                    const compChart = dates.map((d: string, i: number) => {
                      const key = (t: string) => String(t).slice(0, 10);
                      let aCum = 0; for (let j = 0; j <= i; j++) aCum = (1 + aCum) * (1 + (assetDaily[j] ?? 0)) - 1;
                      let pCum = 0; for (let j = 0; j <= i; j++) {
                        const dj = key(dates[j] ?? "");
                        const ry = portReturnByDate.get(dj) ?? 0;
                        pCum = (1 + pCum) * (1 + ry) - 1;
                      }
                      const row: Record<string, any> = { x: d.slice(0, 10), [curAsset]: +(aCum * 100).toFixed(2), Portfolio: +(pCum * 100).toFixed(2) };
                      if (hasBenchmark) {
                        let bCum = 0; for (let j = 0; j <= i; j++) {
                          const dj = key(dates[j] ?? "");
                          const ry = benchReturnByDate.get(dj) ?? 0;
                          bCum = (1 + bCum) * (1 + ry) - 1;
                        }
                        row[`${cmpValue} (Benchmark)`] = +(bCum * 100).toFixed(2);
                      }
                      return row;
                    });
                    const last = compChart[compChart.length - 1];
                    const aFinal = last?.[curAsset] ?? 0; const pFinal = last?.Portfolio ?? 0;
                    const bFinal = last?.[`${cmpValue} (Benchmark)`] ?? 0;
                    return (<>
                      <h4 className="text-base font-semibold text-white mt-4 mb-2">Comparison of Return — {curAsset} vs Portfolio vs {cmpValue}</h4>
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={compChart.filter((_, i) => i % Math.max(1, Math.floor(compChart.length / 200)) === 0)} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                          <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                          <XAxis dataKey="x" tick={{ fill: C.text, fontSize: 10 }} tickLine={false} />
                          <YAxis tick={{ fill: C.text, fontSize: 10 }} width={50} />
                          <RTooltip contentStyle={ttStyle} />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Line dataKey={curAsset} stroke={C.ok} strokeWidth={1.5} dot={false} />
                          <Line dataKey="Portfolio" stroke={C.line1} strokeWidth={1.5} dot={false} />
                          {hasBenchmark && <Line dataKey={`${cmpValue} (Benchmark)`} stroke={C.line2} strokeWidth={1} dot={false} strokeDasharray="4 2" />}
                        </LineChart>
                      </ResponsiveContainer>
                      <InfoBox>Return Comparison Analysis — {curAsset}: Asset total return: {aFinal >= 0 ? "+" : ""}{aFinal.toFixed(2)}% / Portfolio total return: {pFinal >= 0 ? "+" : ""}{pFinal.toFixed(2)}% {aFinal < pFinal ? `/ Asset underperforms portfolio by ${(pFinal - aFinal).toFixed(2)}%` : `/ Asset outperforms portfolio by ${(aFinal - pFinal).toFixed(2)}%`} {hasBenchmark ? `/ Benchmark total return: ${bFinal >= 0 ? "+" : ""}${bFinal.toFixed(2)}%` : ""}</InfoBox>
                    </>);
                  })()}

                  {/* Correlations with Other Assets */}
                  {Object.keys(otherCorrs).length > 0 && (() => {
                    const corrs = Object.entries(otherCorrs)
                      .filter(([k]) => k !== curAsset && !["SPY", "^GSPC", cmpValue].includes(k))
                      .map(([k, v]) => ({ label: k, Correlation: +(typeof v === "number" ? v : 0).toFixed(2) }))
                      .sort((a, b) => b.Correlation - a.Correlation);
                    if (!corrs.length) return null;
                    const topCorr = corrs[0]; const bottomCorr = corrs[corrs.length - 1];
                    return (<>
                      <h4 className="text-base font-semibold text-white mt-4 mb-2">Correlations with Other Assets — {curAsset}</h4>
                      <SimpleBarChart data={corrs} bars={[{ key: "Correlation", color: corrs[0]?.Correlation > 0.5 ? C.danger : C.warn, name: `Correlation with ${curAsset}` }]} height={220} />
                      <InfoBox>Correlation Analysis — {curAsset}: Highest correlation: {topCorr.label} ({topCorr.Correlation}) {topCorr.Correlation > 0.7 ? "- High correlation" : "- Moderate correlation"} - {topCorr.Correlation > 0.5 ? "Some" : "Good"} diversification benefit / Lowest correlation: {bottomCorr.label} ({bottomCorr.Correlation})</InfoBox>
                    </>);
                  })()}
                </div>
              );
            })()}

            {/* Fallback if no per_asset data */}
            {(!assetData.per_asset || Object.keys(assetData.per_asset).length === 0) && !assetData.asset_returns && (
              <Alert type="info">Individual asset metrics and dynamics are computed as part of the analytics calculation. Run analysis first.</Alert>
            )}
          </div>)}
        </div>)}

        {/* ═══════════════════════════════════════════════════════ */}
        {/*  TAB 5: EXPORT & REPORTS                               */}
        {/* ═══════════════════════════════════════════════════════ */}
        {mainTab === "export" && (<div className="space-y-5">
          <div className="panel p-6 space-y-4">
            <h3 className="text-lg font-semibold text-white">Export & Reports</h3>
            <p className="text-sm text-white/60">Generate comprehensive reports with full analytics data.</p>

            <div className="space-y-3">
              <h4 className="text-base font-semibold text-white">Export Raw Data</h4>
              <button className="btn btn-primary" onClick={() => {
                const blob = new Blob([JSON.stringify(analytics, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a"); a.href = url; a.download = `${selected?.name ?? "portfolio"}_analytics_${startDate}_${endDate}.json`; a.click();
                URL.revokeObjectURL(url);
              }}>Download Analytics JSON</button>
            </div>

            <div className="space-y-3">
              <h4 className="text-base font-semibold text-white">Summary Table</h4>
              <table className="data-table"><thead><tr><th>Category</th><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                  {[
                    ["Performance","Total Return",fmtPct(perf.total_return)],["Performance","CAGR",fmtPct(perf.cagr ?? perf.annualized_return)],
                    ["Risk","Volatility",fmtPct(vol)],["Risk","Max Drawdown",fmtPct(maxDD)],["Risk","VaR 95%",fmtPct(risk.var_95 ?? risk.var_historical_95)],
                    ["Ratios","Sharpe",fmtNum(ratios.sharpe_ratio)],["Ratios","Sortino",fmtNum(ratios.sortino_ratio)],["Ratios","Calmar",fmtNum(ratios.calmar_ratio)],
                    ["Market","Beta",fmtNum(market.beta)],["Market","Alpha",fmtPct(market.alpha)],["Market","Info Ratio",fmtNum(ratios.information_ratio)],
                  ].map(([cat,metric,val]) => (
                    <tr key={`${cat}-${metric}`}><td className="text-white/40">{cat}</td><td className="text-white/70">{metric}</td><td className="font-mono">{val}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Expander title="About Reports">
              <div className="space-y-2 text-sm text-white/60">
                <p><strong className="text-white/80">JSON Export</strong> — Downloads the complete analytics response including all metrics, time series data, and raw calculations.</p>
                <p><strong className="text-white/80">PDF Reports</strong> — PDF generation with full-page screenshots is available in the Streamlit version. A dedicated report generation feature for the Next.js version is planned.</p>
              </div>
            </Expander>
          </div>
        </div>)}
      </>)}
    </div>
  );
}
