"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { api } from "@/lib/api";
import { num, pct, usd } from "@/lib/format";
import {
  currentWeightPctFromMarket,
  marketValueByTicker,
  targetWeightPct,
} from "@/lib/portfolio-allocation";
import type { Holding, PnlSummary, Portfolio } from "@/lib/types";

type OverviewRow = Holding & {
  target_label: string;
  current_label: string;
  return_label: string;
};

function growthPct(avgCost: number, marketPrice: number | null | undefined): string {
  if (!marketPrice || avgCost <= 0) return "—";
  const change = ((marketPrice - avgCost) / avgCost) * 100;
  const sign = change >= 0 ? "+" : "";
  return `${sign}${change.toFixed(1)}%`;
}

export default function PortfolioOverviewPage() {
  const id = String(useParams().id);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [pnl, setPnl] = useState<PnlSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id || id === "undefined") return;
    setLoading(true);
    setError(null);
    Promise.all([
      api.get<Portfolio>(`/portfolios/${id}`),
      api.get<Holding[]>(`/portfolios/${id}/holdings`).catch(() => [] as Holding[]),
      api.get<PnlSummary>(`/portfolios/${id}/pnl`).catch(() => null),
    ])
      .then(([p, h, pnlRes]) => {
        setPortfolio(p);
        setHoldings(Array.isArray(h) ? h : []);
        setPnl(pnlRes);
      })
      .catch((err) => {
        setPortfolio(null);
        setError(String(err).replace(/^Error: /, ""));
      })
      .finally(() => setLoading(false));
  }, [id]);

  const overviewRows: OverviewRow[] = useMemo(() => {
    if (!portfolio) return [];
    const marketValues = marketValueByTicker(portfolio.positions, holdings);
    const rows: OverviewRow[] = holdings.map((h) => {
      const pos = portfolio.positions.find((p) => p.ticker === h.ticker);
      const cur = currentWeightPctFromMarket(h.ticker, marketValues);
      return {
        ...h,
        target_label: pos ? targetWeightPct(pos) : "—",
        current_label: cur != null ? `${cur.toFixed(1)}%` : "—",
        return_label: growthPct(h.avg_cost, h.market_price),
      };
    });
    const cashPos = portfolio.positions.find((p) => p.ticker === "CASH");
    if (cashPos && cashPos.shares > 0) {
      const cur = currentWeightPctFromMarket("CASH", marketValues);
      rows.push({
        ticker: "CASH",
        quantity: cashPos.shares,
        avg_cost: 1,
        market_price: 1,
        market_value: cashPos.shares,
        cost_basis: cashPos.shares,
        unrealized_pnl: 0,
        target_label: targetWeightPct(cashPos),
        current_label: cur != null ? `${cur.toFixed(1)}%` : "—",
        return_label: "—",
      });
    }
    return rows.sort((a, b) => {
      if (a.ticker === "CASH") return 1;
      if (b.ticker === "CASH") return -1;
      return a.ticker.localeCompare(b.ticker);
    });
  }, [portfolio, holdings]);

  const totalMarketValue = useMemo(
    () =>
      overviewRows.reduce((s, r) => s + (r.market_value ?? r.quantity ?? 0), 0),
    [overviewRows],
  );

  if (loading) return <p className="alert alert-info">Loading overview…</p>;
  if (error) {
    return (
      <div className="alert alert-error">
        {error}
        <p className="mt-2 text-sm text-white/50">
          Sign in again or open from dashboard.
        </p>
      </div>
    );
  }
  if (!portfolio) {
    return <div className="alert alert-warning">Portfolio not found.</div>;
  }
  return (
    <section className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="metric-card">
          <p className="text-xs text-white/40">Portfolio value (market)</p>
          <p className="text-lg font-semibold text-white">{usd(totalMarketValue)}</p>
        </div>
        <div className="metric-card">
          <p className="text-xs text-white/40">Starting capital</p>
          <p className="text-lg font-semibold text-white">
            {usd(portfolio.starting_capital)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-xs text-white/40">Realized PnL</p>
          <p className="text-lg font-semibold text-white">
            {pnl ? usd(pnl.realized_pnl) : "—"}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-xs text-white/40">Unrealized PnL</p>
          <p className="text-lg font-semibold text-white">
            {pnl ? usd(pnl.unrealized_pnl) : "—"}
          </p>
        </div>
      </div>
      {pnl && (pnl.total_return_twr != null || pnl.total_return_mwr != null) ? (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div className="metric-card">
            <p className="text-xs text-white/40">TWR</p>
            <p className="text-lg font-semibold text-white">
              {pct(pnl.total_return_twr)}
            </p>
          </div>
          <div className="metric-card">
            <p className="text-xs text-white/40">MWR</p>
            <p className="text-lg font-semibold text-white">
              {pct(pnl.total_return_mwr)}
            </p>
          </div>
          <div className="metric-card">
            <p className="text-xs text-white/40">Dividend income</p>
            <p className="text-lg font-semibold text-white">
              {usd(pnl.dividend_income)}
            </p>
          </div>
        </div>
      ) : null}
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">
          Holdings (from transactions)
        </h2>
        <p className="text-xs text-white/40 mb-3">
          Current % and return % use live market prices. Target % is your
          strategy weight (incl. cash).
        </p>
        {overviewRows.length === 0 ? (
          <div className="alert alert-info">No holdings yet. Add a BUY transaction.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Quantity</th>
                <th>Target</th>
                <th>Current</th>
                <th>Return vs avg cost</th>
                <th>Avg cost</th>
                <th>Market price</th>
                <th>Market value</th>
                <th>Unrealized PnL</th>
              </tr>
            </thead>
            <tbody>
              {overviewRows.map((h) => (
                <tr key={h.ticker}>
                  <td className="font-mono text-white">{h.ticker}</td>
                  <td>
                    {h.ticker === "CASH" ? usd(h.quantity) : num(h.quantity, 0)}
                  </td>
                  <td>{h.target_label}</td>
                  <td>{h.current_label}</td>
                  <td>{h.return_label}</td>
                  <td>{h.ticker === "CASH" ? "—" : usd(h.avg_cost)}</td>
                  <td>
                    {h.ticker === "CASH" ? "—" : usd(h.market_price ?? null)}
                  </td>
                  <td>{usd(h.market_value)}</td>
                  <td>
                    {h.ticker === "CASH" ? "—" : usd(h.unrealized_pnl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </section>
  );
}
