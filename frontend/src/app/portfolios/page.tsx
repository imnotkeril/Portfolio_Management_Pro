"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import {
  RebalanceStrategyPanel,
  type StrategyWeightRow,
} from "@/components/rebalance-strategy-panel";
import {
  TransactionForm,
  type TransactionFormPayload,
} from "@/components/transaction-form";
import {
  parseRebalanceInterval,
  rebalanceIntervalLabel,
} from "@/lib/rebalance";
import {
  formatTxShares,
  sortTransactionsForDisplay,
} from "@/lib/format-tx-shares";
import {
  currentWeightPctFromMarket,
  formatShareCount,
  marketValueByTicker,
  marketValueForPosition,
  targetWeightPct,
} from "@/lib/portfolio-allocation";
import type { Holding, Portfolio, Transaction } from "@/lib/types";
import { netDeposits } from "@/lib/transaction-metrics";

/* ───────── helpers ───────── */

function pct(v: number | null) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function usd(v: number) {
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/* ───────── tiny shared components ───────── */

function Tip({ text }: { text: string }) {
  return (
    <span className="tooltip-wrap">
      <span className="tooltip-icon">?</span>
      <span className="tooltip-bubble">{text}</span>
    </span>
  );
}

function Alert({
  type,
  children,
}: {
  type: "success" | "error" | "warning" | "info";
  children: React.ReactNode;
}) {
  return <div className={`alert alert-${type}`}>{children}</div>;
}

function Expander({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="expander" open={defaultOpen || undefined}>
      <summary>{title}</summary>
      <div className="expander-body">{children}</div>
    </details>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card text-center">
      <div className="text-xs text-white/40 mb-1">{label}</div>
      <div className="text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

/* ───────── types ───────── */

type ViewMode = "list" | "view" | "edit";
type TabKey = "overview" | "positions" | "transactions" | "strategies";
type SortKey = "name" | "assets" | "value";

type StrategyApi = {
  portfolio_id: string;
  rebalance_interval_months: number | null;
  targets: Record<string, number>;
  targets_normalized: Record<string, number>;
  total_weight: number;
  is_active: boolean;
};

function buildStrategyWeightRows(
  portfolio: Portfolio,
  drafts: Record<string, number>,
): StrategyWeightRow[] {
  const tickers = new Set<string>();
  for (const p of portfolio.positions) {
    if (p.ticker) tickers.add(p.ticker);
  }
  return [...tickers].map((ticker) => {
    const pos = portfolio.positions.find((p) => p.ticker === ticker);
    const fromPos =
      pos?.weight_target != null && pos.weight_target > 0
        ? pos.weight_target * 100
        : 0;
    return {
      ticker,
      weightPct: drafts[ticker] ?? fromPos,
    };
  });
}

/* =============================================================== */
/*                          MAIN COMPONENT                         */
/* =============================================================== */

export default function PortfoliosPage() {
  const router = useRouter();

  /* --- state --- */
  const [mode, setMode] = useState<ViewMode>("list");
  const [tab, setTab] = useState<TabKey>("overview");
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [message, setMessage] = useState<{
    type: "success" | "error" | "info";
    text: string;
  } | null>(null);

  /* list filters */
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("name");

  /* edit form */
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editCurrency, setEditCurrency] = useState("USD");
  const [editCapital, setEditCapital] = useState(0);
  const [saving, setSaving] = useState(false);

  /* add position */
  const [addTicker, setAddTicker] = useState("");
  const [addShares, setAddShares] = useState(0);
  const [addWeight, setAddWeight] = useState(0);
  const [addingPosition, setAddingPosition] = useState(false);
  const [syncingLedger, setSyncingLedger] = useState(false);
  const [holdings, setHoldings] = useState<Holding[]>([]);

  const [strategyInterval, setStrategyInterval] = useState("");
  const [strategyWeightDrafts, setStrategyWeightDrafts] = useState<
    Record<string, number>
  >({});
  const [savingStrategy, setSavingStrategy] = useState(false);
  const [previewingStrategy, setPreviewingStrategy] = useState(false);

  /* --- load portfolios --- */
  const loadPortfolios = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.get<Portfolio[]>("/portfolios");
      setPortfolios(list);
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPortfolios();
  }, [loadPortfolios]);

  /* --- derived --- */
  const selected = portfolios.find((p) => p.id === selectedId) ?? null;

  const marketValues = useMemo(() => {
    if (!selected) return {};
    return marketValueByTicker(selected.positions, holdings);
  }, [selected, holdings]);

  const strategyWeightRows = useMemo(() => {
    if (!selected) return [];
    return buildStrategyWeightRows(selected, strategyWeightDrafts);
  }, [selected, strategyWeightDrafts]);

  const filtered = useMemo(() => {
    let result = [...portfolios];
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          (p.description ?? "").toLowerCase().includes(q),
      );
    }
    if (sortBy === "name")
      result.sort((a, b) => a.name.localeCompare(b.name));
    else if (sortBy === "assets")
      result.sort((a, b) => b.positions.length - a.positions.length);
    else if (sortBy === "value")
      result.sort((a, b) => b.starting_capital - a.starting_capital);
    return result;
  }, [portfolios, searchQuery, sortBy]);

  /* --- actions --- */

  const syncStrategyFromPortfolio = (p: Portfolio) => {
    const m = p.rebalance_interval_months;
    setStrategyInterval(m != null ? String(m) : "");
    const drafts: Record<string, number> = {};
    for (const pos of p.positions) {
      if (pos.weight_target != null && pos.weight_target > 0) {
        drafts[pos.ticker] = pos.weight_target * 100;
      }
    }
    setStrategyWeightDrafts(drafts);
  };

  const syncPortfolioLedger = useCallback(async (id: string) => {
    setSyncingLedger(true);
    try {
      const portfolio = await api.get<Portfolio>(`/portfolios/${id}`);
      const isLedger = portfolio.ledger_mode === "transactions";

      if (isLedger) {
        const maintainRes = await api.post<{
          maintain_error?: string;
          catchup_rebalance_error?: string;
        }>(`/portfolios/${id}/ledger/maintain`, {});
        const maintainWarn =
          maintainRes.maintain_error ?? maintainRes.catchup_rebalance_error;
        if (maintainWarn) {
          setMessage({
            type: "error",
            text: `Ledger maintain: ${maintainWarn}`,
          });
        }
      }

      const [txs, holdingsRows, portfolioFresh] = await Promise.all([
        api.get<Transaction[]>(
          `/portfolios/${id}/transactions?sync_ledger=${isLedger ? "false" : "true"}`,
        ),
        api.get<Holding[]>(`/portfolios/${id}/holdings`),
        api.get<Portfolio>(`/portfolios/${id}`),
      ]);
      setPortfolios((prev) =>
        prev.map((p) => (p.id === id ? portfolioFresh : p)),
      );
      setTransactions(txs);
      setHoldings(holdingsRows);
      syncStrategyFromPortfolio(portfolioFresh);
    } catch (err) {
      setMessage({
        type: "error",
        text: `Ledger sync failed: ${String(err)}`,
      });
    } finally {
      setSyncingLedger(false);
    }
  }, []);

  const openView = (id: string) => {
    setSelectedId(id);
    setMode("view");
    setTab("overview");
    setMessage(null);
    void syncPortfolioLedger(id);
  };

  const openEdit = (p: Portfolio) => {
    setSelectedId(p.id);
    setEditName(p.name);
    setEditDescription(p.description ?? "");
    setEditCurrency(p.base_currency);
    setEditCapital(p.starting_capital);
    syncStrategyFromPortfolio(p);
    setMode("edit");
    setTab("positions");
    setMessage(null);
  };

  const saveRebalanceStrategy = async () => {
    if (!selectedId || !selected) return;
    const rows = buildStrategyWeightRows(selected, strategyWeightDrafts);
    const targets: Record<string, number> = {};
    for (const row of rows) {
      targets[row.ticker] = row.weightPct / 100;
    }
    setSavingStrategy(true);
    try {
      const res = await api.put<{
        strategy: StrategyApi;
        portfolio: Portfolio;
      }>(`/portfolios/${selectedId}/strategy`, {
        rebalance_interval_months: parseRebalanceInterval(strategyInterval),
        targets,
        replace_targets: true,
      });
      setPortfolios((prev) =>
        prev.map((p) => (p.id === selectedId ? res.portfolio : p)),
      );
      syncStrategyFromPortfolio(res.portfolio);
      setMessage({ type: "success", text: "Rebalancing strategy saved." });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setSavingStrategy(false);
    }
  };

  const previewStrategyRebalance = async () => {
    if (!selectedId) return;
    setPreviewingStrategy(true);
    try {
      const plan = await api.post<{
        trade_count: number;
        message: string;
        complete: boolean;
      }>(`/portfolios/${selectedId}/strategy/preview`, {});
      setMessage({
        type: plan.complete ? "success" : "info",
        text:
          plan.trade_count > 0
            ? `Preview: ${plan.trade_count} trade(s). ${plan.message}`
            : plan.message || "No trades required at current prices.",
      });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setPreviewingStrategy(false);
    }
  };

  const backToList = () => {
    setMode("list");
    setSelectedId(null);
    setMessage(null);
    loadPortfolios();
  };

  const deletePortfolio = async (id: string, name: string) => {
    if (!confirm(`Delete portfolio "${name}"? This cannot be undone.`)) return;
    try {
      await api.delete(`/portfolios/${id}`);
      setMessage({ type: "success", text: `Portfolio "${name}" deleted.` });
      setPortfolios((prev) => prev.filter((p) => p.id !== id));
      if (selectedId === id) backToList();
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    }
  };

  const savePortfolioChanges = async () => {
    if (!selectedId) return;
    setSaving(true);
    try {
      const updated = await api.patch<Portfolio>(
        `/portfolios/${selectedId}`,
        {
          name: editName.trim(),
          description: editDescription.trim(),
          base_currency: editCurrency,
          starting_capital: editCapital,
        },
      );
      setPortfolios((prev) =>
        prev.map((p) => (p.id === selectedId ? updated : p)),
      );
      setMessage({
        type: "success",
        text: `Portfolio "${editName}" updated successfully!`,
      });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setSaving(false);
    }
  };

  const removePosition = async (ticker: string) => {
    if (!selectedId) return;
    try {
      const updated = await api.delete<Portfolio>(
        `/portfolios/${selectedId}/positions/${ticker}`,
      );
      setPortfolios((prev) =>
        prev.map((p) => (p.id === selectedId ? updated : p)),
      );
      setMessage({ type: "success", text: `Position ${ticker} removed.` });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    }
  };

  const handleAddPosition = async () => {
    if (!selectedId || !addTicker.trim()) return;
    setAddingPosition(true);
    try {
      const updated = await api.post<Portfolio>(
        `/portfolios/${selectedId}/positions`,
        {
          ticker: addTicker.trim().toUpperCase(),
          shares: addShares,
          weight_target: addWeight > 0 ? addWeight / 100 : null,
        },
      );
      setPortfolios((prev) =>
        prev.map((p) => (p.id === selectedId ? updated : p)),
      );
      setMessage({
        type: "success",
        text: `Position ${addTicker.toUpperCase()} added!`,
      });
      setAddTicker("");
      setAddShares(0);
      setAddWeight(0);
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setAddingPosition(false);
    }
  };

  const reloadSelectedPortfolio = useCallback(async () => {
    if (!selectedId) return;
    const [updated, holdingsRows] = await Promise.all([
      api.get<Portfolio>(`/portfolios/${selectedId}`),
      api.get<Holding[]>(`/portfolios/${selectedId}/holdings`),
    ]);
    setPortfolios((prev) =>
      prev.map((p) => (p.id === selectedId ? updated : p)),
    );
    setHoldings(holdingsRows);
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    if (tab === "overview" || tab === "positions") {
      void reloadSelectedPortfolio();
    }
  }, [selectedId, tab, reloadSelectedPortfolio]);

  const reloadTransactions = useCallback(async () => {
    if (!selectedId) return;
    setSyncingLedger(true);
    try {
      const rows = await api.get<Transaction[]>(
        `/portfolios/${selectedId}/transactions`,
      );
      setTransactions(rows);
    } finally {
      setSyncingLedger(false);
    }
  }, [selectedId]);

  const handleAddTransaction = async (payload: TransactionFormPayload) => {
    if (!selectedId) return;
    try {
      const tx = await api.post<Transaction>(
        `/portfolios/${selectedId}/transactions`,
        payload,
      );
      setTransactions((prev) => [...prev, tx]);
      await reloadSelectedPortfolio();
      setMessage({ type: "success", text: "Transaction added successfully!" });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
      throw err;
    }
  };

  const deleteTransaction = async (txId: string) => {
    try {
      await api.delete(`/transactions/${txId}`);
      setTransactions((prev) => prev.filter((t) => t.id !== txId));
      await reloadSelectedPortfolio();
      setMessage({ type: "success", text: "Transaction deleted." });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    }
  };

  const sortedTransactions = useMemo(
    () => sortTransactionsForDisplay(transactions),
    [transactions],
  );

  /* ───────── RENDER ───────── */

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold text-white">Portfolio Management</h1>

      {message && (
        <Alert type={message.type}>{message.text}</Alert>
      )}

      {/* ═══════════ LIST VIEW ═══════════ */}
      {mode === "list" && (
        <>
          {/* Action bar */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-semibold text-white">
              Your Portfolios
            </h2>
            <div className="flex gap-2">
              <button
                className="btn btn-primary"
                onClick={() => router.push("/create")}
              >
                + Create New
              </button>
              <button className="btn btn-secondary" onClick={loadPortfolios}>
                Refresh
              </button>
              <button className="btn btn-ghost" disabled>
                Export All
                <Tip text="Export functionality coming soon" />
              </button>
            </div>
          </div>

          {loading && (
            <Alert type="info">Loading portfolios...</Alert>
          )}

          {!loading && portfolios.length === 0 && (
            <div className="panel p-8 text-center space-y-4">
              <Alert type="info">
                No portfolios found. Create your first portfolio to get
                started!
              </Alert>
              <button
                className="btn btn-primary"
                onClick={() => router.push("/create")}
              >
                Create Your First Portfolio
              </button>
            </div>
          )}

          {!loading && portfolios.length > 0 && (
            <>
              {/* Search & Filter */}
              <Expander title="Search & Filter">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">
                      Search portfolios
                      <Tip text="Search by portfolio name or description" />
                    </label>
                    <input
                      className="input"
                      placeholder="Search by name or description..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label">Sort by</label>
                    <select
                      className="input"
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value as SortKey)}
                    >
                      <option value="name">Name</option>
                      <option value="assets">Asset Count</option>
                      <option value="value">Value</option>
                    </select>
                  </div>
                </div>
              </Expander>

              {/* Portfolio table */}
              <div className="panel p-5 space-y-1">
                <div className="text-sm text-white/50 mb-3">
                  {filtered.length} portfolio
                  {filtered.length !== 1 ? "s" : ""}
                  {searchQuery && ` matching "${searchQuery}"`}
                </div>

                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Description</th>
                      <th>Assets</th>
                      <th>Value</th>
                      <th>Currency</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((p) => (
                      <tr key={p.id}>
                        <td className="font-medium text-white">{p.name}</td>
                        <td className="text-white/50 text-xs max-w-40 truncate">
                          {p.description || "—"}
                        </td>
                        <td>{p.positions.length}</td>
                        <td className="font-mono">
                          {usd(p.starting_capital)}
                        </td>
                        <td>{p.base_currency}</td>
                        <td>
                          <div className="flex gap-1.5">
                            <button
                              className="btn btn-secondary !py-1.5 !px-3 !text-xs"
                              onClick={() => openView(p.id)}
                            >
                              View
                            </button>
                            <button
                              className="btn btn-secondary !py-1.5 !px-3 !text-xs"
                              onClick={() => openEdit(p)}
                            >
                              Edit
                            </button>
                            <button
                              className="btn btn-secondary !py-1.5 !px-3 !text-xs"
                              onClick={() =>
                                router.push(`/analysis?id=${p.id}`)
                              }
                            >
                              Analyze
                            </button>
                            <button
                              className="btn btn-danger !py-1.5 !px-3 !text-xs"
                              onClick={() => deletePortfolio(p.id, p.name)}
                            >
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Individual portfolio cards for mobile */}
              <div className="space-y-3 md:hidden">
                {filtered.map((p) => (
                  <div key={p.id} className="panel p-4 space-y-3">
                    <div>
                      <div className="font-semibold text-white">{p.name}</div>
                      <div className="text-xs text-white/50">
                        {p.base_currency} {usd(p.starting_capital)} •{" "}
                        {p.positions.length} positions
                      </div>
                      {p.description && (
                        <div className="text-xs text-white/40 mt-1">
                          {p.description}
                        </div>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        className="btn btn-secondary !text-xs flex-1"
                        onClick={() => openView(p.id)}
                      >
                        View
                      </button>
                      <button
                        className="btn btn-secondary !text-xs flex-1"
                        onClick={() => openEdit(p)}
                      >
                        Edit
                      </button>
                      <button
                        className="btn btn-danger !text-xs flex-1"
                        onClick={() => deletePortfolio(p.id, p.name)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ═══════════ VIEW MODE ═══════════ */}
      {mode === "view" && selected && (
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">
              Portfolio: {selected.name}
            </h2>
            <button className="btn btn-secondary" onClick={backToList}>
              ← Back to List
            </button>
          </div>

          {/* Mode badge */}
          <div className="panel p-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
            <span>
              <strong className="text-white/70">Portfolio Mode:</strong>{" "}
              {selected.ledger_mode === "transactions" ||
              transactions.length > 0 ? (
                <span className="text-[var(--ok)]">
                  With Transactions ({transactions.length})
                </span>
              ) : (
                <span className="text-[var(--info)]">Buy-and-Hold</span>
              )}
            </span>
            <span>
              <strong className="text-white/70">Rebalancing:</strong>{" "}
              {rebalanceIntervalLabel(selected.rebalance_interval_months)}
            </span>
          </div>

          {syncingLedger && (
            <Alert type="info">
              Syncing splits, dividends, and rebalances through today — this may
              take a minute for older portfolios…
            </Alert>
          )}

          {/* Tabs */}
          <div className="tab-bar">
            {(
              [
                ["overview", "Overview"],
                ["positions", "Positions"],
                ["transactions", "Transactions"],
                ["strategies", "Strategies"],
              ] as [TabKey, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                className={`tab-btn ${tab === key ? "active" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* --- Overview tab --- */}
          {tab === "overview" && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard
                  label="Starting Capital"
                  value={usd(selected.starting_capital)}
                />
                <MetricCard label="Currency" value={selected.base_currency} />
                <MetricCard
                  label="Total Assets"
                  value={String(selected.positions.length)}
                />
                <MetricCard
                  label="Cash"
                  value={usd(
                    selected.positions.find((p) => p.ticker === "CASH")?.shares ??
                      0,
                  )}
                />
              </div>

              {selected.description && (
                <div className="panel p-4">
                  <div className="text-xs text-white/40 mb-1">Description</div>
                  <div className="text-sm text-white/70">
                    {selected.description}
                  </div>
                </div>
              )}

              {/* Positions table */}
              {selected.positions.length > 0 && (
                <div>
                  <h3 className="text-base font-semibold text-white mb-1">
                    Asset Allocation
                  </h3>
                  <p className="text-xs text-white/40 mb-3">
                    Current % uses market prices (same as Positions tab).
                  </p>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Target</th>
                        <th>Current (market)</th>
                        <th>Shares</th>
                        <th>Avg cost</th>
                        <th>Market price</th>
                        <th>Market value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.positions.map((pos) => {
                        const holding = holdings.find(
                          (h) => h.ticker === pos.ticker,
                        );
                        const curW = currentWeightPctFromMarket(
                          pos.ticker,
                          marketValues,
                        );
                        const mv = marketValueForPosition(
                          pos.ticker,
                          pos.shares,
                          holding,
                        );
                        return (
                          <tr key={pos.ticker}>
                            <td className="font-mono font-medium text-white">
                              {pos.ticker}
                            </td>
                            <td>{targetWeightPct(pos)}</td>
                            <td>
                              {curW != null ? `${curW.toFixed(1)}%` : "—"}
                            </td>
                            <td>{formatShareCount(pos.ticker, pos.shares)}</td>
                            <td>
                              {pos.purchase_price
                                ? usd(pos.purchase_price)
                                : "—"}
                            </td>
                            <td>
                              {holding?.market_price != null
                                ? usd(holding.market_price)
                                : "—"}
                            </td>
                            <td>{mv != null ? usd(mv) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* --- Positions tab --- */}
          {tab === "positions" && (
            <div className="space-y-4">
              <h3 className="text-base font-semibold text-white">
                Current Positions
              </h3>

              {selected.positions.length === 0 ? (
                <Alert type="info">No positions yet.</Alert>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Shares</th>
                      <th>Target</th>
                      <th>Current (market)</th>
                      <th>Avg cost</th>
                      <th>Market price</th>
                      <th>Market value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.positions.map((pos) => {
                      const holding = holdings.find(
                        (h) => h.ticker === pos.ticker,
                      );
                      const curW = currentWeightPctFromMarket(
                        pos.ticker,
                        marketValues,
                      );
                      const mv = marketValueForPosition(
                        pos.ticker,
                        pos.shares,
                        holding,
                      );
                      return (
                      <tr key={pos.ticker}>
                        <td className="font-mono font-medium text-white">
                          {pos.ticker}
                        </td>
                        <td>{formatShareCount(pos.ticker, pos.shares)}</td>
                        <td>{targetWeightPct(pos)}</td>
                        <td>
                          {curW != null ? `${curW.toFixed(1)}%` : "—"}
                        </td>
                        <td>
                          {pos.purchase_price
                            ? usd(pos.purchase_price)
                            : "—"}
                        </td>
                        <td>
                          {holding?.market_price != null
                            ? usd(holding.market_price)
                            : "—"}
                        </td>
                        <td>{mv != null ? usd(mv) : "—"}</td>
                      </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* --- Transactions tab --- */}
          {tab === "transactions" && (
            <div className="space-y-4">
              {transactions.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <MetricCard
                    label="Total Transactions"
                    value={String(transactions.length)}
                  />
                  <MetricCard
                    label="First Transaction"
                    value={transactions
                      .map((t) => t.transaction_date)
                      .sort()[0] ?? "—"}
                  />
                  <MetricCard
                    label="Last Transaction"
                    value={
                      transactions
                        .map((t) => t.transaction_date)
                        .sort()
                        .reverse()[0] ?? "—"
                    }
                  />
                  <MetricCard
                    label="Total Invested"
                    value={usd(netDeposits(transactions))}
                  />
                </div>
              )}

              {transactions.length === 0 && (
                <Alert type="info">
                  No transactions yet. Add your first transaction below.
                </Alert>
              )}

              <p className="text-xs text-white/40">
                Dividends, splits, and rebalancing sync automatically when this
                portfolio is opened (through today).
              </p>

              <Expander
                title="Add Transaction"
                defaultOpen={transactions.length === 0}
              >
                <TransactionForm onSubmit={handleAddTransaction} />
              </Expander>

              {/* Transaction table */}
              {transactions.length > 0 && (
                <div>
                  <h3 className="text-base font-semibold text-white mb-3">
                    Transaction History
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Amount</th>
                        <th>Fees</th>
                        <th>Notes</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedTransactions.map((tx) => (
                        <tr key={tx.id}>
                          <td className="font-mono text-xs">
                            {tx.transaction_date}
                          </td>
                          <td>
                            <span
                              className={`tag ${
                                tx.transaction_type === "BUY"
                                  ? "!bg-[var(--ok-bg)] !text-[var(--ok)] !border-[rgb(116_241_116/20%)]"
                                  : tx.transaction_type === "SELL"
                                    ? "!bg-[var(--danger-bg)] !text-[var(--danger)] !border-[rgb(250_161_164/20%)]"
                                    : ""
                              }`}
                            >
                              {tx.transaction_type}
                            </span>
                          </td>
                          <td className="font-mono">{tx.ticker}</td>
                          <td>{formatTxShares(tx)}</td>
                          <td>{usd(tx.price)}</td>
                          <td className="font-mono">{usd(tx.amount)}</td>
                          <td>{tx.fees > 0 ? usd(tx.fees) : "—"}</td>
                          <td className="text-xs text-white/40 max-w-24 truncate">
                            {tx.notes || "—"}
                          </td>
                          <td>
                            <button
                              className="btn btn-danger !py-1 !px-2 !text-xs"
                              onClick={() => deleteTransaction(tx.id)}
                            >
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* --- Strategies tab --- */}
          {tab === "strategies" && selected && (
            <RebalanceStrategyPanel
              portfolio={selected}
              intervalValue={strategyInterval}
              onIntervalChange={setStrategyInterval}
              weightRows={strategyWeightRows}
              onWeightChange={(ticker, weightPct) =>
                setStrategyWeightDrafts((prev) => ({
                  ...prev,
                  [ticker]: weightPct,
                }))
              }
              onSave={() => void saveRebalanceStrategy()}
              onPreview={() => void previewStrategyRebalance()}
              saving={savingStrategy}
              previewing={previewingStrategy}
              showSaveButton
            />
          )}
        </div>
      )}

      {/* ═══════════ EDIT MODE ═══════════ */}
      {mode === "edit" && selected && (
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">
              Edit Portfolio: {selected.name}
            </h2>
            <button className="btn btn-secondary" onClick={backToList}>
              ← Back to List
            </button>
          </div>

          {/* Edit form */}
          <div className="panel p-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">
                  Name
                  <Tip text="Change the portfolio name. Must be unique." />
                </label>
                <input
                  className="input"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                />
              </div>
              <div>
                <label className="label">Currency</label>
                <select
                  className="input"
                  value={editCurrency}
                  onChange={(e) => setEditCurrency(e.target.value)}
                >
                  {["USD", "EUR", "GBP", "JPY", "CAD", "AUD"].map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">
                  Description
                  <Tip text="Optional notes about investment strategy" />
                </label>
                <textarea
                  className="input"
                  rows={3}
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                />
              </div>
              <div>
                <label className="label">
                  Starting Capital
                  <Tip text="The initial investment amount for this portfolio" />
                </label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  step={100}
                  value={editCapital}
                  onChange={(e) => setEditCapital(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="flex gap-3">
              <button
                className="btn btn-primary"
                onClick={savePortfolioChanges}
                disabled={saving || !editName.trim()}
              >
                {saving ? "Saving..." : "Save Changes"}
              </button>
              <button className="btn btn-ghost" onClick={backToList}>
                Cancel
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="tab-bar">
            {(
              [
                ["positions", "Positions"],
                ["transactions", "Transactions"],
                ["strategies", "Strategies"],
              ] as [TabKey, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                className={`tab-btn ${tab === key ? "active" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* --- Positions tab (edit) --- */}
          {tab === "positions" && (
            <div className="space-y-5">
              {/* Current positions with remove */}
              {selected.positions.length > 0 && (
                <div>
                  <h3 className="text-base font-semibold text-white mb-3">
                    Current Positions
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Weight Target</th>
                        <th>Purchase Price</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.positions.map((pos) => (
                        <tr key={pos.ticker}>
                          <td className="font-mono font-medium text-white">
                            {pos.ticker}
                          </td>
                          <td>{formatShareCount(pos.ticker, pos.shares)}</td>
                          <td>{targetWeightPct(pos)}</td>
                          <td>
                            {pos.purchase_price
                              ? usd(pos.purchase_price)
                              : "—"}
                          </td>
                          <td>
                            <button
                              className="btn btn-danger !py-1 !px-2 !text-xs"
                              onClick={() => removePosition(pos.ticker)}
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Add position form */}
              <div className="panel p-5 space-y-4">
                <h4 className="text-base font-semibold text-white">
                  Add Position
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="label">
                      Ticker
                      <Tip text="Stock symbol, e.g. AAPL" />
                    </label>
                    <input
                      className="input"
                      placeholder="AAPL"
                      value={addTicker}
                      onChange={(e) => setAddTicker(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label">
                      Shares
                      <Tip text="Number of shares to add. Can be fractional." />
                    </label>
                    <input
                      className="input"
                      type="number"
                      min={0}
                      step={0.01}
                      value={addShares}
                      onChange={(e) => setAddShares(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="label">
                      Weight Target (%)
                      <Tip text="Target percentage allocation. Leave 0 for auto." />
                    </label>
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={100}
                      step={0.1}
                      value={addWeight}
                      onChange={(e) => setAddWeight(Number(e.target.value))}
                    />
                  </div>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleAddPosition}
                  disabled={!addTicker.trim() || addingPosition}
                >
                  {addingPosition ? "Adding..." : "Add Position"}
                </button>
              </div>
            </div>
          )}

          {/* --- Transactions tab (edit) --- */}
          {tab === "transactions" && (
            <div className="space-y-5">
              {/* Summary metrics */}
              {transactions.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <MetricCard
                    label="Total Transactions"
                    value={String(transactions.length)}
                  />
                  <MetricCard
                    label="First Transaction"
                    value={
                      transactions
                        .map((t) => t.transaction_date)
                        .sort()[0] ?? "—"
                    }
                  />
                  <MetricCard
                    label="Last Transaction"
                    value={
                      transactions
                        .map((t) => t.transaction_date)
                        .sort()
                        .reverse()[0] ?? "—"
                    }
                  />
                  <MetricCard
                    label="Total Invested"
                    value={usd(netDeposits(transactions))}
                  />
                </div>
              )}

              {transactions.length === 0 && (
                <Alert type="info">
                  No transactions yet. Add your first transaction below.
                </Alert>
              )}

              <p className="text-xs text-white/40">
                Dividends, splits, and rebalancing sync automatically when this
                portfolio is opened (through today).
              </p>

              <Expander
                title="Add Transaction"
                defaultOpen={transactions.length === 0}
              >
                <TransactionForm onSubmit={handleAddTransaction} />
              </Expander>

              {/* Transaction history */}
              {transactions.length > 0 && (
                <div>
                  <h3 className="text-base font-semibold text-white mb-3">
                    Transaction History
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Amount</th>
                        <th>Fees</th>
                        <th>Notes</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedTransactions.map((tx) => (
                        <tr key={tx.id}>
                          <td className="font-mono text-xs">
                            {tx.transaction_date}
                          </td>
                          <td>
                            <span
                              className={`tag ${
                                tx.transaction_type === "BUY"
                                  ? "!bg-[var(--ok-bg)] !text-[var(--ok)] !border-[rgb(116_241_116/20%)]"
                                  : tx.transaction_type === "SELL"
                                    ? "!bg-[var(--danger-bg)] !text-[var(--danger)] !border-[rgb(250_161_164/20%)]"
                                    : ""
                              }`}
                            >
                              {tx.transaction_type}
                            </span>
                          </td>
                          <td className="font-mono">{tx.ticker}</td>
                          <td>{formatTxShares(tx)}</td>
                          <td>{usd(tx.price)}</td>
                          <td className="font-mono">{usd(tx.amount)}</td>
                          <td>{tx.fees > 0 ? usd(tx.fees) : "—"}</td>
                          <td className="text-xs text-white/40 max-w-24 truncate">
                            {tx.notes || "—"}
                          </td>
                          <td>
                            <button
                              className="btn btn-danger !py-1 !px-2 !text-xs"
                              onClick={() => deleteTransaction(tx.id)}
                            >
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* --- Strategies tab (edit) --- */}
          {tab === "strategies" && selected && (
            <RebalanceStrategyPanel
              portfolio={selected}
              intervalValue={strategyInterval}
              onIntervalChange={setStrategyInterval}
              weightRows={strategyWeightRows}
              onWeightChange={(ticker, weightPct) =>
                setStrategyWeightDrafts((prev) => ({
                  ...prev,
                  [ticker]: weightPct,
                }))
              }
              onSave={() => void saveRebalanceStrategy()}
              onPreview={() => void previewStrategyRebalance()}
              saving={savingStrategy}
              previewing={previewingStrategy}
              showSaveButton
            />
          )}
        </div>
      )}
    </div>
  );
}
