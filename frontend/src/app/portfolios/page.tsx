"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { Portfolio, Position, Transaction } from "@/lib/types";

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

  /* add transaction */
  const [txDate, setTxDate] = useState(
    new Date().toISOString().slice(0, 10),
  );
  const [txType, setTxType] = useState("BUY");
  const [txTicker, setTxTicker] = useState("");
  const [txShares, setTxShares] = useState(0);
  const [txPrice, setTxPrice] = useState(0);
  const [txFees, setTxFees] = useState(0);
  const [txNotes, setTxNotes] = useState("");
  const [addingTx, setAddingTx] = useState(false);

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

  /* --- load transactions when portfolio selected --- */
  useEffect(() => {
    if (!selectedId) {
      setTransactions([]);
      return;
    }
    api
      .get<Transaction[]>(`/portfolios/${selectedId}/transactions`)
      .then(setTransactions)
      .catch(() => setTransactions([]));
  }, [selectedId]);

  /* --- derived --- */
  const selected = portfolios.find((p) => p.id === selectedId) ?? null;

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

  const openView = (id: string) => {
    setSelectedId(id);
    setMode("view");
    setTab("overview");
    setMessage(null);
  };

  const openEdit = (p: Portfolio) => {
    setSelectedId(p.id);
    setEditName(p.name);
    setEditDescription(p.description ?? "");
    setEditCurrency(p.base_currency);
    setEditCapital(p.starting_capital);
    setMode("edit");
    setTab("positions");
    setMessage(null);
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

  const handleAddTransaction = async () => {
    if (!selectedId || !txTicker.trim()) return;
    setAddingTx(true);
    try {
      const tx = await api.post<Transaction>(
        `/portfolios/${selectedId}/transactions`,
        {
          transaction_date: txDate,
          transaction_type: txType,
          ticker: txTicker.trim().toUpperCase(),
          shares: txShares,
          price: txPrice,
          fees: txFees,
          notes: txNotes || null,
        },
      );
      setTransactions((prev) => [...prev, tx]);
      setMessage({ type: "success", text: "Transaction added successfully!" });
      setTxTicker("");
      setTxShares(0);
      setTxPrice(0);
      setTxFees(0);
      setTxNotes("");
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    } finally {
      setAddingTx(false);
    }
  };

  const deleteTransaction = async (txId: string) => {
    try {
      await api.delete(`/transactions/${txId}`);
      setTransactions((prev) => prev.filter((t) => t.id !== txId));
      setMessage({ type: "success", text: "Transaction deleted." });
    } catch (err) {
      setMessage({ type: "error", text: String(err) });
    }
  };

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
          <div className="panel p-3 flex items-center gap-2 text-sm">
            <strong className="text-white/70">Portfolio Mode:</strong>
            {transactions.length > 0 ? (
              <span className="text-[var(--ok)]">
                With Transactions ({transactions.length} transactions)
              </span>
            ) : (
              <span className="text-[var(--info)]">
                Buy-and-Hold (Simple mode)
              </span>
            )}
          </div>

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
                <MetricCard label="Name" value={selected.name} />
                <MetricCard
                  label="Starting Capital"
                  value={usd(selected.starting_capital)}
                />
                <MetricCard label="Currency" value={selected.base_currency} />
                <MetricCard
                  label="Total Assets"
                  value={String(selected.positions.length)}
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
                  <h3 className="text-base font-semibold text-white mb-3">
                    Asset Allocation
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th>Shares</th>
                        <th>Purchase Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.positions.map((pos) => (
                        <tr key={pos.ticker}>
                          <td className="font-mono font-medium text-white">
                            {pos.ticker}
                          </td>
                          <td>{pct(pos.weight_target)}</td>
                          <td>
                            {pos.ticker === "CASH"
                              ? usd(pos.shares)
                              : pos.shares.toFixed(2)}
                          </td>
                          <td>
                            {pos.purchase_price
                              ? usd(pos.purchase_price)
                              : "—"}
                          </td>
                        </tr>
                      ))}
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
                      <th>Weight Target</th>
                      <th>Purchase Price</th>
                      <th>Purchase Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.positions.map((pos) => (
                      <tr key={pos.ticker}>
                        <td className="font-mono font-medium text-white">
                          {pos.ticker}
                        </td>
                        <td>
                          {pos.ticker === "CASH"
                            ? usd(pos.shares)
                            : pos.shares.toFixed(4)}
                        </td>
                        <td>{pct(pos.weight_target)}</td>
                        <td>
                          {pos.purchase_price
                            ? usd(pos.purchase_price)
                            : "—"}
                        </td>
                        <td>{pos.purchase_date ?? "—"}</td>
                      </tr>
                    ))}
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
                    value={usd(
                      transactions
                        .filter((t) =>
                          ["BUY", "DEPOSIT"].includes(t.transaction_type),
                        )
                        .reduce((s, t) => s + t.amount, 0),
                    )}
                  />
                </div>
              )}

              {transactions.length === 0 && (
                <Alert type="info">
                  No transactions yet. Add your first transaction to start
                  tracking trades.
                </Alert>
              )}

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
                      </tr>
                    </thead>
                    <tbody>
                      {transactions.map((tx) => (
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
                          <td>{tx.shares.toFixed(2)}</td>
                          <td>{usd(tx.price)}</td>
                          <td className="font-mono">{usd(tx.amount)}</td>
                          <td>{tx.fees > 0 ? usd(tx.fees) : "—"}</td>
                          <td className="text-xs text-white/40 max-w-24 truncate">
                            {tx.notes || "—"}
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
          {tab === "strategies" && (
            <div className="space-y-3">
              <Alert type="info">
                Strategy management — coming in Phase 5
              </Alert>
              <div className="panel p-4 text-sm text-white/60 space-y-1">
                <p>
                  <strong className="text-white/80">
                    Strategies can be applied to any portfolio mode:
                  </strong>
                </p>
                <ul className="list-disc list-inside space-y-0.5">
                  <li>
                    Buy-and-Hold portfolios can use strategies for backtesting
                  </li>
                  <li>
                    Transaction-based portfolios can also use strategies
                  </li>
                  <li>
                    Strategies generate simulated transactions for analysis
                  </li>
                </ul>
              </div>
            </div>
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
                          <td>
                            {pos.ticker === "CASH"
                              ? usd(pos.shares)
                              : pos.shares.toFixed(4)}
                          </td>
                          <td>{pct(pos.weight_target)}</td>
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
                    value={usd(
                      transactions
                        .filter((t) =>
                          ["BUY", "DEPOSIT"].includes(t.transaction_type),
                        )
                        .reduce((s, t) => s + t.amount, 0),
                    )}
                  />
                </div>
              )}

              {transactions.length === 0 && (
                <Alert type="info">
                  No transactions yet. Add your first transaction below.
                </Alert>
              )}

              {/* Add transaction form */}
              <Expander title="Add Transaction" defaultOpen={transactions.length === 0}>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="label">
                        Date
                        <Tip text="Transaction date" />
                      </label>
                      <input
                        className="input"
                        type="date"
                        value={txDate}
                        onChange={(e) => setTxDate(e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="label">
                        Type
                        <Tip text="BUY, SELL, DEPOSIT, or WITHDRAWAL" />
                      </label>
                      <select
                        className="input"
                        value={txType}
                        onChange={(e) => setTxType(e.target.value)}
                      >
                        <option value="BUY">BUY</option>
                        <option value="SELL">SELL</option>
                        <option value="DEPOSIT">DEPOSIT</option>
                        <option value="WITHDRAWAL">WITHDRAWAL</option>
                      </select>
                    </div>
                    <div>
                      <label className="label">
                        Ticker
                        <Tip text="Stock symbol for this transaction" />
                      </label>
                      <input
                        className="input"
                        placeholder="AAPL"
                        value={txTicker}
                        onChange={(e) => setTxTicker(e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="label">Shares</label>
                      <input
                        className="input"
                        type="number"
                        min={0}
                        step={0.01}
                        value={txShares}
                        onChange={(e) => setTxShares(Number(e.target.value))}
                      />
                    </div>
                    <div>
                      <label className="label">Price per share</label>
                      <input
                        className="input"
                        type="number"
                        min={0}
                        step={0.01}
                        value={txPrice}
                        onChange={(e) => setTxPrice(Number(e.target.value))}
                      />
                    </div>
                    <div>
                      <label className="label">
                        Fees
                        <Tip text="Transaction fees/commissions" />
                      </label>
                      <input
                        className="input"
                        type="number"
                        min={0}
                        step={0.01}
                        value={txFees}
                        onChange={(e) => setTxFees(Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="label">Notes (optional)</label>
                    <input
                      className="input"
                      placeholder="Optional notes..."
                      value={txNotes}
                      onChange={(e) => setTxNotes(e.target.value)}
                    />
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={handleAddTransaction}
                    disabled={!txTicker.trim() || addingTx}
                  >
                    {addingTx ? "Adding..." : "Add Transaction"}
                  </button>
                </div>
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
                      {transactions.map((tx) => (
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
                          <td>{tx.shares.toFixed(2)}</td>
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
          {tab === "strategies" && (
            <div className="space-y-3">
              <Alert type="info">
                Strategy management — coming in Phase 5
              </Alert>
              <div className="panel p-4 text-sm text-white/60 space-y-1">
                <p>
                  <strong className="text-white/80">
                    Strategies can be applied to any portfolio mode:
                  </strong>
                </p>
                <ul className="list-disc list-inside space-y-0.5">
                  <li>
                    Buy-and-Hold portfolios can use strategies for backtesting
                  </li>
                  <li>
                    Transaction-based portfolios can also use strategies
                  </li>
                  <li>
                    Strategies generate simulated transactions for analysis
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
