from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "src" / "app"
D = "d" + "iv"


def w(rel: str, content: str) -> None:
    p = BASE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    print("wrote", rel)


w(
    "dashboard/page.tsx",
    f"""
"use client";

import Link from "next/link";
import {{ useCallback, useEffect, useState }} from "react";
import {{ useRouter }} from "next/navigation";

import {{ api }} from "@/lib/api";
import {{ usd }} from "@/lib/format";
import type {{ Portfolio }} from "@/lib/types";

export default function DashboardPage() {{
  const router = useRouter();
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {{
    setLoading(true);
    setError(null);
    try {{
      const list = await api.get<Portfolio[]>("/portfolios");
      setPortfolios(list);
    }} catch (err) {{
      setError(String(err));
    }} finally {{
      setLoading(false);
    }}
  }}, []);

  useEffect(() => {{
    load();
  }}, [load]);

  const remove = async (id: string, name: string) => {{
    if (!confirm(`Delete portfolio "${{name}}"?`)) return;
    try {{
      await api.delete(`/portfolios/${{id}}`);
      setPortfolios((prev) => prev.filter((p) => p.id !== id));
    }} catch (err) {{
      setError(String(err));
    }}
  }};

  return (
    <{D} className="space-y-6 max-w-5xl mx-auto">
      <{D} className="flex flex-wrap items-center justify-between gap-3">
        <{D}>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-sm text-white/50 mt-1">Your portfolios</p>
        </{D}>
        <{D} className="flex gap-2">
          <button className="btn btn-primary" onClick={{() => router.push("/create")}}>
            + New portfolio
          </button>
          <button className="btn btn-secondary" onClick={{load}}>
            Refresh
          </button>
        </{D}>
      </{D}>

      {{error ? <{D} className="alert alert-error">{{error}}</{D}> : null}}
      {{loading ? <{D} className="alert alert-info">Loading…</{D}> : null}}

      {{!loading && portfolios.length === 0 ? (
        <{D} className="panel p-8 text-center space-y-4">
          <p className="text-white/60">No portfolios yet.</p>
          <button className="btn btn-primary" onClick={{() => router.push("/create")}}>
            Create your first portfolio
          </button>
        </{D}>
      ) : null}}

      {{!loading && portfolios.length > 0 ? (
        <{D} className="grid gap-4 md:grid-cols-2">
          {{portfolios.map((p) => (
            <{D} key={{p.id}} className="panel p-5 space-y-3">
              <{D}>
                <Link
                  href={{`/portfolio/${{p.id}}`}}
                  className="text-lg font-semibold text-white hover:text-violet-200"
                >
                  {{p.name}}
                </Link>
                {{p.description ? (
                  <p className="text-xs text-white/40 mt-1 line-clamp-2">{{p.description}}</p>
                ) : null}}
              </{D}>
              <p className="text-sm text-white/50">
                {{p.positions.length}} positions · {{usd(p.starting_capital)}} {{p.base_currency}}
              </p>
              <{D} className="flex flex-wrap gap-2">
                <Link href={{`/portfolio/${{p.id}}`}} className="btn btn-secondary !text-xs">
                  Open
                </Link>
                <Link
                  href={{`/portfolio/${{p.id}}/transactions`}}
                  className="btn btn-secondary !text-xs"
                >
                  Transactions
                </Link>
                <Link href={{`/analysis?id=${{p.id}}`}} className="btn btn-ghost !text-xs">
                  Analyze
                </Link>
                <button
                  type="button"
                  className="btn btn-danger !text-xs"
                  onClick={{() => remove(p.id, p.name)}}
                >
                  Delete
                </button>
              </{D}>
            </{D}>
          ))}}
        </{D}>
      ) : null}}
    </{D}>
  );
}}
""",
)

w(
    "settings/page.tsx",
    f"""
"use client";

import {{ useAuth }} from "@/contexts/AuthContext";

export default function SettingsPage() {{
  const {{ user, loading, logout }} = useAuth();

  if (loading) {{
    return <{D} className="text-white/60">Loading…</{D}>;
  }}

  return (
    <{D} className="mx-auto max-w-lg space-y-6">
      <h1 className="text-3xl font-bold text-white">Settings</h1>
      <{D} className="panel p-6 space-y-4">
        <{D}>
          <p className="text-xs text-white/40">Email</p>
          <p className="text-white mt-1">{{user?.email ?? "—"}}</p>
        </{D}>
        <button type="button" className="btn btn-secondary" onClick={{() => logout()}}>
          Sign out
        </button>
      </{D}>
    </{D}>
  );
}}
""",
)

w(
    "portfolios/page.tsx",
    """
import { redirect } from "next/navigation";

export default function PortfoliosRedirect() {
  redirect("/dashboard");
}
""",
)

w(
    "portfolio/new/page.tsx",
    """
import { redirect } from "next/navigation";

export default function NewPortfolioPage() {
  redirect("/create");
}
""",
)

w(
    "portfolio/[id]/layout.tsx",
    f"""
"use client";

import {{ useEffect, useState }} from "react";
import {{ useParams }} from "next/navigation";

import {{ PortfolioTabs }} from "@/components/portfolio-tabs";
import {{ api }} from "@/lib/api";
import type {{ Portfolio }} from "@/lib/types";

export default function PortfolioLayout({{
  children,
}}: {{
  children: React.ReactNode;
}}) {{
  const params = useParams();
  const id = String(params.id);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {{
    setNotFound(false);
    api
      .get<Portfolio>(`/portfolios/${{id}}`)
      .then(setPortfolio)
      .catch(() => setNotFound(true));
  }}, [id]);

  if (notFound) {{
    return (
      <{D} className="panel p-8 text-center">
        <h1 className="text-xl text-white">Portfolio not found</h1>
        <p className="text-sm text-white/50 mt-2">It may have been deleted or you lack access.</p>
      </{D}>
    );
  }}

  return (
    <{D} className="space-y-6 max-w-6xl mx-auto">
      <PortfolioTabs portfolioId={{id}} portfolioName={{portfolio?.name}} />
      {{children}}
    </{D}>
  );
}}
""",
)

w(
    "portfolio/[id]/page.tsx",
    f"""
"use client";

import {{ useEffect, useState }} from "react";
import {{ useParams }} from "next/navigation";

import {{ api }} from "@/lib/api";
import {{ pct, usd }} from "@/lib/format";
import type {{ Holding, PnlSummary, Portfolio }} from "@/lib/types";

export default function PortfolioOverviewPage() {{
  const id = String(useParams().id);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [pnl, setPnl] = useState<PnlSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {{
    setLoading(true);
    Promise.all([
      api.get<Portfolio>(`/portfolios/${{id}}`),
      api.get<Holding[]>(`/portfolios/${{id}}/holdings`).catch(() => []),
      api.get<PnlSummary>(`/portfolios/${{id}}/pnl`).catch(() => null),
    ])
      .then(([p, h, pnlRes]) => {{
        setPortfolio(p);
        setHoldings(h);
        setPnl(pnlRes);
      }})
      .finally(() => setLoading(false));
  }}, [id]);

  if (loading) return <{D} className="alert alert-info">Loading overview…</{D}>;
  if (!portfolio) return null;

  return (
    <{D} className="space-y-6">
      <{D} className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <{D} className="metric-card">
          <p className="text-xs text-white/40">Starting capital</p>
          <p className="text-lg font-semibold text-white">{{usd(portfolio.starting_capital)}}</p>
        </{D}>
        <{D} className="metric-card">
          <p className="text-xs text-white/40">Cost basis</p>
          <p className="text-lg font-semibold text-white">{{portfolio.cost_basis_method ?? "fifo"}}</p>
        </{D}>
        <{D} className="metric-card">
          <p className="text-xs text-white/40">Realized PnL</p>
          <p className="text-lg font-semibold text-white">
            {{pnl ? usd(pnl.realized_pnl) : "—"}}
          </p>
        </{D}>
        <{D} className="metric-card">
          <p className="text-xs text-white/40">Unrealized PnL</p>
          <p className="text-lg font-semibold text-white">
            {{pnl ? usd(pnl.unrealized_pnl) : "—"}}
          </p>
        </{D}>
      </{D}>

      {{pnl?.total_return_twr != null ? (
        <{D} className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <{D} className="metric-card">
            <p className="text-xs text-white/40">TWR</p>
            <p className="text-lg font-semibold text-white">{{pct(pnl.total_return_twr)}}</p>
          </{D}>
          <{D} className="metric-card">
            <p className="text-xs text-white/40">MWR</p>
            <p className="text-lg font-semibold text-white">{{pct(pnl.total_return_mwr ?? null)}}</p>
          </{D}>
          <{D} className="metric-card">
            <p className="text-xs text-white/40">Dividend income</p>
            <p className="text-lg font-semibold text-white">
              {{usd(pnl.dividend_income ?? 0)}}
            </p>
          </{D}>
        </{D}>
      ) : null}}

      <section>
        <h2 className="text-lg font-semibold text-white mb-3">Holdings (from transactions)</h2>
        {{holdings.length === 0 ? (
          <{D} className="alert alert-info">No holdings yet. Add a BUY transaction.</{D}>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Quantity</th>
                <th>Avg cost</th>
                <th>Market value</th>
                <th>Unrealized PnL</th>
              </tr>
            </thead>
            <tbody>
              {{holdings.map((h) => (
                <tr key={{h.ticker}}>
                  <td className="font-mono text-white">{{h.ticker}}</td>
                  <td>{{h.quantity.toFixed(4)}}</td>
                  <td>{{usd(h.avg_cost)}}</td>
                  <td>{{h.market_value != null ? usd(h.market_value) : "—"}}</td>
                  <td>
                    {{h.unrealized_pnl != null ? usd(h.unrealized_pnl) : "—"}}
                  </td>
                </tr>
              ))}}
            </tbody>
          </table>
        )}}
      </section>
    </{D}>
  );
}}
""",
)

w(
    "portfolio/[id]/transactions/page.tsx",
    f"""
"use client";

import {{ FormEvent, useCallback, useEffect, useState }} from "react";
import {{ useParams }} from "next/navigation";

import {{ api }} from "@/lib/api";
import {{ usd }} from "@/lib/format";
import type {{ Transaction }} from "@/lib/types";

const TX_TYPES = [
  "BUY",
  "SELL",
  "DIVIDEND",
  "SPLIT",
  "DEPOSIT",
  "WITHDRAWAL",
] as const;

export default function PortfolioTransactionsPage() {{
  const portfolioId = String(useParams().id);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [txDate, setTxDate] = useState(new Date().toISOString().slice(0, 10));
  const [txType, setTxType] = useState<string>("BUY");
  const [txTicker, setTxTicker] = useState("");
  const [txShares, setTxShares] = useState(0);
  const [txPrice, setTxPrice] = useState(0);
  const [txFees, setTxFees] = useState(0);
  const [txNotes, setTxNotes] = useState("");
  const [reinvest, setReinvest] = useState(false);
  const [splitRatio, setSplitRatio] = useState(2);

  const load = useCallback(async () => {{
    setLoading(true);
    try {{
      const rows = await api.get<Transaction[]>(
        `/portfolios/${{portfolioId}}/transactions`,
      );
      setTransactions(rows);
    }} catch (err) {{
      setError(String(err));
    }} finally {{
      setLoading(false);
    }}
  }}, [portfolioId]);

  useEffect(() => {{
    load();
  }}, [load]);

  const onSubmit = async (e: FormEvent) => {{
    e.preventDefault();
    if (!txTicker.trim()) return;
    setSaving(true);
    setError(null);
    try {{
      const body: Record<string, unknown> = {{
        transaction_date: txDate,
        transaction_type: txType,
        ticker: txTicker.trim().toUpperCase(),
        shares: txShares,
        price: txPrice,
        fees: txFees,
        notes: txNotes || null,
      }};
      if (txType === "DIVIDEND") body.reinvest = reinvest;
      if (txType === "SPLIT") body.split_ratio = splitRatio;
      const tx = await api.post<Transaction>(
        `/portfolios/${{portfolioId}}/transactions`,
        body,
      );
      setTransactions((prev) => [...prev, tx]);
      setTxTicker("");
      setTxShares(0);
      setTxPrice(0);
      setTxFees(0);
      setTxNotes("");
    }} catch (err) {{
      setError(String(err));
    }} finally {{
      setSaving(false);
    }}
  }};

  const remove = async (txId: string) => {{
    try {{
      await api.delete(`/portfolios/${{portfolioId}}/transactions/${{txId}}`);
      setTransactions((prev) => prev.filter((t) => t.id !== txId));
    }} catch (err) {{
      setError(String(err));
    }}
  }};

  return (
    <{D} className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Transactions</h2>
      {{error ? <{D} className="alert alert-error">{{error}}</{D}> : null}}
      {{loading ? <{D} className="alert alert-info">Loading…</{D}> : null}}

      <form onSubmit={{onSubmit}} className="panel p-5 space-y-4">
        <h3 className="font-medium text-white">Add transaction</h3>
        <{D} className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <{D}>
            <label className="label">Date</label>
            <input
              type="date"
              className="input"
              value={{txDate}}
              onChange={{(e) => setTxDate(e.target.value)}}
            />
          </{D}>
          <{D}>
            <label className="label">Type</label>
            <select
              className="input"
              value={{txType}}
              onChange={{(e) => setTxType(e.target.value)}}
            >
              {{TX_TYPES.map((t) => (
                <option key={{t}} value={{t}}>
                  {{t}}
                </option>
              ))}}
            </select>
          </{D}>
          <{D}>
            <label className="label">Ticker</label>
            <input
              className="input"
              value={{txTicker}}
              onChange={{(e) => setTxTicker(e.target.value)}}
              placeholder="AAPL"
            />
          </{D}>
          <{D}>
            <label className="label">Shares</label>
            <input
              type="number"
              className="input"
              min={{0}}
              step={{0.0001}}
              value={{txShares}}
              onChange={{(e) => setTxShares(Number(e.target.value))}}
            />
          </{D}>
          <{D}>
            <label className="label">Price</label>
            <input
              type="number"
              className="input"
              min={{0}}
              step={{0.01}}
              value={{txPrice}}
              onChange={{(e) => setTxPrice(Number(e.target.value))}}
            />
          </{D}>
          <{D}>
            <label className="label">Fees</label>
            <input
              type="number"
              className="input"
              min={{0}}
              step={{0.01}}
              value={{txFees}}
              onChange={{(e) => setTxFees(Number(e.target.value))}}
            />
          </{D}>
        </{D}>
        {{txType === "DIVIDEND" ? (
          <label className="flex items-center gap-2 text-sm text-white/70">
            <input
              type="checkbox"
              checked={{reinvest}}
              onChange={{(e) => setReinvest(e.target.checked)}}
            />
            Reinvest dividend
          </label>
        ) : null}}
        {{txType === "SPLIT" ? (
          <{D}>
            <label className="label">Split ratio</label>
            <input
              type="number"
              className="input"
              min={{0.01}}
              step={{0.01}}
              value={{splitRatio}}
              onChange={{(e) => setSplitRatio(Number(e.target.value))}}
            />
          </{D}>
        ) : null}}
        <{D}>
          <label className="label">Notes</label>
          <input
            className="input"
            value={{txNotes}}
            onChange={{(e) => setTxNotes(e.target.value)}}
          />
        </{D}>
        <button type="submit" className="btn btn-primary" disabled={{saving || !txTicker.trim()}}>
          {{saving ? "Saving…" : "Add transaction"}}
        </button>
      </form>

      {{transactions.length > 0 ? (
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Type</th>
              <th>Ticker</th>
              <th>Shares</th>
              <th>Price</th>
              <th>Amount</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {{transactions.map((tx) => (
              <tr key={{tx.id}}>
                <td className="font-mono text-xs">{{tx.transaction_date}}</td>
                <td>{{tx.transaction_type}}</td>
                <td className="font-mono">{{tx.ticker}}</td>
                <td>{{tx.shares.toFixed(4)}}</td>
                <td>{{usd(tx.price)}}</td>
                <td>{{usd(tx.amount)}}</td>
                <td>
                  <button
                    type="button"
                    className="btn btn-danger !py-1 !px-2 !text-xs"
                    onClick={{() => remove(tx.id)}}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}}
          </tbody>
        </table>
      ) : (
        !loading && <{D} className="alert alert-info">No transactions yet.</{D}>
      )}}
    </{D}>
  );
}}
""",
)
