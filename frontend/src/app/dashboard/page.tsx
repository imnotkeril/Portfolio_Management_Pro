"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { api } from "@/lib/api";
import { usd } from "@/lib/format";
import type { Portfolio } from "@/lib/types";

export default function DashboardPage() {
  const router = useRouter();
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await api.get<Portfolio[]>("/portfolios");
      setPortfolios(list);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const remove = async (id: string, name: string) => {
    if (!confirm(`Delete portfolio "${name}"?`)) return;
    try {
      await api.delete(`/portfolios/${id}`);
      setPortfolios((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-sm text-white/50 mt-1">
            Quick view — full management in{" "}
            <Link href="/portfolios" className="text-violet-300 hover:text-white">
              Portfolio List
            </Link>
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-primary" onClick={() => router.push("/create")}>
            + New portfolio
          </button>
          <button className="btn btn-secondary" onClick={load}>
            Refresh
          </button>
        </div>
      </div>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <div className="alert alert-info">Loading…</div> : null}

      {!loading && portfolios.length === 0 ? (
        <div className="panel p-8 text-center space-y-4">
          <p className="text-white/60">No portfolios yet.</p>
          <button className="btn btn-primary" onClick={() => router.push("/create")}>
            Create your first portfolio
          </button>
        </div>
      ) : null}

      {!loading && portfolios.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          {portfolios.map((p) => (
            <div key={p.id} className="panel p-5 space-y-3">
              <div>
                <Link
                  href={`/portfolio/${p.id}`}
                  className="text-lg font-semibold text-white hover:text-violet-200"
                >
                  {p.name}
                </Link>
                {p.description ? (
                  <p className="text-xs text-white/40 mt-1 line-clamp-2">{p.description}</p>
                ) : null}
              </div>
              <p className="text-sm text-white/50">
                {(p.positions?.length ?? 0)} positions · {usd(p.starting_capital)}{" "}
                {p.base_currency}
              </p>
              <div className="flex flex-wrap gap-2">
                <Link href={`/portfolio/${p.id}`} className="btn btn-secondary !text-xs">
                  Open
                </Link>
                <Link
                  href={`/portfolio/${p.id}/transactions`}
                  className="btn btn-secondary !text-xs"
                >
                  Transactions
                </Link>
                <Link href={`/analysis?id=${p.id}`} className="btn btn-ghost !text-xs">
                  Analyze
                </Link>
                <button
                  type="button"
                  className="btn btn-danger !text-xs"
                  onClick={() => remove(p.id, p.name)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
