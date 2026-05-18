"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";

import {
  TransactionForm,
  type TransactionFormPayload,
} from "@/components/transaction-form";
import { api } from "@/lib/api";
import { usd } from "@/lib/format";
import type { Transaction } from "@/lib/types";

function formatTxShares(tx: Transaction): string {
  if (tx.ticker === "CASH") return usd(tx.shares);
  return String(Math.floor(tx.shares));
}

export default function PortfolioTransactionsPage() {
  const portfolioId = String(useParams().id);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await api.get<Transaction[]>(
        `/portfolios/${portfolioId}/transactions`,
      );
      setTransactions(rows);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [portfolioId]);

  useEffect(() => {
    load();
  }, [load]);

  const onSubmit = async (payload: TransactionFormPayload) => {
    setError(null);
    try {
      const tx = await api.post<Transaction>(
        `/portfolios/${portfolioId}/transactions`,
        payload,
      );
      setTransactions((prev) => [...prev, tx]);
    } catch (err) {
      setError(String(err));
      throw err;
    }
  };

  const remove = async (txId: string) => {
    try {
      await api.delete(`/transactions/${txId}`);
      setTransactions((prev) => prev.filter((t) => t.id !== txId));
    } catch (err) {
      setError(String(err));
    }
  };

  const totalInvested = transactions
    .filter((t) => ["BUY", "DEPOSIT"].includes(t.transaction_type))
    .reduce((s, t) => s + t.amount, 0);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Transactions</h2>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <div className="alert alert-info">Loading…</div> : null}

      {transactions.length > 0 ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="metric-card">
            <p className="text-xs text-white/40">Total transactions</p>
            <p className="text-lg font-semibold text-white">
              {transactions.length}
            </p>
          </div>
          <div className="metric-card">
            <p className="text-xs text-white/40">Total invested</p>
            <p className="text-lg font-semibold text-white">
              {usd(totalInvested)}
            </p>
          </div>
        </div>
      ) : null}

      <div className="panel p-5">
        <h3 className="font-medium text-white mb-4">Add transaction</h3>
        <TransactionForm onSubmit={onSubmit} disabled={loading} />
      </div>

      {transactions.length > 0 ? (
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
              <th></th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((tx) => (
              <tr key={tx.id}>
                <td className="font-mono text-xs">{tx.transaction_date}</td>
                <td>{tx.transaction_type}</td>
                <td className="font-mono">{tx.ticker}</td>
                <td>{formatTxShares(tx)}</td>
                <td>{usd(tx.price)}</td>
                <td>{usd(tx.amount)}</td>
                <td>{tx.fees > 0 ? usd(tx.fees) : "—"}</td>
                <td>
                  <button
                    type="button"
                    className="btn btn-danger !py-1 !px-2 !text-xs"
                    onClick={() => remove(tx.id)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        !loading && (
          <div className="alert alert-info">No transactions yet.</div>
        )
      )}
    </div>
  );
}
