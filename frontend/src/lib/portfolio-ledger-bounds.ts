import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Portfolio, Transaction } from "@/lib/types";

export type LedgerDateBounds = {
  first: string;
  last: string;
  count: number;
};

export function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

export function defaultAnalysisStart(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().slice(0, 10);
}

/** Earliest/latest transaction dates for transaction-mode portfolios. */
export async function fetchLedgerBounds(
  portfolioId: string,
): Promise<LedgerDateBounds | null> {
  const txs = await api.get<Transaction[]>(
    `/portfolios/${portfolioId}/transactions`,
  );
  if (!txs.length) return null;
  const dates = txs.map((t) => t.transaction_date).sort();
  return {
    first: dates[0]!,
    last: dates[dates.length - 1]!,
    count: txs.length,
  };
}

export function isTransactionsMode(portfolio: Portfolio | null): boolean {
  return portfolio?.ledger_mode === "transactions";
}

export function clampStartDate(
  start: string,
  firstTx: string | undefined,
): string {
  if (!firstTx) return start;
  return start < firstTx ? firstTx : start;
}

/** When a transaction-mode portfolio is selected, default dates to ledger inception. */
export function useLedgerAnalysisDates(
  portfolioId: string,
  portfolio: Portfolio | null,
  setStartDate: (v: string) => void,
  setEndDate: (v: string) => void,
): string | null {
  const [fetched, setFetched] = useState<{ id: string; first: string } | null>(
    null,
  );
  const active = Boolean(
    portfolioId && portfolio && isTransactionsMode(portfolio),
  );
  const ledgerFirstTx =
    active && fetched?.id === portfolioId ? fetched.first : null;

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    fetchLedgerBounds(portfolioId)
      .then((bounds) => {
        if (cancelled || !bounds) return;
        setFetched({ id: portfolioId, first: bounds.first });
        setStartDate(bounds.first);
        setEndDate(todayIso());
      })
      .catch(() => {
        if (!cancelled) setFetched(null);
      });
    return () => {
      cancelled = true;
    };
  }, [active, portfolioId, setStartDate, setEndDate]);

  return ledgerFirstTx;
}
