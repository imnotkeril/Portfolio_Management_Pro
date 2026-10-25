import type { Transaction } from "@/lib/types";

function usd(v: number) {
  return `$${v.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/** Human-readable shares / cash / dividend columns for transaction tables. */
export function formatTxShares(tx: Transaction): string {
  if (tx.ticker === "CASH") return usd(tx.shares);
  if (tx.transaction_type === "DIVIDEND") {
    const dps = tx.price;
    return `${Math.floor(tx.shares)} sh × ${usd(dps)}`;
  }
  return String(Math.floor(tx.shares));
}

export function sortTransactionsForDisplay(
  txs: Transaction[],
): Transaction[] {
  const typeOrder: Record<string, number> = {
    DEPOSIT: 0,
    WITHDRAWAL: 1,
    SPLIT: 2,
    BUY: 3,
    SELL: 4,
    DIVIDEND: 5,
  };
  return [...txs].sort((a, b) => {
    const d = a.transaction_date.localeCompare(b.transaction_date);
    if (d !== 0) return d;
    const ta = typeOrder[a.transaction_type] ?? 99;
    const tb = typeOrder[b.transaction_type] ?? 99;
    if (ta !== tb) return ta - tb;
    return a.ticker.localeCompare(b.ticker);
  });
}
