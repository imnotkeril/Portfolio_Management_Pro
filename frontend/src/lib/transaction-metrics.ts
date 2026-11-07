import type { Transaction } from "@/lib/types";

/** Net cash deposited (DEPOSIT minus WITHDRAWAL) — not double-counted with BUY. */
export function netDeposits(transactions: Transaction[]): number {
  return transactions.reduce((sum, t) => {
    if (t.transaction_type === "DEPOSIT") return sum + t.amount;
    if (t.transaction_type === "WITHDRAWAL") return sum - t.amount;
    return sum;
  }, 0);
}
