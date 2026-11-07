import { fetchTickerPrice } from "@/lib/ticker-price-api";
import type { Holding, PnlSummary, Position } from "@/lib/types";

/**
 * Buy-and-hold portfolios have positions but no ledger transactions, so
 * GET /portfolios/{id}/holdings (built from transactions) is empty. Build the
 * equivalent holdings from the positions, pricing them with live quotes
 * (falls back to the purchase price when a quote is unavailable).
 */
export async function buyHoldHoldings(positions: Position[]): Promise<Holding[]> {
  const stocks = positions.filter((p) => p.ticker !== "CASH" && p.shares > 0);
  const quotes = await Promise.allSettled(
    stocks.map((p) => fetchTickerPrice(p.ticker)),
  );
  return stocks.map((p, i) => {
    const q = quotes[i];
    const live =
      q.status === "fulfilled" && q.value.valid && (q.value.price ?? 0) > 0
        ? (q.value.price as number)
        : null;
    const avgCost = p.purchase_price ?? 0;
    const price = live ?? (avgCost > 0 ? avgCost : null);
    const costBasis = p.shares * avgCost;
    const marketValue = price != null ? p.shares * price : null;
    return {
      ticker: p.ticker,
      quantity: p.shares,
      avg_cost: avgCost,
      market_price: price,
      market_value: marketValue,
      cost_basis: costBasis,
      unrealized_pnl:
        marketValue != null && avgCost > 0 ? marketValue - costBasis : null,
    };
  });
}

/** Unrealized PnL for holdings built without a ledger (realized is always 0). */
export function buyHoldPnl(rows: Holding[], base: PnlSummary | null): PnlSummary {
  return {
    ...(base ?? { realized_pnl: 0, unrealized_pnl: 0 }),
    unrealized_pnl: rows.reduce((s, r) => s + (r.unrealized_pnl ?? 0), 0),
  };
}
