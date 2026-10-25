/** Build portfolio positions with floor shares and cash remainder (Streamlit parity). */

export type WeightRow = { ticker: string; weight: number };

export type BuiltPosition = {
  ticker: string;
  shares: number;
  weight_target: number;
  purchase_price: number;
};

export function buildPositionsWithCash(params: {
  rows: WeightRow[];
  prices: Record<string, number>;
  totalCapital: number;
  cashAllocationPct: number;
  floorShares?: boolean;
}): { positions: BuiltPosition[]; totalInvested: number; totalCash: number } {
  const { rows, prices, totalCapital, cashAllocationPct, floorShares = true } =
    params;

  const plannedCash = totalCapital * (cashAllocationPct / 100);
  const investable = Math.max(0, totalCapital - plannedCash);

  const stockRows = rows.filter((r) => r.ticker.toUpperCase() !== "CASH");
  const positions: BuiltPosition[] = [];
  let totalInvested = 0;

  for (const row of stockRows) {
    const ticker = row.ticker.toUpperCase();
    const price = prices[ticker];
    if (!price || price <= 0) continue;

    const targetDollars = investable * row.weight;
    let shares = targetDollars / price;
    if (floorShares) {
      shares = Math.floor(shares);
    }
    if (shares < 1) continue;

    const invested = shares * price;
    totalInvested += invested;
    positions.push({
      ticker,
      shares,
      weight_target: row.weight,
      purchase_price: price,
    });
  }

  const remainderCash = Math.max(0, totalCapital - totalInvested - plannedCash);
  const totalCash = plannedCash + remainderCash;
  const cashTargetWeight = cashAllocationPct / 100;
  const stockTargetScale = Math.max(0, 1 - cashTargetWeight);

  for (const p of positions) {
    const row = stockRows.find((r) => r.ticker.toUpperCase() === p.ticker);
    if (row) {
      p.weight_target = row.weight * stockTargetScale;
    }
  }

  if (totalCash >= 1) {
    positions.push({
      ticker: "CASH",
      shares: Math.round(totalCash * 100) / 100,
      weight_target: cashTargetWeight,
      purchase_price: 1,
    });
  }

  return { positions, totalInvested, totalCash };
}

export function formatShareCount(ticker: string, shares: number): string {
  if (ticker === "CASH") {
    return `$${shares.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }
  return String(Math.floor(shares));
}

export function positionMarketValue(
  ticker: string,
  shares: number,
  purchasePrice: number | null,
): number {
  if (ticker === "CASH") return shares;
  return shares * (purchasePrice ?? 0);
}

export function currentWeightPct(
  pos: {
    ticker: string;
    shares: number;
    purchase_price: number | null;
  },
  all: typeof pos[],
): number | null {
  const total = all.reduce(
    (s, p) => s + positionMarketValue(p.ticker, p.shares, p.purchase_price),
    0,
  );
  if (total <= 0) return null;
  const v = positionMarketValue(pos.ticker, pos.shares, pos.purchase_price);
  return (v / total) * 100;
}

/** Market values by ticker plus cash from positions cache. */
export function marketValueByTicker(
  positions: {
    ticker: string;
    shares: number;
    purchase_price?: number | null;
  }[],
  holdings: {
    ticker: string;
    market_value?: number | null;
    market_price?: number | null;
  }[],
): Record<string, number> {
  const values: Record<string, number> = {};
  for (const h of holdings) {
    if (h.market_value != null && h.market_value > 0) {
      values[h.ticker] = h.market_value;
    }
  }
  for (const p of positions) {
    if (p.ticker === "CASH") continue;
    if (values[p.ticker] != null && values[p.ticker] > 0) continue;
    const h = holdings.find((row) => row.ticker === p.ticker);
    if (h?.market_price != null && h.market_price > 0 && p.shares > 0) {
      values[p.ticker] = p.shares * h.market_price;
      continue;
    }
    if (p.purchase_price != null && p.purchase_price > 0 && p.shares > 0) {
      values[p.ticker] = p.shares * p.purchase_price;
    }
  }
  const cash = positions.find((p) => p.ticker === "CASH");
  if (cash && cash.shares > 0) {
    values.CASH = cash.shares;
  }
  return values;
}

export function currentWeightPctFromMarket(
  ticker: string,
  valuesByTicker: Record<string, number>,
): number | null {
  const total = Object.values(valuesByTicker).reduce((s, v) => s + v, 0);
  if (total <= 0) return null;
  const v = valuesByTicker[ticker];
  if (v == null || v <= 0) return null;
  return (v / total) * 100;
}

/** Market value for a position row (holdings API parity). */
export function marketValueForPosition(
  ticker: string,
  shares: number,
  holding?: {
    market_value?: number | null;
    market_price?: number | null;
  } | null,
): number | null {
  if (ticker === "CASH") return shares > 0 ? shares : null;
  if (holding?.market_value != null && holding.market_value > 0) {
    return holding.market_value;
  }
  if (
    holding?.market_price != null &&
    holding.market_price > 0 &&
    shares > 0
  ) {
    return shares * holding.market_price;
  }
  return null;
}

export function targetWeightPct(
  pos: { weight_target: number | null },
): string {
  if (pos.weight_target == null || pos.weight_target <= 0) return "—";
  return `${(pos.weight_target * 100).toFixed(1)}%`;
}

/** @deprecated Prefer targetWeightPct + currentWeightPct for positions table */
export function displayWeightPct(
  pos: { ticker: string; shares: number; weight_target: number | null; purchase_price: number | null },
  all: typeof pos[],
): string {
  const current = currentWeightPct(pos, all);
  if (current != null) return `${current.toFixed(1)}%`;
  return "—";
}
