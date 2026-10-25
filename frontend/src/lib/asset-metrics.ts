/** Helpers for /analytics/assets rows (market weights from shares × price). */

export type AssetMetricRow = {
  ticker?: string;
  weight?: number;
  shares?: number;
  price?: number;
  sector?: string;
  industry?: string;
  name?: string;
  currency?: string;
  change_pct?: number;
};

export function rowMarketValue(row: AssetMetricRow): number {
  const shares = Number(row.shares) || 0;
  const price = Number(row.price) || 0;
  if (shares <= 0) return 0;
  if (row.ticker === "CASH") return shares;
  return shares * price;
}

/** Portfolio weight in percent; uses weight column or derives from shares × price. */
export function assetRowWeightPct(
  row: AssetMetricRow,
  allRows: AssetMetricRow[],
): number {
  const fromApi = Number(row.weight);
  if (fromApi > 0) return fromApi;

  const total = allRows.reduce((s, r) => s + rowMarketValue(r), 0);
  if (total <= 0) return 0;
  return (rowMarketValue(row) / total) * 100;
}

export function sectorLabel(sector: string | undefined | null): string {
  const s = sector?.trim();
  if (!s || s === "N/A") return "Other";
  return s;
}
