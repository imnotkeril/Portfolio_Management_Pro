export function usd(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v)) return "—";
  return `$${v.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function num(v: number | null | undefined, digits = 4) {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

/** API metrics are decimals: 0.0563 → 5.63%, 5.63 → 563%. */
export function percentDecimal(
  v: number | null | undefined,
  digits = 2,
): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function pct(v: number | null | undefined): string {
  return percentDecimal(v, 1);
}
