/** Portfolio rebalance schedule (target weights live on each position). */

export const REBALANCE_INTERVAL_OPTIONS = [
  { value: "", label: "No rebalancing" },
  { value: "1", label: "Every 1 month" },
  { value: "3", label: "Every 3 months" },
  { value: "6", label: "Every 6 months" },
  { value: "12", label: "Every year" },
] as const;

export function rebalanceIntervalLabel(
  months: number | null | undefined,
): string {
  if (months == null) return "Off";
  const opt = REBALANCE_INTERVAL_OPTIONS.find(
    (o) => o.value === String(months),
  );
  return opt?.label ?? `Every ${months} months`;
}

export function parseRebalanceInterval(
  value: string,
): number | null {
  if (!value) return null;
  const n = Number(value);
  if ([1, 3, 6, 12].includes(n)) return n;
  return null;
}
