/** Parse weight context from rebalance transaction notes. */

export type RebalanceWeightInfo = {
  weightBeforePct: number;
  weightTargetPct: number;
};

const WEIGHT_PATTERN =
  /weight\s+([\d.]+)%\s*->\s*target\s+([\d.]+)%/i;

export function parseRebalanceWeights(
  notes: string | null | undefined,
): RebalanceWeightInfo | null {
  if (!notes) return null;
  const m = notes.match(WEIGHT_PATTERN);
  if (!m) return null;
  return {
    weightBeforePct: Number(m[1]),
    weightTargetPct: Number(m[2]),
  };
}

export function formatRebalanceWeights(notes: string | null | undefined): string {
  const w = parseRebalanceWeights(notes);
  if (!w) return "";
  return `${w.weightBeforePct.toFixed(1)}% → ${w.weightTargetPct.toFixed(1)}%`;
}
