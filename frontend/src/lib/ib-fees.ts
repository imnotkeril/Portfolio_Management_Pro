/** Interactive Brokers US stock commission estimate (IB Pro tier). */

export function estimateIbCommission(
  shares: number,
  pricePerShare: number,
): number {
  if (shares <= 0 || pricePerShare <= 0) return 0;
  const tradeValue = shares * pricePerShare;
  const perShareFee = 0.005 * shares;
  const minimum = 1.0;
  const maximum = tradeValue * 0.01;
  return Math.round(Math.min(Math.max(perShareFee, minimum), maximum) * 100) / 100;
}

/** Cash movements typically have no IB equity commission. */
export function defaultFeesForTransaction(
  type: string,
  shares: number,
  price: number,
): number {
  const t = type.toUpperCase();
  if (t === "DEPOSIT" || t === "WITHDRAWAL") return 0;
  if (t === "DIVIDEND" || t === "SPLIT") return 0;
  return estimateIbCommission(shares, price);
}
