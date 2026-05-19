"""Interactive Brokers US equity commission estimate."""


def estimate_ib_commission(shares: float, price_per_share: float) -> float:
    if shares <= 0 or price_per_share <= 0:
        return 0.0
    trade_value = shares * price_per_share
    per_share_fee = 0.005 * shares
    minimum = 1.0
    maximum = trade_value * 0.01
    return round(min(max(per_share_fee, minimum), maximum) * 100) / 100
