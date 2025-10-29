"""Text parsing utilities for portfolio creation."""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def parse_ticker_weights_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse text input for tickers and weights.

    Supports formats:
    - "AAPL:0.4, MSFT:0.3, GOOGL:0.3"
    - "AAPL 40%, MSFT 30%, GOOGL 30%"
    - "AAPL 0.4\nMSFT 0.3\nGOOGL 0.3"
    - "AAPL, MSFT, GOOGL" (equal weights)

    Args:
        text: Text input with tickers and weights

    Returns:
        List of dictionaries with 'ticker' and 'weight' keys
    """
    assets = []
    text = text.strip()

    if not text:
        return assets

    # Split by commas or newlines
    lines = []
    if ',' in text:
        lines = [line.strip() for line in text.split(',') if line.strip()]
    else:
        lines = [line.strip() for line in text.split('\n') if line.strip()]

    for line in lines:
        if not line:
            continue

        # Pattern 1: TICKER:WEIGHT or TICKER: WEIGHT
        colon_match = re.match(r'^([A-Za-z0-9.-]+):\s*([0-9.%]+)$', line)

        # Pattern 2: TICKER WEIGHT% or TICKER WEIGHT
        percent_match = re.match(r'^([A-Za-z0-9.-]+)\s+([0-9.]+)%?$', line)

        # Pattern 3: Just TICKER (equal weight)
        ticker_only_match = re.match(r'^([A-Za-z0-9.-]+)$', line)

        if colon_match:
            ticker, weight_str = colon_match.groups()
            weight = float(weight_str.rstrip('%'))
            # Normalize: if weight > 1, assume it's a percentage
            if weight > 1.0:
                weight = weight / 100.0
            assets.append({
                'ticker': ticker.strip().upper(),
                'weight': weight
            })
        elif percent_match:
            ticker, weight_str = percent_match.groups()
            weight = float(weight_str)
            # Normalize: if weight > 1, assume it's a percentage
            # Also normalize if line ends with %
            if weight > 1.0 or line.endswith('%'):
                weight = weight / 100.0
            assets.append({
                'ticker': ticker.strip().upper(),
                'weight': weight
            })
        elif ticker_only_match:
            ticker = ticker_only_match.group(1)
            assets.append({
                'ticker': ticker.strip().upper(),
                'weight': 0.0
            })

    # If all weights are 0, assign equal weights
    if assets and all(asset['weight'] == 0.0 for asset in assets):
        equal_weight = 1.0 / len(assets)
        for asset in assets:
            asset['weight'] = equal_weight

    return assets

