"""Utilities for comparing portfolio metrics with benchmark."""

from typing import Dict, Optional, Tuple


def compare_metrics(
    portfolio_value: Optional[float],
    benchmark_value: Optional[float],
) -> Dict[str, any]:
    """
    Compare portfolio metric with benchmark metric.

    Args:
        portfolio_value: Portfolio metric value
        benchmark_value: Benchmark metric value

    Returns:
        Dictionary with comparison data:
        - difference: Absolute difference
        - percentage_diff: Percentage difference
        - is_better: Whether portfolio is better (for positive metrics)
        - status: 'better', 'worse', 'neutral', or 'no_benchmark'
    """
    if portfolio_value is None:
        return {
            "difference": None,
            "percentage_diff": None,
            "is_better": None,
            "status": "no_data",
        }

    if benchmark_value is None:
        return {
            "difference": None,
            "percentage_diff": None,
            "is_better": None,
            "status": "no_benchmark",
        }

    difference = portfolio_value - benchmark_value

    # Calculate percentage difference
    if benchmark_value != 0:
        percentage_diff = (difference / abs(benchmark_value)) * 100
    else:
        percentage_diff = None

    # Determine status (neutral if very small difference)
    if abs(difference) < 0.0001:
        status = "neutral"
        is_better = None
    else:
        is_better = difference > 0
        status = "better" if is_better else "worse"

    return {
        "difference": difference,
        "percentage_diff": percentage_diff,
        "is_better": is_better,
        "status": status,
    }


def determine_metric_direction(metric_name: str) -> str:
    """
    Determine if higher is better or lower is better for a metric.

    Args:
        metric_name: Name of the metric

    Returns:
        'higher' if higher is better, 'lower' if lower is better
    """
    metric_lower = metric_name.lower()

    # Lower is better
    if any(
        keyword in metric_lower
        for keyword in [
            "drawdown",
            "volatility",
            "deviation",
            "var",
            "cvar",
            "risk",
            "pain",
            "ulcer",
            "tracking_error",
            "downside",
        ]
    ):
        return "lower"

    # Higher is better (default for most metrics)
    return "higher"


def is_metric_better(
    portfolio_value: float,
    benchmark_value: float,
    metric_name: str,
) -> bool:
    """
    Determine if portfolio metric is better than benchmark.

    Args:
        portfolio_value: Portfolio metric value
        benchmark_value: Benchmark metric value
        metric_name: Name of the metric

    Returns:
        True if portfolio is better, False otherwise
    """
    direction = determine_metric_direction(metric_name)

    if direction == "lower":
        return portfolio_value < benchmark_value
    else:
        return portfolio_value > benchmark_value


def format_comparison_delta(
    difference: Optional[float],
    is_percentage: bool = False,
    decimals: int = 2,
) -> str:
    """
    Format difference for display.

    Args:
        difference: Difference value
        is_percentage: Whether the value is already a percentage
        decimals: Number of decimal places

    Returns:
        Formatted string with sign (e.g., "+2.17%", "-0.15")
    """
    if difference is None:
        return "N/A"

    sign = "+" if difference > 0 else ""

    if is_percentage:
        return f"{sign}{difference:.{decimals}f}%"
    else:
        # Multiply by 100 to show as percentage
        return f"{sign}{difference * 100:.{decimals}f}%"


def get_comparison_color(
    status: str,
    metric_name: str,
) -> str:
    """
    Get color for comparison based on status and metric type.

    Args:
        status: Comparison status ('better', 'worse', 'neutral', 'no_benchmark')
        metric_name: Name of the metric

    Returns:
        Color code ('green', 'red', 'gray', 'blue')
    """
    if status == "no_benchmark" or status == "no_data":
        return "gray"

    if status == "neutral":
        return "blue"

    direction = determine_metric_direction(metric_name)

    # For "lower is better" metrics, invert colors
    if direction == "lower":
        return "red" if status == "better" else "green"
    else:
        return "green" if status == "better" else "red"


def calculate_outperformance(
    portfolio_return: Optional[float],
    benchmark_return: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate outperformance vs benchmark.

    Args:
        portfolio_return: Portfolio return (as decimal, e.g., 0.1234)
        benchmark_return: Benchmark return (as decimal)

    Returns:
        Tuple of (outperformance, description)
    """
    if portfolio_return is None or benchmark_return is None:
        return None, "N/A"

    outperformance = portfolio_return - benchmark_return

    if abs(outperformance) < 0.0001:
        description = "In line with benchmark"
    elif outperformance > 0:
        pct = outperformance * 100
        description = f"Outperformed by {pct:.2f}%"
    else:
        pct = abs(outperformance) * 100
        description = f"Underperformed by {pct:.2f}%"

    return outperformance, description
