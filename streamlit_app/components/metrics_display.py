"""Metrics display component for showing portfolio metrics."""

from typing import Any, Dict

import streamlit as st

from streamlit_app.utils.formatters import format_percentage


def render_metrics_section(
    metrics: Dict[str, Any],
    category: str,
    columns: int = 3,
) -> None:
    """
    Render a section of metrics in a grid layout.

    Args:
        metrics: Dictionary of metric name -> value
        category: Category name (e.g., "Performance", "Risk")
        columns: Number of columns (default: 3)
    """
    st.subheader(category)

    # Filter out None values
    valid_metrics = {
        k: v for k, v in metrics.items() if v is not None
    }

    if not valid_metrics:
        st.info(f"No {category.lower()} metrics available")
        return

    # Create columns
    cols = st.columns(columns)

    # Display metrics
    for idx, (metric_name, value) in enumerate(valid_metrics.items()):
        col = cols[idx % columns]

        # Format metric name (convert snake_case to Title Case)
        display_name = metric_name.replace("_", " ").title()

        # Determine color and format value based on type
        if isinstance(value, (int, float)):
            # Format value
            if (
                "return" in metric_name.lower() or
                "ratio" in metric_name.lower()
            ):
                formatted_value = format_percentage(value)
            elif abs(value) < 0.01:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}"

            col.metric(display_name, formatted_value)
        else:
            col.metric(display_name, str(value))
