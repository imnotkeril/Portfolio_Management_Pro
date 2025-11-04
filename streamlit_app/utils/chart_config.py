"""Chart configuration for Plotly charts."""

# Color palette (from requirements)
COLORS = {
    "primary": "#BF9FFB",  # Purple
    "secondary": "#90BFF9",  # Blue
    "background": "#0D1015",  # Dark background
    "success": "#00CC96",  # Green
    "danger": "#EF553B",  # Red
    "warning": "#FFA15A",  # Orange
    "info": "#19D3F3",  # Cyan
    "cash": "#C8C8C8",  # Gray for cash
    "text": "#FFFFFF",  # White text
}

DEFAULT_LAYOUT = {
    "height": 500,
    "margin": dict(l=50, r=50, t=50, b=50),
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": dict(size=11, color="#FFFFFF"),
    "xaxis": dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
        showline=True,
        linecolor="rgba(255,255,255,0.2)",
    ),
    "yaxis": dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
        showline=True,
        linecolor="rgba(255,255,255,0.2)",
    ),
}


def get_chart_layout(**overrides) -> dict:
    """
    Get default chart layout with optional overrides.

    Args:
        **overrides: Layout parameters to override

    Returns:
        Dictionary with Plotly layout settings
    """
    layout = DEFAULT_LAYOUT.copy()
    layout.update(overrides)
    return layout


