"""Chart configuration for Plotly charts."""

# Color palette (from requirements)
COLORS = {
    "primary": "#BF9FFB",  # Purple
    "secondary": "#90BFF9",  # Blue
    "background": "#0D1015",  # Dark background
    "success": "#00CC96",  # Green
    "danger": "#EF553B",  # Red
    "error": "#EF553B",  # Red (alias for danger)
    "warning": "#FFA15A",  # Orange
    "info": "#19D3F3",  # Cyan
    "cash": "#C8C8C8",  # Gray for cash
    "text": "#FFFFFF",  # White text
}

# Method-specific colors (pastel colors for forecasts)
METHOD_COLORS = {
    "arima": "#A8D5E2",  # Pastel blue
    "prophet": "#B5E5CF",  # Pastel green
    "xgboost": "#FFD4A3",  # Pastel orange
    "random_forest": "#E2B5E8",  # Pastel purple
    "lstm": "#FFB3BA",  # Pastel pink
    "tcn": "#BAFFC9",  # Pastel mint
    "garch": "#FFDFBA",  # Pastel peach
    "arima_garch": "#D4A5FF",  # Pastel lavender
    "svm": "#FFCCCB",  # Pastel coral
    "ensemble": "#C8E6C9",  # Pastel light green
}

def get_method_color(method_name: str) -> str:
    """
    Get color for a forecasting method.
    
    Args:
        method_name: Name of the forecasting method (case-insensitive)
        
    Returns:
        Color hex code for the method
    """
    method_lower = method_name.lower()
    # Handle variations in method names
    if "arima" in method_lower and "garch" in method_lower:
        return METHOD_COLORS.get("arima_garch", COLORS["primary"])
    elif "arima" in method_lower:
        return METHOD_COLORS.get("arima", COLORS["secondary"])
    elif "prophet" in method_lower:
        return METHOD_COLORS.get("prophet", COLORS["success"])
    elif "xgboost" in method_lower or "xgb" in method_lower:
        return METHOD_COLORS.get("xgboost", COLORS["warning"])
    elif "random" in method_lower and "forest" in method_lower:
        return METHOD_COLORS.get("random_forest", COLORS["primary"])
    elif "lstm" in method_lower:
        return METHOD_COLORS.get("lstm", COLORS["danger"])
    elif "tcn" in method_lower:
        return METHOD_COLORS.get("tcn", COLORS["info"])
    elif "garch" in method_lower:
        return METHOD_COLORS.get("garch", COLORS["warning"])
    elif "svm" in method_lower:
        return METHOD_COLORS.get("svm", COLORS["error"])
    elif "ensemble" in method_lower:
        return METHOD_COLORS.get("ensemble", COLORS["success"])
    else:
        # Default color if method not found
        return COLORS["primary"]

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
