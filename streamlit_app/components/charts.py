"""Chart components for portfolio visualization using Plotly."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

from streamlit_app.utils.chart_config import COLORS, get_chart_layout

logger = logging.getLogger(__name__)


def plot_cumulative_returns(
    data: Dict[str, pd.Series],
    linear_scale: bool = True,
) -> go.Figure:
    """
    Plot cumulative returns chart.

    Args:
        data: Dictionary with 'portfolio' and optionally 'benchmark' Series
        linear_scale: Whether to use linear scale (True) or log scale (False)

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Portfolio line
    if "portfolio" in data and not data["portfolio"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["portfolio"].index,
                y=data["portfolio"].values * 100,  # Convert to percentage
                mode="lines",
                name="Portfolio",
                line=dict(color=COLORS["primary"], width=2),
            )
        )

    # Benchmark line
    if "benchmark" in data and not data["benchmark"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["benchmark"].index,
                y=data["benchmark"].values * 100,  # Convert to percentage
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2),
            )
        )

    layout = get_chart_layout(
        title="Cumulative Returns",
        yaxis=dict(
            title="Cumulative Return (%)",
            tickformat=",.1f",
            type="linear" if linear_scale else "log",
        ),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_drawdown(
    data: Dict[str, pd.Series],
) -> go.Figure:
    """
    Plot drawdown chart.

    Args:
        data: Dictionary with 'drawdown' Series and 'max_drawdown' info

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    drawdown_series = data.get("drawdown", pd.Series())
    max_dd_info = data.get("max_drawdown", {})

    if not drawdown_series.empty:
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100,  # Convert to percentage
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(250, 161, 164, 0.3)",  # Red gradient
                line=dict(color=COLORS["danger"], width=2),
                name="Drawdown",
            )
        )

        # Max drawdown annotation
        if max_dd_info and "date" in max_dd_info and "value" in max_dd_info:
            fig.add_annotation(
                x=max_dd_info["date"],
                y=max_dd_info["value"] * 100,
                text=f"Max DD: {max_dd_info['value']*100:.2f}%",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS["danger"],
            )

    layout = get_chart_layout(
        title="Drawdown Analysis",
        yaxis=dict(title="Drawdown (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_rolling_metric(
    data: Dict[str, pd.Series],
    metric_name: str,
    threshold: Optional[float] = None,
) -> go.Figure:
    """
    Plot rolling metric chart.

    Args:
        data: Dictionary with 'metric' Series
        metric_name: Name of the metric
        threshold: Optional threshold line value

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    metric_series = data.get("metric", pd.Series())
    if not metric_series.empty:
        fig.add_trace(
            go.Scatter(
                x=metric_series.index,
                y=metric_series.values,
                mode="lines",
                name=metric_name.title(),
                line=dict(color=COLORS["primary"], width=2),
            )
        )

        # Threshold line (e.g., Sharpe = 1.0)
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=COLORS["warning"],
                annotation_text=f"Threshold: {threshold:.2f}",
            )

    layout = get_chart_layout(
        title="Rolling " + metric_name,
        yaxis=dict(title=metric_name.title()),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_return_distribution(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot return distribution histogram.

    Args:
        data: Dictionary with histogram data and statistics

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    counts = data.get("counts", np.array([]))
    edges = data.get("edges", np.array([]))
    mean = data.get("mean", 0.0)
    std = data.get("std", 0.0)

    if len(counts) > 0 and len(edges) > 1:
        # Create bins centers for x-axis
        # Convert to percentage
        bin_centers = (edges[:-1] + edges[1:]) / 2 * 100

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                name="Return Distribution",
                marker_color=COLORS["primary"],
                opacity=0.7,
            )
        )

        # Mean line
        if mean != 0:
            mean_pct = mean * 100
            fig.add_vline(
                x=mean_pct,
                line_dash="dash",
                line_color=COLORS["success"],
                annotation_text=f"Mean: {mean_pct:.2f}%",
            )

        # VaR lines
        var_95 = data.get("var_95")
        if var_95 is not None:
            fig.add_vline(
                x=var_95 * 100,
                line_dash="dot",
                line_color=COLORS["danger"],
                annotation_text=f"VaR 95%: {var_95*100:.2f}%",
            )

        # Normal distribution overlay
        if std > 0:
            x_norm = np.linspace(edges[0], edges[-1], 100) * 100
            pdf_values = stats.norm.pdf(x_norm / 100, loc=mean, scale=std)
            y_norm = pdf_values * len(counts) * (edges[1] - edges[0])
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode="lines",
                    name="Normal Distribution",
                    line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                )
            )

    layout = get_chart_layout(
        title="Return Distribution",
        yaxis=dict(title="Frequency"),
        xaxis=dict(title="Return (%)", tickformat=",.1f"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_qq_plot(
    data: Dict[str, np.ndarray],
) -> go.Figure:
    """
    Plot Q-Q plot (quantile-quantile plot).

    Args:
        data: Dictionary with 'theoretical' and 'sample' quantiles

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    theoretical = data.get("theoretical", np.array([]))
    sample = data.get("sample", np.array([]))

    if len(theoretical) > 0 and len(sample) > 0:
        # Convert to percentage
        theoretical_pct = theoretical * 100
        sample_pct = sample * 100

        fig.add_trace(
            go.Scatter(
                x=theoretical_pct,
                y=sample_pct,
                mode="markers",
                name="Data Points",
                marker=dict(color=COLORS["primary"], size=4),
            )
        )

        # 45-degree reference line
        min_val = min(theoretical_pct.min(), sample_pct.min())
        max_val = max(theoretical_pct.max(), sample_pct.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="45° Line (Normal)",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            )
        )

    layout = get_chart_layout(
        title="Q-Q Plot (Normal Distribution)",
        yaxis=dict(title="Sample Quantiles (%)", tickformat=",.1f"),
        xaxis=dict(title="Theoretical Quantiles (%)", tickformat=",.1f"),
        hovermode="closest",
    )

    fig.update_layout(**layout)
    return fig


def plot_monthly_heatmap(
    data: Dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Plot monthly returns heatmap.

    Args:
        data: Dictionary with 'heatmap' DataFrame (years x months)

    Returns:
        Plotly Figure
    """
    heatmap_df = data.get("heatmap", pd.DataFrame())

    if heatmap_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Prepare data for heatmap
    years = heatmap_df.index.values
    months = heatmap_df.columns.values
    z_values = heatmap_df.values

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=months,
            y=years,
            colorscale=[
                [0, COLORS["danger"]],
                [0.5, "#FFFFFF"],
                [1, COLORS["success"]],
            ],
            text=heatmap_df.round(2).values,
            texttemplate="%{text}%",
            textfont={"size": 10},
            colorbar=dict(title="Return (%)"),
        )
    )

    layout = get_chart_layout(
        title="Monthly Returns Heatmap",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year"),
    )

    fig.update_layout(**layout)
    return fig


def plot_rolling_sharpe(
    data: Dict[str, pd.Series],
    window: int = 126,
) -> go.Figure:
    """
    Plot rolling Sharpe ratio with benchmark comparison.

    Args:
        data: Dictionary with 'portfolio' and optionally 'benchmark' Series
        window: Rolling window size in days

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Portfolio rolling Sharpe
    if "portfolio" in data and not data["portfolio"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["portfolio"].index,
                y=data["portfolio"].values,
                mode="lines",
                name="Portfolio",
                line=dict(color=COLORS["primary"], width=2),
            )
        )

    # Benchmark rolling Sharpe
    if "benchmark" in data and not data["benchmark"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["benchmark"].index,
                y=data["benchmark"].values,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2),
            )
        )

    # Reference line (Sharpe = 1.0)
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=COLORS["warning"],
        annotation_text="Sharpe = 1.0",
    )

    layout = get_chart_layout(
        title=f"Rolling Sharpe Ratio ({window}-day window)",
        yaxis=dict(title="Sharpe Ratio", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_rolling_beta_alpha(
    data: Dict[str, pd.Series],
    window: int = 126,
) -> go.Figure:
    """
    Plot rolling Beta and Alpha.

    Args:
        data: Dictionary with 'beta' and 'alpha' Series
        window: Rolling window size in days

    Returns:
        Plotly Figure (dual y-axis)
    """
    fig = go.Figure()

    # Beta on primary y-axis
    if "beta" in data and not data["beta"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["beta"].index,
                y=data["beta"].values,
                mode="lines",
                name="Beta",
                line=dict(color=COLORS["primary"], width=2),
                yaxis="y",
            )
        )

    # Alpha on secondary y-axis
    if "alpha" in data and not data["alpha"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["alpha"].index,
                y=data["alpha"].values,
                mode="lines",
                name="Alpha",
                line=dict(color=COLORS["success"], width=2),
                yaxis="y2",
            )
        )

    # Beta reference line at 1.0
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=COLORS["warning"],
        annotation_text="Beta = 1.0",
    )

    layout = get_chart_layout(
        title=f"Rolling Beta & Alpha ({window}-day window)",
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    # Add dual y-axis
    layout["yaxis"] = dict(title="Beta", side="left", tickformat=",.2f")
    layout["yaxis2"] = dict(
        title="Alpha (%)",
        side="right",
        overlaying="y",
        tickformat=",.2f",
    )

    fig.update_layout(**layout)
    return fig


def plot_underwater(
    data: Dict[str, pd.Series],
) -> go.Figure:
    """
    Plot underwater plot (drawdown from peak).

    Args:
        data: Dictionary with 'underwater' Series and optionally 'benchmark' Series

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    underwater_series = data.get("underwater", pd.Series())

    if not underwater_series.empty:
        fig.add_trace(
            go.Scatter(
                x=underwater_series.index,
                y=underwater_series.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(250, 161, 164, 0.3)",
                line=dict(color=COLORS["danger"], width=2),
                name="Portfolio",
            )
        )

    # Benchmark underwater (if available)
    benchmark_underwater = data.get("benchmark", pd.Series())
    if not benchmark_underwater.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_underwater.index,
                y=benchmark_underwater.values,
                mode="lines",
                line=dict(color=COLORS["secondary"], width=2),
                name="Benchmark",
            )
        )

    layout = get_chart_layout(
        title="Underwater Plot (Drawdown from Peak)",
        yaxis=dict(title="Drawdown (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_yearly_returns(
    data: Dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Plot yearly returns comparison bar chart.

    Args:
        data: Dictionary with 'yearly' DataFrame

    Returns:
        Plotly Figure
    """
    yearly_df = data.get("yearly", pd.DataFrame())

    if yearly_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Portfolio bars
    if "Portfolio" in yearly_df.columns:
        fig.add_trace(
            go.Bar(
                x=yearly_df.index,
                y=yearly_df["Portfolio"],
                name="Portfolio",
                marker_color=COLORS["primary"],
            )
        )

    # Benchmark bars
    if "Benchmark" in yearly_df.columns:
        fig.add_trace(
            go.Bar(
                x=yearly_df.index,
                y=yearly_df["Benchmark"],
                name="Benchmark",
                marker_color=COLORS["secondary"],
            )
        )

    layout = get_chart_layout(
        title="Yearly Returns Comparison",
        yaxis=dict(title="Return (%)", tickformat=",.1f"),
        xaxis=dict(title="Year"),
        barmode="group",
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_best_worst_periods(
    data: Dict[str, pd.DataFrame],
    period_type: str = "days",
) -> go.Figure:
    """
    Plot best and worst periods side by side.

    Args:
        data: Dictionary with 'best_days' and 'worst_days' DataFrames
        period_type: 'days', 'weeks', or 'months'

    Returns:
        Plotly Figure
    """
    best = data.get("best_days", pd.DataFrame())
    worst = data.get("worst_days", pd.DataFrame())

    fig = go.Figure()

    # Best periods (green)
    if not best.empty:
        fig.add_trace(
            go.Bar(
                x=best.index.astype(str),
                y=best["Return"],
                name=f"Best {period_type}",
                marker_color=COLORS["success"],
            )
        )

    # Worst periods (red)
    if not worst.empty:
        fig.add_trace(
            go.Bar(
                x=worst.index.astype(str),
                y=worst["Return"],
                name=f"Worst {period_type}",
                marker_color=COLORS["danger"],
            )
        )

    layout = get_chart_layout(
        title=f"Best & Worst {period_type.title()}",
        yaxis=dict(title="Return (%)", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_asset_allocation(
    asset_data: Dict[str, float],
) -> go.Figure:
    """
    Plot asset allocation pie chart.

    Args:
        asset_data: Dictionary mapping asset ticker to weight (%)

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if not asset_data:
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Filter out 0% items and build color map (CASH always gray)
    labels = []
    values = []
    colors = []
    has_cash = False
    for k, v in asset_data.items():
        val = float(v) if v is not None else 0.0
        if str(k).upper() == "CASH":
            has_cash = True
            if val <= 0:
                val = 1e-6  # show tiny slice for 0%
            labels.append(k)
            values.append(val)
            colors.append("#9E9E9E")
        elif val and abs(val) > 1e-6:
            labels.append(k)
            values.append(val)
            colors.append(None)
    if not has_cash:
        labels.append("CASH")
        values.append(1e-6)
        colors.append("#9E9E9E")

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
            hole=0.5,
            marker=dict(colors=colors),
        )
    )

    layout = get_chart_layout(
        title="Asset Allocation",
    )
    layout["height"] = 320

    fig.update_layout(**layout)
    return fig


def plot_sector_allocation(
    sector_data: Dict[str, float],
) -> go.Figure:
    """
    Plot sector allocation pie chart.

    Args:
        sector_data: Dictionary mapping sector to weight (%)

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if not sector_data:
        fig.add_annotation(
            text="No sector data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    labels = []
    values = []
    colors = []
    has_cash_sector = False
    for k, v in sector_data.items():
        val = float(v) if v is not None else 0.0
        if str(k).lower() == "cash":
            has_cash_sector = True
            if val <= 0:
                val = 1e-6
            labels.append(k)
            values.append(val)
            colors.append("#9E9E9E")
        elif val and abs(val) > 1e-6:
            labels.append(k)
            values.append(val)
            colors.append(None)
    if not has_cash_sector:
        labels.append("Cash")
        values.append(1e-6)
        colors.append("#9E9E9E")

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
            hole=0.5,
            marker=dict(colors=colors),
        )
    )

    layout = get_chart_layout(
        title="Sector Allocation",
    )
    layout["height"] = 320

    fig.update_layout(**layout)
    return fig