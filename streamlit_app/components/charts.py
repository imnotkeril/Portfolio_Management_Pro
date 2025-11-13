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
    bar_color: str = "blue",  # "blue" for daily, "green" for monthly
) -> go.Figure:
    """
    Plot return distribution histogram.

    Args:
        data: Dictionary with histogram data and statistics
        bar_color: Color for bars ("blue" or "green")

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

        # Bar color based on parameter
        if bar_color == "green":
            bar_color_hex = COLORS["success"]  # Green
        else:
            bar_color_hex = COLORS["primary"]  # Purple (portfolio color)

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                name="Return Distribution",
                marker_color=bar_color_hex,
                opacity=0.7,
            )
        )

        # Mean line (blue dashed)
        mean_pct = mean * 100
        fig.add_vline(
            x=mean_pct,
            line_dash="dash",
            line_color=COLORS["secondary"],  # Blue
            annotation_text=f"Mean: {mean_pct:.2f}%",
            annotation_position="top",
        )

        # VaR lines (90%, 95%, 99%) - both negative and positive sides
        var_90 = data.get("var_90")
        var_95 = data.get("var_95")
        var_99 = data.get("var_99")
        
        # Positive VaR (upper tail)
        var_90_pos = data.get("var_90_pos")
        var_95_pos = data.get("var_95_pos")
        var_99_pos = data.get("var_99_pos")
        
        # Negative VaR (lower tail) - percentage at bottom, value at top
        if var_90 is not None:
            var_90_val = var_90 * 100
            # Add annotation at top with value
            fig.add_annotation(
                x=var_90_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_90_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["danger"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            # Add vline with percentage label at bottom
            fig.add_vline(
                x=var_90_val,
                line_dash="dot",
                line_color=COLORS["danger"],
                annotation_text="90%",
                annotation_position="bottom",
                line_width=1,
            )
        if var_95 is not None:
            var_95_val = var_95 * 100
            fig.add_annotation(
                x=var_95_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_95_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["danger"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            fig.add_vline(
                x=var_95_val,
                line_dash="dot",
                line_color=COLORS["danger"],
                annotation_text="95%",
                annotation_position="bottom",
                line_width=1,
            )
        if var_99 is not None:
            var_99_val = var_99 * 100
            fig.add_annotation(
                x=var_99_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_99_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["danger"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            fig.add_vline(
                x=var_99_val,
                line_dash="dot",
                line_color=COLORS["danger"],
                annotation_text="99%",
                annotation_position="bottom",
                line_width=1,
            )
        
        # Positive VaR (upper tail) - percentage at bottom, value at top
        if var_90_pos is not None:
            var_90_pos_val = var_90_pos * 100
            fig.add_annotation(
                x=var_90_pos_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_90_pos_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["success"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            fig.add_vline(
                x=var_90_pos_val,
                line_dash="dot",
                line_color=COLORS["success"],
                annotation_text="90%",
                annotation_position="bottom",
                line_width=1,
            )
        if var_95_pos is not None:
            var_95_pos_val = var_95_pos * 100
            fig.add_annotation(
                x=var_95_pos_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_95_pos_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["success"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            fig.add_vline(
                x=var_95_pos_val,
                line_dash="dot",
                line_color=COLORS["success"],
                annotation_text="95%",
                annotation_position="bottom",
                line_width=1,
            )
        if var_99_pos is not None:
            var_99_pos_val = var_99_pos * 100
            fig.add_annotation(
                x=var_99_pos_val,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{var_99_pos_val:.2f}%",
                showarrow=False,
                font=dict(size=10, color=COLORS["success"]),
                bgcolor="rgba(0,0,0,0.7)",
            )
            fig.add_vline(
                x=var_99_pos_val,
                line_dash="dot",
                line_color=COLORS["success"],
                annotation_text="99%",
                annotation_position="bottom",
                line_width=1,
            )

        # Normal distribution overlay (orange dashed) - #FFCC80
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
                    line=dict(color=COLORS["warning"], width=2, dash="dash"),  # Orange #FFCC80
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
                marker=dict(color=COLORS["primary"], size=4),  # Purple dots
            )
        )

        # 45-degree reference line (red)
        min_val = min(theoretical_pct.min(), sample_pct.min())
        max_val = max(theoretical_pct.max(), sample_pct.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="45° Line (Normal)",
                line=dict(color=COLORS["danger"], width=2, dash="dash"),  # Red
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

    # Reference line (Sharpe = 1.0) - white dashed
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="white",
        line_width=1,
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
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot underwater plot (drawdown from peak).

    Args:
        data: Dictionary with 'underwater' Series, optionally 'benchmark' Series,
              'max_drawdown' dict, and 'current_drawdown' dict

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    underwater_series = data.get("underwater", pd.Series())

    if not underwater_series.empty:
        # Portfolio drawdown - red color
        fig.add_trace(
            go.Scatter(
                x=underwater_series.index,
                y=underwater_series.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(239, 83, 80, 0.3)",  # Red with transparency
                line=dict(color=COLORS["danger"], width=2),  # Red - portfolio
                name="Portfolio",
            )
        )

    # Benchmark underwater (if available) - orange color
    benchmark_underwater = data.get("benchmark", pd.Series())
    if not benchmark_underwater.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_underwater.index,
                y=benchmark_underwater.values,
                mode="lines",
                line=dict(color=COLORS["warning"], width=2),  # Orange #FFCC80 - benchmark
                name="Benchmark",
            )
        )

    # Add zero line
    if not underwater_series.empty:
        fig.add_hline(
            y=0,
            line=dict(color="black", width=2),
            annotation_text="Peak",
            annotation_position="right"
        )

    # Annotate max drawdown
    max_dd = data.get("max_drawdown")
    if max_dd and "date" in max_dd and "value" in max_dd:
        fig.add_annotation(
            x=max_dd["date"],
            y=max_dd["value"],
            text=f"Max DD: {max_dd['value']:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["danger"],
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            bordercolor=COLORS["danger"],
            borderwidth=2,
            font=dict(color="white", size=11)  # White text
        )

    # Highlight current drawdown (if exists)
    current_dd = data.get("current_drawdown")
    if current_dd and "date" in current_dd and "value" in current_dd:
        fig.add_annotation(
            x=current_dd["date"],
            y=current_dd["value"],
            text=f"Current: {current_dd['value']:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["warning"],
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            bordercolor=COLORS["warning"],
            borderwidth=2,
            font=dict(color="white", size=11)  # White text
        )

    layout = get_chart_layout(
        title="Underwater Plot (Drawdown from Peak)",
        yaxis=dict(title="Drawdown (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_drawdown_periods(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot drawdown periods with shaded zones.

    Args:
        data: Dictionary with 'cumulative_returns' Series and 'drawdown_zones' list

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    cum_returns = data.get("cumulative_returns", pd.Series())
    drawdown_zones = data.get("drawdown_zones", [])

    # Plot cumulative returns line
    if not cum_returns.empty:
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                name="Cumulative Return",
            )
        )

    # Add shaded regions for drawdown periods
    for zone in drawdown_zones:
        # Add shaded region
        fig.add_vrect(
            x0=zone["start"],
            x1=zone["end"],
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        
        # Add label in the middle of the zone
        mid_date = zone["start"] + (zone["end"] - zone["start"]) / 2
        if mid_date in cum_returns.index:
            mid_value = cum_returns.loc[mid_date]
        else:
            # Find closest date
            closest_idx = cum_returns.index.get_indexer([mid_date], method="nearest")[0]
            mid_value = cum_returns.iloc[closest_idx]
        
        fig.add_annotation(
            x=mid_date,
            y=mid_value,
            text=f"{zone['depth']*100:.1f}%",
            showarrow=False,
            bgcolor="rgba(220, 53, 69, 0.8)",
            bordercolor=COLORS["danger"],
            borderwidth=2,
            font=dict(size=11, color="white", family="Arial Black")
        )

    layout = get_chart_layout(
        title="Drawdown Periods",
        yaxis=dict(title="Cumulative Return (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_drawdown_recovery(
    drawdown_data: dict,
) -> go.Figure:
    """
    Plot drawdown recovery timeline for a single drawdown.

    Args:
        drawdown_data: Dictionary with drawdown information

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    start = drawdown_data["start_date"]
    bottom = drawdown_data["bottom_date"]
    recovery = drawdown_data["recovery_date"]
    
    # Create timeline points
    dates = [start, bottom]
    values = [0, drawdown_data["depth"] * 100]  # Convert to %
    colors = ["green", "red"]
    labels = ["Peak", f"Trough ({drawdown_data['duration_days']}d)"]
    
    if recovery:
        dates.append(recovery)
        values.append(0)
        colors.append("yellow")
        labels.append(f"Recovery ({drawdown_data['recovery_days']}d)")

    # Plot timeline
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(color="gray", width=2),
            marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
            showlegend=False,
        )
    )

    # Add annotations for each point
    for date, value, label in zip(dates, values, labels):
        fig.add_annotation(
            x=date,
            y=value,
            text=label,
            showarrow=True,
            arrowhead=2,
            yshift=20 if value < 0 else -20,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="black", size=11)
        )

    # Add zero line
    fig.add_hline(
        y=0,
        line=dict(color="black", width=1, dash="dash"),
    )

    layout = get_chart_layout(
        title=f"Drawdown #{drawdown_data['number']}: {start} to {recovery if recovery else 'Ongoing'}",
        yaxis=dict(title="Drawdown (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        showlegend=False,
        height=300,
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


def plot_active_returns_area(
    active_returns: pd.Series,
) -> go.Figure:
    """
    Plot daily active returns as area chart.
    
    Args:
        active_returns: Series of active returns (portfolio - benchmark)
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if active_returns.empty:
        return fig
    
    # Separate positive and negative
    positive = active_returns.copy()
    positive[positive < 0] = 0
    negative = active_returns.copy()
    negative[negative > 0] = 0
    
    # Convert hex colors to rgba for transparency
    # #74F174 -> rgba(116, 241, 116, 0.3)
    # #EF5350 -> rgba(239, 83, 80, 0.3)
    success_rgba = "rgba(116, 241, 116, 0.3)"
    danger_rgba = "rgba(239, 83, 80, 0.3)"
    success_line_rgba = "rgba(116, 241, 116, 0.5)"
    danger_line_rgba = "rgba(239, 83, 80, 0.5)"
    
    # Positive area (green) - #74F174
    fig.add_trace(
        go.Scatter(
            x=positive.index,
            y=positive.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor=success_rgba,
            line=dict(color=success_line_rgba, width=1),
            name="Positive Alpha",
            showlegend=False,
        )
    )
    
    # Negative area (red) - #EF5350
    fig.add_trace(
        go.Scatter(
            x=negative.index,
            y=negative.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor=danger_rgba,
            line=dict(color=danger_line_rgba, width=1),
            name="Negative Alpha",
            showlegend=False,
        )
    )
    
    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        line_width=1,
    )
    
    layout = get_chart_layout(
        title="Daily Active Returns (Portfolio - Benchmark)",
        yaxis=dict(title="Active Return (%)", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_period_returns_bar(
    data: pd.DataFrame,
) -> go.Figure:
    """
    Plot period returns comparison as side-by-side bar chart.
    
    Args:
        data: DataFrame with Period, Portfolio, Benchmark columns
        (values should be in decimal format, e.g., 0.15 = 15%)
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if data.empty:
        return fig
    
    periods = data["Period"].values
    # Convert to percentage (values are in decimal format)
    portfolio = data["Portfolio"].fillna(0).values * 100
    benchmark = data["Benchmark"].fillna(0).values * 100
    
    # Portfolio bars
    fig.add_trace(
        go.Bar(
            x=periods,
            y=portfolio,
            name="Portfolio",
            marker_color=COLORS["primary"],
            text=[f"{v:.2f}%" for v in portfolio],
            textposition="outside",
        )
    )
    
    # Benchmark bars
    fig.add_trace(
        go.Bar(
            x=periods,
            y=benchmark,
            name="Benchmark",
            marker_color=COLORS["secondary"],
            text=[f"{v:.2f}%" for v in benchmark],
            textposition="outside",
        )
    )
    
    layout = get_chart_layout(
        title="Return by Periods",
        yaxis=dict(title="Return (%)", tickformat=",.1f"),
        xaxis=dict(title="Period"),
        hovermode="x unified",
        barmode="group",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_return_quantiles_box(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Plot side-by-side box plots for return quantiles.
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if portfolio_returns.empty:
        return fig
    
    # Portfolio box plot (blue)
    fig.add_trace(
        go.Box(
            y=portfolio_returns.values * 100,
            name="Portfolio",
            marker_color=COLORS["primary"],  # Purple
            boxmean="sd",
        )
    )
    
    # Benchmark box plot (orange)
    if benchmark_returns is not None and not benchmark_returns.empty:
        fig.add_trace(
            go.Box(
                y=benchmark_returns.values * 100,
                name="Benchmark",
                marker_color=COLORS["secondary"],  # Blue (benchmark)
                boxmean="sd",
            )
        )
    
    layout = get_chart_layout(
        title="Return Quantiles - Portfolio vs Benchmark",
        yaxis=dict(title="Return (%)", tickformat=",.2f"),
        xaxis=dict(title=""),
        hovermode="y",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_seasonal_bar(
    data: pd.DataFrame,
    title: str,
) -> go.Figure:
    """
    Plot seasonal analysis bar chart (day of week, month, or quarter).
    
    Args:
        data: DataFrame with Portfolio and optionally Benchmark columns
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if data.empty:
        return fig
    
    categories = data.index.values
    portfolio = data["Portfolio"].values
    
    # Portfolio bars
    fig.add_trace(
        go.Bar(
            x=categories,
            y=portfolio,
            name="Portfolio",
            marker_color=COLORS["primary"],
            text=[f"{v:.2f}%" for v in portfolio],
            textposition="outside",
        )
    )
    
    # Benchmark bars if available
    if "Benchmark" in data.columns:
        benchmark = data["Benchmark"].values
        fig.add_trace(
            go.Bar(
                x=categories,
                y=benchmark,
                name="Benchmark",
                marker_color=COLORS["secondary"],
                text=[f"{v:.2f}%" for v in benchmark],
                textposition="outside",
            )
        )
    
    layout = get_chart_layout(
        title=title,
        yaxis=dict(title="Avg Return (%)", tickformat=",.2f"),
        xaxis=dict(title=""),
        hovermode="x unified",
        barmode="group" if "Benchmark" in data.columns else "group",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_outlier_scatter(
    portfolio_returns: pd.Series,
    outlier_data: Dict[str, any],
) -> go.Figure:
    """
    Plot outlier returns scatter plot with reference bands.
    
    Args:
        portfolio_returns: Portfolio returns
        outlier_data: Dictionary with outliers DataFrame, z_scores, mean, std
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if portfolio_returns.empty:
        return fig
    
    mean = outlier_data.get("mean", 0)
    std = outlier_data.get("std", 0)
    
    if std == 0:
        return fig
    
    # Normal returns (not outliers)
    z_scores = outlier_data.get("z_scores", pd.Series())
    if isinstance(z_scores, pd.Series) and not z_scores.empty:
        normal_mask = abs(z_scores) <= 2.0
        normal_returns = portfolio_returns[normal_mask]
        
        if not normal_returns.empty:
            fig.add_trace(
                go.Scatter(
                    x=normal_returns.index,
                    y=normal_returns.values * 100,
                    mode="markers",
                    marker=dict(
                        color="rgba(191, 159, 251, 0.5)",  # Slightly brighter
                        size=4,
                    ),
                    name="Normal Returns",
                )
            )
    
    # Outliers
    outliers_df = outlier_data.get("outliers", pd.DataFrame())
    if not outliers_df.empty:
        positive_outliers = outliers_df[outliers_df["Return"] > 0]
        negative_outliers = outliers_df[outliers_df["Return"] < 0]
        
        if not positive_outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=positive_outliers["Date"],
                    y=positive_outliers["Return"],
                    mode="markers",
                    marker=dict(
                        color="green",
                        size=8,
                        symbol="star",
                    ),
                    name="Positive Outliers",
                )
            )
        
        if not negative_outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=negative_outliers["Date"],
                    y=negative_outliers["Return"],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=8,
                        symbol="star",
                    ),
                    name="Negative Outliers",
                )
            )
    
    # Reference bands (±1σ, ±2σ, ±3σ)
    mean_pct = mean * 100
    std_pct = std * 100
    
    for i, sigma in enumerate([1, 2, 3]):
        color = ["rgba(255,255,255,0.3)", "rgba(255,255,255,0.5)", "rgba(255,255,255,0.7)"][i]
        upper = mean_pct + sigma * std_pct
        lower = mean_pct - sigma * std_pct
        
        # Upper band
        fig.add_hline(
            y=upper,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=f"+{sigma}σ",
        )
        
        # Lower band
        fig.add_hline(
            y=lower,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=f"-{sigma}σ",
        )
    
    # Mean line
    fig.add_hline(
        y=mean_pct,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="Mean",
    )
    
    layout = get_chart_layout(
        title="Outlier Returns Analysis",
        yaxis=dict(title="Return (%)", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="closest",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_win_rate(
    rolling_data: pd.Series,
    benchmark_rolling: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Plot rolling 12-month win rate chart.
    
    Args:
        rolling_data: Series of rolling win rate percentages
        benchmark_rolling: Optional benchmark rolling win rate
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if rolling_data.empty:
        return fig
    
    # Portfolio line
    fig.add_trace(
        go.Scatter(
            x=rolling_data.index,
            y=rolling_data.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=2),
        )
    )
    
    # Benchmark line if available
    if benchmark_rolling is not None and not benchmark_rolling.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_rolling.index,
                y=benchmark_rolling.values,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2),
            )
        )
    
    # 50% reference line
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="50%",
    )
    
    layout = get_chart_layout(
        title="12-Month Rolling Win Rate",
        yaxis=dict(title="Win Rate (%)", tickformat=",.1f", range=[0, 100]),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_capture_ratio(
    data: Dict[str, float],
) -> go.Figure:
    """
    Plot capture ratio horizontal bar chart.
    
    Args:
        data: Dictionary with 'up_capture', 'down_capture', 'capture_ratio'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data:
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    
    up_capture = data.get("up_capture", 0.0) * 100  # Convert to percentage
    down_capture = data.get("down_capture", 0.0) * 100
    
    # Add horizontal bars
    fig.add_trace(
        go.Bar(
            y=["Up Capture"],
            x=[up_capture],
            orientation="h",
            marker=dict(color=COLORS["success"]),
            name="Up Capture",
            text=[f"{up_capture:.1f}%"],
            textposition="outside",
        )
    )
    
    fig.add_trace(
        go.Bar(
            y=["Down Capture"],
            x=[down_capture],
            orientation="h",
            marker=dict(color=COLORS["danger"]),
            name="Down Capture",
            text=[f"{down_capture:.1f}%"],
            textposition="outside",
        )
    )
    
    # Add reference line at 100%
    fig.add_vline(
        x=100,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="100%",
        annotation_position="top",
    )
    
    layout = get_chart_layout(
        title="Capture Ratios - Asymmetry Analysis",
        xaxis=dict(title="Capture (%)", tickformat=",.0f"),
        yaxis=dict(title=""),
        showlegend=False,
        height=250,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_risk_return_scatter(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot risk/return scatter chart with CML.
    
    Args:
        data: Dictionary with portfolio, benchmark, and CML data
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "portfolio" not in data:
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    
    # Portfolio point
    portfolio = data.get("portfolio", {})
    fig.add_trace(
        go.Scatter(
            x=[portfolio.get("volatility", 0.0) * 100],
            y=[portfolio.get("return", 0.0) * 100],
            mode="markers+text",
            name="Portfolio",
            marker=dict(size=15, color=COLORS["primary"]),
            text=["Portfolio"],
            textposition="top center",
            textfont=dict(size=12, color="white"),
        )
    )
    
    # Benchmark point if available
    if "benchmark" in data:
        benchmark = data["benchmark"]
        fig.add_trace(
            go.Scatter(
                x=[benchmark.get("volatility", 0.0) * 100],
                y=[benchmark.get("return", 0.0) * 100],
                mode="markers+text",
                name="Benchmark",
                marker=dict(size=15, color=COLORS["secondary"]),
                text=["Benchmark"],
                textposition="top center",
                textfont=dict(size=12, color="white"),
            )
        )
    
    # Capital Market Line (CML)
    if "cml" in data:
        cml = data["cml"]
        vol_points = [v * 100 for v in cml.get("volatility", [])]
        ret_points = [r * 100 for r in cml.get("return", [])]
        fig.add_trace(
            go.Scatter(
                x=vol_points,
                y=ret_points,
                mode="lines",
                name="Capital Market Line",
                line=dict(color=COLORS["success"], width=2, dash="dash"),
            )
        )
    
    # Add risk-free rate point (0 volatility)
    rf_rate = data.get("risk_free_rate", 0.0) * 100
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[rf_rate],
            mode="markers",
            name="Risk-Free Rate",
            marker=dict(size=10, color="gray", symbol="diamond"),
        )
    )
    
    # Quadrant labels can be added here if needed in the future
    # For now, chart is clean without annotations
    layout = get_chart_layout(
        title="Risk/Return Scatter",
        xaxis=dict(title="Volatility (Annual, %)", tickformat=",.1f", rangemode="tozero"),
        yaxis=dict(title="Annualized Return (%)", tickformat=",.1f"),
        hovermode="closest",
        height=500,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_var_distribution(
    returns: pd.Series,
    var_value: float,
    cvar_value: float,
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Plot return distribution histogram with VaR and CVaR lines.
    Uses same structure as plot_return_distribution but without extra VaR percentile lines.

    Args:
        returns: Series of returns
        var_value: VaR value (negative)
        cvar_value: CVaR value (negative)
        confidence_level: Confidence level (0.95 for 95%)

    Returns:
        Plotly Figure
    """
    # Use the same data preparation as plot_return_distribution
    from core.analytics_engine.chart_data import get_return_distribution_data
    
    dist_data = get_return_distribution_data(returns, bins=50)
    
    # Create clean figure without extra VaR lines
    fig = go.Figure()
    
    counts = dist_data.get("counts", np.array([]))
    edges = dist_data.get("edges", np.array([]))
    mean = dist_data.get("mean", 0.0)
    std = dist_data.get("std", 0.0)
    
    if len(counts) > 0 and len(edges) > 1:
        bin_centers = (edges[:-1] + edges[1:]) / 2 * 100
        
        # Add histogram bars (purple)
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                name="Return Distribution",
                marker_color=COLORS["primary"],  # Purple
                opacity=0.7,
            )
        )
        
        # Mean line (blue dashed)
        mean_pct = mean * 100
        fig.add_vline(
            x=mean_pct,
            line_dash="dash",
            line_color=COLORS["secondary"],  # Blue
            annotation_text=f"Mean: {mean_pct:.2f}%",
            annotation_position="top",
        )
        
        # Normal distribution overlay (orange dashed)
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
                    line=dict(color=COLORS["warning"], width=2, dash="dash"),  # Orange
                )
            )
    
    # Add VaR line (red dotted, thicker)
    var_pct = var_value * 100
    fig.add_vline(
        x=var_pct,
        line=dict(color=COLORS["danger"], width=3, dash="dot"),
        annotation_text=f"VaR {int(confidence_level*100)}%",
        annotation_position="bottom",
    )
    fig.add_annotation(
        x=var_pct,
        y=1.0,
        xref="x",
        yref="paper",
        text=f"{var_pct:.2f}%",
        showarrow=False,
        font=dict(size=10, color=COLORS["danger"]),
        bgcolor="rgba(0,0,0,0.7)",
    )

    # Add CVaR line (orange dotted, thicker)
    cvar_pct = cvar_value * 100
    fig.add_vline(
        x=cvar_pct,
        line=dict(color=COLORS["warning"], width=3, dash="dot"),
        annotation_text=f"CVaR {int(confidence_level*100)}%",
        annotation_position="bottom",
    )
    fig.add_annotation(
        x=cvar_pct,
        y=1.0,
        xref="x",
        yref="paper",
        text=f"{cvar_pct:.2f}%",
        showarrow=False,
        font=dict(size=10, color=COLORS["warning"]),
        bgcolor="rgba(0,0,0,0.7)",
    )

    # Add shaded area for returns beyond VaR
    returns_pct = returns * 100
    tail_returns = returns_pct[returns_pct < var_pct]
    if not tail_returns.empty:
        fig.add_vrect(
            x0=returns_pct.min(),
            x1=var_pct,
            fillcolor="rgba(220, 53, 69, 0.2)",
            layer="below",
            line_width=0,
            annotation_text=f"{int((1-confidence_level)*100)}% of days",
            annotation_position="top left",
        )

    layout = get_chart_layout(
        title=f"VaR {int(confidence_level*100)}% on Return Distribution",
        yaxis=dict(title="Frequency"),
        xaxis=dict(title="Daily Return (%)", tickformat=",.1f"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_rolling_volatility(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling volatility chart with statistics table.
    
    Args:
        data: Dictionary with portfolio/benchmark series and statistics
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "portfolio" not in data:
        return fig
    
    portfolio_vol = data.get("portfolio", pd.Series())
    benchmark_vol = data.get("benchmark", pd.Series())
    window = data.get("window", 63)
    
    # Portfolio line (blue)
    if not portfolio_vol.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_vol.index,
                y=portfolio_vol.values * 100,  # Convert to percentage
                mode="lines",
                name="Portfolio",
                line=dict(color=COLORS["primary"], width=2),
            )
        )
    
    # Benchmark line (blue solid)
    if not benchmark_vol.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_vol.index,
                y=benchmark_vol.values * 100,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2),  # Blue - solid line
            )
        )
    
    # Reference line at 20% volatility (white dashed)
    fig.add_hline(
        y=20.0,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="20%",
    )
    
    layout = get_chart_layout(
        title=f"Rolling Volatility ({window} days)",
        yaxis=dict(title="Volatility (annualized %)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_var(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling VaR chart.

    Args:
        data: Dictionary with portfolio/benchmark series and statistics

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if not data or "portfolio" not in data:
        return fig

    portfolio_var = data.get("portfolio", pd.Series())
    benchmark_var = data.get("benchmark", pd.Series())
    window = data.get("window", 63)
    confidence_level = data.get("confidence_level", 0.95)

    # Portfolio line
    if not portfolio_var.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_var.index,
                y=portfolio_var.values * 100,  # Convert to percentage
                mode="lines",
                name="Portfolio VaR",
                line=dict(color=COLORS["danger"], width=2),
            )
        )

    # Benchmark line (dashed)
    if not benchmark_var.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_var.index,
                y=benchmark_var.values * 100,
                mode="lines",
                name="Benchmark VaR",
                line=dict(
                    color=COLORS["secondary"], width=2, dash="dash"
                ),
            )
        )

    layout = get_chart_layout(
        title=(
            f"Rolling VaR ({window} days, "
            f"{int(confidence_level*100)}% confidence)"
        ),
        yaxis=dict(title="VaR (%)", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )

    fig.update_layout(**layout)
    return fig


def plot_rolling_sortino(
    data: Dict[str, pd.Series],
    window: int = 63,
) -> go.Figure:
    """
    Plot rolling Sortino ratio chart.
    
    Args:
        data: Dictionary with 'portfolio' and optionally 'benchmark' Series
        window: Rolling window size in days
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Portfolio line (blue solid)
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
    
    # Benchmark line (blue solid)
    if "benchmark" in data and not data["benchmark"].empty:
        fig.add_trace(
            go.Scatter(
                x=data["benchmark"].index,
                y=data["benchmark"].values,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2),  # Blue - solid line
            )
        )
    
    # Zero Sortino reference line (white dashed)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="Sortino = 0",
    )
    
    layout = get_chart_layout(
        title=f"Rolling Sortino Ratio ({window} days)",
        yaxis=dict(title="Sortino Ratio", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_beta(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling beta chart with shaded zones.
    
    Args:
        data: Dictionary with 'beta' series and 'window'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "beta" not in data:
        return fig
    
    beta_series = data.get("beta", pd.Series())
    window = data.get("window", 63)
    
    if beta_series.empty:
        return fig
    
    # Add shaded zones
    # High beta zone (β > 1.2) - light red
    fig.add_hrect(
        y0=1.2,
        y1=beta_series.max() * 1.1 if beta_series.max() > 1.2 else 1.5,
        fillcolor="rgba(244, 67, 54, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="High Beta",
        annotation_position="top right",
    )
    
    # Normal zone (0.8 < β < 1.2) - light gray
    fig.add_hrect(
        y0=0.8,
        y1=1.2,
        fillcolor="rgba(158, 158, 158, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Normal",
        annotation_position="top right",
    )
    
    # Low beta zone (β < 0.8) - light green
    fig.add_hrect(
        y0=beta_series.min() * 0.9 if beta_series.min() < 0.8 else 0,
        y1=0.8,
        fillcolor="rgba(76, 175, 80, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Low Beta",
        annotation_position="top right",
    )
    
    # Beta line (purple)
    fig.add_trace(
        go.Scatter(
            x=beta_series.index,
            y=beta_series.values,
            mode="lines",
            name="Beta",
            line=dict(color=COLORS["primary"], width=2),  # Purple
        )
    )
    
    # Reference lines
    # Beta = 1.0 (white dashed)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="Beta = 1.0",
    )
    
    # Beta = 0.0 (gray dashed)
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )
    
    layout = get_chart_layout(
        title=f"Rolling Beta ({window} days)",
        yaxis=dict(title="Beta", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_alpha(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling alpha chart with shaded areas.
    
    Args:
        data: Dictionary with 'alpha' series and 'window'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "alpha" not in data:
        return fig
    
    alpha_series = data.get("alpha", pd.Series())
    window = data.get("window", 63)
    
    if alpha_series.empty:
        return fig
    
    # Separate positive and negative values for different coloring
    positive = alpha_series.copy()
    positive[positive < 0] = 0
    negative = alpha_series.copy()
    negative[negative > 0] = 0
    
    # Positive alpha area (light green)
    fig.add_trace(
        go.Scatter(
            x=positive.index,
            y=positive.values * 100,  # Convert to percentage
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.3)",
            line=dict(color=COLORS["success"], width=2),
            name="Positive Alpha",
            showlegend=True,
        )
    )
    
    # Negative alpha area (light red)
    fig.add_trace(
        go.Scatter(
            x=negative.index,
            y=negative.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(244, 67, 54, 0.3)",
            line=dict(color=COLORS["danger"], width=2),
            name="Negative Alpha",
            showlegend=True,
        )
    )
    
    # Alpha = 0 reference line (black dashed)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        line_width=2,
        annotation_text="Alpha = 0",
    )
    
    layout = get_chart_layout(
        title=f"Rolling Alpha ({window} days)",
        yaxis=dict(title="Alpha (%)", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_active_return(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling active return area chart.
    
    Args:
        data: Dictionary with 'active_return' series, 'window', and 'stats'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "active_return" not in data:
        return fig
    
    active_return = data.get("active_return", pd.Series())
    window = data.get("window", 63)
    stats = data.get("stats", {})
    
    if active_return.empty:
        return fig
    
    # Active return as single purple line (no positive/negative split)
    fig.add_trace(
        go.Scatter(
            x=active_return.index,
            y=active_return.values * 100,  # Convert to percentage
            mode="lines",
            name="Active Return",
            line=dict(color=COLORS["primary"], width=2),  # Purple - portfolio
        )
    )
    
    # Zero line (black dashed)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        line_width=2,
    )
    
    layout = get_chart_layout(
        title=f"Rolling Active Return ({window} days) - Portfolio - Benchmark",
        yaxis=dict(title="Active Return (%)", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_bull_bear_returns_comparison(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot bull/bear market returns comparison as side-by-side bars.
    
    Args:
        data: Dictionary with 'bull' and 'bear' market data
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "bull" not in data or "bear" not in data:
        return fig
    
    bull = data.get("bull", {})
    bear = data.get("bear", {})
    
    # Data for plotting
    categories = ["Bullish Market", "Bearish Market"]
    portfolio_returns = [
        bull.get("portfolio_return", 0),
        bear.get("portfolio_return", 0)
    ]
    benchmark_returns = [
        bull.get("benchmark_return", 0),
        bear.get("benchmark_return", 0)
    ]
    
    # Portfolio bars
    fig.add_trace(
        go.Bar(
            x=categories,
            y=portfolio_returns,
            name="Portfolio",
            marker_color=COLORS["primary"],
            text=[f"{v:.2f}%" for v in portfolio_returns],
            textposition="outside",
        )
    )
    
    # Benchmark bars
    fig.add_trace(
        go.Bar(
            x=categories,
            y=benchmark_returns,
            name="Benchmark",
            marker_color=COLORS["secondary"],
            text=[f"{v:.2f}%" for v in benchmark_returns],
            textposition="outside",
        )
    )
    
    layout = get_chart_layout(
        title="Average Daily Returns in Bull vs Bear Markets",
        yaxis=dict(title="Avg Daily Return (%)", tickformat=",.2f"),
        xaxis=dict(title=""),
        hovermode="x unified",
        barmode="group",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_bull_bear_rolling_beta(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling beta in different market periods.
    
    Args:
        data: Dictionary with rolling beta for bull/bear markets
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "rolling_beta" not in data:
        return fig
    
    rolling_data = data.get("rolling_beta", {})
    bull_beta = rolling_data.get("bull", pd.Series())
    bear_beta = rolling_data.get("bear", pd.Series())
    window = rolling_data.get("window", 126)
    
    # Bull market beta (green line)
    if not bull_beta.empty:
        fig.add_trace(
            go.Scatter(
                x=bull_beta.index,
                y=bull_beta.values,
                mode="lines",
                name="Beta in Bullish Market",
                line=dict(color=COLORS["success"], width=2),
            )
        )
    
    # Bear market beta (red line)
    if not bear_beta.empty:
        fig.add_trace(
            go.Scatter(
                x=bear_beta.index,
                y=bear_beta.values,
                mode="lines",
                name="Beta in Bearish Market",
                line=dict(color=COLORS["danger"], width=2),
            )
        )
    
    # Beta = 1.0 reference line (white dashed)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text="Beta = 1.0",
    )
    
    layout = get_chart_layout(
        title=f"Rolling Beta in Different Market Periods ({window} days)",
        yaxis=dict(title="Beta", tickformat=",.2f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_impact_on_return(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot impact on total return bar chart.
    
    Args:
        data: Dictionary with 'tickers', 'contributions', 'returns', 'weights'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "tickers" not in data:
        return fig
    
    tickers = data.get("tickers", [])
    contributions = data.get("contributions", [])
    
    # Color scale based on contribution value
    colors = []
    for contrib in contributions:
        if contrib > 4:
            colors.append("#1B5E20")  # Dark Green
        elif contrib > 2:
            colors.append("#4CAF50")  # Green
        elif contrib > 1:
            colors.append("#8BC34A")  # Light Green
        elif contrib > 0:
            colors.append("#FFC107")  # Yellow
        elif contrib > -1:
            colors.append("#FF9800")  # Orange
        else:
            colors.append("#F44336")  # Red
    
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=contributions,
            marker=dict(color=colors),
            text=[f"{c:.2f}%" for c in contributions],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>",
        )
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=COLORS["text"],
        line_width=1,
    )
    
    layout = get_chart_layout(
        title="Impact of Assets on Total Return",
        yaxis=dict(title="Weighted Return Contribution (%)", tickformat=",.2f"),
        xaxis=dict(title="Assets"),
        showlegend=False,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_impact_on_risk(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot impact on portfolio risk bar chart.
    
    Args:
        data: Dictionary with 'tickers', 'risk_contributions'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "tickers" not in data:
        return fig
    
    tickers = data.get("tickers", [])
    risk_contributions = data.get("risk_contributions", [])
    
    # Yellow to Red gradient based on contribution
    colors = []
    max_contrib = max(risk_contributions) if risk_contributions else 1
    for contrib in risk_contributions:
        # Normalize to 0-1 range
        normalized = contrib / max_contrib if max_contrib > 0 else 0
        # Gradient from yellow to red
        if normalized > 0.8:
            colors.append("#D32F2F")  # Dark Red
        elif normalized > 0.6:
            colors.append("#F44336")  # Red
        elif normalized > 0.4:
            colors.append("#FF5722")  # Red-Orange
        elif normalized > 0.2:
            colors.append("#FF9800")  # Orange
        else:
            colors.append("#FFC107")  # Yellow
    
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=risk_contributions,
            marker=dict(color=colors),
            text=[f"{c:.1f}%" for c in risk_contributions],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>",
        )
    )
    
    layout = get_chart_layout(
        title="Impact of Assets on Overall Portfolio Risk",
        yaxis=dict(title="Risk Contribution (%)", tickformat=",.1f"),
        xaxis=dict(title="Assets"),
        showlegend=False,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_risk_vs_weight_comparison(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot comparison of risk impact, return impact vs asset weight (grouped bar chart).
    
    Args:
        data: Dictionary with 'tickers', 'risk_impact', 'return_impact', 'weights'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "tickers" not in data:
        return fig
    
    tickers = data.get("tickers", [])
    risk_impact = data.get("risk_impact", [])
    return_impact = data.get("return_impact", [])
    weights = data.get("weights", [])
    
    # Risk Impact bars (red)
    fig.add_trace(
        go.Bar(
            name="Impact on Risk",
            x=tickers,
            y=risk_impact,
            marker=dict(color=COLORS["danger"]),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Risk Impact: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    
    # Return Impact bars (green)
    if return_impact:
        fig.add_trace(
            go.Bar(
                name="Impact on Return",
                x=tickers,
                y=return_impact,
                marker=dict(color=COLORS["success"]),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Return Impact: %{y:.2f}%<extra></extra>"
                ),
            )
        )
    
    # Weight bars (purple)
    fig.add_trace(
        go.Bar(
            name="Weight in Portfolio",
            x=tickers,
            y=weights,
            marker=dict(color=COLORS["primary"]),  # Purple
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Weight: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    
    layout = get_chart_layout(
        title="Comparison of Risk & Return Impact and Asset Weighting",
        yaxis=dict(title="Percentage (%)", tickformat=",.1f"),
        xaxis=dict(title="Assets"),
        barmode="group",
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
) -> go.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if correlation_matrix.empty:
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    
    # Create heatmap with custom color scale (green to red through white)
    colorscale = [
        [0.0, COLORS["success"]],  # Green (-1.0)
        [0.5, "#FFFFFF"],           # White (0.0)
        [1.0, COLORS["danger"]],    # Red (+1.0)
    ]
    
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=colorscale,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10, "color": "black"},
            colorbar=dict(title="Correlation"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    
    layout = get_chart_layout(
        title="Correlation Matrix - All Assets + Benchmark",
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        height=600,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_correlation_with_benchmark(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot bar chart of asset correlations with benchmark.
    
    Args:
        data: Dictionary with tickers, correlations, betas
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "tickers" not in data:
        return fig
    
    tickers = data.get("tickers", [])
    correlations = data.get("correlations", [])
    
    if not tickers or not correlations:
        return fig
    
    # Color gradient: Red (high) to Blue (low)
    colors = []
    for corr in correlations:
        if corr > 0.7:
            colors.append("#DC2626")  # Dark Red
        elif corr > 0.5:
            colors.append("#F87171")  # Light Red
        elif corr > 0.3:
            colors.append("#FCD34D")  # Yellow
        elif corr > 0.1:
            colors.append("#60A5FA")  # Light Blue
        else:
            colors.append("#1E3A8A")  # Dark Blue
    
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=correlations,
            marker=dict(color=colors),
            text=[f"{c:.2f}" for c in correlations],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Correlation: %{y:.2f}<extra></extra>",
        )
    )
    
    # Reference line at 0.7 (high correlation threshold)
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="white",  # White
        line_width=2,
        annotation_text="High Correlation (0.7)",
        annotation_position="right",
    )
    
    layout = get_chart_layout(
        title="Asset Correlation with SPY",
        yaxis=dict(title="Correlation Coefficient", tickformat=",.2f", range=[-1, 1]),
        xaxis=dict(title="Assets"),
        showlegend=False,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_clustered_correlation_matrix(
    clustered_matrix: pd.DataFrame,
) -> go.Figure:
    """
    Plot clustered correlation matrix heatmap.
    
    Args:
        clustered_matrix: Reordered correlation matrix DataFrame
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if clustered_matrix.empty:
        return fig
    
    # Same color scale as regular correlation matrix
    colorscale = [
        [0.0, "#1E3A8A"],      # Dark Blue (-1.0)
        [0.25, "#3B82F6"],     # Light Blue (-0.5)
        [0.5, "#FFFFFF"],      # White (0.0)
        [0.75, "#F87171"],     # Light Red (+0.5)
        [1.0, "#DC2626"],      # Dark Red (+1.0)
    ]
    
    fig.add_trace(
        go.Heatmap(
            z=clustered_matrix.values,
            x=clustered_matrix.columns,
            y=clustered_matrix.index,
            colorscale=colorscale,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=clustered_matrix.round(2).values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10, "color": "black"},
            colorbar=dict(title="Correlation"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    
    layout = get_chart_layout(
        title="Clustered Correlation Matrix",
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        height=600,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list,
    n_clusters: int = 3,
) -> go.Figure:
    """
    Plot horizontal hierarchical clustering dendrogram.
    
    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy
        labels: List of asset tickers
        n_clusters: Number of clusters to highlight
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    try:
        from scipy.cluster.hierarchy import dendrogram, fcluster
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Calculate cluster assignments
        cluster_assignments = fcluster(
            linkage_matrix, n_clusters, criterion="maxclust"
        )
        
        # Create vertical dendrogram using matplotlib (more standard)
        # Larger figure size for better visibility
        num_assets = len(labels)
        fig_width = 14  # Increased width
        fig_height = max(10, num_assets * 0.8)  # More height per asset
        plt.figure(figsize=(fig_width, fig_height))
        
        # Set dark theme colors
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_facecolor("#1E1E1E")
        fig_pyplot = plt.gcf()
        fig_pyplot.patch.set_facecolor("#1E1E1E")
        
        # Create dendrogram (vertical orientation - standard)
        dendro_data = dendrogram(
            linkage_matrix,
            labels=labels,
            orientation="left",  # Labels on left, tree grows right
            leaf_font_size=12,
            leaf_rotation=0,
            ax=ax,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
        )
        
        # Calculate cut line distance for optimal clusters
        if len(linkage_matrix) >= n_clusters:
            # Get distance at which we get n_clusters
            cut_dist = linkage_matrix[-n_clusters+1, 2]
        else:
            cut_dist = linkage_matrix[-1, 2] * 0.7
        
        # Add cut line for optimal clusters (vertical line)
        plt.axvline(x=cut_dist, color="yellow", linestyle="--", linewidth=2.5, 
                   label=f"Cut line ({n_clusters} clusters)", alpha=0.8)
        
        plt.title("Asset Clustering Dendrogram", color="white", fontsize=14, pad=20, fontweight="bold")
        plt.xlabel("Distance", color="white", fontsize=12, fontweight="bold")
        plt.ylabel("Assets", color="white", fontsize=12, fontweight="bold")
        plt.xticks(color="white", fontsize=10)
        plt.yticks(color="white", fontsize=10)
        plt.legend(loc="upper right", facecolor="#1E1E1E", edgecolor="yellow", 
                  labelcolor="white", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.2, color="gray", linestyle=":")
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Convert to image with higher resolution
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, 
                   facecolor="#1E1E1E", edgecolor="none")
        plt.close()
        buf.seek(0)
        
        # Add image to plotly
        import base64
        img_str = base64.b64encode(buf.read()).decode()
        
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_str}",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
            )
        )
        
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        
    except Exception as e:
        logger.warning(f"Error creating dendrogram: {e}")
        fig.add_annotation(
            text="Dendrogram visualization not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    
    # Dynamic height based on number of assets
    num_assets = len(labels) if labels else 8
    chart_height = max(600, num_assets * 80)
    
    layout = get_chart_layout(
        title="Asset Clustering Dendrogram",
        height=chart_height,
    )
    
    fig.update_layout(**layout)
    return fig


def plot_asset_price_dynamics(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot normalized asset price dynamics (multi-line chart).
    
    Args:
        data: Dictionary with price_series (ticker -> Series)
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "price_series" not in data:
        return fig
    
    price_series = data.get("price_series", {})
    
    if not price_series:
        return fig
    
    # Color palette for assets (portfolio first, then others)
    asset_colors = [
        COLORS["primary"],  # Purple - portfolio
        COLORS["secondary"],  # Blue
        COLORS["success"],  # Green
        COLORS["warning"],  # Orange
        COLORS["additional"],  # Yellow
        COLORS["danger"],  # Red
    ]
    
    # Plot each asset
    for idx, (ticker, series) in enumerate(price_series.items()):
        if ticker == "SPY":
            # Benchmark as dashed orange line
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=f"{ticker} (Benchmark)",
                    line=dict(
                        color=COLORS["secondary"],
                        width=2,
                        dash="dash",
                    ),
                )
            )
        else:
            # Assets as solid lines
            color = asset_colors[idx % len(asset_colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=ticker,
                    line=dict(color=color, width=2),
                )
            )
    
    # Add final returns to legend
    for ticker, series in price_series.items():
        if not series.empty:
            final_return = series.iloc[-1]
            # Update hover template to show final return
            fig.data[list(price_series.keys()).index(ticker)].hovertemplate = (
                f"<b>{ticker}</b><br>"
                f"Date: %{{x}}<br>"
                f"Return: %{{y:.2f}}%<br>"
                f"Final: {final_return:.2f}%<extra></extra>"
            )
    
    layout = get_chart_layout(
        title="Asset Price Change (% from Start Date)",
        yaxis=dict(title="% Change from Start", tickformat=",.1f"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_rolling_correlation_with_benchmark(
    data: Dict[str, any],
) -> go.Figure:
    """
    Plot rolling correlation with benchmark for multiple assets.
    
    Args:
        data: Dictionary with rolling_correlations, portfolio_avg_correlation, window
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not data or "rolling_correlations" not in data:
        return fig
    
    rolling_correlations = data.get("rolling_correlations", {})
    portfolio_avg_corr = data.get("portfolio_avg_correlation")
    window = data.get("window", 60)
    
    # Color palette
    asset_colors = [
        COLORS["primary"],  # Purple
        COLORS["secondary"],  # Blue
        COLORS["success"],  # Green
        COLORS["warning"],  # Orange
        COLORS["additional"],  # Yellow
        COLORS["danger"],  # Red
    ]
    
    # Plot each asset's rolling correlation
    for idx, (ticker, series) in enumerate(rolling_correlations.items()):
        color = asset_colors[idx % len(asset_colors)]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=ticker,
                line=dict(color=color, width=1.5),
            )
        )
    
    # Plot portfolio average correlation (thick purple line)
    if portfolio_avg_corr is not None and not portfolio_avg_corr.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_avg_corr.index,
                y=portfolio_avg_corr.values,
                mode="lines",
                name="Portfolio Avg Correlation",
                line=dict(color=COLORS["primary"], width=3),  # Purple - portfolio
            )
        )
    
    # Reference lines
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color=COLORS["danger"],  # Red
        line_width=1,
        annotation_text="High (0.7)",
    )
    
    fig.add_hline(
        y=0.3,
        line_dash="dash",
        line_color=COLORS["success"],  # Green (keep as is)
        line_width=1,
        annotation_text="Low (0.3)",
    )
    
    layout = get_chart_layout(
        title=f"Rolling Correlation with SPY (Window: {window} days)",
        yaxis=dict(title="Correlation Coefficient", tickformat=",.2f", range=[-1, 1]),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    
    fig.update_layout(**layout)
    return fig


def plot_detailed_asset_price_volume(
    prices: pd.Series,
    returns: pd.Series,
    ma50: Optional[pd.Series] = None,
    ma200: Optional[pd.Series] = None,
    ticker: str = "",
) -> go.Figure:
    """
    Plot asset price chart with moving averages and volume.
    
    Args:
        prices: Price series
        returns: Returns series (for volume coloring)
        ma50: Optional 50-day moving average
        ma200: Optional 200-day moving average
        ticker: Asset ticker
        
    Returns:
        Plotly Figure with subplots
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} - Price and Moving Averages", "Volume"),
    )
    
    # Price line (purple)
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="Price",
            line=dict(color=COLORS["primary"], width=2),  # Purple
        ),
        row=1,
        col=1,
    )
    
    # Moving averages
    if ma50 is not None and not ma50.empty:
        fig.add_trace(
            go.Scatter(
                x=ma50.index,
                y=ma50.values,
                mode="lines",
                name="MA50",
                line=dict(color=COLORS["secondary"], width=1.5, dash="dash"),  # Blue
            ),
            row=1,
            col=1,
        )
    
    if ma200 is not None and not ma200.empty:
        fig.add_trace(
            go.Scatter(
                x=ma200.index,
                y=ma200.values,
                mode="lines",
                name="MA200",
                line=dict(color=COLORS["warning"], width=1.5, dash="dash"),  # Orange
            ),
            row=1,
            col=1,
        )
    
    # Volume bars (colored by return direction)
    if not returns.empty:
        # Align returns with prices
        aligned_returns = returns.reindex(prices.index, method="ffill").fillna(0)
        
        # Create volume bars (use returns magnitude as proxy for volume)
        volume = abs(aligned_returns) * 1000  # Scale for visualization
        
        # Green for up days, red for down days (from palette)
        colors = [COLORS["success"] if r >= 0 else COLORS["danger"] for r in aligned_returns]
        
        fig.add_trace(
            go.Bar(
                x=prices.index,
                y=volume.values,
                name="Volume",
                marker_color=colors,
                opacity=0.6,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    
    # Update layout for subplots (use proper axis references)
    base_layout = get_chart_layout(
        title=f"{ticker} - Price and Volume Chart",
        height=600,
    )
    
    # Update base layout
    fig.update_layout(**base_layout)
    
    # Update axes for subplots
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def plot_asset_correlation_bar(
    correlations: Dict[str, float],
    ticker: str,
) -> go.Figure:
    """
    Plot bar chart of asset correlations with other assets.
    
    Args:
        correlations: Dictionary of ticker -> correlation
        ticker: Asset ticker being analyzed
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if not correlations:
        return fig
    
    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    tickers = [t[0] for t in sorted_corr]
    values = [t[1] for t in sorted_corr]
    
    # Color gradient
    colors = []
    for val in values:
        if val > 0.7:
            colors.append("#DC2626")  # Dark Red
        elif val > 0.5:
            colors.append("#F87171")  # Light Red
        elif val > 0.3:
            colors.append("#FCD34D")  # Yellow
        elif val > 0.1:
            colors.append("#60A5FA")  # Light Blue
        else:
            colors.append("#1E3A8A")  # Dark Blue
    
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=values,
            marker=dict(color=colors),
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Correlation: %{y:.2f}<extra></extra>",
        )
    )
    
    layout = get_chart_layout(
        title=f"Correlations with Other Assets - {ticker}",
        yaxis=dict(title="Correlation with " + ticker, tickformat=",.2f", range=[-1, 1]),
        xaxis=dict(title="Assets"),
        showlegend=False,
    )
    
    fig.update_layout(**layout)
    return fig