"""Triple chart section component for portfolio dynamics."""

import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from streamlit_app.components.period_filter import filter_series_by_period
from streamlit_app.utils.chart_config import COLORS

logger = logging.getLogger(__name__)


def create_triple_dynamics_chart(
    portfolio_returns: pd.Series,
    portfolio_values: pd.Series,  # noqa: ARG001
    benchmark_returns: Optional[pd.Series] = None,
    period: str = "All",
) -> go.Figure:
    """
    Create a triple chart showing cumulative returns, drawdown, and daily returns.

    Args:
        portfolio_returns: Portfolio returns series
        portfolio_values: Portfolio values series
        benchmark_returns: Optional benchmark returns series
        period: Time period filter ('6M', '1Y', '2Y', 'All')

    Returns:
        Plotly Figure with 3 subplots
    """
    # Filter data by period
    portfolio_returns_filtered = filter_series_by_period(
        portfolio_returns, period
    )

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_returns_filtered = filter_series_by_period(
            benchmark_returns, period
        )
    else:
        benchmark_returns_filtered = None

    # Calculate cumulative returns
    portfolio_cum_returns = (1 + portfolio_returns_filtered).cumprod() - 1

    if benchmark_returns_filtered is not None:
        benchmark_cum_returns = (1 + benchmark_returns_filtered).cumprod() - 1
    else:
        benchmark_cum_returns = None

    # Calculate drawdown
    cumulative = (1 + portfolio_returns_filtered).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    if benchmark_returns_filtered is not None:
        bench_cumulative = (1 + benchmark_returns_filtered).cumprod()
        bench_running_max = bench_cumulative.cummax()
        bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
    else:
        bench_drawdown = None

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Cumulative Returns",
            "Drawdowns",
            "Daily Returns",
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Row 1: Cumulative Returns
    fig.add_trace(
        go.Scatter(
            x=portfolio_cum_returns.index,
            y=portfolio_cum_returns.values * 100,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=2),
            legendgroup="portfolio",
        ),
        row=1,
        col=1,
    )

    if benchmark_cum_returns is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_cum_returns.index,
                y=benchmark_cum_returns.values * 100,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                legendgroup="benchmark",
            ),
            row=1,
            col=1,
        )

    # Row 2: Drawdowns
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            fill="tozeroy",
            name="Portfolio DD",
            line=dict(color=COLORS["danger"], width=1.5),
            fillcolor="rgba(250, 161, 164, 0.3)",
            legendgroup="portfolio",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if bench_drawdown is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_drawdown.index,
                y=bench_drawdown.values * 100,
                mode="lines",
                name="Benchmark DD",
                line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
                legendgroup="benchmark",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Row 3: Daily Returns
    colors = [
        COLORS["success"] if r > 0 else COLORS["danger"]
        for r in portfolio_returns_filtered.values
    ]

    fig.add_trace(
        go.Bar(
            x=portfolio_returns_filtered.index,
            y=portfolio_returns_filtered.values * 100,
            name="Daily Return",
            marker_color=colors,
            legendgroup="portfolio",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Update axes
    fig.update_yaxes(title_text="Return (%)", row=1, col=1, tickformat=",.1f")
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=",.1f")
    fig.update_yaxes(title_text="Return (%)", row=3, col=1, tickformat=",.2f")

    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )

    return fig


def render_triple_chart_section(
    portfolio_returns: pd.Series,
    portfolio_values: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    original_start_date=None,
) -> None:
    """
    Render triple chart section with period filter.

    Args:
        portfolio_returns: Portfolio returns series
        portfolio_values: Portfolio values series
        benchmark_returns: Optional benchmark returns series
        original_start_date: Original analysis start date
    """
    st.subheader("Portfolio Dynamics")

    # Period filter
    col1, col2 = st.columns([3, 1])

    with col2:
        period = st.radio(
            "Period:",
            options=["6M", "1Y", "2Y", "All"],
            index=1,  # Default to 1Y
            horizontal=True,
            key="triple_chart_period",
        )

    # Create and display chart
    try:
        fig = create_triple_dynamics_chart(
            portfolio_returns=portfolio_returns,
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            period=period,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating triple chart: {e}", exc_info=True)
        st.error(f"Error creating chart: {e}")

