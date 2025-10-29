"""Portfolio analysis page showing comprehensive analytics."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from core.analytics_engine.chart_data import (
    get_cumulative_returns_data,
    get_drawdown_data,
    get_monthly_heatmap_data,
    get_qq_plot_data,
    get_return_distribution_data,
    get_rolling_metric_data,
)
from core.exceptions import InsufficientDataError, PortfolioNotFoundError
from services.analytics_service import AnalyticsService
from services.portfolio_service import PortfolioService
from streamlit_app.components.charts import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_qq_plot,
    plot_return_distribution,
    plot_rolling_metric,
)
from streamlit_app.components.metrics_display import render_metrics_section

logger = logging.getLogger(__name__)


def render_portfolio_analysis() -> None:
    """Render the portfolio analysis page."""
    # Get portfolio ID from session state or query params
    portfolio_id = (
        st.session_state.get("selected_portfolio_id")
        or st.query_params.get("id")
    )

    if not portfolio_id:
        st.error("No portfolio selected")
        st.info("Please select a portfolio from the list")
        if st.button("Go to Portfolio List"):
            st.switch_page("pages/portfolio_list.py")
        return

    # Initialize services
    portfolio_service = PortfolioService()
    analytics_service = AnalyticsService()

    try:
        # Fetch portfolio
        portfolio = portfolio_service.get_portfolio(portfolio_id)

        # Page header
        portfolio_title = f"Analysis: {portfolio.name}"
        st.title(portfolio_title)

        st.markdown("---")

        # Analysis parameters
        st.subheader("Analysis Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Default to 1 year ago
            default_start = date.today() - timedelta(days=365)
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=date.today(),
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                max_value=date.today(),
            )

        with col3:
            help_text = (
                "Enter a benchmark ticker for comparison "
                "(e.g., SPY for S&P 500)"
            )
            benchmark_ticker = st.text_input(
                "Benchmark Ticker (Optional)",
                placeholder="SPY",
                help=help_text,
            ).strip().upper() or None

        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        # Calculate button
        if st.button("Calculate Metrics", type="primary"):
            _calculate_and_display_metrics(
                analytics_service,
                portfolio_id,
                start_date,
                end_date,
                benchmark_ticker,
            )

        # Display cached metrics and charts if available
        metrics_key = f"metrics_{portfolio_id}"
        if metrics_key in st.session_state:
            # Create tabs for Metrics and Charts
            tab_names = ["Metrics", "Charts"]
            tabs = st.tabs(tab_names)

            with tabs[0]:
                _display_metrics(
                    st.session_state[metrics_key],
                    portfolio_id,
                )

            with tabs[1]:
                _display_charts(
                    analytics_service,
                    portfolio_id,
                    start_date,
                    end_date,
                    benchmark_ticker,
                )

    except PortfolioNotFoundError:
        st.error(f"Portfolio {portfolio_id} not found")
        if st.button("Go to Portfolio List"):
            st.switch_page("pages/portfolio_list.py")
    except Exception as e:
        logger.error(f"Error loading portfolio analysis: {e}", exc_info=True)
        st.error(f"Error loading portfolio analysis: {e}")


def _calculate_and_display_metrics(
    analytics_service: AnalyticsService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
    benchmark_ticker: Optional[str],
) -> None:
    """Calculate and display portfolio metrics."""
    try:
        with st.spinner("Calculating metrics... This may take a moment."):
            metrics = analytics_service.calculate_portfolio_metrics(
                portfolio_id=portfolio_id,
                start_date=start_date,
                end_date=end_date,
                benchmark_ticker=benchmark_ticker,
            )

        # Store in session state
        st.session_state[f"metrics_{portfolio_id}"] = metrics

        # Create tabs for Metrics and Charts
        tabs = st.tabs(["Metrics", "Charts"])

        with tabs[0]:
            _display_metrics(metrics, portfolio_id)

        with tabs[1]:
            _display_charts(
                analytics_service,
                portfolio_id,
                start_date,
                end_date,
                benchmark_ticker,
            )

    except InsufficientDataError as e:
        st.error(f"Insufficient data for analysis: {e}")
        info_msg = (
            "Please ensure the portfolio has positions with "
            "sufficient price history for the selected date range."
        )
        st.info(info_msg)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        st.error(f"Error calculating metrics: {e}")


def _display_metrics(metrics: dict, portfolio_id: str) -> None:
    """Display calculated metrics in organized sections."""
    st.markdown("---")

    # Categorize metrics
    performance_metrics = {
        k: v
        for k, v in metrics.items()
        if any(
            keyword in k.lower()
            for keyword in [
                "return",
                "cagr",
                "win",
                "payoff",
                "profit",
                "expectancy",
                "best",
                "worst",
            ]
        )
    }

    risk_metrics = {
        k: v
        for k, v in metrics.items()
        if any(
            keyword in k.lower()
            for keyword in [
                "volatility",
                "drawdown",
                "var",
                "cvar",
                "deviation",
                "skewness",
                "kurtosis",
                "ulcer",
                "pain",
            ]
        )
    }

    ratio_metrics = {
        k: v
        for k, v in metrics.items()
        if any(
            keyword in k.lower()
            for keyword in [
                "ratio",
                "sharpe",
                "sortino",
                "calmar",
                "sterling",
                "burke",
                "treynor",
                "information",
                "modigliani",
                "omega",
                "kappa",
                "gain",
                "martin",
                "tail",
                "common",
                "rachev",
            ]
        )
    }

    market_metrics = {
        k: v
        for k, v in metrics.items()
        if any(
            keyword in k.lower()
            for keyword in [
                "beta",
                "alpha",
                "correlation",
                "capture",
                "tracking",
                "active",
                "jensen",
                "batting",
                "relative",
                "rolling",
                "timing",
                "r_squared",
            ]
        )
    }

    # Display sections
    if performance_metrics:
        render_metrics_section(
            performance_metrics, "Performance Metrics", columns=3
        )

    if risk_metrics:
        st.markdown("---")
        render_metrics_section(risk_metrics, "Risk Metrics", columns=3)

    if ratio_metrics:
        st.markdown("---")
        render_metrics_section(
            ratio_metrics, "Risk-Adjusted Ratios", columns=3
        )

    if market_metrics:
        st.markdown("---")
        render_metrics_section(
            market_metrics, "Market-Related Metrics", columns=3
        )

    # Export option
    st.markdown("---")
    st.subheader("Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to CSV"):
            df = pd.DataFrame([metrics])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_metrics_{portfolio_id}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Export to JSON"):
            import json

            json_data = json.dumps(metrics, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"portfolio_metrics_{portfolio_id}.json",
                mime="application/json",
            )


def _display_charts(
    analytics_service: AnalyticsService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
    benchmark_ticker: Optional[str],
) -> None:
    """Display portfolio charts."""
    try:
        with st.spinner("Preparing chart data..."):
            # Fetch portfolio data using AnalyticsService
            portfolio_service = analytics_service._portfolio_service
            portfolio = portfolio_service.get_portfolio(portfolio_id)
            positions = portfolio.get_all_positions()
            tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]

            if not tickers:
                st.warning("Portfolio has no valid positions for charting")
                return

            # Fetch portfolio prices and calculate returns
            portfolio_prices = analytics_service._fetch_portfolio_prices(
                tickers, start_date, end_date
            )

            if portfolio_prices.empty:
                st.error("No price data available for the selected date range")
                return

            portfolio_returns = analytics_service._calculate_portfolio_returns(
                portfolio_prices, positions
            )

            if portfolio_returns.empty:
                st.error("Unable to calculate returns from price data")
                return

            portfolio_values = analytics_service._calculate_portfolio_values(
                portfolio_prices, positions, portfolio.starting_capital
            )

            # Fetch benchmark data if provided
            benchmark_returns = None
            if benchmark_ticker:
                try:
                    data_service = analytics_service._data_service
                    benchmark_prices = data_service.fetch_historical_prices(
                        benchmark_ticker,
                        start_date,
                        end_date,
                        use_cache=True,
                        save_to_db=True,
                    )

                    has_data = (
                        not benchmark_prices.empty
                        and "Adjusted_Close" in benchmark_prices.columns
                    )
                    if has_data:
                        benchmark_close = benchmark_prices["Adjusted_Close"]
                        benchmark_returns = (
                            benchmark_close.pct_change().dropna()
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch benchmark: {e}")

        # Chart selector
        chart_type = st.selectbox(
            "Select Chart Type:",
            [
                "Cumulative Returns",
                "Drawdown",
                "Return Distribution",
                "Q-Q Plot",
                "Rolling Sharpe Ratio",
                "Monthly Heatmap",
            ],
        )

        # Render selected chart
        if chart_type == "Cumulative Returns":
            # Linear/Log scale toggle
            scale_type = st.radio("Scale:", ["Linear", "Log"], horizontal=True)
            chart_data = get_cumulative_returns_data(
                portfolio_returns, benchmark_returns
            )
            fig = plot_cumulative_returns(
                chart_data, linear_scale=(scale_type == "Linear")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Drawdown":
            chart_data = get_drawdown_data(portfolio_values)
            fig = plot_drawdown(chart_data)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Return Distribution":
            chart_data = get_return_distribution_data(
                portfolio_returns, bins=50
            )
            fig = plot_return_distribution(chart_data)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Q-Q Plot":
            chart_data = get_qq_plot_data(portfolio_returns)
            fig = plot_qq_plot(chart_data)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Rolling Sharpe Ratio":
            window = st.slider("Rolling Window (days):", 20, 120, 30, step=10)
            chart_data = get_rolling_metric_data(
                portfolio_returns, "sharpe", window=window
            )
            threshold_value = 1.0
            fig = plot_rolling_metric(
                chart_data, "Sharpe Ratio", threshold=threshold_value
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Monthly Heatmap":
            chart_data = get_monthly_heatmap_data(portfolio_returns)
            fig = plot_monthly_heatmap(chart_data)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error displaying charts: {e}", exc_info=True)
        st.error(f"Error displaying charts: {e}")


if __name__ == "__main__":
    render_portfolio_analysis()
