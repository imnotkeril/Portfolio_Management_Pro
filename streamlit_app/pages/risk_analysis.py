"""Risk Analysis page for advanced risk metrics."""

import logging
from datetime import date, timedelta

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from services.portfolio_service import PortfolioService
from services.risk_service import RiskService
from core.scenario_engine.historical_scenarios import get_all_scenarios
from core.scenario_engine.custom_scenarios import (
    create_custom_scenario,
    validate_scenario,
)
from core.scenario_engine.scenario_chain import create_scenario_chain
from core.exceptions import ValidationError
from core.analytics_engine.risk_metrics import calculate_cvar
from streamlit_app.utils.chart_config import COLORS
from streamlit_app.components.charts import plot_var_distribution

logger = logging.getLogger(__name__)


def render_risk_analysis_page() -> None:
    """Render risk analysis page."""
    st.title("Risk Analysis")
    st.markdown(
        "Advanced risk analysis including VaR, Monte Carlo simulations, "
        "and stress testing."
    )

    # Initialize services
    portfolio_service = PortfolioService()
    risk_service = RiskService()

    # Get portfolios
    portfolios = portfolio_service.list_portfolios()

    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return

    # Portfolio selection
    portfolio_names = [p.name for p in portfolios]
    selected_name = st.selectbox(
        "Select Portfolio",
        portfolio_names,
        key="risk_analysis_portfolio",
    )

    selected_portfolio = next(
        p for p in portfolios if p.name == selected_name
    )

    # Date range selection
    st.subheader("Analysis Period")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=365),
            key="risk_start_date",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            key="risk_end_date",
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "VaR Analysis",
            "Monte Carlo Simulation",
            "Historical Scenarios",
            "Custom Scenario",
            "Scenario Chain",
        ]
    )

    with tab1:
        _render_var_analysis(
            risk_service,
            selected_portfolio.id,
            start_date,
            end_date,
        )

    with tab2:
        _render_monte_carlo(
            risk_service,
            selected_portfolio.id,
            start_date,
            end_date,
        )

    with tab3:
        _render_historical_scenarios(
            risk_service,
            selected_portfolio.id,
        )

    with tab4:
        _render_custom_scenario(
            risk_service,
            selected_portfolio.id,
        )

    with tab5:
        _render_scenario_chain(
            risk_service,
            selected_portfolio.id,
        )


def _render_var_analysis(
    risk_service: RiskService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
) -> None:
    """Render VaR analysis section."""
    st.subheader("Value at Risk (VaR) Analysis")

    # Confidence level selector
    confidence_level = st.slider(
        "Confidence Level",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key="var_confidence",
        help="Confidence level for VaR calculation (90%, 95%, or 99%)",
    )
    conf_decimal = confidence_level / 100.0

    # Time horizon
    time_horizon = st.number_input(
        "Time Horizon (days)",
        min_value=1,
        max_value=30,
        value=1,
        key="var_horizon",
    )

    # Calculate VaR
    if st.button("Calculate VaR", key="calculate_var"):
        try:
            with st.spinner("Calculating VaR..."):
                var_results = risk_service.calculate_var_analysis(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date,
                    confidence_level=conf_decimal,
                    include_monte_carlo=True,
                    num_simulations=10000,
                    time_horizon=time_horizon,
                )

                # Display results
                st.success("VaR calculation completed!")

                var_data = var_results["var_results"]

                # Get portfolio returns for visualization
                portfolio_returns = risk_service._get_portfolio_returns(
                    portfolio_id, start_date, end_date
                )

                # Calculate CVaR
                cvar_value = calculate_cvar(
                    portfolio_returns, conf_decimal
                )

                # Calculate metrics for cards (95% and 99%)
                var_95_hist = (
                    var_data.get("historical")
                    if var_data.get("historical")
                    else 0
                )
                cvar_95 = calculate_cvar(portfolio_returns, 0.95)

                var_99_hist = None
                cvar_99 = None
                if (
                    portfolio_returns is not None
                    and not portfolio_returns.empty
                ):
                    from core.analytics_engine.risk_metrics import (
                        calculate_var,
                    )
                    var_99_hist = calculate_var(
                        portfolio_returns, 0.99, "historical"
                    )
                    cvar_99 = calculate_cvar(portfolio_returns, 0.99)

                # Display metrics cards
                st.markdown("### Key Risk Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "VaR (95%)",
                        f"{var_95_hist * 100:.2f}%" if var_95_hist else "N/A",
                        help="Maximum loss at 95% confidence"
                    )
                with col2:
                    st.metric(
                        "CVaR (95%)",
                        f"{cvar_95 * 100:.2f}%" if cvar_95 else "N/A",
                        help="Expected loss beyond VaR at 95% confidence"
                    )
                with col3:
                    st.metric(
                        "VaR (99%)",
                        f"{var_99_hist * 100:.2f}%" if var_99_hist else "N/A",
                        help="Maximum loss at 99% confidence"
                    )
                with col4:
                    st.metric(
                        "CVaR (99%)",
                        f"{cvar_99 * 100:.2f}%" if cvar_99 else "N/A",
                        help="Expected loss beyond VaR at 99% confidence"
                    )

                st.markdown("---")

                # Create comparison table
                comparison_data = []
                methods = [
                    "historical",
                    "parametric",
                    "cornish_fisher",
                    "monte_carlo",
                ]
                method_names = {
                    "historical": "Historical",
                    "parametric": "Parametric",
                    "cornish_fisher": "Cornish-Fisher",
                    "monte_carlo": "Monte Carlo",
                }

                for method in methods:
                    var_value = var_data.get(method)
                    if (
                        var_value is not None
                        and not isinstance(var_value, str)
                    ):
                        comparison_data.append({
                            "Method": method_names[method],
                            f"VaR ({confidence_level}%)": f"{var_value:.4f}",
                            "VaR (%)": f"{var_value * 100:.2f}%",
                        })
                
                # Add CVaR to comparison table
                if cvar_value is not None:
                    comparison_data.append({
                        "Method": "CVaR (Expected Shortfall)",
                        f"VaR ({confidence_level}%)": (
                            f"{cvar_value:.4f}"
                        ),
                        "VaR (%)": f"{cvar_value * 100:.2f}%",
                    })

                if comparison_data:
                    st.markdown("### VaR Methods Comparison")
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)

                    # Add distribution chart
                    if (
                        portfolio_returns is not None
                        and not portfolio_returns.empty
                    ):
                        st.markdown("---")
                        st.markdown(
                            "### VaR on Return Distribution"
                        )
                        try:
                            var_hist = var_data.get("historical", 0)
                            if var_hist is not None:
                                fig_dist = plot_var_distribution(
                                    portfolio_returns,
                                    var_hist,
                                    cvar_value,
                                    conf_decimal,
                                )
                                st.plotly_chart(
                                    fig_dist, use_container_width=True
                                )
                                pct_days = int((1 - conf_decimal) * 100)
                                st.caption(
                                    f"**Interpretation:** {pct_days}% of "
                                    f"days have returns less than "
                                    f"{var_hist*100:.2f}%. On those worst "
                                    f"days, the average loss is "
                                    f"{cvar_value*100:.2f}% (CVaR)."
                                )
                        except Exception as e:
                            logger.warning(
                                f"Error plotting VaR distribution: {e}"
                            )

                    # Add VaR Sensitivity Chart
                    st.markdown("---")
                    st.markdown("### VaR Sensitivity Analysis")
                    try:
                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.risk_metrics import (
                                calculate_var,
                            )

                            confidence_levels = [0.90, 0.95, 0.99]
                            var_sensitivity = []
                            cvar_sensitivity = []

                            for cl in confidence_levels:
                                var_val = calculate_var(
                                    portfolio_returns, cl, "historical"
                                )
                                cvar_val = calculate_cvar(
                                    portfolio_returns, cl
                                )
                                var_sensitivity.append(var_val * 100)
                                cvar_sensitivity.append(cvar_val * 100)

                            fig_sensitivity = go.Figure()
                            fig_sensitivity.add_trace(
                                go.Scatter(
                                    x=[
                                        int(cl * 100)
                                        for cl in confidence_levels
                                    ],
                                    y=var_sensitivity,
                                    mode="lines+markers",
                                    name="VaR",
                                    line=dict(
                                        color=COLORS["danger"], width=3
                                    ),
                                    marker=dict(size=10),
                                )
                            )
                            fig_sensitivity.add_trace(
                                go.Scatter(
                                    x=[
                                        int(cl * 100)
                                        for cl in confidence_levels
                                    ],
                                    y=cvar_sensitivity,
                                    mode="lines+markers",
                                    name="CVaR",
                                    line=dict(
                                        color=COLORS["warning"], width=3
                                    ),
                                    marker=dict(size=10),
                                )
                            )

                            fig_sensitivity.update_layout(
                                title=(
                                    "VaR and CVaR at Different "
                                    "Confidence Levels"
                                ),
                                xaxis_title="Confidence Level (%)",
                                yaxis_title="Risk (%)",
                                height=400,
                                template="plotly_dark",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                ),
                            )

                            st.plotly_chart(
                                fig_sensitivity, use_container_width=True
                            )
                            st.caption(
                                "**Interpretation:** Higher confidence "
                                "levels (99%) show more extreme potential "
                                "losses. CVaR is always more negative than "
                                "VaR, representing the average loss in tail "
                                "events."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error creating VaR sensitivity chart: {e}"
                        )

                    # Add Rolling VaR Chart
                    st.markdown("---")
                    st.markdown("### Rolling VaR Analysis")
                    try:
                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.chart_data import (
                                get_rolling_var_data,
                            )
                            from streamlit_app.components.charts import (
                                plot_rolling_var,
                            )

                            # Window selector
                            rolling_window = st.slider(
                                "Rolling Window (days)",
                                min_value=30,
                                max_value=252,
                                value=63,
                                step=21,
                                key="rolling_var_window",
                                help="Window size for rolling VaR calculation",
                            )

                            var_rolling_data = get_rolling_var_data(
                                portfolio_returns,
                                None,  # No benchmark for now
                                window=rolling_window,
                                confidence_level=conf_decimal,
                            )

                            if var_rolling_data:
                                fig_rolling = plot_rolling_var(
                                    var_rolling_data
                                )
                                st.plotly_chart(
                                    fig_rolling, use_container_width=True
                                )

                                # Statistics
                                stats = var_rolling_data.get(
                                    "portfolio_stats", {}
                                )
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Avg VaR",
                                        f"{stats.get('avg', 0)*100:.2f}%",
                                    )
                                with col2:
                                    st.metric(
                                        "Median VaR",
                                        f"{stats.get('median', 0)*100:.2f}%",
                                    )
                                with col3:
                                    st.metric(
                                        "Min VaR",
                                        f"{stats.get('min', 0)*100:.2f}%",
                                    )
                                with col4:
                                    st.metric(
                                        "Max VaR",
                                        f"{stats.get('max', 0)*100:.2f}%",
                                    )

                                st.caption(
                                    "**Interpretation:** Rolling VaR shows "
                                    "how maximum potential loss changes over "
                                    "time. Higher values indicate periods of "
                                    "increased risk."
                                )
                    except Exception as e:
                        logger.warning(
                            f"Error creating rolling VaR chart: {e}"
                        )

                    # Add Portfolio VaR with Covariance Matrix
                    st.markdown("---")
                    st.markdown("### Portfolio VaR with Covariance Matrix")
                    try:
                        # Get portfolio positions and price data
                        portfolio_service = PortfolioService()
                        portfolio = portfolio_service.get_portfolio(
                            portfolio_id
                        )
                        positions = portfolio.positions if portfolio else []

                        if positions:
                            from services.data_service import DataService
                            data_service = DataService()

                            # Get tickers (exclude CASH)
                            tickers = [
                                pos.ticker
                                for pos in positions
                                if pos.ticker != "CASH"
                            ]

                            if len(tickers) >= 2:
                                # Fetch price data
                                all_prices = []
                                for ticker in tickers:
                                    try:
                                        prices = (
                                            data_service.fetch_historical_prices(
                                                ticker,
                                                start_date,
                                                end_date,
                                                use_cache=True,
                                                save_to_db=False,
                                            )
                                        )
                                        if not prices.empty:
                                            prices["Ticker"] = ticker
                                            all_prices.append(prices)
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to fetch {ticker}: {e}"
                                        )

                                if all_prices:
                                    # Combine and pivot
                                    combined = pd.concat(
                                        all_prices, ignore_index=True
                                    )
                                    price_data = combined.pivot_table(
                                        index="Date",
                                        columns="Ticker",
                                        values="Adjusted_Close",
                                        aggfunc="first",
                                    )

                                    # Calculate returns
                                    returns_df = price_data.pct_change().dropna()

                                    # Calculate weights
                                    total_value = sum(
                                        pos.shares
                                        * price_data[pos.ticker].iloc[0]
                                        for pos in positions
                                        if pos.ticker in price_data.columns
                                        and pos.ticker != "CASH"
                                    )

                                    weights = np.array([
                                        (
                                            pos.shares
                                            * price_data[pos.ticker].iloc[0]
                                            / total_value
                                        )
                                        if pos.ticker in price_data.columns
                                        and pos.ticker != "CASH"
                                        else 0.0
                                        for pos in positions
                                        if pos.ticker != "CASH"
                                    ])

                                    # Calculate Portfolio VaR
                                    from core.risk_engine.var_calculator import (
                                        calculate_portfolio_var_covariance,
                                    )

                                    portfolio_var_result = (
                                        calculate_portfolio_var_covariance(
                                            returns_df,
                                            weights,
                                            conf_decimal,
                                            time_horizon,
                                        )
                                    )

                                    # Display Portfolio VaR
                                    st.metric(
                                        "Portfolio VaR (Covariance Method)",
                                        f"{portfolio_var_result['portfolio_var']*100:.2f}%",
                                        help=(
                                            "VaR calculated using covariance "
                                            "matrix, accounting for "
                                            "correlations between assets"
                                        ),
                                    )

                                    # VaR Decomposition
                                    st.markdown("#### VaR Decomposition")
                                    decomposition_data = []
                                    for ticker in returns_df.columns:
                                        if ticker in portfolio_var_result[
                                            "component_var"
                                        ]:
                                            decomposition_data.append({
                                                "Asset": ticker,
                                                "Weight (%)": (
                                                    weights[
                                                        list(returns_df.columns).index(ticker)
                                                    ]
                                                    * 100
                                                    if ticker in returns_df.columns
                                                    else 0.0
                                                ),
                                                "Component VaR (%)": (
                                                    portfolio_var_result[
                                                        "component_var"
                                                    ][ticker]
                                                    * 100
                                                ),
                                                "Contribution (%)": (
                                                    portfolio_var_result[
                                                        "var_contribution_pct"
                                                    ][ticker]
                                                ),
                                                "Marginal VaR (%)": (
                                                    portfolio_var_result[
                                                        "marginal_var"
                                                    ][ticker]
                                                    * 100
                                                ),
                                            })

                                    if decomposition_data:
                                        import pandas as pd
                                        decomp_df = pd.DataFrame(
                                            decomposition_data
                                        )
                                        st.dataframe(
                                            decomp_df, use_container_width=True
                                        )

                                        # Bar chart of contributions
                                        fig_decomp = go.Figure()
                                        fig_decomp.add_trace(
                                            go.Bar(
                                                x=decomp_df["Asset"],
                                                y=decomp_df["Contribution (%)"],
                                                marker=dict(
                                                    color=COLORS["primary"]
                                                ),
                                                text=[
                                                    f"{v:.1f}%"
                                                    for v in decomp_df[
                                                        "Contribution (%)"
                                                    ]
                                                ],
                                                textposition="outside",
                                            )
                                        )
                                        fig_decomp.update_layout(
                                            title=(
                                                "VaR Contribution by Asset "
                                                "(%)"
                                            ),
                                            xaxis_title="Asset",
                                            yaxis_title="Contribution (%)",
                                            height=400,
                                            template="plotly_dark",
                                        )
                                        st.plotly_chart(
                                            fig_decomp, use_container_width=True
                                        )

                                        st.caption(
                                            "**Component VaR:** Contribution "
                                            "of each asset to total portfolio "
                                            "VaR. **Marginal VaR:** Change in "
                                            "portfolio VaR for a 1% increase "
                                            "in asset weight."
                                        )
                    except Exception as e:
                        logger.warning(
                            f"Error calculating Portfolio VaR: {e}"
                        )

                    # Visual comparison
                    try:
                        fig = go.Figure()
                        methods_list = [d["Method"] for d in comparison_data]

                        # Extract and convert VaR values safely
                        var_values = []
                        for d in comparison_data:
                            var_str = d.get(f"VaR ({confidence_level}%)", "0")
                            try:
                                # Remove commas and convert to float
                                var_float = float(
                                    str(var_str).replace(",", "")
                                )
                                var_values.append(var_float)
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Could not convert VaR value "
                                    f"'{var_str}': {e}"
                                )
                                var_values.append(0.0)

                        if not var_values or all(v == 0.0 for v in var_values):
                            raise ValueError("No valid VaR values to plot")

                        # Format: percentages above, absolute values below
                        var_values_pct = [v * 100 for v in var_values]
                        var_values_abs = [abs(v) for v in var_values]

                        fig.add_trace(
                            go.Bar(
                                x=methods_list,
                                y=var_values_abs,
                                marker_color=COLORS["error"],
                                text=[f"{p:.1f}%" for p in var_values_pct],
                                textposition="outside",
                                textfont=dict(size=12),
                            )
                        )

                        title_text = (
                            f"VaR Comparison ({confidence_level}% Confidence)"
                        )
                        fig.update_layout(
                            title=title_text,
                            xaxis_title="Method",
                            yaxis_title="VaR (absolute value)",
                            height=400,
                            template="plotly_dark",
                        )

                        # Add absolute values below bars
                        fig.add_annotation(
                            text="Values shown: percentages above bars, "
                            "absolute values on Y-axis",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as chart_error:
                        logger.warning(
                            f"Error creating VaR chart: {chart_error}"
                        )
                        st.warning(
                            "Could not display VaR comparison chart, "
                            "but calculations completed successfully."
                        )
                else:
                    st.warning(
                        "No VaR values could be calculated. "
                        "Please check your portfolio data."
                    )

        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            st.error(f"Error calculating VaR: {error_msg}")
            logger.exception(f"VaR calculation error: {error_msg}")


def _render_monte_carlo(
    risk_service: RiskService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
) -> None:
    """Render Monte Carlo simulation section."""
    st.subheader("Monte Carlo Simulation")

    st.info(
        "Monte Carlo simulation forecasts portfolio value **forward** "
        "in time based on historical return patterns. "
        "Time horizon = number of days **into the future** to simulate."
    )

    # Simulation parameters
    col1, col2 = st.columns(2)

    with col1:
        time_horizon = st.number_input(
            "Time Horizon (days forward)",
            min_value=1,
            max_value=252,
            value=30,
            key="mc_horizon",
            help="Number of days into the future to simulate",
        )

        num_simulations = st.selectbox(
            "Number of Simulations",
            options=[1000, 5000, 10000, 50000],
            index=2,
            key="mc_simulations",
        )

    with col2:
        initial_value = st.number_input(
            "Initial Portfolio Value",
            min_value=1.0,
            value=100000.0,
            step=10000.0,
            key="mc_initial",
        )

        model = st.selectbox(
            "Model",
            options=["gbm", "jump_diffusion"],
            key="mc_model",
            help="GBM: Geometric Brownian Motion\n"
            "Jump Diffusion: Includes jump events",
        )

    # Show paths option
    show_paths = st.checkbox(
        "Show Paths Chart (Spaghetti Chart)",
        value=False,
        key="mc_show_paths",
        help="Display individual simulation paths",
    )

    # Run simulation
    if st.button("Run Simulation", key="run_mc"):
        try:
            with st.spinner(
                f"Running {num_simulations:,} simulations..."
            ):
                results = risk_service.run_monte_carlo_simulation(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date,
                    time_horizon=time_horizon,
                    num_simulations=num_simulations,
                    initial_value=initial_value,
                    model=model,
                )

                st.success("Simulation completed!")

                # Display statistics
                st.markdown("### Simulation Statistics")
                stats = results["statistics"]
                percentiles = results["percentiles"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"${stats['mean']:,.2f}")
                    st.metric("Median", f"${stats['median']:,.2f}")
                with col2:
                    st.metric("Std Dev", f"${stats['std']:,.2f}")
                    st.metric("Min", f"${stats['min']:,.2f}")
                with col3:
                    st.metric("Max", f"${stats['max']:,.2f}")

                # Percentiles
                st.markdown("### Percentile Outcomes")
                percentile_data = [
                    {"Percentile": f"{p}%", "Value": f"${v:,.2f}"}
                    for p, v in percentiles.items()
                ]
                import pandas as pd
                df = pd.DataFrame(percentile_data)
                st.dataframe(df, use_container_width=True)

                # Distribution histogram
                final_values = results["final_values"]
                fig = go.Figure()

                fig.add_trace(
                    go.Histogram(
                        x=final_values,
                        nbinsx=50,
                        marker_color=COLORS["primary"],
                        name="Final Values",
                    )
                )

                # Add percentile lines with % above, values below
                for p, v in percentiles.items():
                    fig.add_vline(
                        x=v,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=f"{p}%",
                        annotation_position="top",
                    )
                    # Add value below
                    fig.add_annotation(
                        x=v,
                        y=0,
                        text=f"${int(v):,}",
                        showarrow=False,
                        yshift=-30,
                        font=dict(size=10, color="yellow"),
                    )

                fig.update_layout(
                    title="Monte Carlo Simulation: Final Value Distribution",
                    xaxis_title="Portfolio Value ($)",
                    yaxis_title="Frequency",
                    height=500,
                    template="plotly_dark",
                )

                st.plotly_chart(fig, use_container_width=True)

                # VaR/CVaR from simulations
                st.markdown("---")
                st.markdown("### VaR & CVaR from Simulations")
                try:

                    # Calculate returns from final values
                    returns_sim = (
                        np.array(final_values) / initial_value - 1.0
                    )

                    # Calculate VaR and CVaR at different levels
                    confidence_levels_mc = [0.90, 0.95, 0.99]
                    var_cvar_data = []

                    for cl in confidence_levels_mc:
                        # VaR = percentile
                        var_val = np.percentile(
                            returns_sim, (1 - cl) * 100
                        )
                        # CVaR = mean of returns below VaR
                        tail_returns = returns_sim[
                            returns_sim <= var_val
                        ]
                        cvar_val = (
                            tail_returns.mean()
                            if len(tail_returns) > 0
                            else var_val
                        )

                        var_cvar_data.append({
                            "Confidence": f"{int(cl*100)}%",
                            "VaR (%)": f"{var_val*100:.2f}%",
                            "VaR ($)": (
                                f"${var_val*initial_value:,.2f}"
                            ),
                            "CVaR (%)": f"{cvar_val*100:.2f}%",
                            "CVaR ($)": (
                                f"${cvar_val*initial_value:,.2f}"
                            ),
                        })

                    import pandas as pd
                    var_cvar_df = pd.DataFrame(var_cvar_data)
                    st.dataframe(var_cvar_df, use_container_width=True)

                    # Compare with historical VaR
                    st.markdown("---")
                    st.markdown("### Comparison with Historical VaR")
                    try:
                        portfolio_returns = (
                            risk_service._get_portfolio_returns(
                                portfolio_id, start_date, end_date
                            )
                        )

                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.risk_metrics import (
                                calculate_var,
                            )

                            comparison_data = []
                            for cl in confidence_levels_mc:
                                # Historical VaR
                                hist_var = calculate_var(
                                    portfolio_returns, cl, "historical"
                                )
                                # Monte Carlo VaR
                                mc_var = np.percentile(
                                    returns_sim, (1 - cl) * 100
                                )

                                comparison_data.append({
                                    "Confidence": f"{int(cl*100)}%",
                                    "Historical VaR (%)": (
                                        f"{hist_var*100:.2f}%"
                                    ),
                                    "MC VaR (%)": (
                                        f"{mc_var*100:.2f}%"
                                    ),
                                    "Difference": (
                                        f"{(mc_var - hist_var)*100:.2f}%"
                                    ),
                                })

                            comp_df = pd.DataFrame(comparison_data)
                            st.dataframe(comp_df, use_container_width=True)

                            # Visual comparison
                            fig_comp = go.Figure()
                            fig_comp.add_trace(
                                go.Bar(
                                    x=[int(cl * 100) for cl in confidence_levels_mc],
                                    y=[
                                        calculate_var(
                                            portfolio_returns, cl, "historical"
                                        )
                                        * 100
                                        for cl in confidence_levels_mc
                                    ],
                                    name="Historical VaR",
                                    marker_color=COLORS["danger"],
                                )
                            )
                            fig_comp.add_trace(
                                go.Bar(
                                    x=[int(cl * 100) for cl in confidence_levels_mc],
                                    y=[
                                        np.percentile(
                                            returns_sim, (1 - cl) * 100
                                        )
                                        * 100
                                        for cl in confidence_levels_mc
                                    ],
                                    name="Monte Carlo VaR",
                                    marker_color=COLORS["warning"],
                                )
                            )

                            fig_comp.update_layout(
                                title=(
                                    "Historical VaR vs Monte Carlo VaR"
                                ),
                                xaxis_title="Confidence Level (%)",
                                yaxis_title="VaR (%)",
                                barmode="group",
                                height=400,
                                template="plotly_dark",
                            )
                            st.plotly_chart(
                                fig_comp, use_container_width=True
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error comparing with historical VaR: {e}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error calculating VaR/CVaR from simulations: {e}"
                    )

                # Confidence intervals on distribution chart
                st.markdown("---")
                st.markdown("### Distribution with Confidence Intervals")
                try:
                    fig_ci = go.Figure()

                    # Histogram
                    fig_ci.add_trace(
                        go.Histogram(
                            x=final_values,
                            nbinsx=50,
                            marker_color=COLORS["primary"],
                            opacity=0.7,
                            name="Final Values",
                        )
                    )

                    # Add confidence intervals
                    ci_levels = [0.90, 0.95, 0.99]
                    ci_colors = [
                        COLORS["success"],
                        COLORS["warning"],
                        COLORS["danger"],
                    ]

                    for ci, color in zip(ci_levels, ci_colors):
                        lower = np.percentile(
                            final_values, (1 - ci) / 2 * 100
                        )
                        upper = np.percentile(
                            final_values, (1 + ci) / 2 * 100
                        )

                        # Shaded area
                        fig_ci.add_vrect(
                            x0=lower,
                            x1=upper,
                            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
                            layer="below",
                            line_width=0,
                            annotation_text=f"{int(ci*100)}% CI",
                            annotation_position="top left",
                        )

                        # Lines without annotations to avoid overlap
                        fig_ci.add_vline(
                            x=lower,
                            line_dash="dash",
                            line_color=color,
                        )
                        fig_ci.add_vline(
                            x=upper,
                            line_dash="dash",
                            line_color=color,
                        )
                        
                        # Add annotations separately with better positioning
                        # Get histogram max height for positioning
                        hist_counts, _ = np.histogram(final_values, bins=50)
                        max_height = np.max(hist_counts)

                        # Left annotation (lower bound)
                        lower_pct = int((1 - ci) / 2 * 100)
                        fig_ci.add_annotation(
                            x=lower,
                            y=max_height * 0.95,
                            text=f"{lower_pct}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=color,
                            font=dict(color=color, size=10),
                            bgcolor="rgba(0, 0, 0, 0.7)",
                            bordercolor=color,
                            borderwidth=1,
                            xanchor="right",
                            yanchor="top",
                        )

                        # Right annotation (upper bound)
                        upper_pct = int((1 + ci) / 2 * 100)
                        fig_ci.add_annotation(
                            x=upper,
                            y=max_height * 0.95,
                            text=f"{upper_pct}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=color,
                            font=dict(color=color, size=10),
                            bgcolor="rgba(0, 0, 0, 0.7)",
                            bordercolor=color,
                            borderwidth=1,
                            xanchor="left",
                            yanchor="top",
                        )

                    fig_ci.update_layout(
                        title=(
                            "Final Value Distribution with "
                            "Confidence Intervals"
                        ),
                        xaxis_title="Portfolio Value ($)",
                        yaxis_title="Frequency",
                        height=500,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_ci, use_container_width=True)

                except Exception as e:
                    logger.warning(
                        f"Error creating confidence intervals chart: {e}"
                    )

                # Extreme scenarios analysis
                st.markdown("---")
                st.markdown("### Extreme Scenarios Analysis")
                try:
                    # Worst case scenarios
                    worst_5_pct = np.percentile(final_values, 5)
                    worst_1_pct = np.percentile(final_values, 1)
                    worst_0_1_pct = np.percentile(final_values, 0.1)

                    # Best case scenarios
                    best_5_pct = np.percentile(final_values, 95)
                    best_1_pct = np.percentile(final_values, 99)
                    best_0_1_pct = np.percentile(final_values, 99.9)

                    extreme_data = [
                        {
                            "Scenario": "Worst 5%",
                            "Value": f"${worst_5_pct:,.2f}",
                            "Return": (
                                f"${worst_5_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_5_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Worst 1%",
                            "Value": f"${worst_1_pct:,.2f}",
                            "Return": (
                                f"${worst_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Worst 0.1%",
                            "Value": f"${worst_0_1_pct:,.2f}",
                            "Return": (
                                f"${worst_0_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_0_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 5%",
                            "Value": f"${best_5_pct:,.2f}",
                            "Return": f"${best_5_pct - initial_value:,.2f}",
                            "Return %": (
                                f"{((best_5_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 1%",
                            "Value": f"${best_1_pct:,.2f}",
                            "Return": f"${best_1_pct - initial_value:,.2f}",
                            "Return %": (
                                f"{((best_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 0.1%",
                            "Value": f"${best_0_1_pct:,.2f}",
                            "Return": (
                                f"${best_0_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((best_0_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                    ]

                    extreme_df = pd.DataFrame(extreme_data)
                    st.dataframe(extreme_df, use_container_width=True)

                    # Visual comparison
                    fig_extreme = go.Figure()
                    scenarios = [d["Scenario"] for d in extreme_data]
                    values = [
                        float(d["Value"].replace("$", "").replace(",", ""))
                        for d in extreme_data
                    ]

                    colors_extreme = [
                        COLORS["danger"]
                        if "Worst" in s
                        else COLORS["success"]
                        for s in scenarios
                    ]

                    fig_extreme.add_trace(
                        go.Bar(
                            x=scenarios,
                            y=values,
                            marker_color=colors_extreme,
                            text=[f"${v:,.0f}" for v in values],
                            textposition="outside",
                        )
                    )

                    # Add initial value line
                    fig_extreme.add_hline(
                        y=initial_value,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=f"Initial: ${initial_value:,.0f}",
                    )

                    fig_extreme.update_layout(
                        title="Extreme Scenarios: Best vs Worst Cases",
                        xaxis_title="Scenario",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_extreme, use_container_width=True)

                except Exception as e:
                    logger.warning(
                        f"Error analyzing extreme scenarios: {e}"
                    )

                # Spaghetti chart moved here (after Final Value Distribution)
                if show_paths and results.get("simulated_paths"):
                    st.markdown("---")
                    st.markdown("### Simulation Paths (Spaghetti Chart)")
                    paths = results["simulated_paths"]
                    time_horizon = results["time_horizon"]

                    fig_paths = go.Figure()

                    # Convert paths to numpy array for max/min calculation
                    paths_array = np.array(paths)
                    
                    # Calculate max and min paths
                    max_path = np.max(paths_array, axis=0)
                    min_path = np.min(paths_array, axis=0)

                    # Plot a sample of paths (max 100 for performance)
                    num_paths_to_show = min(100, len(paths))
                    step = (
                        len(paths) // num_paths_to_show
                        if num_paths_to_show > 0
                        else 1
                    )

                    for i in range(0, len(paths), step):
                        path_values = paths[i]
                        days = list(range(time_horizon))
                        fig_paths.add_trace(
                            go.Scatter(
                                x=days,
                                y=path_values,
                                mode="lines",
                                line=dict(
                                    width=0.5,
                                    color="rgba(191, 159, 251, 0.3)",
                                ),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                    # Add max path
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=max_path,
                            mode="lines",
                            line=dict(width=2, color="green"),
                            name="Max Path",
                        )
                    )
                    
                    # Add min path
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=min_path,
                            mode="lines",
                            line=dict(width=2, color="red"),
                            name="Min Path",
                        )
                    )

                    # Add current value line
                    fig_paths.add_hline(
                        y=initial_value,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=f"Initial: ${initial_value:,.0f}",
                    )

                    # Add percentile paths
                    from core.risk_engine.monte_carlo import (
                        simulate_portfolio_paths,
                    )

                    # Get returns for percentile calculation
                    returns = risk_service._get_portfolio_returns(
                        portfolio_id, start_date, end_date
                    )

                    # Re-run simulation to get paths for percentiles
                    percentile_result = simulate_portfolio_paths(
                        returns,
                        time_horizon,
                        num_simulations,
                        initial_value,
                        model,
                    )

                    # Add median path (50th percentile)
                    median_path = percentile_result.simulated_paths[
                        np.argmin(
                            np.abs(
                                percentile_result.final_values
                                - percentile_result.percentiles[50.0]
                            )
                        )
                    ]
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=median_path,
                            mode="lines",
                            line=dict(width=2, color="orange"),
                            name="Median (50th percentile)",
                        )
                    )

                    fig_paths.update_layout(
                        title=f"Monte Carlo Simulation: {time_horizon} Days",
                        xaxis_title="Day",
                        yaxis_title="Portfolio Value ($)",
                        yaxis=dict(
                            tickformat="$,.0f",
                            tickmode="linear",
                            tick0=0,
                            dtick=10000,
                        ),
                        height=500,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_paths, use_container_width=True)
                elif show_paths:
                    st.warning(
                        "Path data not available. Please ensure simulation "
                        "returns path information."
                    )

        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            logger.exception("Monte Carlo simulation error")


def _render_stress_tests(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render stress testing section."""
    st.subheader("Stress Testing")

    # Get available scenarios
    scenarios = risk_service.get_available_scenarios()

    if not scenarios:
        st.warning("No stress scenarios available.")
        return

    # Scenario selection
    scenario_names = list(scenarios.keys())
    selected_scenarios = st.multiselect(
        "Select Scenarios to Test",
        options=scenario_names,
        key="stress_scenarios",
        help="Select one or more historical scenarios to test",
    )

    if not selected_scenarios:
        st.info("Please select at least one scenario.")
        return

    # Display scenario descriptions
    with st.expander("Scenario Descriptions", expanded=False):
        for key in selected_scenarios:
            scenario = scenarios[key]
            st.markdown(f"**{scenario.name}**")
            st.markdown(f"*{scenario.description}*")
            st.markdown(
                f"Period: {scenario.start_date} to {scenario.end_date}"
            )
            st.markdown(
                f"Market Impact: {scenario.market_impact_pct * 100:.1f}%"
            )
            st.markdown("---")

    # Run stress tests
    if st.button("Run Stress Tests", key="run_stress"):
        try:
            with st.spinner("Running stress tests..."):
                results = risk_service.run_stress_test(
                    portfolio_id=portfolio_id,
                    scenario_names=selected_scenarios,
                )

                st.success("Stress tests completed!")

                # Display results
                st.markdown("### Stress Test Results")

                # Create results table
                import pandas as pd
                results_data = []
                for r in results:
                    # Safely access nested dictionaries
                    worst_pos = r.get("worst_position", {})
                    worst_ticker = worst_pos.get("ticker", "N/A")
                    worst_impact_pct = worst_pos.get("impact_pct", 0.0)

                    results_data.append({
                        "Scenario": r.get("scenario_name", "Unknown"),
                        "Impact %": (
                            f"{r.get('portfolio_impact_pct', 0.0) * 100:.2f}%"
                        ),
                        "Impact ($)": (
                            f"${r.get('portfolio_impact_value', 0.0):,.2f}"
                        ),
                        "Worst Position": worst_ticker,
                        "Worst Impact": (
                            f"{worst_impact_pct * 100:.2f}%"
                        ),
                        "Recovery (days)": r.get("recovery_time_days", "N/A"),
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)

                # Visual comparison
                fig = go.Figure()

                scenario_names_list = [r["scenario_name"] for r in results]
                impacts = [r["portfolio_impact_pct"] * 100 for r in results]

                fig.add_trace(
                    go.Bar(
                        x=scenario_names_list,
                        y=impacts,
                        marker_color=[
                            COLORS["error"] if i < 0 else COLORS["success"]
                            for i in impacts
                        ],
                        text=[f"{i:.2f}%" for i in impacts],
                        textposition="outside",
                    )
                )

                fig.update_layout(
                    title="Portfolio Impact by Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=400,
                    template="plotly_dark",
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error running stress tests: {str(e)}")
            logger.exception("Stress test error")


def _render_historical_scenarios(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render historical scenarios section."""
    st.subheader("Historical Scenarios")

    # Get all available scenarios
    scenarios = get_all_scenarios()

    if not scenarios:
        st.warning("No historical scenarios available.")
        return

    # Scenario selection
    scenario_keys = list(scenarios.keys())
    scenario_display = [
        scenarios[key].name for key in scenario_keys
    ]

    selected_indices = st.multiselect(
        "Select Scenarios",
        options=range(len(scenario_keys)),
        format_func=lambda x: scenario_display[x],
        key="historical_scenarios",
    )

    if not selected_indices:
        st.info("Please select at least one scenario.")
        return

    selected_keys = [scenario_keys[i] for i in selected_indices]

    # Display scenario info
    with st.expander("Selected Scenarios", expanded=True):
        for key in selected_keys:
            scenario = scenarios[key]
            st.markdown(f"**{scenario.name}**")
            st.markdown(f"*{scenario.description}*")
            st.markdown(
                f"Period: {scenario.start_date} to {scenario.end_date}"
            )
            st.markdown(
                f"Market Impact: {scenario.market_impact_pct * 100:.1f}%"
            )
            if scenario.recovery_period_days:
                st.markdown(
                    f"Recovery Period: {scenario.recovery_period_days} days"
                )
            st.markdown("---")

    # Run scenarios
    if st.button("Run Scenarios", key="run_historical"):
        try:
            with st.spinner("Running scenarios..."):
                results = risk_service.run_stress_test(
                    portfolio_id=portfolio_id,
                    scenario_names=selected_keys,
                )

                st.success("Scenario analysis completed!")

                # Display results
                st.markdown("### Scenario Results")

                import pandas as pd
                results_data = []
                for r in results:
                    # Safely access nested dictionaries
                    worst_pos = r.get("worst_position", {})
                    worst_ticker = worst_pos.get("ticker", "N/A")
                    worst_impact_pct = worst_pos.get("impact_pct", 0.0)

                    results_data.append({
                        "Scenario": r.get("scenario_name", "Unknown"),
                        "Impact %": (
                            f"{r.get('portfolio_impact_pct', 0.0) * 100:.2f}%"
                        ),
                        "Impact ($)": (
                            f"${r.get('portfolio_impact_value', 0.0):,.2f}"
                        ),
                        "Worst Position": worst_ticker,
                        "Worst Impact": (
                            f"{worst_impact_pct * 100:.2f}%"
                        ),
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)

                # Visual comparison
                fig = go.Figure()

                scenario_names_list = [r["scenario_name"] for r in results]
                impacts = [r["portfolio_impact_pct"] * 100 for r in results]

                # For negative values, use 'outside' and adjust with
                # annotations if needed
                # go.Bar only supports: 'inside', 'outside', 'auto', 'none'
                fig.add_trace(
                    go.Bar(
                        x=scenario_names_list,
                        y=impacts,
                        marker_color=[
                            COLORS["error"] if i < 0 else COLORS["success"]
                            for i in impacts
                        ],
                        text=[f"{i:.2f}%" for i in impacts],
                        textposition="outside",
                        textfont=dict(size=12),
                    )
                )

                # Add annotations for negative values below the bars
                for i, (name, impact) in enumerate(
                    zip(scenario_names_list, impacts)
                ):
                    if impact < 0:
                        fig.add_annotation(
                            x=name,
                            y=impact,
                            text=f"{impact:.2f}%",
                            showarrow=False,
                            yshift=-25,  # Position below the bar
                            font=dict(size=12, color=COLORS["error"]),
                            yanchor="top",
                        )

                fig.update_layout(
                    title="Portfolio Impact by Historical Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=500,
                    template="plotly_dark",
                    margin=dict(b=100),  # Extra bottom margin for labels
                )

                st.plotly_chart(fig, use_container_width=True)

                # Portfolio Recovery Chart
                st.markdown("---")
                st.markdown("### Portfolio Recovery Timeline")
                try:
                    fig_recovery = go.Figure()

                    for r in results:
                        scenario_name = r.get("scenario_name", "Unknown")
                        impact_pct = r.get("portfolio_impact_pct", 0.0)
                        recovery_days = r.get("recovery_time_days", None)

                        # Simulate recovery path
                        if recovery_days and recovery_days > 0:
                            days = list(
                                range(
                                    0,
                                    recovery_days + 1,
                                    max(1, recovery_days // 20),
                                )
                            )
                            # Linear recovery (simplified)
                            recovery_path = [
                                1.0 + impact_pct * (1 - d / recovery_days)
                                for d in days
                            ]
                            recovery_path = [
                                max(0, v) for v in recovery_path
                            ]  # No negative

                            fig_recovery.add_trace(
                                go.Scatter(
                                    x=days,
                                    y=[v * 100 for v in recovery_path],
                                    mode="lines+markers",
                                    name=scenario_name,
                                    line=dict(width=2),
                                )
                            )

                    # Add baseline (100%)
                    fig_recovery.add_hline(
                        y=100,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Baseline (100%)",
                    )

                    fig_recovery.update_layout(
                        title="Portfolio Recovery Timeline",
                        xaxis_title="Days After Event",
                        yaxis_title="Portfolio Value (% of Initial)",
                        height=500,
                        template="plotly_dark",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_recovery, use_container_width=True)
                    st.caption(
                        "**Note:** Recovery timeline is estimated based on "
                        "scenario recovery period. Actual recovery may vary."
                    )

                except Exception as e:
                    logger.warning(f"Error creating recovery chart: {e}")

                # Enhanced Scenario Comparison
                st.markdown("---")
                st.markdown("### Enhanced Scenario Comparison")
                try:
                    # Get scenario details
                    scenario_details = {}
                    for key in selected_keys:
                        scenario = scenarios[key]
                        scenario_details[key] = {
                            "name": scenario.name,
                            "start_date": scenario.start_date,
                            "end_date": scenario.end_date,
                            "duration_days": (
                                scenario.end_date - scenario.start_date
                            ).days,
                            "market_impact": scenario.market_impact_pct,
                        }

                    # Create comparison table
                    comparison_data = []
                    for r in results:
                        scenario_key = next(
                            (
                                k
                                for k in selected_keys
                                if scenarios[k].name == r.get("scenario_name")
                            ),
                            None,
                        )
                        if scenario_key:
                            details = scenario_details[scenario_key]
                            comparison_data.append({
                                "Scenario": details["name"],
                                "Period": (
                                    f"{details['start_date']} to "
                                    f"{details['end_date']}"
                                ),
                                "Duration (days)": details["duration_days"],
                                "Market Impact": (
                                    f"{details['market_impact']*100:.1f}%"
                                ),
                                "Portfolio Impact": (
                                    f"{r.get('portfolio_impact_pct', 0)*100:.2f}%"
                                ),
                                "Recovery (days)": (
                                    r.get("recovery_time_days", "N/A")
                                ),
                            })

                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)

                        # Visual comparison with multiple metrics
                        fig_comp = go.Figure()

                        scenario_names = [
                            d["Scenario"] for d in comparison_data
                        ]
                        impacts = [
                            float(d["Portfolio Impact"].replace("%", ""))
                            for d in comparison_data
                        ]
                        durations = [
                            d["Duration (days)"] for d in comparison_data
                        ]

                        fig_comp.add_trace(
                            go.Bar(
                                x=scenario_names,
                                y=impacts,
                                name="Portfolio Impact (%)",
                                marker_color=COLORS["danger"],
                                yaxis="y",
                            )
                        )

                        fig_comp.add_trace(
                            go.Scatter(
                                x=scenario_names,
                                y=durations,
                                name="Duration (days)",
                                mode="lines+markers",
                                line=dict(color=COLORS["warning"], width=3),
                                marker=dict(size=10),
                                yaxis="y2",
                            )
                        )

                        fig_comp.update_layout(
                            title=(
                                "Scenario Comparison: Impact vs Duration"
                            ),
                            xaxis_title="Scenario",
                            yaxis=dict(
                                title="Portfolio Impact (%)",
                                side="left",
                            ),
                            yaxis2=dict(
                                title="Duration (days)",
                                side="right",
                                overlaying="y",
                            ),
                            height=500,
                            template="plotly_dark",
                            barmode="group",
                        )

                        st.plotly_chart(fig_comp, use_container_width=True)

                except Exception as e:
                    logger.warning(
                        f"Error creating enhanced comparison: {e}"
                    )

                # Position Breakdown
                st.markdown("---")
                st.markdown("### Position Impact Breakdown")
                try:
                    # Get portfolio positions
                    portfolio_service = PortfolioService()
                    portfolio = portfolio_service.get_portfolio(portfolio_id)
                    positions = portfolio.positions if portfolio else []

                    if positions:
                        # Create breakdown for each scenario
                        for r in results:
                            scenario_name = r.get("scenario_name", "Unknown")
                            details = r.get("details", {})
                            position_impacts = details.get(
                                "position_impacts", {}
                            )

                            if position_impacts:
                                st.markdown(f"#### {scenario_name}")

                                breakdown_data = []
                                for (
                                    ticker,
                                    impact,
                                ) in position_impacts.items():
                                    # Find position
                                    pos = next(
                                        (
                                            p
                                            for p in positions
                                            if p.ticker == ticker
                                        ),
                                        None,
                                    )
                                    if pos:
                                        breakdown_data.append({
                                            "Ticker": ticker,
                                            "Weight": (
                                                f"{pos.weight*100:.2f}%"
                                                if hasattr(pos, "weight")
                                                else "N/A"
                                            ),
                                            "Impact (%)": (
                                                f"{impact*100:.2f}%"
                                            ),
                                            "Impact Type": (
                                                "Loss" if impact < 0 else "Gain"
                                            ),
                                        })

                                if breakdown_data:
                                    breakdown_df = pd.DataFrame(breakdown_data)
                                    st.dataframe(
                                        breakdown_df, use_container_width=True
                                    )

                                    # Bar chart
                                    fig_breakdown = go.Figure()
                                    fig_breakdown.add_trace(
                                        go.Bar(
                                            x=breakdown_df["Ticker"],
                                            y=[
                                                float(v.replace("%", ""))
                                                for v in breakdown_df[
                                                    "Impact (%)"
                                                ]
                                            ],
                                            marker=dict(
                                                color=[
                                                    COLORS["danger"]
                                                    if float(v.replace("%", ""))
                                                    < 0
                                                    else COLORS["success"]
                                                    for v in breakdown_df[
                                                        "Impact (%)"
                                                    ]
                                                ]
                                            ),
                                            text=breakdown_df["Impact (%)"],
                                            textposition="outside",
                                        )
                                    )

                                    fig_breakdown.update_layout(
                                        title=(
                                            f"Position Impact: {scenario_name}"
                                        ),
                                        xaxis_title="Asset",
                                        yaxis_title="Impact (%)",
                                        height=400,
                                        template="plotly_dark",
                                    )

                                    st.plotly_chart(
                                        fig_breakdown, use_container_width=True
                                    )

                except Exception as e:
                    logger.warning(
                        f"Error creating position breakdown: {e}"
                    )

                # Timeline Visualization
                st.markdown("---")
                st.markdown("### Historical Timeline Visualization")
                try:
                    # Create timeline with all scenarios
                    timeline_data = []
                    for key in selected_keys:
                        scenario = scenarios[key]
                        timeline_data.append({
                            "name": scenario.name,
                            "start": scenario.start_date,
                            "end": scenario.end_date,
                            "impact": scenario.market_impact_pct,
                        })

                    # Sort by start date
                    timeline_data.sort(key=lambda x: x["start"])

                    # Create Gantt-like chart
                    fig_timeline = go.Figure()

                    colors_timeline = [
                        COLORS["danger"]
                        if d["impact"] < -0.2
                        else COLORS["warning"]
                        if d["impact"] < -0.1
                        else COLORS["info"]
                        for d in timeline_data
                    ]

                    for i, data in enumerate(timeline_data):
                        duration = (data["end"] - data["start"]).days
                        fig_timeline.add_trace(
                            go.Bar(
                                x=[duration],
                                y=[data["name"]],
                                orientation="h",
                                marker_color=colors_timeline[i],
                                text=(
                                    f"{data['start']} to {data['end']} "
                                    f"({duration} days)"
                                ),
                                textposition="inside",
                                name=data["name"],
                            )
                        )

                    fig_timeline.update_layout(
                        title="Historical Scenarios Timeline",
                        xaxis_title="Duration (days)",
                        yaxis_title="Scenario",
                        height=300 + len(timeline_data) * 50,
                        template="plotly_dark",
                        showlegend=False,
                    )

                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Timeline with impact magnitude
                    fig_timeline2 = go.Figure()

                    for data in timeline_data:
                        fig_timeline2.add_trace(
                            go.Scatter(
                                x=[data["start"], data["end"]],
                                y=[
                                    data["impact"] * 100,
                                    data["impact"] * 100,
                                ],
                                mode="lines+markers",
                                name=data["name"],
                                line=dict(
                                    width=5,
                                    color=(
                                        COLORS["danger"]
                                        if data["impact"] < -0.2
                                        else COLORS["warning"]
                                        if data["impact"] < -0.1
                                        else COLORS["info"]
                                    ),
                                ),
                                marker=dict(size=10),
                            )
                        )

                    fig_timeline2.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="No Impact",
                    )

                    fig_timeline2.update_layout(
                        title="Market Impact Over Time",
                        xaxis_title="Date",
                        yaxis_title="Market Impact (%)",
                        height=400,
                        template="plotly_dark",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_timeline2, use_container_width=True)

                except Exception as e:
                    logger.warning(
                        f"Error creating timeline visualization: {e}"
                    )

        except KeyError as e:
            error_msg = f"Missing data in results: {str(e)}"
            st.error(f"Error running scenarios: {error_msg}")
            logger.exception(f"Scenario analysis KeyError: {error_msg}")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            st.error(f"Error running scenarios: {error_msg}")
            logger.exception(f"Scenario analysis error: {error_msg}")


def _render_custom_scenario(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render custom scenario builder."""
    st.subheader("Custom Scenario Builder")

    # Scenario inputs
    scenario_name = st.text_input(
        "Scenario Name",
        key="custom_name",
        placeholder="e.g., Tech Recession 2024",
    )

    scenario_description = st.text_area(
        "Description",
        key="custom_description",
        placeholder="Describe the scenario...",
    )

    # Market impact
    market_impact = st.slider(
        "Market Impact (%)",
        min_value=-100.0,
        max_value=100.0,
        value=-20.0,
        step=1.0,
        key="custom_market",
        help="Overall market impact percentage",
    )

    # Sector impacts (simplified - would need sector mapping)
    st.markdown("### Sector Impacts (Optional)")
    st.info(
        "Sector-specific impacts can be added here. "
        "This is a simplified version."
    )

    # Asset-specific impacts
    st.markdown("### Asset-Specific Impacts (Optional)")
    asset_impacts_text = st.text_area(
        "Asset Impacts",
        key="custom_assets",
        placeholder=(
            "Format: TICKER:IMPACT%\nExample:\nAAPL:-30%\nMSFT:-25%"
        ),
        help="Enter one ticker:impact per line",
    )

    # Parse asset impacts
    asset_impacts = {}
    if asset_impacts_text:
        for line in asset_impacts_text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    ticker = parts[0].strip().upper()
                    try:
                        impact = (
                            float(parts[1].strip().replace("%", "")) / 100.0
                        )
                        asset_impacts[ticker] = impact
                    except ValueError:
                        pass

    # Create and run scenario
    if st.button("Create & Run Scenario", key="run_custom"):
        if not scenario_name:
            st.error("Please enter a scenario name.")
            return

        try:
            # Create custom scenario
            custom_scenario = create_custom_scenario(
                name=scenario_name,
                description=scenario_description,
                market_impact_pct=market_impact / 100.0,
                asset_impacts=asset_impacts,
            )

            # Validate scenario
            is_valid, error_msg = validate_scenario(custom_scenario)
            if not is_valid:
                st.error(f"Invalid scenario: {error_msg}")
                return

            # Run scenario
            with st.spinner("Running custom scenario..."):
                result = risk_service.run_custom_scenario(
                    portfolio_id=portfolio_id,
                    scenario=custom_scenario,
                )

                st.success("Custom scenario completed!")

                # Display results
                st.markdown("### Scenario Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Portfolio Impact",
                        f"{result['portfolio_impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Impact Value",
                        f"${result['portfolio_impact_value']:,.2f}",
                    )
                with col2:
                    st.metric(
                        "Worst Position",
                        result["worst_position"]["ticker"],
                        f"{result['worst_position']['impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Best Position",
                        result["best_position"]["ticker"],
                        f"{result['best_position']['impact_pct'] * 100:.2f}%",
                    )

        except ValidationError as e:
            st.error(f"Validation error: {str(e)}")
        except Exception as e:
            st.error(f"Error running custom scenario: {str(e)}")
            logger.exception("Custom scenario error")


def _render_scenario_chain(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render scenario chain builder."""
    st.subheader("Scenario Chain Builder")

    st.info(
        "Create a chain of scenarios to analyze cumulative portfolio "
        "impacts from multiple sequential events."
    )

    # Get all scenarios
    scenarios = get_all_scenarios()

    if not scenarios:
        st.warning("No scenarios available.")
        return

    # Chain name
    chain_name = st.text_input(
        "Chain Name",
        key="chain_name",
        placeholder="e.g., Double Crisis (2008 + 2020)",
    )

    chain_description = st.text_area(
        "Description",
        key="chain_description",
        placeholder="Describe the scenario chain...",
    )

    # Scenario selection for chain
    scenario_keys = list(scenarios.keys())
    scenario_display = [
        scenarios[key].name for key in scenario_keys
    ]

    selected_indices = st.multiselect(
        "Select Scenarios (in order)",
        options=range(len(scenario_keys)),
        format_func=lambda x: scenario_display[x],
        key="chain_scenarios",
        help="Select scenarios in the order they should be applied",
    )

    if not selected_indices:
        st.info("Please select at least one scenario.")
        return

    selected_keys = [scenario_keys[i] for i in selected_indices]
    selected_scenarios = [scenarios[key] for key in selected_keys]

    # Display chain
    if selected_scenarios:
        st.markdown("### Scenario Chain")
        for i, scenario in enumerate(selected_scenarios, 1):
            st.markdown(f"{i}. **{scenario.name}**")
            st.markdown(f"   Impact: {scenario.market_impact_pct * 100:.1f}%")

    # Run chain
    if st.button("Run Scenario Chain", key="run_chain"):
        if not chain_name:
            st.error("Please enter a chain name.")
            return

        try:
            # Create chain
            chain = create_scenario_chain(
                name=chain_name,
                description=chain_description,
                scenarios=selected_scenarios,
            )

            # Run chain
            with st.spinner("Running scenario chain..."):
                results = risk_service.run_scenario_chain(
                    portfolio_id=portfolio_id,
                    chain=chain,
                )

                st.success("Scenario chain completed!")

                # Display results
                st.markdown("### Chain Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Cumulative Impact",
                        f"{results['cumulative_impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Final Portfolio Value",
                        f"${results['final_portfolio_value']:,.2f}",
                    )
                with col2:
                    st.metric(
                        "Impact Value",
                        f"${results['cumulative_impact_value']:,.2f}",
                    )
                    st.metric(
                        "Worst Scenario",
                        results.get("worst_scenario", "N/A"),
                    )

                # Individual scenario results (table only, no charts)
                st.markdown("### Individual Scenario Results")
                import pandas as pd
                scenario_results = results["scenario_results"]

                results_data = []
                for r in scenario_results:
                    cum_pct = f"{r['cumulative_impact_pct'] * 100:.2f}%"
                    results_data.append({
                        "Scenario": r["scenario_name"],
                        "Impact %": f"{r['impact_pct'] * 100:.2f}%",
                        "Cumulative %": cum_pct,
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error running scenario chain: {str(e)}")
            logger.exception("Scenario chain error")


# Main entry point
if __name__ == "__main__":
    render_risk_analysis_page()
