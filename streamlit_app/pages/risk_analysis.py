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
from streamlit_app.utils.chart_config import COLORS

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

                if comparison_data:
                    st.markdown("### VaR Methods Comparison")
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)

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

                # Spaghetti chart if requested
                if show_paths and results.get("simulated_paths"):
                    st.markdown("### Simulation Paths (Spaghetti Chart)")
                    paths = results["simulated_paths"]
                    time_horizon = results["time_horizon"]

                    fig_paths = go.Figure()

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

                    # Add current value line
                    fig_paths.add_hline(
                        y=initial_value,
                        line_dash="dash",
                        line_color="red",
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
                            line=dict(width=2, color="yellow"),
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
                    title="Portfolio Impact by Historical Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=400,
                    template="plotly_dark",
                )  # noqa: E501

                st.plotly_chart(fig, use_container_width=True)

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
