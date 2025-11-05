"""Portfolio Optimization page."""

import logging
from datetime import date, timedelta
from typing import Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services.portfolio_service import PortfolioService
from services.optimization_service import OptimizationService
from services.analytics_service import AnalyticsService
from core.exceptions import CalculationError, InsufficientDataError
from streamlit_app.utils.chart_config import COLORS

logger = logging.getLogger(__name__)

# Method display names
METHOD_NAMES = {
    "equal_weight": "Equal Weight (1/N)",
    "mean_variance": "Mean-Variance (Markowitz)",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe Ratio",
    "max_return": "Maximum Return",
    "risk_parity": "Risk Parity",
    "kelly_criterion": "Kelly Criterion",
    "min_tracking_error": "Minimum Tracking Error",
    "max_alpha": "Maximum Alpha",
}


def render_optimization_page() -> None:
    """Render portfolio optimization page."""
    st.title("Portfolio Optimization")
    st.markdown(
        "Optimize your portfolio weights using various optimization methods."
    )
    
    # Initialize services
    portfolio_service = PortfolioService()
    optimization_service = OptimizationService()
    
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
        key="optimization_portfolio",
    )
    
    selected_portfolio = next(
        p for p in portfolios if p.name == selected_name
    )
    
    # Display current portfolio info
    with st.expander("Current Portfolio Information", expanded=False):
        positions = selected_portfolio.get_all_positions()
        
        if not positions:
            st.error("Portfolio has no positions.")
            return
        
        # Get current weights
        from services.data_service import DataService
        data_service = DataService()
        
        # Fetch current prices for all positions
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        prices = {}
        for ticker in tickers:
            current_price = data_service.fetch_current_price(ticker)
            if current_price:
                prices[ticker] = current_price
        
        # Add CASH price (always 1.0) if CASH position exists
        if any(pos.ticker == "CASH" for pos in positions):
            prices["CASH"] = 1.0
        
        # Calculate portfolio value
        current_value = selected_portfolio.calculate_current_value(prices)
        
        current_weights_data = []
        for pos in positions:
            if pos.ticker == "CASH":
                # Handle CASH position
                current_price = 1.0  # CASH is always 1.0
                pos_value = pos.shares  # CASH shares = dollar amount
            else:
                current_price = prices.get(pos.ticker)
                if not current_price:
                    continue
                pos_value = current_price * pos.shares
            
            weight = (pos_value / current_value) if current_value > 0 else 0.0
            current_weights_data.append({
                "Ticker": pos.ticker,
                "Shares": f"{pos.shares:,.2f}",
                "Price": f"${current_price:.2f}",
                "Value": f"${pos_value:,.2f}",
                "Weight": f"{weight:.2%}",
            })
        
        if current_weights_data:
            st.dataframe(
                pd.DataFrame(current_weights_data),
                use_container_width=True,
            )
    
    # Optimization parameters
    st.header("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        
        # Get saved dates if available, otherwise use defaults
        saved_start = st.session_state.get("optimization_start_date")
        saved_end = st.session_state.get("optimization_end_date")
        
        # Use saved dates as default if optimization was run,
        # otherwise use defaults
        start_date_value = saved_start if saved_start else default_start
        end_date_value = saved_end if saved_end else default_end
        
        start_date = st.date_input(
            "Start Date",
            value=start_date_value,
            max_value=date.today(),
            key="opt_start_date",
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=end_date_value,
            min_value=start_date,
            max_value=date.today(),
            key="opt_end_date",
        )
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    # Method selection
    available_methods = optimization_service.get_available_methods()
    method_display = [
        METHOD_NAMES.get(m, m.replace("_", " ").title())
        for m in available_methods
    ]
    
    selected_method_idx = st.selectbox(
        "Optimization Method",
        range(len(available_methods)),
        format_func=lambda x: method_display[x],
        key="opt_method",
    )
    
    selected_method = available_methods[selected_method_idx]
    
    # Method description
    method_descriptions = {
        "equal_weight": "Equal weight allocation (1/N) - simplest method",
        "mean_variance": "Markowitz mean-variance optimization",
        "min_variance": "Minimize portfolio variance",
        "max_sharpe": "Maximize Sharpe ratio (risk-adjusted return)",
        "max_return": "Maximize expected return",
        "risk_parity": "Equal risk contribution from each asset",
        "kelly_criterion": "Maximize long-term growth rate",
        "min_tracking_error": "Minimize tracking error vs benchmark",
        "max_alpha": "Maximize alpha (excess return) vs benchmark",
    }
    
    st.info(method_descriptions.get(selected_method, ""))
    
    # Benchmark selection (for visualization and methods that require it)
    benchmark_ticker = None
    benchmark_for_viz = None
    
    if selected_method in ["min_tracking_error", "max_alpha"]:
        # Required for these methods
        presets = ["SPY", "QQQ", "VTI", "DIA", "IWM"]
        benchmark_ticker = st.selectbox(
            "Benchmark Ticker (Required)",
            options=presets,
            index=0,
            key="opt_benchmark_required",
            help="Select benchmark ticker for this optimization method",
        )
        benchmark_for_viz = benchmark_ticker
    else:
        # Optional for visualization
        st.markdown("**Benchmark (Optional, for visualization)**")
        presets = ["None", "SPY", "QQQ", "VTI", "DIA", "IWM"]
        benchmark_choice = st.selectbox(
            "Benchmark",
            options=presets,
            index=0,
            key="opt_benchmark_optional",
            help="Select benchmark for comparison on charts",
        )
        if benchmark_choice != "None":
            benchmark_for_viz = benchmark_choice
    
    # Constraints
    st.subheader("Constraints (Optional)")
    
    constraints = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_min_weight = st.checkbox("Minimum Weight", key="opt_min_weight")
        if use_min_weight:
            min_weight = st.slider(
                "Min Weight %",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key="opt_min_weight_val",
            ) / 100.0
            constraints["min_weight"] = min_weight
    
    with col2:
        use_max_weight = st.checkbox("Maximum Weight", key="opt_max_weight")
        if use_max_weight:
            max_weight = st.slider(
                "Max Weight %",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
                key="opt_max_weight_val",
            ) / 100.0
            constraints["max_weight"] = max_weight
    
    long_only = st.checkbox(
        "Long Only (No Short Positions)",
        value=True,
        key="opt_long_only",
    )
    constraints["long_only"] = long_only
    
    # Kelly Criterion fraction
    if selected_method == "kelly_criterion":
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.25,
            max_value=1.0,
            value=0.5,
            step=0.25,
            key="opt_kelly_fraction",
            help="1.0 = Full Kelly, 0.5 = Half Kelly, 0.25 = Quarter Kelly",
        )
        constraints["kelly_fraction"] = kelly_fraction
    
    # Run optimization
    st.header("Run Optimization")
    
    if st.button("Optimize Portfolio", type="primary", use_container_width=True):
        with st.spinner("Running optimization... This may take a moment."):
            try:
                result = optimization_service.optimize_portfolio(
                    portfolio_id=selected_portfolio.id,
                    method=selected_method,
                    start_date=start_date,
                    end_date=end_date,
                    constraints=constraints if constraints else None,
                    benchmark_ticker=benchmark_ticker,
                )
                
                # Store result in session state
                st.session_state["optimization_result"] = result
                st.session_state["optimization_portfolio_id"] = (
                    selected_portfolio.id
                )
                st.session_state["optimization_start_date"] = start_date
                st.session_state["optimization_end_date"] = end_date
                st.session_state["optimization_benchmark"] = benchmark_for_viz
                
                st.success("Optimization completed successfully!")
                st.rerun()
            
            except InsufficientDataError as e:
                st.error(f"Insufficient data: {str(e)}")
            except CalculationError as e:
                st.error(f"Calculation error: {str(e)}")
            except Exception as e:
                logger.exception("Optimization failed")
                st.error(f"Optimization failed: {str(e)}")
    
    # Display results if available
    if "optimization_result" in st.session_state:
        result = st.session_state["optimization_result"]
        portfolio_id = st.session_state.get("optimization_portfolio_id")
        # CRITICAL: Use CURRENT dates from date_input, not saved ones!
        # This ensures metrics are calculated for the period user selected
        saved_benchmark = st.session_state.get("optimization_benchmark")
        
        if result.success:
            _display_optimization_results(
                result,
                portfolio_id,
                portfolio_service,
                optimization_service,
                start_date,  # Use CURRENT start_date from date_input
                end_date,    # Use CURRENT end_date from date_input
                constraints,
                saved_benchmark,
            )
        else:
            st.error(f"Optimization failed: {result.message}")


def _display_optimization_results(
    result,
    portfolio_id: str,
    portfolio_service: PortfolioService,
    optimization_service: OptimizationService,
    start_date: date,
    end_date: date,
    constraints: Dict,
    benchmark_for_viz: str = None,
) -> None:
    """Display optimization results."""
    st.header("Optimization Results")
    
    # Show date range info
    st.caption(
        f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to "
        f"{end_date.strftime('%Y-%m-%d')} "
        f"({(end_date - start_date).days} days)"
    )
    
    # Check if dates differ from optimization dates (if stored)
    saved_opt_start = st.session_state.get("optimization_start_date")
    saved_opt_end = st.session_state.get("optimization_end_date")
    if saved_opt_start and saved_opt_end:
        if saved_opt_start != start_date or saved_opt_end != end_date:
            st.info(
                f"ℹ️ **Note:** Metrics are calculated for the selected period "
                f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}), "
                f"which may differ from the optimization period "
                f"({saved_opt_start.strftime('%Y-%m-%d')} to {saved_opt_end.strftime('%Y-%m-%d')})."
            )
    
    # Comparison: Optimized vs Current Portfolio
    # Use EXACT same logic as portfolio_analysis.py overview tab
    st.subheader("Comparison: Optimized vs Current Portfolio")
    
    risk_free_rate = 0.0435
    
    # Get current portfolio returns for the SELECTED period
    current_returns = None
    try:
        analytics_service = AnalyticsService()
        current_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            start_date,
            end_date,
        )
        current_returns = current_analytics.get("portfolio_returns")
    except Exception as e:
        logger.warning(
            f"Could not calculate current portfolio returns: {e}"
        )
    
    # Calculate optimized portfolio returns for the SELECTED period
    optimized_returns = None
    try:
        # Get optimized portfolio historical returns (SAME logic as charts)
        from services.data_service import DataService
        data_service = DataService()
        
        optimal_weights_dict = result.get_weights_dict()
        tickers_list = result.tickers
        
        # Fetch historical prices and calculate returns
        asset_returns_df = pd.DataFrame()
        for ticker in tickers_list:
            if ticker == "CASH":
                dr = pd.bdate_range(
                    start=start_date, end=end_date, normalize=True
                )
                daily_return = (1 + risk_free_rate) ** (1.0 / 252) - 1
                cash_returns = pd.Series(
                    data=daily_return,
                    index=dr,
                    name=ticker,
                )
                asset_returns_df[ticker] = cash_returns
            else:
                try:
                    prices = data_service.fetch_historical_prices(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        save_to_db=False,
                    )
                    if not prices.empty:
                        if "Date" in prices.columns:
                            prices.set_index("Date", inplace=True)
                        prices.index = pd.to_datetime(
                            prices.index, errors="coerce"
                        )
                        prices.index = prices.index.tz_localize(None)
                        asset_returns = (
                            prices["Adjusted_Close"].pct_change().dropna()
                        )
                        asset_returns_df[ticker] = asset_returns
                except Exception:
                    pass
        
        if not asset_returns_df.empty:
            # Align all returns to common index
            asset_returns_df = asset_returns_df.sort_index()
            common_index = asset_returns_df.index
            
            # Calculate optimized portfolio returns: weighted sum
            optimized_returns = pd.Series(0.0, index=common_index)
            for ticker, weight in optimal_weights_dict.items():
                if ticker in asset_returns_df.columns:
                    optimized_returns += (
                        asset_returns_df[ticker] * weight
                    )
            
            optimized_returns = optimized_returns.dropna()
            
            # CRITICAL: Align optimized returns with current portfolio
            # to ensure same date range (SAME as charts)
            if (current_returns is not None and
                    not current_returns.empty):
                aligned_index = current_returns.index.intersection(
                    optimized_returns.index
                )
                optimized_returns = optimized_returns.reindex(aligned_index)
                current_returns = current_returns.reindex(aligned_index)
    except Exception as e:
        logger.warning(
            f"Could not calculate optimized portfolio returns: {e}"
        )
    
    # Calculate metrics using EXACT same logic as portfolio_analysis.py
    # overview tab (_render_overview_tab function)
    from core.analytics_engine.performance import (
        calculate_annualized_return,
    )
    from core.analytics_engine.risk_metrics import (
        calculate_volatility,
        calculate_max_drawdown,
    )
    from core.analytics_engine.ratios import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )
    
    # Calculate optimized portfolio metrics
    optimized_metrics = {}
    if optimized_returns is not None and not optimized_returns.empty:
        optimized_metrics["total_return"] = float(
            (1 + optimized_returns).prod() - 1
        )
        optimized_metrics["annualized_return"] = float(
            calculate_annualized_return(optimized_returns)
        )
        vol_result = calculate_volatility(optimized_returns)
        if isinstance(vol_result, dict):
            optimized_metrics["volatility"] = float(
                vol_result.get("annual", 0.0)
            )
        else:
            optimized_metrics["volatility"] = float(vol_result)
        optimized_metrics["sharpe_ratio"] = float(
            calculate_sharpe_ratio(
                optimized_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        optimized_metrics["sortino_ratio"] = float(
            calculate_sortino_ratio(
                optimized_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        dd_result = calculate_max_drawdown(optimized_returns)
        optimized_metrics["max_drawdown"] = float(
            dd_result[0] if isinstance(dd_result, tuple) else dd_result
        )
    else:
        # Fallback to optimization result
        optimized_metrics = {
            "total_return": 0.0,
            "annualized_return": float(result.expected_return or 0.0),
            "volatility": float(result.volatility or 0.0),
            "sharpe_ratio": float(result.sharpe_ratio or 0.0),
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Calculate current portfolio metrics (same logic)
    current_metrics = {}
    if current_returns is not None and not current_returns.empty:
        current_metrics["total_return"] = float(
            (1 + current_returns).prod() - 1
        )
        current_metrics["annualized_return"] = float(
            calculate_annualized_return(current_returns)
        )
        vol_result = calculate_volatility(current_returns)
        if isinstance(vol_result, dict):
            current_metrics["volatility"] = float(
                vol_result.get("annual", 0.0)
            )
        else:
            current_metrics["volatility"] = float(vol_result)
        current_metrics["sharpe_ratio"] = float(
            calculate_sharpe_ratio(
                current_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        current_metrics["sortino_ratio"] = float(
            calculate_sortino_ratio(
                current_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        dd_result = calculate_max_drawdown(current_returns)
        current_metrics["max_drawdown"] = float(
            dd_result[0] if isinstance(dd_result, tuple) else dd_result
        )
    else:
        current_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Display metrics using same component as overview
    from streamlit_app.components.metric_card_comparison import (
        render_metric_cards_row,
    )
    
    # Row 1: Total Return, Sharpe Ratio, Volatility, Max Drawdown
    # (same format as overview, optimized vs current)
    metrics_row1 = [
        {
            "label": "Total Return",
            "portfolio_value": optimized_metrics.get("total_return", 0),
            "benchmark_value": current_metrics.get("total_return"),
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "Sharpe Ratio",
            "portfolio_value": optimized_metrics.get("sharpe_ratio", 0),
            "benchmark_value": current_metrics.get("sharpe_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Volatility",
            "portfolio_value": optimized_metrics.get("volatility", 0),
            "benchmark_value": current_metrics.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": optimized_metrics.get("max_drawdown", 0),
            "benchmark_value": current_metrics.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
        },
    ]
    render_metric_cards_row(metrics_row1, columns_per_row=4)
    
    # Current vs Optimal weights comparison
    st.subheader("Current vs Optimal Allocation")
    
    # Get current weights
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    from services.data_service import DataService
    data_service = DataService()
    
    # Fetch current prices
    positions = portfolio.get_all_positions()
    tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
    prices = {}
    for ticker in tickers:
        current_price = data_service.fetch_current_price(ticker)
        if current_price:
            prices[ticker] = current_price
    
    # Add CASH price (always 1.0) if CASH position exists
    if any(pos.ticker == "CASH" for pos in positions):
        prices["CASH"] = 1.0
    
    # Calculate portfolio value
    current_value = portfolio.calculate_current_value(prices)
    
    current_weights = {}
    for pos in positions:
        if pos.ticker in result.tickers:
            if pos.ticker == "CASH":
                pos_value = pos.shares  # CASH shares = dollar amount
            else:
                current_price = prices.get(pos.ticker)
                if not current_price:
                    continue
                pos_value = current_price * pos.shares
            
            current_weights[pos.ticker] = (
                pos_value / current_value if current_value > 0 else 0.0
            )
    
    # Create comparison DataFrame
    comparison_data = []
    for ticker in result.tickers:
        current_w = current_weights.get(ticker, 0.0)
        optimal_w = result.get_weights_dict().get(ticker, 0.0)
        diff = optimal_w - current_w
        
        comparison_data.append({
            "Ticker": ticker,
            "Current Weight": f"{current_w:.2%}",
            "Optimal Weight": f"{optimal_w:.2%}",
            "Difference": f"{diff:+.2%}",
        })
    
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )
    
    # Visual comparison
    fig = go.Figure()
    
    tickers = result.tickers
    current_weights_array = [
        current_weights.get(t, 0.0) for t in tickers
    ]
    optimal_weights_array = [result.get_weights_dict()[t] for t in tickers]
    
    fig.add_trace(
        go.Bar(
            name="Current",
            x=tickers,
            y=[w * 100 for w in current_weights_array],
            marker_color="#BF9FFB",
        )
    )
    
    fig.add_trace(
        go.Bar(
            name="Optimal",
            x=tickers,
            y=[w * 100 for w in optimal_weights_array],
            marker_color="#90BFF9",
        )
    )
    
    fig.update_layout(
        title="Current vs Optimal Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        barmode="group",
        height=400,
        template="plotly_dark",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade list
    st.subheader("Trade List")
    
    try:
        trades = optimization_service.generate_trade_list(
            portfolio_id, result
        )
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Format columns
            trades_df["Weight Change"] = (
                trades_df["optimal_weight"]
                - trades_df["current_weight"]
            ).apply(lambda x: f"{x:+.2%}")
            
            trades_df["Shares"] = trades_df["shares"].apply(
                lambda x: f"{x:,.2f}"
            )
            trades_df["Value"] = trades_df["value"].apply(
                lambda x: f"${x:,.2f}"
            )
            
            display_df = trades_df[
                ["ticker", "action", "Shares", "Value", "Weight Change"]
            ].rename(
                columns={
                    "ticker": "Ticker",
                    "action": "Action",
                }
            )
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            total_trade_value = trades_df["value"].sum()
            st.info(f"Total Trade Value: ${total_trade_value:,.2f}")
        else:
            st.info("No trades required - portfolio is already optimal.")
    
    except Exception as e:
        logger.exception("Error generating trade list")
        st.warning(f"Could not generate trade list: {str(e)}")
    
    # Performance Charts Section
    st.subheader("Performance Charts")
    
    # Initialize variables for Efficient Frontier
    current_analytics = None
    benchmark_returns = None
    
    # Get historical data for charts
    try:
        analytics_service = AnalyticsService()
        
        # Get current portfolio returns (with optional benchmark)
        current_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            start_date,
            end_date,
            benchmark_ticker=benchmark_for_viz if benchmark_for_viz else None,
        )
        current_returns = current_analytics.get("portfolio_returns")
        current_values = current_analytics.get("portfolio_values")
        benchmark_returns = current_analytics.get("benchmark_returns")
        
        # Calculate optimized portfolio historical returns
        # Get asset returns for the period
        from services.data_service import DataService
        data_service = DataService()
        
        optimal_weights_dict = result.get_weights_dict()
        tickers_list = result.tickers
        
        # Fetch historical prices for all assets
        asset_returns_df = pd.DataFrame()
        for ticker in tickers_list:
            if ticker == "CASH":
                # CASH returns = risk-free rate / periods
                dr = pd.bdate_range(start=start_date, end=end_date, normalize=True)
                daily_return = (1 + 0.0435) ** (1.0 / 252) - 1
                cash_returns = pd.Series(
                    data=daily_return,
                    index=dr,
                    name=ticker,
                )
                asset_returns_df[ticker] = cash_returns
            else:
                try:
                    prices = data_service.fetch_historical_prices(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        save_to_db=False,
                    )
                    if not prices.empty:
                        if "Date" in prices.columns:
                            prices.set_index("Date", inplace=True)
                        prices.index = pd.to_datetime(prices.index, errors="coerce")
                        prices.index = prices.index.tz_localize(None)
                        
                        asset_returns = prices["Adjusted_Close"].pct_change().dropna()
                        asset_returns_df[ticker] = asset_returns
                except Exception as e:
                    logger.warning(f"Could not fetch returns for {ticker}: {e}")
        
        # Align all returns to common index
        if not asset_returns_df.empty:
            asset_returns_df = asset_returns_df.sort_index()
            common_index = asset_returns_df.index
            
            # Calculate optimized portfolio returns: weighted sum
            optimized_returns = pd.Series(0.0, index=common_index)
            for ticker, weight in optimal_weights_dict.items():
                if ticker in asset_returns_df.columns:
                    optimized_returns += asset_returns_df[ticker] * weight
            
            optimized_returns = optimized_returns.dropna()
            
            # Calculate optimized portfolio values (cumulative)
            optimized_values = (1 + optimized_returns).cumprod()
            if current_values is not None and not current_values.empty:
                initial_value = float(current_values.iloc[0])
                optimized_values = optimized_values * initial_value
            else:
                optimized_values = optimized_values * 10000  # Default $10k
            
            # Align current and optimized returns
            if current_returns is not None and not current_returns.empty:
                aligned_index = current_returns.index.intersection(
                    optimized_returns.index
                )
                current_aligned = current_returns.reindex(aligned_index)
                optimized_aligned = optimized_returns.reindex(aligned_index)
                benchmark_aligned = None
                if benchmark_returns is not None and not benchmark_returns.empty:
                    benchmark_aligned = benchmark_returns.reindex(aligned_index)
            else:
                current_aligned = None
                optimized_aligned = optimized_returns
                benchmark_aligned = benchmark_returns if benchmark_returns is not None and not benchmark_returns.empty else None
            
            # 1. Returns Chart
            st.markdown("**Portfolio Returns Comparison**")
            fig_returns = go.Figure()
            
            if current_aligned is not None and not current_aligned.empty:
                current_cumulative = (1 + current_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=current_cumulative.index,
                        y=current_cumulative * 100,
                        mode="lines",
                        name="Current Portfolio",
                        line=dict(
                            color=COLORS["primary"], width=2
                        ),  # Фиолетовый
                    )
                )
            
            if optimized_aligned is not None and not optimized_aligned.empty:
                optimized_cumulative = (1 + optimized_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=optimized_cumulative.index,
                        y=optimized_cumulative * 100,
                        mode="lines",
                        name="Optimized Portfolio",
                        line=dict(
                            color=COLORS["success"], width=2
                        ),  # Зеленый
                    )
                )
            
            if (
                benchmark_aligned is not None
                and not benchmark_aligned.empty
                and benchmark_for_viz
            ):
                benchmark_cumulative = (1 + benchmark_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative * 100,
                        mode="lines",
                        name=f"Benchmark ({benchmark_for_viz})",
                        line=dict(
                            color=COLORS["secondary"], width=2
                        ),  # Синий
                    )
                )
            
            fig_returns.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # 2. Drawdown Chart
            st.markdown("**Drawdown Comparison**")
            fig_drawdown = go.Figure()
            
            # Drawdown: оптимизированный - красное заполнение
            if optimized_values is not None and not optimized_values.empty:
                optimized_peak = optimized_values.expanding().max()
                optimized_dd = (
                    (optimized_values - optimized_peak) / optimized_peak * 100
                )
                fig_drawdown.add_trace(
                    go.Scatter(
                        x=optimized_dd.index,
                        y=optimized_dd,
                        mode="lines",
                        name="Optimized Portfolio",
                        line=dict(
                            color=COLORS["danger"], width=2
                        ),  # Красный
                        fill="tozeroy",
                        fillcolor="rgba(239, 85, 59, 0.3)",  # Красное заполнение
                    )
                )
            
            # Drawdown: текущий - оранжевая линия
            if current_values is not None and not current_values.empty:
                current_peak = current_values.expanding().max()
                current_dd = (current_values - current_peak) / current_peak * 100
                fig_drawdown.add_trace(
                    go.Scatter(
                        x=current_dd.index,
                        y=current_dd,
                        mode="lines",
                        name="Current Portfolio",
                        line=dict(
                            color=COLORS["warning"], width=2
                        ),  # Оранжевый
                    )
                )
            
            # Drawdown: бенчмарк - синяя линия
            if (
                benchmark_returns is not None
                and not benchmark_returns.empty
                and benchmark_for_viz
            ):
                if benchmark_aligned is not None:
                    benchmark_values = (1 + benchmark_aligned).cumprod()
                    if current_values is not None and not current_values.empty:
                        initial_value = float(current_values.iloc[0])
                        benchmark_values = benchmark_values * initial_value
                    else:
                        benchmark_values = benchmark_values * 10000
                    
                    benchmark_peak = benchmark_values.expanding().max()
                    benchmark_dd = (
                        (benchmark_values - benchmark_peak) / benchmark_peak * 100
                    )
                    fig_drawdown.add_trace(
                        go.Scatter(
                            x=benchmark_dd.index,
                            y=benchmark_dd,
                            mode="lines",
                            name=f"Benchmark ({benchmark_for_viz})",
                            line=dict(
                                color=COLORS["secondary"], width=2
                            ),  # Синий
                        )
                    )
            
            fig_drawdown.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig_drawdown, use_container_width=True)
            
    except Exception as e:
        logger.exception("Error generating performance charts")
        st.warning(f"Could not generate performance charts: {str(e)}")
    
    # Efficient Frontier
    st.subheader("Efficient Frontier")
    if st.checkbox("Show Efficient Frontier", value=True, key="show_frontier"):
        with st.spinner("Generating efficient frontier..."):
            try:
                frontier_data = optimization_service.generate_efficient_frontier(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date,
                    n_points=150,  # More points for smoother curve
                    constraints=constraints if constraints else None,
                )
                
                # Plot frontier
                fig = go.Figure()
                
                # Get frontier points
                volatilities_list = frontier_data["volatilities"]
                returns_list = frontier_data["returns"]
                
                if not volatilities_list or not returns_list:
                    st.warning("No frontier data available")
                    return
                
                # Convert to numpy arrays for easier manipulation
                import numpy as np
                vols = np.array(volatilities_list)
                rets = np.array(returns_list)
                
                # Filter only efficient part (upper part of curve)
                # Efficient frontier: for each volatility, take max return
                # Sort by volatility
                sorted_indices = np.argsort(vols)
                sorted_vols = vols[sorted_indices]
                sorted_rets = rets[sorted_indices]
                
                # Filter to keep only efficient part (non-decreasing returns)
                # Start from min volatility, keep points where return >= previous
                efficient_indices = [0]
                for i in range(1, len(sorted_vols)):
                    if sorted_rets[i] >= sorted_rets[efficient_indices[-1]]:
                        efficient_indices.append(i)
                    else:
                        # Check if this point has higher return than any previous
                        # at same or lower volatility (shouldn't happen, but safety)
                        prev_idx = efficient_indices[-1]
                        if sorted_rets[i] > sorted_rets[prev_idx]:
                            efficient_indices.append(i)
                
                efficient_vols = sorted_vols[efficient_indices]
                efficient_rets = sorted_rets[efficient_indices]
                
                # Convert to percentages for display
                x_plot = efficient_vols * 100
                y_plot = efficient_rets * 100
                
                # Add efficient frontier line with smooth interpolation
                fig.add_trace(
                    go.Scatter(
                        x=x_plot,
                        y=y_plot,
                        mode="lines",
                        name="Efficient Frontier",
                        line=dict(
                            color=COLORS["primary"],
                            width=3,
                            smoothing=1.0,  # Smooth the line
                        ),
                        showlegend=True,
                        hovertemplate=(
                            "Volatility: %{x:.2f}%<br>"
                            "Return: %{y:.2f}%<extra></extra>"
                        ),
                    )
                )
                
                # Get tangency (max Sharpe) and min variance portfolios
                tangency_portfolio = frontier_data.get("tangency_portfolio")
                min_var_portfolio = frontier_data.get("min_variance_portfolio")
                
                # Max Sharpe Ratio point (tangency portfolio)
                if tangency_portfolio:
                    tang_vol = tangency_portfolio.get("volatility")
                    tang_ret = tangency_portfolio.get("expected_return")
                    if tang_vol is not None and tang_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[float(tang_vol) * 100],
                                y=[float(tang_ret) * 100],
                                mode="markers",
                                name="Max Sharpe Ratio",
                                marker=dict(
                                    color="#FFD700",  # Gold
                                    size=16,
                                    symbol="triangle-up",
                                    line=dict(color="#000000", width=2),
                                ),
                                hovertemplate=(
                                    "Max Sharpe Ratio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Min Variance point
                if min_var_portfolio:
                    min_var_vol = min_var_portfolio.get("volatility")
                    min_var_ret = min_var_portfolio.get("expected_return")
                    if min_var_vol is not None and min_var_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[float(min_var_vol) * 100],
                                y=[float(min_var_ret) * 100],
                                mode="markers",
                                name="Min Volatility",
                                marker=dict(
                                    color="#4CAF50",  # Green
                                    size=16,
                                    symbol="triangle-down",
                                    line=dict(color="#000000", width=2),
                                ),
                                hovertemplate=(
                                    "Min Volatility<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Optimized portfolio - use metrics from optimized_returns
                if optimized_returns is not None and not optimized_returns.empty:
                    opt_vol = optimized_metrics.get("volatility", 0.0)
                    opt_ret = optimized_metrics.get("annualized_return", 0.0)
                    if opt_vol > 0 and opt_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[opt_vol * 100],
                                y=[opt_ret * 100],
                                mode="markers",
                                name="Optimized Portfolio",
                                marker=dict(
                                    color=COLORS["success"],  # Green
                                    size=18,
                                    symbol="star",
                                    line=dict(color="#FFFFFF", width=2),
                                ),
                                hovertemplate=(
                                    "Optimized Portfolio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Current portfolio - use metrics from current_returns
                if current_returns is not None and not current_returns.empty:
                    curr_vol = current_metrics.get("volatility", 0.0)
                    curr_ret = current_metrics.get("annualized_return", 0.0)
                    if curr_vol > 0 and curr_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[curr_vol * 100],
                                y=[curr_ret * 100],
                                mode="markers",
                                name="Current Portfolio",
                                marker=dict(
                                    color=COLORS["primary"],  # Purple
                                    size=18,
                                    symbol="star",
                                    line=dict(color="#FFFFFF", width=2),
                                ),
                                hovertemplate=(
                                    "Current Portfolio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Benchmark
                if (
                    benchmark_returns is not None
                    and not benchmark_returns.empty
                    and benchmark_for_viz
                ):
                    try:
                        from core.analytics_engine.performance import (
                            calculate_annualized_return,
                        )
                        from core.analytics_engine.risk_metrics import (
                            calculate_volatility,
                        )
                        
                        bench_ret = calculate_annualized_return(benchmark_returns)
                        bench_vol_dict = calculate_volatility(benchmark_returns)
                        bench_vol = (
                            bench_vol_dict.get("annual", 0.0)
                            if isinstance(bench_vol_dict, dict)
                            else bench_vol_dict
                        )
                        
                        if bench_vol > 0 and bench_ret is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=[float(bench_vol) * 100],
                                    y=[float(bench_ret) * 100],
                                    mode="markers",
                                    name=f"Benchmark ({benchmark_for_viz})",
                                    marker=dict(
                                        color=COLORS["secondary"],  # Blue
                                        size=14,
                                        symbol="diamond",
                                        line=dict(color="#FFFFFF", width=2),
                                    ),
                                    hovertemplate=(
                                        f"Benchmark ({benchmark_for_viz})<br>"
                                        "Volatility: %{x:.2f}%<br>"
                                        "Return: %{y:.2f}%<extra></extra>"
                                    ),
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not add benchmark to frontier: {e}"
                        )
                
                # Update layout with better styling
                fig.update_layout(
                    title=dict(
                        text="Efficient Frontier",
                        font=dict(size=20),
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Volatility (Annualized, %)",
                            font=dict(size=14),
                        ),
                        gridcolor="rgba(128, 128, 128, 0.3)",
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Expected Return (Annualized, %)",
                            font=dict(size=14),
                        ),
                        gridcolor="rgba(128, 128, 128, 0.3)",
                    ),
                    height=600,
                    template="plotly_dark",
                    hovermode="closest",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0, 0, 0, 0.5)",
                        bordercolor="rgba(255, 255, 255, 0.3)",
                        borderwidth=1,
                    ),
                    margin=dict(l=60, r=20, t=60, b=60),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                logger.exception("Error generating efficient frontier")
                st.warning(f"Could not generate efficient frontier: {str(e)}")
    
    # Apply optimization
    st.subheader("Apply Optimization")
    
    st.warning(
        "⚠️ Applying optimization will update your portfolio weights. "
        "This action cannot be undone."
    )
    
    if st.button("Apply Optimization to Portfolio", type="primary"):
        # TODO: Implement apply optimization
        st.info("Apply optimization feature will be implemented soon.")


if __name__ == "__main__":
    render_optimization_page()

