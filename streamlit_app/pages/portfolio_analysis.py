"""Portfolio Analysis page - Full implementation according to specification."""

import json
import logging
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from services.portfolio_service import PortfolioService
from services.analytics_service import AnalyticsService
from services.report_service import ReportService
from core.analytics_engine.chart_data import (
    get_cumulative_returns_data,
    get_return_distribution_data,
    get_monthly_heatmap_data,
    get_rolling_sharpe_data,
    get_rolling_beta_alpha_data,
    get_underwater_plot_data,
    get_yearly_returns_data,
    get_period_returns_comparison_data,
    get_three_month_rolling_periods_data,
    get_seasonal_analysis_data,
    get_monthly_active_returns_data,
    get_win_rate_statistics_data,
    get_outlier_analysis_data,
    get_statistical_tests_data,
    get_qq_plot_data,
    get_capture_ratio_data,
    get_risk_return_scatter_data,
    get_drawdown_periods_data,
    get_drawdown_recovery_data,
)
from core.analytics_engine.advanced_metrics import (
    calculate_expected_returns,
    calculate_common_performance_periods,
    calculate_probabilistic_sharpe_ratio,
    calculate_smart_sharpe,
    calculate_smart_sortino,
    calculate_kelly_criterion,
    calculate_risk_of_ruin,
)
from streamlit_app.components.charts import (
    plot_cumulative_returns,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_rolling_sharpe,
    plot_rolling_beta_alpha,
    plot_underwater,
    plot_yearly_returns,
    plot_asset_allocation,
    plot_sector_allocation,
    plot_active_returns_area,
    plot_period_returns_bar,
    plot_qq_plot,
    plot_return_quantiles_box,
    plot_seasonal_bar,
    plot_outlier_scatter,
    plot_rolling_win_rate,
    plot_capture_ratio,
    plot_risk_return_scatter,
    plot_drawdown_periods,
    plot_drawdown_recovery,
)
from streamlit_app.components.position_table import render_position_table
from streamlit_app.components.metric_card_comparison import render_metric_cards_row
from streamlit_app.components.comparison_table import render_comparison_table

logger = logging.getLogger(__name__)


def show():
    """Render Portfolio Analysis page."""
    st.title("Portfolio Analysis")

    # Services
    portfolio_service = PortfolioService()
    analytics_service = AnalyticsService()

    # Get portfolios
    portfolios = portfolio_service.list_portfolios()

    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return

    # === ANALYSIS PARAMETERS SECTION ===
    st.subheader("Analysis Parameters")

    # Row 1: Start Date (left) + End Date (right)
    col1, col2 = st.columns([2, 2])

    portfolio_names = {p.name: p.id for p in portfolios}
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=default_end,
            key="start_date_input",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=start_date,
            max_value=default_end,
            key="end_date_input",
        )

    # Row 2: Portfolio (left) + Comparison selector (right)
    col1, col2 = st.columns([2, 2])

    with col1:
        selected_name = st.selectbox(
            "Portfolio",
            options=list(portfolio_names.keys()),
            key="portfolio_analysis_selector",
        )
        portfolio_id = portfolio_names[selected_name]

    with col2:
        st.markdown("**Comparison**")
        cmp_type = st.radio(
            "",
            options=["None", "Index ETF", "Portfolio"],
            horizontal=True,
            label_visibility="collapsed",
            key="cmp_type_radio",
        )
        comparison_type = None
        comparison_value = None
        if cmp_type == "Index ETF":
            presets = ["SPY", "QQQ", "VTI", "DIA", "IWM"]
            comparison_value = st.selectbox("Index ETF", options=presets, key="cmp_etf_select")
            comparison_type = "ticker"
        elif cmp_type == "Portfolio":
            other_names = [name for name in portfolio_names.keys() if name != selected_name]
            if other_names:
                sel_other = st.selectbox("Portfolio", options=other_names, key="cmp_portfolio_select")
                comparison_value = portfolio_names[sel_other]
                comparison_type = "portfolio"
            else:
                st.info("No other portfolios available for comparison")
        else:
            st.caption("No comparison selected")
        benchmark_ticker = None

    # Row 3: Risk-Free Rate + Buttons
    col1, col2, col3 = st.columns([2, 3, 3])

    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.35,
            step=0.01,
            key="risk_free_rate_input",
        ) / 100

    with col2:
        if st.button("📈 Calculate Metrics", type="primary", use_container_width=True):
            with st.spinner("Calculating analytics..."):
                try:
                    result = analytics_service.calculate_portfolio_metrics(
                        portfolio_id=portfolio_id,
                        benchmark_ticker=benchmark_ticker,
                        comparison_type=comparison_type,
                        comparison_value=comparison_value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    st.session_state.portfolio_analytics = result
                    st.session_state.portfolio_id = portfolio_id
                    st.session_state.selected_name = selected_name
                    st.success("Analytics calculated!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error calculating analytics: {e}")
                    st.error(f"Error: {str(e)}")

    with col3:
        if st.button("🔄 Update Prices", type="secondary", use_container_width=True):
            st.info("Price update functionality coming soon...")

    st.markdown("---")

    # Check if analytics available
    if "portfolio_analytics" not in st.session_state:
        st.info("👆 Click 'Calculate Metrics' to start analysis")
        return

    analytics = st.session_state.portfolio_analytics
    portfolio_id = st.session_state.get("portfolio_id", portfolio_id)
    selected_name = st.session_state.get("selected_name", selected_name)
    
    # Extract data
    portfolio_returns = analytics.get("portfolio_returns")
    # Use comparison series as benchmark for rendering
    comparison_returns = analytics.get("comparison_returns")
    benchmark_returns = comparison_returns if comparison_returns is not None else analytics.get("benchmark_returns")
    portfolio_values = analytics.get("portfolio_values")
    perf = analytics.get("performance", {})
    risk = analytics.get("risk", {})
    ratios = analytics.get("ratios", {})
    market = analytics.get("market", {})
    
    # Benchmark metrics will be calculated in _render_overview_tab
    
    # Get portfolio positions
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
        positions = portfolio.get_all_positions() if portfolio else []
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        positions = []
    
    # === MAIN TABS ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Performance",
        "Risk",
        "Assets & Correlations",
        "Export & Reports"
    ])
    
    # === TAB 1: OVERVIEW ===
    with tab1:
        _render_overview_tab(
            perf, risk, ratios, market,
            portfolio_returns, benchmark_returns, portfolio_values,
            positions, start_date, end_date, risk_free_rate
        )
    
    # === TAB 2: PERFORMANCE ===
    with tab2:
        _render_performance_tab(
            perf, portfolio_returns, benchmark_returns, portfolio_values,
            risk_free_rate, start_date, end_date
        )
    
    # === TAB 3: RISK ===
    with tab3:
        _render_risk_tab(
            risk, ratios, market,
            portfolio_returns, benchmark_returns, portfolio_values,
            risk_free_rate, start_date, end_date
        )
    
    # === TAB 4: ASSETS & CORRELATIONS ===
    with tab4:
        _render_assets_tab(
            positions, portfolio_returns, benchmark_returns,
            portfolio_id, portfolio_service
        )
    
    # === TAB 5: EXPORT & REPORTS ===
    with tab5:
        _render_export_tab(
            analytics, selected_name, portfolio_id
        )


def _render_overview_tab(
    perf, risk, ratios, market,
    portfolio_returns, benchmark_returns, portfolio_values,
    positions, start_date, end_date, risk_free_rate=0.0435
):
    """Render Overview tab content."""
    # Section 1.1: Key Performance Cards
    st.subheader("Key Performance Metrics")
    
    # Prepare benchmark metrics for cards (reuse calc below when possible)
    bm_for_cards = {}
    common_idx = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align by overlapping dates only, no zero-filling to avoid distortions
        try:
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_bench_cards = benchmark_returns.loc[common_idx]
        except Exception:
            aligned_bench_cards = benchmark_returns.reindex(
                portfolio_returns.index, method="ffill"
            ).dropna()
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.risk_metrics import (
            calculate_volatility,
            calculate_max_drawdown,
        )
        from core.analytics_engine.ratios import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )
        max_dd_cards = calculate_max_drawdown(aligned_bench_cards)
        bm_for_cards = {
            "total_return": float((1 + aligned_bench_cards).prod() - 1),
            "annualized_return": float(calculate_annualized_return(aligned_bench_cards)),
            "volatility": float(calculate_volatility(aligned_bench_cards).get("annual", 0.0)),
            "max_drawdown": float(max_dd_cards[0] if isinstance(max_dd_cards, tuple) else max_dd_cards),
            "sharpe_ratio": float(calculate_sharpe_ratio(aligned_bench_cards, risk_free_rate=risk_free_rate) or 0),
            "sortino_ratio": float(calculate_sortino_ratio(aligned_bench_cards, risk_free_rate=risk_free_rate) or 0),
        }

    # Prepare portfolio metrics for cards/table early
    portfolio_metrics_flat = {
        "total_return": perf.get("total_return", 0),
        "cagr": perf.get("cagr", perf.get("annualized_return", 0)),
        "annualized_return": perf.get("annualized_return", 0),
        "volatility": (risk.get("volatility", {}) or {}).get("annual", risk.get("volatility", 0)) if isinstance(risk.get("volatility", 0), dict) else risk.get("volatility", 0),
        "sharpe_ratio": ratios.get("sharpe_ratio", 0),
        "sortino_ratio": ratios.get("sortino_ratio", 0),
        "max_drawdown": risk.get("max_drawdown", 0),
        "calmar_ratio": ratios.get("calmar_ratio", 0),
        "beta": market.get("beta", 0),
        "alpha": market.get("alpha", 0),
        "up_capture": market.get("up_capture", None),
        "down_capture": market.get("down_capture", None),
    }

    # Fallbacks if backend returned zeros: compute from portfolio_returns
    try:
        if portfolio_returns is not None and not portfolio_returns.empty:
            from core.analytics_engine.performance import calculate_annualized_return
            from core.analytics_engine.ratios import calculate_sharpe_ratio
            if abs(portfolio_metrics_flat.get("total_return", 0)) < 1e-8:
                portfolio_metrics_flat["total_return"] = float((1 + portfolio_returns).prod() - 1)
            if abs(portfolio_metrics_flat.get("annualized_return", 0)) < 1e-8:
                portfolio_metrics_flat["annualized_return"] = float(
                    calculate_annualized_return(portfolio_returns)
                )
            if abs(portfolio_metrics_flat.get("sharpe_ratio", 0)) < 1e-6:
                portfolio_metrics_flat["sharpe_ratio"] = float(
                    calculate_sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate) or 0
                )
            # Ensure CAGR equals annualized return for consistency in table
            portfolio_metrics_flat["cagr"] = portfolio_metrics_flat.get("annualized_return", portfolio_metrics_flat.get("cagr", 0))
            # Volatility & MaxDD fallback from returns for strict equality with benchmark
            try:
                from core.analytics_engine.risk_metrics import (
                    calculate_volatility as _calc_vol,
                    calculate_max_drawdown as _calc_dd,
                )
                # If есть бенчмарк — пересчитываем по общему диапазону дат
                series_for_calc = (
                    portfolio_returns.loc[common_idx]
                    if common_idx is not None and len(common_idx) > 1
                    else portfolio_returns
                )
                vol = _calc_vol(series_for_calc)
                if isinstance(vol, dict):
                    vol_val = float(vol.get("annual", 0.0))
                else:
                    vol_val = float(vol)
                if vol_val == 0 or portfolio_metrics_flat.get("volatility") in (0, None):
                    portfolio_metrics_flat["volatility"] = vol_val
                dd = _calc_dd(series_for_calc)
                dd_val = float(dd[0] if isinstance(dd, tuple) else dd)
                if dd_val != 0 and portfolio_metrics_flat.get("max_drawdown", 0) == 0:
                    portfolio_metrics_flat["max_drawdown"] = dd_val
            except Exception:
                pass
    except Exception:
        pass

    # Row 1: Total Return, CAGR, Volatility, Max Drawdown
    metrics_row1 = [
        {
            "label": "Total Return",
            "portfolio_value": portfolio_metrics_flat.get("total_return", 0),
            "benchmark_value": bm_for_cards.get("total_return"),
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "CAGR",
            "portfolio_value": portfolio_metrics_flat.get("annualized_return", 0),
            "benchmark_value": bm_for_cards.get("annualized_return"),
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "Volatility",
            "portfolio_value": portfolio_metrics_flat.get("volatility", 0),
            "benchmark_value": bm_for_cards.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": risk.get("max_drawdown", 0),
            "benchmark_value": bm_for_cards.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
        },
    ]
    render_metric_cards_row(metrics_row1, columns_per_row=4)

    # Row 2: Sharpe Ratio, Sortino Ratio, Beta, Alpha
    st.markdown("---")
    metrics_row2 = [
        {
            "label": "Sharpe Ratio",
            "portfolio_value": portfolio_metrics_flat.get("sharpe_ratio", 0),
            "benchmark_value": bm_for_cards.get("sharpe_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "benchmark_value": bm_for_cards.get("sortino_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Beta",
            "portfolio_value": market.get("beta", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "ratio",
            "higher_is_better": None,  # Special: closer to 1.0 is better
        },
        {
            "label": "Alpha",
            "portfolio_value": market.get("alpha", 0),
            "benchmark_value": 0.0,
            "format": "percent",
            "higher_is_better": True,
        },
    ]
    render_metric_cards_row(metrics_row2, columns_per_row=4)
    
    # Section 1.2: Portfolio Dynamics (3 graphs stacked)
    st.markdown("---")
    st.subheader("Portfolio Performance")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Enforce selected period on series for plotting (normalize tz)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        try:
            pr_index = portfolio_returns.index.tz_localize(None)
        except Exception:
            pr_index = portfolio_returns.index
        pr = portfolio_returns[(pr_index >= start_ts) & (pr_index < end_ts)]
        br = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                br_index = benchmark_returns.index.tz_localize(None)
            except Exception:
                br_index = benchmark_returns.index
            br = benchmark_returns[(br_index >= start_ts) & (br_index < end_ts)]
        # Cumulative Returns
        cum_data = get_cumulative_returns_data(pr, br)
        if cum_data:
            fig = plot_cumulative_returns(cum_data)
            st.plotly_chart(fig, use_container_width=True, key="overview_cumulative_returns")
        
        # Underwater Plot (Drawdowns)
        if portfolio_values is not None and not portfolio_values.empty:
            # Calculate benchmark values from benchmark returns if available
            benchmark_values = None
            if br is not None and not br.empty:
                aligned_bench = br.reindex(
                    portfolio_values.index, method="ffill"
                ).fillna(0)
                # Start from same initial value as portfolio
                initial_value = float(portfolio_values.iloc[0])
                benchmark_values = (1 + aligned_bench).cumprod() * initial_value
            
            underwater_data = get_underwater_plot_data(
                portfolio_values, benchmark_values
            )
            if underwater_data:
                fig = plot_underwater(underwater_data)
                st.plotly_chart(fig, use_container_width=True, key="overview_underwater")
        
        # Daily Returns (bar chart) - без бенчмарка
        st.subheader("Daily Returns")
        daily_df = pd.DataFrame({
            "Date": pr.index,
            "Return": pr.values * 100,
        })
        daily_df["Color"] = daily_df["Return"].apply(
            lambda x: "#4CAF50" if x >= 0 else "#F44336"
        )
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_df["Date"],
            y=daily_df["Return"],
            marker_color=daily_df["Color"],
            name="Daily Returns",
        ))
        fig.update_layout(
            title="Daily Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode="x unified",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_daily_returns")
    
    # Section 1.3: Portfolio Structure
    st.markdown("---")
    st.subheader("Portfolio Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution by Assets**")
        if positions:
            # Use weight_target if available, otherwise equal weight
            weights = []
            for pos in positions:
                if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                    weights.append(pos.weight_target)
                else:
                    weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
            
            total_weight = sum(weights)
            if total_weight > 0:
                # Build mapping ticker -> weight % for donut
                alloc_data = {}
                for pos, w in zip(positions, weights):
                    pct = (w / total_weight * 100)
                    alloc_data[pos.ticker] = alloc_data.get(pos.ticker, 0.0) + pct
                fig = plot_asset_allocation(alloc_data)
                st.plotly_chart(fig, use_container_width=True, key="overview_asset_allocation")
        else:
            st.info("No positions available")

    with col2:
        st.markdown("**Distribution by Sectors**")
        from core.data_manager.ticker_validator import TickerValidator
        validator = TickerValidator()
        sector_to_weight: dict[str, float] = {}
        if positions:
            # Build weights like in asset allocation
            weights = []
            for pos in positions:
                if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                    weights.append(pos.weight_target)
                else:
                    weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
            total_weight = sum(weights)
            tickers = [pos.ticker for pos in positions]
            for tkr, w in zip(tickers, weights):
                if tkr == "CASH":
                    sector = "Cash"
                else:
                    try:
                        info = validator.get_ticker_info(tkr)
                        sector = info.sector or "Other"
                    except Exception:
                        sector = "Other"
                pct = (w / total_weight * 100) if total_weight > 0 else 0.0
                sector_to_weight[sector] = sector_to_weight.get(sector, 0.0) + pct
        if sector_to_weight:
            fig = plot_sector_allocation(sector_to_weight)
            st.plotly_chart(fig, use_container_width=True, key="overview_sector_allocation")
        else:
            st.info("No sector data available")
    
    # Section 1.4: Comparison Table
    st.markdown("---")
    st.subheader("Portfolio vs Comparison")
    
    # Calculate benchmark metrics if benchmark data available
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
    
    # portfolio_metrics_flat already prepared above
    
    # Calculate comparison metrics
    benchmark_metrics_flat = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            # Align by strict intersection to avoid distortions
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_bench = benchmark_returns.loc[common_idx]
            if not aligned_bench.empty:
                # Calculate benchmark metrics
                max_dd_result = calculate_max_drawdown(aligned_bench)
                max_dd_value = max_dd_result[0] if isinstance(
                    max_dd_result, tuple
                ) else max_dd_result
                
                benchmark_metrics_flat = {
                    "total_return": float((1 + aligned_bench).prod() - 1),
                    "annualized_return": float(
                        calculate_annualized_return(aligned_bench)
                    ),
                    "cagr": float(calculate_annualized_return(aligned_bench)),
                    "volatility": float(calculate_volatility(aligned_bench).get("annual", 0.0)),
                    "max_drawdown": float(max_dd_value),
                    "sharpe_ratio": float(
                        calculate_sharpe_ratio(
                            aligned_bench, risk_free_rate=risk_free_rate
                        ) or 0
                    ),
                    "sortino_ratio": float(
                        calculate_sortino_ratio(
                            aligned_bench, risk_free_rate=risk_free_rate
                        ) or 0
                    ),
                    "calmar_ratio": float((calculate_annualized_return(aligned_bench))/abs(max_dd_value)) if max_dd_value not in (0, None) else 0.0,
                    "beta": 1.0,  # Benchmark beta is always 1.0 vs itself
                    "alpha": 0.0,  # Benchmark alpha is 0 vs itself
                    "up_capture": 1.0,
                    "down_capture": 1.0,
                }
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
    
    render_comparison_table(
        portfolio_metrics=portfolio_metrics_flat,
        benchmark_metrics=benchmark_metrics_flat if benchmark_metrics_flat else None,
        title="Key Metrics Comparison",
        height=480,
    )
    
    # Section 1.5: Time in Market Metadata
    st.markdown("---")
    st.subheader("Analysis Metadata")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        trading_days = len(portfolio_returns)
        total_days = (end_date - start_date).days + 1
        data_quality = (trading_days / total_days * 100) if total_days > 0 else 0
        
        metadata_text = f"""
**Analysis Period:** {start_date} to {end_date} ({total_days} days)
**Trading Days:** {trading_days}
**Time in Market:** {trading_days}/{total_days} days ({data_quality:.1f}%)
**Data Quality:** {data_quality:.1f}% (no missing data)
**Last Updated:** {date.today()} {pd.Timestamp.now().strftime('%H:%M:%S')}
        """
        st.markdown(metadata_text)


def _render_performance_tab(perf, portfolio_returns, benchmark_returns, portfolio_values, risk_free_rate, start_date, end_date):
    """Render Performance tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Returns Analysis",
        "Periodic Analysis",
        "Distribution"
    ])
    
    # Calculate benchmark values if needed
    benchmark_values = None
    if benchmark_returns is not None and not benchmark_returns.empty and portfolio_values is not None:
        aligned_bench = benchmark_returns.reindex(portfolio_values.index, method="ffill").fillna(0)
        initial_value = float(portfolio_values.iloc[0])
        benchmark_values = (1 + aligned_bench).cumprod() * initial_value
    
    with sub_tab1:
        _render_returns_analysis(perf, portfolio_returns, benchmark_returns, portfolio_values, benchmark_values)
    
    with sub_tab2:
        _render_periodic_analysis(portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_distribution_analysis(portfolio_returns, benchmark_returns)


def _render_returns_analysis(perf, portfolio_returns, benchmark_returns, portfolio_values, benchmark_values):
    """Sub-tab 2.1: Returns Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return

    # Graph 2.1.1: Cumulative Returns
    st.subheader("Cumulative Returns")
    cum_data = get_cumulative_returns_data(portfolio_returns, benchmark_returns)
    if cum_data:
        fig = plot_cumulative_returns(cum_data)
        st.plotly_chart(fig, use_container_width=True, key="returns_cumulative")
    
    # Graph 2.1.2: Daily Active Returns (Area Chart)
    st.markdown("---")
    st.subheader("Daily Active Returns")
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = benchmark_returns.reindex(portfolio_returns.index, method="ffill").fillna(0)
        active_returns = portfolio_returns - aligned
        
        # Area chart
        fig = plot_active_returns_area(active_returns)
        st.plotly_chart(fig, use_container_width=True, key="returns_active_returns_area")

        # Stats box
        pos_days = (active_returns > 0).sum()
        total_days = len(active_returns)
        st.info(f"""
**Avg Daily Active Return:** {active_returns.mean()*100:.2f}%
**Positive Days:** {pos_days} ({pos_days/total_days*100:.1f}%)
**Negative Days:** {total_days - pos_days} ({(total_days-pos_days)/total_days*100:.1f}%)
**Max Daily Alpha:** {active_returns.max()*100:.2f}%
**Min Daily Alpha:** {active_returns.min()*100:.2f}%
        """)
    
    # Table 2.1.3: Return by Periods
    st.markdown("---")
    st.subheader("Return by Periods")
    periods_data = get_period_returns_comparison_data(
        portfolio_returns, benchmark_returns, portfolio_values, benchmark_values
    )
    if periods_data.get("periods") is not None:
        periods_df = periods_data["periods"].copy()
        
        # Format for display (keep original values for calculations)
        from streamlit_app.components.comparison_table import (
            _calculate_percentage_change,
        )
        
        display_data = []
        for _, row in periods_df.iterrows():
            port_val = row["Portfolio"]
            bench_val = row["Benchmark"]
            
            # Format values (already in decimal format)
            port_formatted = f"{port_val * 100:.2f}%" if pd.notna(port_val) else "—"
            bench_formatted = f"{bench_val * 100:.2f}%" if pd.notna(bench_val) else "—"
            
            # Calculate relative difference like in comparison_table
            diff_formatted = "—"
            if pd.notna(port_val) and pd.notna(bench_val):
                # Use same logic as comparison_table for returns
                diff_pct = _calculate_percentage_change(
                    "total_return", port_val, bench_val
                )
                # Format without emoji circles
                if abs(diff_pct) < 0.01:
                    diff_formatted = "—"
                else:
                    sign = "+" if diff_pct > 0 else ""
                    diff_formatted = f"{sign}{diff_pct:.2f}%"
            
            display_data.append({
                "Period": row["Period"],
                "Portfolio": port_formatted,
                "Benchmark": bench_formatted,
                "Difference": diff_formatted,
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Bar chart (use original values, not multiplied)
        fig = plot_period_returns_bar(periods_df)
        st.plotly_chart(fig, use_container_width=True, key="returns_periods_bar")
    
    # Table 2.1.4: Expected Returns
    st.markdown("---")
    st.subheader("Expected Returns (Mean Historical)")
    expected_port = calculate_expected_returns(portfolio_returns)
    expected_bench = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        expected_bench = calculate_expected_returns(benchmark_returns)
    
    if expected_port:
        timeframes = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        port_vals = [
            expected_port.get("expected_daily", 0) * 100,
            expected_port.get("expected_weekly", 0) * 100,
            expected_port.get("expected_monthly", 0) * 100,
            expected_port.get("expected_quarterly", 0) * 100,
            expected_port.get("expected_yearly", 0) * 100,
        ]
        bench_vals = [0.0] * 5
        if expected_bench:
            bench_vals = [
                expected_bench.get("expected_daily", 0) * 100,
                expected_bench.get("expected_weekly", 0) * 100,
                expected_bench.get("expected_monthly", 0) * 100,
                expected_bench.get("expected_quarterly", 0) * 100,
                expected_bench.get("expected_yearly", 0) * 100,
            ]
        
        expected_df = pd.DataFrame({
            "Timeframe": timeframes,
            "Portfolio": [f"{v:.2f}%" for v in port_vals],
            "Benchmark": [f"{v:.2f}%" for v in bench_vals],
            "Difference": [f"{p - b:.2f}%" for p, b in zip(port_vals, bench_vals)],
        })
        st.dataframe(expected_df, use_container_width=True, hide_index=True)
        st.caption("Note: Based on arithmetic mean of historical returns")
    
    # Metric 2.1.5: Common Performance Periods (CPP)
    st.markdown("---")
    st.subheader("Common Performance Periods (CPP)")
    if benchmark_returns is not None and not benchmark_returns.empty:
        cpp_data = calculate_common_performance_periods(portfolio_returns, benchmark_returns)
        if cpp_data:
            same_dir_pct = cpp_data.get("same_direction_pct", 0) * 100
            cpp_index = cpp_data.get("cpp_index", 0)
            st.info(f"""
**Same Direction:** {same_dir_pct:.1f}% of days  
(Portfolio and Benchmark moved in same direction)  

**CPP Index:** {cpp_index:.2f}  
(Correlation of directional moves)  

**Interpretation:** Portfolio is {'highly' if cpp_index > 0.7 else 'moderately' if cpp_index > 0.4 else 'lowly'} correlated with market direction
            """)
    
    # Table 2.1.6: Best/Worst Periods (3-month rolling)
    st.markdown("---")
    st.subheader("The Best and Worst Periods")
    
    # Best 3-Month Periods
    st.markdown("**Best 3-Month Periods:**")
    rolling_periods = get_three_month_rolling_periods_data(
        portfolio_returns, benchmark_returns, top_n=3
    )
    if rolling_periods.get("best") is not None and not rolling_periods["best"].empty:
        best_df = rolling_periods["best"].copy()
        best_df["#"] = range(1, len(best_df) + 1)
        best_df["Portfolio"] = best_df["Portfolio"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Benchmark"] = best_df["Benchmark"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Difference"] = best_df["Difference"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Start"] = best_df["Start"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        best_df["End"] = best_df["End"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        display_best = best_df[["#", "Start", "End", "Portfolio", "Benchmark", "Difference"]]
        st.dataframe(display_best, use_container_width=True, hide_index=True)
    
    # Worst 3-Month Periods
    st.markdown("**Worst 3-Month Periods:**")
    if rolling_periods.get("worst") is not None and not rolling_periods["worst"].empty:
        worst_df = rolling_periods["worst"].copy()
        worst_df["#"] = range(1, len(worst_df) + 1)
        worst_df["Portfolio"] = worst_df["Portfolio"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Benchmark"] = worst_df["Benchmark"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Difference"] = worst_df["Difference"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Start"] = worst_df["Start"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        worst_df["End"] = worst_df["End"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        display_worst = worst_df[["#", "Start", "End", "Portfolio", "Benchmark", "Difference"]]
        st.dataframe(display_worst, use_container_width=True, hide_index=True)


def _render_periodic_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.2: Periodic Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Graph 2.2.1: Annual Returns (EOY)
    st.subheader("Annual Returns (EOY)")
    yearly_data = get_yearly_returns_data(portfolio_returns, benchmark_returns)
    if yearly_data.get("yearly") is not None and not yearly_data["yearly"].empty:
        fig = plot_yearly_returns({"yearly": yearly_data["yearly"]})
        st.plotly_chart(fig, use_container_width=True, key="periodic_yearly")
    
    # Heatmap 2.2.2: Monthly Returns Calendar
    st.markdown("---")
    st.subheader("Monthly Returns Calendar (%)")
    heatmap_data = get_monthly_heatmap_data(portfolio_returns)
    if heatmap_data.get("heatmap") is not None and not heatmap_data["heatmap"].empty:
        fig = plot_monthly_heatmap({"heatmap": heatmap_data["heatmap"]})
        st.plotly_chart(fig, use_container_width=True, key="periodic_monthly")
    
    # Heatmap 2.2.3: Monthly Active Returns
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.markdown("---")
        st.subheader("Monthly Active Returns (%) - Portfolio vs Benchmark")
        active_heatmap_data = get_monthly_active_returns_data(portfolio_returns, benchmark_returns)
        if active_heatmap_data.get("heatmap") is not None and not active_heatmap_data["heatmap"].empty:
            fig = plot_monthly_heatmap({"heatmap": active_heatmap_data["heatmap"]})
            st.plotly_chart(fig, use_container_width=True, key="periodic_monthly_active")
    
    # Charts 2.2.4: Seasonal Analysis
    st.markdown("---")
    st.subheader("Seasonal Analysis")
    seasonal_data = get_seasonal_analysis_data(portfolio_returns, benchmark_returns)
    
    if seasonal_data.get("day_of_week") is not None and not seasonal_data["day_of_week"].empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = plot_seasonal_bar(seasonal_data["day_of_week"], "Avg Return by Day of Week (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_day")
        
        with col2:
            fig = plot_seasonal_bar(seasonal_data["month"], "Avg Return by Month (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_month")
        
        with col3:
            fig = plot_seasonal_bar(seasonal_data["quarter"], "Avg Return by Quarter (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_quarter")


def _render_distribution_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.3: Distribution."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Charts 2.3.1: Return Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Daily Returns")
        dist_data_daily = get_return_distribution_data(portfolio_returns, bins=50)
        if dist_data_daily:
            fig = plot_return_distribution(dist_data_daily, bar_color="blue")
            st.plotly_chart(fig, use_container_width=True, key="distribution_daily")
    
    with col2:
        st.subheader("Distribution of Monthly Returns")
        monthly_returns = portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        if not monthly_returns.empty:
            dist_data_monthly = get_return_distribution_data(monthly_returns, bins=30)
            if dist_data_monthly:
                fig = plot_return_distribution(dist_data_monthly, bar_color="blue")
                st.plotly_chart(fig, use_container_width=True, key="distribution_monthly")
    
    # Chart 2.3.2: Q-Q Plot
    st.markdown("---")
    st.subheader("Q-Q Plot")
    qq_data = get_qq_plot_data(portfolio_returns)
    if qq_data:
        fig = plot_qq_plot(qq_data)
        st.plotly_chart(fig, use_container_width=True, key="distribution_qq")
        st.caption("Points closer to line = more normal distribution. Deviation shows fat tails / skewness.")
    
    # Chart 2.3.3: Return Quantiles Box Plots
    st.markdown("---")
    st.subheader("Return Quantiles Box Plots")
    fig = plot_return_quantiles_box(portfolio_returns, benchmark_returns)
    st.plotly_chart(fig, use_container_width=True, key="distribution_box")
    
    # Table 2.3.4: Win Rate Statistics (Comprehensive)
    st.markdown("---")
    st.subheader("Win Rate Statistics - Comprehensive")
    win_rate_data = get_win_rate_statistics_data(portfolio_returns, benchmark_returns)
    
    if win_rate_data.get("stats"):
        stats = win_rate_data["stats"]
        port_stats = stats.get("portfolio", {})
        bench_stats = stats.get("benchmark", {})
        
        # Build benchmark and difference columns
        bench_values = [
            f"{bench_stats.get('win_rate_daily', 0)*100:.1f}%" if bench_stats.get('win_rate_daily') is not None else "N/A",
            f"{bench_stats.get('win_rate_weekly', 0)*100:.1f}%" if bench_stats.get('win_rate_weekly') is not None else "N/A",
            f"{bench_stats.get('win_rate_monthly', 0)*100:.1f}%" if bench_stats.get('win_rate_monthly') is not None else "N/A",
            f"{bench_stats.get('win_rate_quarterly', 0)*100:.1f}%" if bench_stats.get('win_rate_quarterly') is not None else "N/A",
            f"{bench_stats.get('win_rate_yearly', 0)*100:.1f}%" if bench_stats.get('win_rate_yearly') is not None else "N/A",
            f"{bench_stats.get('avg_win_daily', 0)*100:.2f}%" if bench_stats.get('avg_win_daily') is not None else "N/A",
            f"{bench_stats.get('avg_loss_daily', 0)*100:.2f}%" if bench_stats.get('avg_loss_daily') is not None else "N/A",
            f"{bench_stats.get('avg_win_monthly', 0)*100:.2f}%" if bench_stats.get('avg_win_monthly') is not None else "N/A",
            f"{bench_stats.get('avg_loss_monthly', 0)*100:.2f}%" if bench_stats.get('avg_loss_monthly') is not None else "N/A",
            f"{bench_stats.get('best_daily', 0)*100:.2f}%" if bench_stats.get('best_daily') is not None else "N/A",
            f"{bench_stats.get('worst_daily', 0)*100:.2f}%" if bench_stats.get('worst_daily') is not None else "N/A",
            f"{bench_stats.get('best_monthly', 0)*100:.2f}%" if bench_stats.get('best_monthly') is not None else "N/A",
            f"{bench_stats.get('worst_monthly', 0)*100:.2f}%" if bench_stats.get('worst_monthly') is not None else "N/A",
        ]
        
        # Calculate differences
        diff_values = []
        port_vals = [
            port_stats.get('win_days_pct', 0),
            port_stats.get('win_weeks_pct', 0),
            port_stats.get('win_months_pct', 0),
            port_stats.get('win_quarters_pct', 0),
            port_stats.get('win_years_pct', 0),
            port_stats.get('avg_up_day', 0),
            port_stats.get('avg_down_day', 0),
            port_stats.get('avg_up_month', 0),
            port_stats.get('avg_down_month', 0),
            port_stats.get('best_day', 0),
            port_stats.get('worst_day', 0),
            port_stats.get('best_month', 0),
            port_stats.get('worst_month', 0),
        ]
        bench_vals = [
            bench_stats.get('win_rate_daily', 0) * 100 if bench_stats.get('win_rate_daily') is not None else None,
            bench_stats.get('win_rate_weekly', 0) * 100 if bench_stats.get('win_rate_weekly') is not None else None,
            bench_stats.get('win_rate_monthly', 0) * 100 if bench_stats.get('win_rate_monthly') is not None else None,
            bench_stats.get('win_rate_quarterly', 0) * 100 if bench_stats.get('win_rate_quarterly') is not None else None,
            bench_stats.get('win_rate_yearly', 0) * 100 if bench_stats.get('win_rate_yearly') is not None else None,
            bench_stats.get('avg_win_daily', 0) * 100 if bench_stats.get('avg_win_daily') is not None else None,
            bench_stats.get('avg_loss_daily', 0) * 100 if bench_stats.get('avg_loss_daily') is not None else None,
            bench_stats.get('avg_win_monthly', 0) * 100 if bench_stats.get('avg_win_monthly') is not None else None,
            bench_stats.get('avg_loss_monthly', 0) * 100 if bench_stats.get('avg_loss_monthly') is not None else None,
            bench_stats.get('best_daily', 0) * 100 if bench_stats.get('best_daily') is not None else None,
            bench_stats.get('worst_daily', 0) * 100 if bench_stats.get('worst_daily') is not None else None,
            bench_stats.get('best_monthly', 0) * 100 if bench_stats.get('best_monthly') is not None else None,
            bench_stats.get('worst_monthly', 0) * 100 if bench_stats.get('worst_monthly') is not None else None,
        ]
        
        for p, b in zip(port_vals, bench_vals):
            if b is not None:
                diff = p - b
                diff_values.append(f"{diff:+.2f}%")
            else:
                diff_values.append("N/A")
        
        win_rate_df = pd.DataFrame({
            "Timeframe": ["Win Days %", "Win Weeks %", "Win Months %", "Win Quarters %", "Win Years %",
                          "Avg Up Day", "Avg Down Day", "Avg Up Month", "Avg Down Month",
                          "Best Day", "Worst Day", "Best Month", "Worst Month"],
            "Portfolio": [
                f"{port_stats.get('win_days_pct', 0):.1f}%",
                f"{port_stats.get('win_weeks_pct', 0):.1f}%",
                f"{port_stats.get('win_months_pct', 0):.1f}%",
                f"{port_stats.get('win_quarters_pct', 0):.1f}%",
                f"{port_stats.get('win_years_pct', 0):.1f}%",
                f"{port_stats.get('avg_up_day', 0):.2f}%",
                f"{port_stats.get('avg_down_day', 0):.2f}%",
                f"{port_stats.get('avg_up_month', 0):.2f}%",
                f"{port_stats.get('avg_down_month', 0):.2f}%",
                f"{port_stats.get('best_day', 0):.2f}%",
                f"{port_stats.get('worst_day', 0):.2f}%",
                f"{port_stats.get('best_month', 0):.2f}%",
                f"{port_stats.get('worst_month', 0):.2f}%",
            ],
            "Benchmark": bench_values,
            "Difference": diff_values,
        })
        # Calculate required height (13 rows + header)
        row_height = 35  # Approximate height per row
        header_height = 40
        total_height = 13 * row_height + header_height
        st.dataframe(win_rate_df, use_container_width=True, hide_index=True, height=total_height)
        
        # 12-Month Rolling Win Rate Chart
        rolling_win_rate = win_rate_data.get("rolling", pd.Series())
        if not rolling_win_rate.empty:
            bench_rolling = None
            if benchmark_returns is not None:
                # Calculate benchmark rolling win rate
                bench_win_rate_data = get_win_rate_statistics_data(benchmark_returns)
                bench_rolling = bench_win_rate_data.get("rolling", pd.Series())
            
            fig = plot_rolling_win_rate(rolling_win_rate, bench_rolling)
            st.plotly_chart(fig, use_container_width=True, key="distribution_rolling_win_rate")
    
    # Section 2.3.5: Outlier Analysis
    st.markdown("---")
    st.subheader("Outlier Analysis - Tail Events")
    outlier_data = get_outlier_analysis_data(portfolio_returns, outlier_threshold=2.0)
    
    if outlier_data.get("stats"):
        stats = outlier_data["stats"]
        st.info(f"""
**Outlier Definition:** Beyond 2 standard deviations  

**Outlier Win Ratio:** {stats.get('outlier_win_ratio', 0):.2f}  
(Avg outlier win / Avg normal win)  

**Outlier Loss Ratio:** {stats.get('outlier_loss_ratio', 0):.2f}  
(Avg outlier loss / Avg normal loss)  

**Interpretation:**  
{'✓' if stats.get('outlier_win_ratio', 0) > 0 else '✗'} Big wins are {stats.get('outlier_win_ratio', 0):.2f}x larger than typical wins  
{'✓' if stats.get('outlier_loss_ratio', 0) > 0 else '✗'} Big losses are {stats.get('outlier_loss_ratio', 0):.2f}x larger than typical losses
        """)
        
        # Outlier Scatter Plot
        fig = plot_outlier_scatter(portfolio_returns, outlier_data)
        st.plotly_chart(fig, use_container_width=True, key="distribution_outlier_scatter")
    
    # Table 2.3.6: Statistical Tests
    st.markdown("---")
    st.subheader("Statistical Tests - Distribution Analysis")
    stats_tests = get_statistical_tests_data(portfolio_returns)
    
    if stats_tests:
        shapiro = stats_tests.get("shapiro_wilk", {})
        jb = stats_tests.get("jarque_bera", {})
        skewness = stats_tests.get("skewness", 0)
        kurtosis = stats_tests.get("kurtosis", 0)
        sample_size = stats_tests.get("sample_size", 0)
        
        # Format p-values more accurately
        def format_pvalue(p):
            if p < 0.0001:
                return f"{p:.2e}"  # Scientific notation
            else:
                return f"{p:.4f}"
        
        tests_df = pd.DataFrame({
            "Test": ["Shapiro-Wilk", "Jarque-Bera", "Skewness", "Kurtosis (Excess)"],
            "Statistic": [
                f"{shapiro.get('statistic', 0):.4f}",
                f"{jb.get('statistic', 0):.4f}",
                f"{skewness:.3f}",
                f"{kurtosis:+.3f}",
            ],
            "p-value": [
                format_pvalue(shapiro.get('pvalue', 1)),
                format_pvalue(jb.get('pvalue', 1)),
                "-",
                "-",
            ],
        })
        st.dataframe(tests_df, use_container_width=True, hide_index=True)
        
        # Show sample size info
        shapiro_sample = shapiro.get('sample_size', sample_size)
        if sample_size > 1000:
            st.caption(f"ℹ️ Sample size: {sample_size:,} observations" + 
                      (f" (Shapiro-Wilk used random sample of {shapiro_sample:,})" if shapiro_sample < sample_size else ""))
        
        # Interpretation
        shapiro_p = shapiro.get("pvalue", 1.0)
        jb_p = jb.get("pvalue", 1.0)
        is_normal = shapiro_p >= 0.05 and jb_p >= 0.05
        
        skew_interpretation = "Slight negative skew" if skewness < -0.1 else "Slight positive skew" if skewness > 0.1 else "Symmetric"
        kurtosis_interpretation = "Leptokurtic (fat tails)" if kurtosis > 0.5 else "Platykurtic (thin tails)" if kurtosis < -0.5 else "Normal tails"
        
        # Add note about large sample sensitivity
        interpretation_text = f"""
**Interpretation:**  
{'✅' if is_normal else '❌'} **Shapiro-Wilk:** Distribution is {'NOT' if shapiro_p < 0.05 else ''} normal (p {'<' if shapiro_p < 0.05 else '≥'} 0.05)  
{'✅' if is_normal else '❌'} **Jarque-Bera:** Distribution is {'NOT' if jb_p < 0.05 else ''} normal (p {'<' if jb_p < 0.05 else '≥'} 0.05)  
⚠ **Skewness:** {skew_interpretation} ({skewness:.3f})  
⚠ **Kurtosis:** {kurtosis_interpretation} ({kurtosis:+.3f})
"""
        
        if sample_size > 1000 and not is_normal:
            interpretation_text += """
**Note:** Large samples (>1000) make normality tests very sensitive - they detect even minor deviations.  
Financial returns typically show fat tails and slight skewness, so rejection of normality is expected and not necessarily a concern.
"""
        
        st.info(interpretation_text)


def _render_risk_tab(risk, ratios, market, portfolio_returns, benchmark_returns, portfolio_values, risk_free_rate, start_date, end_date):
    """Render Risk tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Key Metrics",
        "Drawdown Analysis",
        "VaR & CVaR",
        "Rolling Risk Metrics"
    ])
    
    # Calculate benchmark values if needed
    benchmark_values = None
    if benchmark_returns is not None and not benchmark_returns.empty and portfolio_values is not None:
        aligned_bench = benchmark_returns.reindex(portfolio_values.index, method="ffill").fillna(0)
        initial_value = float(portfolio_values.iloc[0])
        benchmark_values = (1 + aligned_bench).cumprod() * initial_value
    
    with sub_tab1:
        _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate)
    
    with sub_tab2:
        _render_drawdown_analysis(risk, portfolio_values, benchmark_values, portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_var_analysis(risk)
    
    with sub_tab4:
        _render_rolling_risk(portfolio_returns, benchmark_returns, risk_free_rate)


def _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate):
    """Sub-tab 3.1: Key Risk Metrics."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Recalculate portfolio risk metrics from returns to ensure correctness
    from core.analytics_engine.risk_metrics import (
        calculate_volatility,
        calculate_max_drawdown,
        calculate_var,
        calculate_cvar,
    )
    
    portfolio_vol = calculate_volatility(portfolio_returns)
    portfolio_max_dd_tuple = calculate_max_drawdown(portfolio_returns)
    portfolio_max_dd = portfolio_max_dd_tuple[0] if isinstance(portfolio_max_dd_tuple, tuple) else portfolio_max_dd_tuple
    
    # Calculate benchmark metrics for comparison
    benchmark_risk_metrics = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            from core.analytics_engine.ratios import (
                calculate_sortino_ratio,
                calculate_calmar_ratio,
            )
            
            bench_vol = calculate_volatility(benchmark_returns)
            bench_vol_annual = bench_vol.get("annual", 0.0) if isinstance(bench_vol, dict) else bench_vol
            bench_max_dd = calculate_max_drawdown(benchmark_returns)
            bench_max_dd_val = bench_max_dd[0] if isinstance(bench_max_dd, tuple) else bench_max_dd
            
            benchmark_risk_metrics = {
                "volatility": float(bench_vol_annual),
                "max_drawdown": float(bench_max_dd_val),
                "sortino_ratio": float(calculate_sortino_ratio(benchmark_returns, risk_free_rate) or 0),
                "calmar_ratio": float(calculate_calmar_ratio(benchmark_returns) or 0),
                "var_95": float(calculate_var(benchmark_returns, confidence_level=0.95, method="historical") or 0),
                "cvar_95": float(calculate_cvar(benchmark_returns, confidence_level=0.95) or 0),
                "up_capture": 1.0,
                "down_capture": 1.0,
            }
        except Exception as e:
            logger.warning(f"Error calculating benchmark risk metrics: {e}")
    
    # Section 3.1.1: Risk Metric Cards (8 cards in 2 rows)
    st.subheader("Risk Metrics")
    
    # Use recalculated values
    vol_annual = portfolio_vol.get("annual", 0.0) if isinstance(portfolio_vol, dict) else portfolio_vol
    
    # Row 1: Volatility, Max Drawdown, Sortino Ratio, Calmar Ratio
    risk_metrics_row1 = [
        {
            "label": "Volatility",
            "portfolio_value": float(vol_annual),
            "benchmark_value": benchmark_risk_metrics.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": float(portfolio_max_dd),
            "benchmark_value": benchmark_risk_metrics.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "benchmark_value": benchmark_risk_metrics.get("sortino_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Calmar Ratio",
            "portfolio_value": ratios.get("calmar_ratio", 0),
            "benchmark_value": benchmark_risk_metrics.get("calmar_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
    ]
    render_metric_cards_row(risk_metrics_row1, columns_per_row=4)
    
    st.markdown("---")
    
    # Row 2: VaR (95%), CVaR (95%), Up Capture, Down Capture
    risk_metrics_row2 = [
        {
            "label": "VaR (95%)",
            "portfolio_value": risk.get("var_95", 0),
            "benchmark_value": benchmark_risk_metrics.get("var_95"),
            "format": "percent",
            "higher_is_better": False,  # Less negative is better
        },
        {
            "label": "CVaR (95%)",
            "portfolio_value": risk.get("cvar_95", 0),
            "benchmark_value": benchmark_risk_metrics.get("cvar_95"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Up Capture",
            "portfolio_value": market.get("up_capture", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "Down Capture",
            "portfolio_value": market.get("down_capture", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "percent",
            "higher_is_better": False,  # Lower is better
        },
    ]
    render_metric_cards_row(risk_metrics_row2, columns_per_row=4)
    
    # Section 3.1.2: Probabilistic Sharpe Ratio
    st.markdown("---")
    st.subheader("Probabilistic Sharpe Ratio")
    
    psr_95 = calculate_probabilistic_sharpe_ratio(portfolio_returns, risk_free_rate, benchmark_sharpe=1.0)
    psr_99 = calculate_probabilistic_sharpe_ratio(portfolio_returns, risk_free_rate, benchmark_sharpe=0.0)
    observed_sharpe = ratios.get("sharpe_ratio", 0)
    
    if psr_95 is not None:
        psr_95_pct = psr_95 * 100
        psr_99_pct = psr_99 * 100 if psr_99 is not None else 0
        
        st.info(f"""
**Observed Sharpe Ratio:** {observed_sharpe:.2f}

**Probabilistic Sharpe Ratio (PSR):**  
- At 95% confidence: {psr_95_pct:.1f}%  
- At 99% confidence: {psr_99_pct:.1f}%

**Interpretation:**  
✓ {psr_95_pct:.1f}% probability that true Sharpe > 1.0  
{'✓ High statistical significance' if psr_95 > 0.85 else '⚠ Moderate statistical significance'}  
{'✓ Sharpe is likely NOT due to luck' if psr_95 > 0.80 else '⚠ Sharpe may be influenced by luck'}
        """)
    else:
        st.warning("Insufficient data for Probabilistic Sharpe Ratio calculation")
    
    # Section 3.1.3: Smart Sharpe & Sortino
    st.markdown("---")
    st.subheader("Smart Sharpe & Sortino")
    
    smart_sharpe = calculate_smart_sharpe(portfolio_returns, risk_free_rate)
    smart_sortino = calculate_smart_sortino(portfolio_returns, risk_free_rate)
    observed_sortino = ratios.get("sortino_ratio", 0)
    
    if smart_sharpe is not None and smart_sortino is not None:
        sharpe_adjustment = observed_sharpe - smart_sharpe
        sortino_adjustment = observed_sortino - smart_sortino
        sortino_conservative = observed_sortino / np.sqrt(2)
        
        smart_ratios_df = pd.DataFrame({
            "Ratio": [
                "Sharpe Ratio",
                "Smart Sharpe (Autocorrelation adj.)",
                "Sortino Ratio",
                "Smart Sortino",
                "Sortino/√2 (Conservative Est.)",
        ],
        "Value": [
                f"{observed_sharpe:.2f}",
                f"{smart_sharpe:.2f}",
                f"{observed_sortino:.2f}",
                f"{smart_sortino:.2f}",
                f"{sortino_conservative:.2f}",
            ],
            "Adjustment": [
                "—",
                f"{sharpe_adjustment:+.2f}",
                "—",
                f"{sortino_adjustment:+.2f}",
                "—",
            ],
        })
        
        st.dataframe(smart_ratios_df, use_container_width=True, hide_index=True, height=230)
        st.caption("Note: Smart ratios adjust for autocorrelation and non-normality")
    else:
        st.warning("Insufficient data for Smart Sharpe/Sortino calculation")
    
    # Chart 3.1.4: Capture Ratio Visualization
    st.markdown("---")
    st.subheader("Capture Ratio Visualization")
    
    up_capture = market.get("up_capture")
    down_capture = market.get("down_capture")
    
    if up_capture is not None and down_capture is not None:
        capture_data = get_capture_ratio_data(up_capture, down_capture)
        if capture_data:
            fig = plot_capture_ratio(capture_data)
            st.plotly_chart(fig, use_container_width=True, key="risk_capture_ratio")
            
            capture_ratio = capture_data.get("capture_ratio", 0)
            st.info(f"""
**Capture Ratio:** {capture_ratio:.2f} (Up/Down)

**Interpretation:**  
✓ Portfolio captures {up_capture*100:.0f}% of market upside  
✓ Portfolio captures only {down_capture*100:.0f}% of market downside  
{'✓ Strong asymmetry' if capture_ratio > 1.2 else '⚠ Moderate asymmetry'} ({capture_ratio:.2f}) → {'Favorable risk/reward' if capture_ratio > 1.1 else 'Neutral risk/reward'}
            """)
    else:
        st.info("Capture ratios require benchmark comparison")
    
    # Chart 3.1.5: Risk/Return Scatter
    st.markdown("---")
    st.subheader("Risk/Return Scatter")
    
    scatter_data = get_risk_return_scatter_data(
        portfolio_returns, benchmark_returns, risk_free_rate
    )
    if scatter_data:
        fig = plot_risk_return_scatter(scatter_data)
        st.plotly_chart(fig, use_container_width=True, key="risk_return_scatter")
    else:
        st.warning("Unable to generate risk/return scatter plot")
    
    # Section 3.1.6: Information Ratio Breakdown
    st.markdown("---")
    st.subheader("Information Ratio Breakdown")
    
    # Recalculate IR from scratch to ensure correctness
    if benchmark_returns is not None and not benchmark_returns.empty:
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.ratios import calculate_information_ratio
        from core.analytics_engine.market_metrics import calculate_tracking_error
        
        try:
            # Calculate components
            port_return = calculate_annualized_return(portfolio_returns)
            bench_return = calculate_annualized_return(benchmark_returns)
            active_return = port_return - bench_return
            
            # Recalculate tracking error and IR
            tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns) or 0
            info_ratio = calculate_information_ratio(portfolio_returns, benchmark_returns) or 0
            
            st.info(f"""
**Active Return (Portfolio - Benchmark):** {active_return*100:+.2f}%  
**Tracking Error (Std of Active Return):** {tracking_error*100:.2f}%  
**Information Ratio (AR / TE):** {info_ratio:.2f}
            """)
            
            # Stacked bar visualization
            import plotly.graph_objects as go
            from streamlit_app.utils.chart_config import COLORS, get_chart_layout
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=["Return Breakdown"],
                x=[bench_return * 100],
                orientation="h",
                name="Benchmark Return",
                marker=dict(color=COLORS["secondary"]),
                text=[f"{bench_return*100:.1f}%"],
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                y=["Return Breakdown"],
                x=[active_return * 100],
                orientation="h",
                name="Active Return",
                marker=dict(color=COLORS["success"] if active_return > 0 else COLORS["danger"]),
                text=[f"{active_return*100:+.1f}%"],
                textposition="inside",
            ))
            
            layout = get_chart_layout(
                title="Return Breakdown",
                xaxis=dict(title="Return (%)", tickformat=",.1f"),
                yaxis=dict(title=""),
                barmode="stack",
                height=200,
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, key="info_ratio_breakdown")
            
            st.caption(f"""
**Interpretation:**  
{'✓ High IR' if info_ratio > 1.0 else '⚠ Moderate IR'} ({info_ratio:.2f}) → {'Consistent alpha generation' if info_ratio > 0.75 else 'Inconsistent alpha'}  
✓ Active return is {abs(active_return)/abs(port_return)*100:.0f}% of total return
            """)
        except Exception as e:
            logger.warning(f"Error calculating Information Ratio: {e}")
            st.info("Unable to calculate Information Ratio")
    else:
        st.info("Information Ratio requires benchmark comparison")
    
    # Section 3.1.7: Kelly Criterion & Risk of Ruin
    st.markdown("---")
    st.subheader("Kelly Criterion & Risk of Ruin")
    
    kelly_data = calculate_kelly_criterion(portfolio_returns)
    risk_of_ruin = calculate_risk_of_ruin(portfolio_returns)
    
    if kelly_data:
        kelly_full = kelly_data.get("kelly_full", 0) * 100
        kelly_half = kelly_data.get("kelly_half", 0) * 100
        kelly_quarter = kelly_data.get("kelly_quarter", 0) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Kelly Criterion - Position Sizing**")
            st.info(f"""
**Kelly Criterion:** {kelly_full:.1f}%  
(Optimal leverage for max growth)

**Half-Kelly:** {kelly_half:.1f}%  
(Conservative, reduces volatility)

**Quarter-Kelly:** {kelly_quarter:.1f}%  
(Very conservative)
            """)
        
        with col2:
            if risk_of_ruin:
                st.markdown("**Risk of Ruin Analysis**")
                ruin_df = pd.DataFrame({
                    "Drawdown": ["-10%", "-20%", "-25%", "-30%", "-50%"],
                    "Probability": [
                        f"{risk_of_ruin.get('ruin_10pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_20pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_25pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_30pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_50pct', 0)*100:.1f}%",
                    ],
                    "Est. Recovery": ["~3 mo", "~8 mo", "~12 mo", "~18 mo", "~5 yr"],
                })
                st.dataframe(ruin_df, use_container_width=True, hide_index=True, height=210)
                st.caption("Note: Recovery times are approximate estimates")
    else:
        st.warning("Insufficient data for Kelly Criterion / Risk of Ruin calculation")
    
    # Table 3.1.8: Complete Risk Metrics Table (28 metrics as per spec)
    st.markdown("---")
    st.subheader("Complete Risk Metrics Table")
    
    # Recalculate all metrics from portfolio returns
    from core.analytics_engine.risk_metrics import (
        calculate_volatility as calc_vol,
        calculate_max_drawdown as calc_dd,
        calculate_var,
        calculate_cvar,
        calculate_downside_deviation,
    )
    
    # Portfolio metrics
    port_vol = calc_vol(portfolio_returns)
    port_dd_tuple = calc_dd(portfolio_returns)
    port_dd_val = port_dd_tuple[0] if isinstance(port_dd_tuple, tuple) else port_dd_tuple
    port_dd_date = port_dd_tuple[1] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 1 else None
    port_dd_trough = port_dd_tuple[2] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 2 else None
    port_dd_duration = port_dd_tuple[3] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 3 else None
    
    # Calculate all VaR/CVaR metrics
    port_var_90 = calculate_var(portfolio_returns, 0.90, method="historical") or 0
    port_var_95 = calculate_var(portfolio_returns, 0.95, method="historical") or 0
    port_var_99 = calculate_var(portfolio_returns, 0.99, method="historical") or 0
    port_var_95_param = calculate_var(portfolio_returns, 0.95, method="parametric") or 0
    port_var_95_cf = calculate_var(portfolio_returns, 0.95, method="cornish_fisher") or 0
    
    port_cvar_90 = calculate_cvar(portfolio_returns, 0.90) or 0
    port_cvar_95 = calculate_cvar(portfolio_returns, 0.95) or 0
    port_cvar_99 = calculate_cvar(portfolio_returns, 0.99) or 0
    
    port_downside = calculate_downside_deviation(portfolio_returns) or 0
    
    # Calculate remaining metrics
    from core.analytics_engine.risk_metrics import (
        calculate_semi_deviation,
        calculate_skewness,
        calculate_kurtosis,
        calculate_top_drawdowns,
        calculate_current_drawdown,
        calculate_average_drawdown,
        calculate_ulcer_index,
        calculate_pain_index,
    )
    
    port_semi = calculate_semi_deviation(portfolio_returns) or 0
    port_skew = calculate_skewness(portfolio_returns) or 0
    port_kurt = calculate_kurtosis(portfolio_returns) or 0
    
    # Calculate drawdown metrics using top_drawdowns
    port_current_dd = calculate_current_drawdown(portfolio_returns) or 0
    port_avg_dd = calculate_average_drawdown(portfolio_returns) or 0
    port_ulcer = calculate_ulcer_index(portfolio_returns) or 0
    port_pain = calculate_pain_index(portfolio_returns) or 0
    
    # Calculate avg DD duration and recovery from top drawdowns
    top_dds = calculate_top_drawdowns(portfolio_returns, top_n=5)
    avg_dd_duration = 0
    avg_recovery_time = 0
    max_recovery_time = 0
    if top_dds:
        avg_dd_duration = sum(dd['duration_days'] for dd in top_dds) / len(top_dds)
        recoveries = [dd['recovery_days'] for dd in top_dds if dd['recovery_days']]
        if recoveries:
            avg_recovery_time = sum(recoveries) / len(recoveries)
            max_recovery_time = max(recoveries)
    
    # Benchmark metrics
    bench_metrics = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            bench_vol = calc_vol(benchmark_returns)
            bench_dd_tuple = calc_dd(benchmark_returns)
            bench_dd_val = bench_dd_tuple[0] if isinstance(bench_dd_tuple, tuple) else bench_dd_tuple
            bench_dd_date = bench_dd_tuple[1] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 1 else None
            bench_dd_trough = bench_dd_tuple[2] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 2 else None
            bench_dd_duration = bench_dd_tuple[3] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 3 else None
            
            # Calculate all benchmark metrics
            bench_current_dd = calculate_current_drawdown(benchmark_returns) or 0
            bench_avg_dd = calculate_average_drawdown(benchmark_returns) or 0
            bench_ulcer = calculate_ulcer_index(benchmark_returns) or 0
            bench_pain = calculate_pain_index(benchmark_returns) or 0
            bench_semi = calculate_semi_deviation(benchmark_returns) or 0
            bench_skew = calculate_skewness(benchmark_returns) or 0
            bench_kurt = calculate_kurtosis(benchmark_returns) or 0
            
            # Calculate benchmark drawdown durations
            bench_top_dds = calculate_top_drawdowns(benchmark_returns, top_n=5)
            bench_avg_dd_duration = 0
            bench_avg_recovery = 0
            bench_max_recovery = 0
            if bench_top_dds:
                bench_avg_dd_duration = sum(dd['duration_days'] for dd in bench_top_dds) / len(bench_top_dds)
                bench_recoveries = [dd['recovery_days'] for dd in bench_top_dds if dd['recovery_days']]
                if bench_recoveries:
                    bench_avg_recovery = sum(bench_recoveries) / len(bench_recoveries)
                    bench_max_recovery = max(bench_recoveries)
            
            bench_metrics = {
                'daily_vol': bench_vol.get('daily', 0),
                'weekly_vol': bench_vol.get('weekly', 0),
                'monthly_vol': bench_vol.get('monthly', 0),
                'annual_vol': bench_vol.get('annual', 0),
                'max_dd': bench_dd_val,
                'max_dd_date': bench_dd_date,
                'max_dd_trough': bench_dd_trough,
                'max_dd_duration': bench_dd_duration,
                'current_dd': bench_current_dd,
                'avg_dd': bench_avg_dd,
                'avg_dd_duration': bench_avg_dd_duration,
                'avg_recovery': bench_avg_recovery,
                'max_recovery': bench_max_recovery,
                'ulcer': bench_ulcer,
                'pain': bench_pain,
                'var_90': calculate_var(benchmark_returns, 0.90, method="historical") or 0,
                'var_95': calculate_var(benchmark_returns, 0.95, method="historical") or 0,
                'var_99': calculate_var(benchmark_returns, 0.99, method="historical") or 0,
                'var_95_param': calculate_var(benchmark_returns, 0.95, method="parametric") or 0,
                'var_95_cf': calculate_var(benchmark_returns, 0.95, method="cornish_fisher") or 0,
                'cvar_90': calculate_cvar(benchmark_returns, 0.90) or 0,
                'cvar_95': calculate_cvar(benchmark_returns, 0.95) or 0,
                'cvar_99': calculate_cvar(benchmark_returns, 0.99) or 0,
                'downside': calculate_downside_deviation(benchmark_returns) or 0,
                'semi': bench_semi,
                'skew': bench_skew,
                'kurt': bench_kurt,
            }
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics for table: {e}")
    
    # Extract portfolio volatilities
    p_daily = port_vol.get('daily', 0) if isinstance(port_vol, dict) else 0
    p_weekly = port_vol.get('weekly', 0) if isinstance(port_vol, dict) else 0
    p_monthly = port_vol.get('monthly', 0) if isinstance(port_vol, dict) else 0
    p_annual = port_vol.get('annual', 0) if isinstance(port_vol, dict) else 0
    
    # Build table with all 28 metrics
    metrics_data = [
        ("Daily Volatility", f"{p_daily*100:.2f}%", 
         f"{bench_metrics.get('daily_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_daily - bench_metrics.get('daily_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Weekly Volatility", f"{p_weekly*100:.2f}%",
         f"{bench_metrics.get('weekly_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_weekly - bench_metrics.get('weekly_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Monthly Volatility", f"{p_monthly*100:.2f}%",
         f"{bench_metrics.get('monthly_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_monthly - bench_metrics.get('monthly_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Annual Volatility", f"{p_annual*100:.2f}%",
         f"{bench_metrics.get('annual_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_annual - bench_metrics.get('annual_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Max Drawdown", f"{port_dd_val*100:.2f}%",
         f"{bench_metrics.get('max_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_dd_val - bench_metrics.get('max_dd', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Max DD Peak Date", str(port_dd_date)[:10] if port_dd_date else "—",
         str(bench_metrics.get('max_dd_date'))[:10] if bench_metrics and bench_metrics.get('max_dd_date') else "—",
         "—"),
        ("Max DD Trough Date", str(port_dd_trough)[:10] if port_dd_trough else "—",
         str(bench_metrics.get('max_dd_trough'))[:10] if bench_metrics and bench_metrics.get('max_dd_trough') else "—",
         "—"),
        ("Max DD Duration (days)", str(port_dd_duration) if port_dd_duration else "—",
         str(bench_metrics.get('max_dd_duration')) if bench_metrics and bench_metrics.get('max_dd_duration') else "—",
         f"{int(port_dd_duration or 0) - int(bench_metrics.get('max_dd_duration') or 0):+d}" if bench_metrics and port_dd_duration and bench_metrics.get('max_dd_duration') else "—"),
        ("Current Drawdown", f"{port_current_dd*100:.2f}%",
         f"{bench_metrics.get('current_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_current_dd - bench_metrics.get('current_dd', 0))*100:+.2f}%" if bench_metrics and port_current_dd != 0 else "—"),
        ("Average Drawdown", f"{port_avg_dd*100:.2f}%",
         f"{bench_metrics.get('avg_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_avg_dd - bench_metrics.get('avg_dd', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Avg DD Duration", f"{int(avg_dd_duration)} days" if avg_dd_duration else "—",
         f"{int(bench_metrics.get('avg_dd_duration', 0))} days" if bench_metrics and bench_metrics.get('avg_dd_duration') else "—",
         f"{int(avg_dd_duration or 0) - int(bench_metrics.get('avg_dd_duration') or 0):+d} days" if bench_metrics and avg_dd_duration and bench_metrics.get('avg_dd_duration') else "—"),
        ("Avg Recovery Time", f"{int(avg_recovery_time)} days" if avg_recovery_time else "—",
         f"{int(bench_metrics.get('avg_recovery', 0))} days" if bench_metrics and bench_metrics.get('avg_recovery') else "—",
         f"{int(avg_recovery_time or 0) - int(bench_metrics.get('avg_recovery') or 0):+d} days" if bench_metrics and avg_recovery_time and bench_metrics.get('avg_recovery') else "—"),
        ("Max Recovery Time", f"{int(max_recovery_time)} days" if max_recovery_time else "—",
         f"{int(bench_metrics.get('max_recovery', 0))} days" if bench_metrics and bench_metrics.get('max_recovery') else "—",
         f"{int(max_recovery_time or 0) - int(bench_metrics.get('max_recovery') or 0):+d} days" if bench_metrics and max_recovery_time and bench_metrics.get('max_recovery') else "—"),
        ("Ulcer Index", f"{port_ulcer*100:.2f}%",
         f"{bench_metrics.get('ulcer', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_ulcer - bench_metrics.get('ulcer', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Pain Index", f"{port_pain*100:.2f}%",
         f"{bench_metrics.get('pain', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_pain - bench_metrics.get('pain', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 90% (Historical)", f"{port_var_90*100:.2f}%",
         f"{bench_metrics.get('var_90', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_90 - bench_metrics.get('var_90', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Historical)", f"{port_var_95*100:.2f}%",
         f"{bench_metrics.get('var_95', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95 - bench_metrics.get('var_95', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 99% (Historical)", f"{port_var_99*100:.2f}%",
         f"{bench_metrics.get('var_99', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_99 - bench_metrics.get('var_99', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Parametric)", f"{port_var_95_param*100:.2f}%",
         f"{bench_metrics.get('var_95_param', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95_param - bench_metrics.get('var_95_param', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Cornish-Fisher)", f"{port_var_95_cf*100:.2f}%",
         f"{bench_metrics.get('var_95_cf', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95_cf - bench_metrics.get('var_95_cf', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 90%", f"{port_cvar_90*100:.2f}%",
         f"{bench_metrics.get('cvar_90', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_90 - bench_metrics.get('cvar_90', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 95%", f"{port_cvar_95*100:.2f}%",
         f"{bench_metrics.get('cvar_95', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_95 - bench_metrics.get('cvar_95', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 99%", f"{port_cvar_99*100:.2f}%",
         f"{bench_metrics.get('cvar_99', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_99 - bench_metrics.get('cvar_99', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Downside Deviation", f"{port_downside*100:.2f}%",
         f"{bench_metrics.get('downside', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_downside - bench_metrics.get('downside', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Semi-Deviation", f"{port_semi*100:.2f}%",
         f"{bench_metrics.get('semi', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_semi - bench_metrics.get('semi', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Skewness", f"{port_skew:.3f}",
         f"{bench_metrics.get('skew', 0):.3f}" if bench_metrics else "—",
         f"{(port_skew - bench_metrics.get('skew', 0)):+.3f}" if bench_metrics else "—"),
        ("Kurtosis (Excess)", f"{port_kurt:+.3f}",
         f"{bench_metrics.get('kurt', 0):+.3f}" if bench_metrics else "—",
         f"{(port_kurt - bench_metrics.get('kurt', 0)):+.3f}" if bench_metrics else "—"),
    ]
    
    complete_risk_df = pd.DataFrame(metrics_data, columns=["Metric", "Portfolio", "Benchmark", "Difference"])
    
    # Calculate height for all rows (no scroll)
    row_height = 35
    header_height = 40
    total_height = len(complete_risk_df) * row_height + header_height
    
    st.dataframe(complete_risk_df, use_container_width=True, hide_index=True, height=total_height)


def _render_drawdown_analysis(risk, portfolio_values, benchmark_values, portfolio_returns, benchmark_returns):
    """Sub-tab 3.2: Drawdown Analysis."""
    st.subheader("Drawdown Analysis")
    
    if portfolio_values is None or portfolio_values.empty:
        st.warning("No portfolio values data available for drawdown analysis")
        return
    
    # Import needed functions
    from core.analytics_engine.risk_metrics import calculate_top_drawdowns, calculate_drawdown_duration
    from core.analytics_engine.performance import calculate_annualized_return
    
    # Chart 3.2.1: Underwater Plot
    st.markdown("### Underwater Plot")
    underwater_data = get_underwater_plot_data(portfolio_values, benchmark_values)
    if underwater_data:
        fig = plot_underwater(underwater_data)
        st.plotly_chart(fig, use_container_width=True, key="drawdown_underwater")
    
    st.markdown("---")
    
    # Chart 3.2.2: Drawdown Periods
    st.markdown("### Drawdown Periods")
    drawdown_periods_data = get_drawdown_periods_data(portfolio_values, threshold=0.05)
    if drawdown_periods_data:
        fig = plot_drawdown_periods(drawdown_periods_data)
        st.plotly_chart(fig, use_container_width=True, key="drawdown_periods")
    
    st.markdown("---")
    
    # Chart 3.2.3: Drawdown Recovery Visualization
    st.markdown("### Drawdown Recovery Timeline")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        recovery_data = get_drawdown_recovery_data(portfolio_returns, top_n=3)
        
        if recovery_data:
            for dd in recovery_data:
                # Create expander for each drawdown
                with st.expander(
                    f"Drawdown #{dd['number']}: {dd['start_date']} to {dd['recovery_date'] if dd['recovery_date'] else 'Ongoing'}",
                    expanded=(dd['number'] == 1)  # Expand first one
                ):
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Depth", f"{dd['depth']*100:.2f}%")
                    with col2:
                        st.metric("Duration", f"{dd['duration_days']} days")
                    with col3:
                        if dd['recovery_days']:
                            st.metric("Recovery Time", f"{dd['recovery_days']} days")
                        else:
                            st.metric("Recovery Time", "Not recovered")
                    with col4:
                        total_days = dd['duration_days'] + (dd['recovery_days'] if dd['recovery_days'] else 0)
                        st.metric("Total Duration", f"{total_days} days")
                    
                    # Show timeline chart
                    fig = plot_drawdown_recovery(dd)
                    st.plotly_chart(fig, use_container_width=True, key=f"recovery_{dd['number']}")
                    
                    # Show value information
                    if dd['peak_value'] and dd['trough_value']:
                        st.caption(
                            f"**Values:** Peak: {dd['peak_value']:.2f} → "
                            f"Trough: {dd['trough_value']:.2f} ({dd['depth']*100:.2f}%)"
                            + (f" → Recovery: {dd['recovery_value']:.2f}" if dd['recovery_value'] else " → Not recovered")
                        )
        else:
            st.info("No significant drawdowns found (threshold: 0.1%)")
    
    st.markdown("---")
    
    # Table 3.2.4: Top 5 Drawdowns
    st.markdown("### Top 5 Drawdowns")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        top_drawdowns = calculate_top_drawdowns(portfolio_returns, top_n=5)
        
        if top_drawdowns:
            # Create DataFrame for top drawdowns
            dd_data = []
            for i, dd in enumerate(top_drawdowns, 1):
                dd_data.append({
                    "#": i,
                    "Start (Peak)": dd['start_date'].strftime("%Y-%m-%d") if dd['start_date'] else "—",
                    "Bottom (Trough)": dd['bottom_date'].strftime("%Y-%m-%d") if dd['bottom_date'] else "—",
                    "Recovery (End)": dd['recovery_date'].strftime("%Y-%m-%d") if dd['recovery_date'] else "Ongoing",
                    "Depth (%)": f"{dd['depth']*100:.2f}%",
                    "Duration (days)": dd['duration_days'],
                    "Recovery (days)": dd['recovery_days'] if dd['recovery_days'] else "—",
                })
            
            dd_df = pd.DataFrame(dd_data)
            st.dataframe(dd_df, use_container_width=True, hide_index=True)
            
            # Benchmark Comparison (if available)
            st.markdown("#### Benchmark Comparison")
            
            comparison_data = []
            
            # Portfolio metrics
            avg_dd_depth = sum(dd['depth'] for dd in top_drawdowns) / len(top_drawdowns) if top_drawdowns else 0
            avg_dd_duration = sum(dd['duration_days'] for dd in top_drawdowns) / len(top_drawdowns) if top_drawdowns else 0
            avg_recovery = sum(dd['recovery_days'] for dd in top_drawdowns if dd['recovery_days']) / len([dd for dd in top_drawdowns if dd['recovery_days']]) if any(dd['recovery_days'] for dd in top_drawdowns) else None
            max_recovery = max((dd['recovery_days'] for dd in top_drawdowns if dd['recovery_days']), default=None)
            
            comparison_data.append({
                "#": "1",
                "Metric": "Avg Drawdown Depth",
                "Portfolio": f"{avg_dd_depth*100:.2f}%",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "2",
                "Metric": "Avg Drawdown Duration",
                "Portfolio": f"{avg_dd_duration:.0f} days",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "3",
                "Metric": "Avg Recovery Time",
                "Portfolio": f"{avg_recovery:.0f} days" if avg_recovery else "—",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "4",
                "Metric": "Max Recovery Time",
                "Portfolio": f"{max_recovery:.0f} days" if max_recovery else "—",
                "Benchmark": "—"
            })
            
            # Try to calculate benchmark metrics
            if benchmark_returns is not None and not benchmark_returns.empty:
                try:
                    bench_drawdowns = calculate_top_drawdowns(benchmark_returns, top_n=5)
                    
                    if bench_drawdowns:
                        bench_avg_dd_depth = sum(dd['depth'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_dd_duration = sum(dd['duration_days'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_recovery = sum(dd['recovery_days'] for dd in bench_drawdowns if dd['recovery_days']) / len([dd for dd in bench_drawdowns if dd['recovery_days']]) if any(dd['recovery_days'] for dd in bench_drawdowns) else None
                        bench_max_recovery = max((dd['recovery_days'] for dd in bench_drawdowns if dd['recovery_days']), default=None)
                        
                        comparison_data[0]["Benchmark"] = f"{bench_avg_dd_depth*100:.2f}%"
                        comparison_data[1]["Benchmark"] = f"{bench_avg_dd_duration:.0f} days"
                        comparison_data[2]["Benchmark"] = f"{bench_avg_recovery:.0f} days" if bench_avg_recovery else "—"
                        comparison_data[3]["Benchmark"] = f"{bench_max_recovery:.0f} days" if bench_max_recovery else "—"
                except Exception as e:
                    logger.warning(f"Error calculating benchmark drawdown metrics: {e}")
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant drawdowns found")
    else:
        st.warning("No portfolio returns data available")


def _render_var_analysis(risk):
    """Sub-tab 3.3: VaR & CVaR."""
    st.subheader("Value at Risk (VaR) & Conditional VaR")
    
    # Trust Level Slider
    confidence_level = st.slider(
        "Confidence Level",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key="var_confidence_level"
    )
    
    # VaR Comparison Table
    st.markdown("---")
    st.subheader("VaR Methods Comparison")
    var_df = pd.DataFrame({
        "Method": ["Historical", "Parametric", "Monte Carlo", "Cornish-Fisher"],
        f"VaR ({confidence_level}%)": [
            f"{risk.get('var_95', 0)*100:.2f}%",
            f"{risk.get('var_95_parametric', risk.get('var_95', 0))*100:.2f}%",
            f"{risk.get('var_95_mc', risk.get('var_95', 0))*100:.2f}%",
            f"{risk.get('var_95_cf', risk.get('var_95', 0))*100:.2f}%",
        ],
        "CVaR": [
            f"{risk.get('cvar_95', 0)*100:.2f}%",
            f"{risk.get('cvar_95', 0)*100:.2f}%",
            f"{risk.get('cvar_95', 0)*100:.2f}%",
            f"{risk.get('cvar_95', 0)*100:.2f}%",
        ]
    })
    st.dataframe(var_df, use_container_width=True, hide_index=True)


def _render_rolling_risk(portfolio_returns, benchmark_returns, risk_free_rate):
    """Sub-tab 3.4: Rolling Risk Metrics."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Window Size Slider
    window_size = st.slider(
        "Rolling Window Size (days)",
        min_value=21,
        max_value=252,
        value=63,
        step=21,
        key="rolling_window_size"
    )
    
    # Rolling Sharpe
    st.subheader("Rolling Sharpe Ratio")
    sharpe_data = get_rolling_sharpe_data(
        portfolio_returns, benchmark_returns,
        window=window_size, risk_free_rate=risk_free_rate
    )
    if sharpe_data:
        fig = plot_rolling_sharpe(sharpe_data)
        st.plotly_chart(fig, use_container_width=True, key="rolling_sharpe")
    
    # Rolling Beta & Alpha
    st.markdown("---")
    st.subheader("Rolling Beta & Alpha")
    if benchmark_returns is not None and not benchmark_returns.empty:
        beta_alpha_data = get_rolling_beta_alpha_data(
            portfolio_returns, benchmark_returns,
            window=window_size, risk_free_rate=risk_free_rate
        )
        if beta_alpha_data:
            fig = plot_rolling_beta_alpha(beta_alpha_data)
            st.plotly_chart(fig, use_container_width=True, key="rolling_beta_alpha")

    # Rolling Volatility
    st.markdown("---")
    st.subheader("Rolling Volatility")
    st.info("Rolling volatility chart coming soon...")


def _render_assets_tab(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service):
    """Render Assets & Correlations tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Asset Overview & Impact",
        "Correlations",
        "Asset Details & Dynamics"
    ])
    
    with sub_tab1:
        _render_asset_overview(positions)
    
    with sub_tab2:
        _render_correlations(positions, portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_asset_details(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service)


def _render_asset_overview(positions):
    """Sub-tab 4.1: Asset Overview & Impact."""
    st.subheader("Portfolio Positions")
    
    if positions:
        render_position_table(positions)
        
        # Asset Allocation
        st.markdown("---")
        st.subheader("Asset Allocation")
        # Use weight_target if available, otherwise equal weight
        weights = []
        for pos in positions:
            if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                weights.append(pos.weight_target)
            else:
                weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
        
        total_weight = sum(weights)
        if total_weight > 0:
            # Build mapping ticker -> weight %
            alloc_data = {}
            for pos, w in zip(positions, weights):
                pct = (w / total_weight * 100)
                alloc_data[pos.ticker] = alloc_data.get(pos.ticker, 0.0) + pct
            fig = plot_asset_allocation(alloc_data)
            st.plotly_chart(fig, use_container_width=True, key="assets_allocation")

        # Impact on Return/Risk
        st.markdown("---")
        st.subheader("Impact Analysis")
        st.info("Impact on return and risk charts coming soon...")
    else:
        st.info("No positions found")


def _render_correlations(positions, portfolio_returns, benchmark_returns):
    """Sub-tab 4.2: Correlations."""
    st.subheader("Correlation Analysis")
    
    if not positions:
        st.info("No positions available for correlation analysis")
        return
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.info("Correlation matrix heatmap implementation coming soon...")
    
    # Correlation with Benchmark
    st.markdown("---")
    st.subheader("Correlation with Benchmark")
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.info("Asset correlation with benchmark chart coming soon...")
    
    # Cluster Analysis
    st.markdown("---")
    st.subheader("Cluster Analysis")
    st.info("Correlation clustering dendrogram coming soon...")


def _render_asset_details(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service):
    """Sub-tab 4.3: Asset Details & Dynamics."""
    st.subheader("Asset Price Dynamics")
    
    if not positions:
        st.info("No positions available")
        return
    
    # Asset Multi-Select
    ticker_list = [pos.ticker for pos in positions]
    selected_tickers = st.multiselect(
        "Select Assets for Comparison",
        options=ticker_list,
        default=ticker_list[:5] if len(ticker_list) > 5 else ticker_list,
        key="asset_selector"
    )
    
    st.info("Asset price dynamics chart coming soon...")
    
    # Detailed Single Asset Analysis
    st.markdown("---")
    st.subheader("Detailed Asset Analysis")
    selected_ticker = st.selectbox(
        "Select asset for detailed analysis",
        options=ticker_list,
        key="detailed_asset_selector"
    )
    
    st.info(f"Detailed analysis for {selected_ticker} coming soon...")


def _render_export_tab(analytics, portfolio_name, portfolio_id):
    """Render Export & Reports tab."""
    st.subheader("Export Analytics Data")
    
    report_service = ReportService()
    
    # Extract metrics for export
    perf = analytics.get("performance", {})
    risk = analytics.get("risk", {})
    ratios = analytics.get("ratios", {})
    market = analytics.get("market", {})
    
    metrics_for_export = {
        "performance": perf,
        "risk": risk,
        "ratios": ratios,
        "market": market,
    }
    
    # Multiple Metrics Comparison Table
    st.subheader("Complete Metrics Comparison")
    st.info("Full 70+ metrics comparison table coming soon...")
    
    # Export Options
    st.markdown("---")
    st.subheader("Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = report_service.generate_csv_report(metrics_for_export)
        st.download_button(
            label="📄 Export to CSV",
            data=csv_data,
            file_name=f"portfolio_metrics_{portfolio_id[:8]}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        json_report = report_service.generate_json_report(
            portfolio_name, metrics_for_export, analytics.get("portfolio_returns")
        )
        json_data = json.dumps(json_report, indent=2, default=str)
        st.download_button(
            label="📄 Export to JSON",
            data=json_data,
            file_name=f"portfolio_metrics_{portfolio_id[:8]}.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )
    
    # PDF Report
    st.markdown("---")
    st.subheader("PDF Tearsheet Generation")
    st.info("PDF tearsheet generation (quantstats-style) coming soon...")
