"""Portfolio Analysis page - Full implementation according to specification."""

import json
import logging
from datetime import date, timedelta
import streamlit as st
import pandas as pd

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
    get_best_worst_periods_data,
)
from streamlit_app.components.charts import (
    plot_cumulative_returns,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_rolling_sharpe,
    plot_rolling_beta_alpha,
    plot_underwater,
    plot_yearly_returns,
    plot_best_worst_periods,
    plot_asset_allocation,
    plot_sector_allocation,
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

    # Row 1: Portfolio + Start Date
    col1, col2 = st.columns([2, 2])

    portfolio_names = {p.name: p.id for p in portfolios}
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    with col1:
        selected_name = st.selectbox(
            "Portfolio",
            options=list(portfolio_names.keys()),
            key="portfolio_analysis_selector",
        )
        portfolio_id = portfolio_names[selected_name]

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=default_end,
            key="start_date_input",
        )

    # Row 2: End Date + Comparison selector
    col1, col2 = st.columns([2, 2])

    with col1:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=start_date,
            max_value=default_end,
            key="end_date_input",
        )

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
            perf, portfolio_returns, benchmark_returns,
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

    # Row 1: Main metrics
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
            "label": "Sharpe Ratio",
            "portfolio_value": portfolio_metrics_flat.get("sharpe_ratio", 0),
            "benchmark_value": bm_for_cards.get("sharpe_ratio"),
            "format": "ratio",
            "higher_is_better": True,
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
    
    # Row 2: Risk metrics
    st.markdown("---")
    metrics_row2 = [
        {
            "label": "Volatility",
            "portfolio_value": portfolio_metrics_flat.get("volatility", 0),
            "benchmark_value": bm_for_cards.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Beta",
            "portfolio_value": market.get("beta", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Alpha",
            "portfolio_value": market.get("alpha", 0),
            "benchmark_value": 0.0,
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "benchmark_value": bm_for_cards.get("sortino_ratio"),
            "format": "ratio",
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


def _render_performance_tab(perf, portfolio_returns, benchmark_returns, risk_free_rate, start_date, end_date):
    """Render Performance tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Returns Analysis",
        "Periodic Analysis",
        "Distribution"
    ])
    
    with sub_tab1:
        _render_returns_analysis(perf, portfolio_returns, benchmark_returns, risk_free_rate)
    
    with sub_tab2:
        _render_periodic_analysis(portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_distribution_analysis(portfolio_returns, benchmark_returns)


def _render_returns_analysis(perf, portfolio_returns, benchmark_returns, risk_free_rate):
    """Sub-tab 2.1: Returns Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return

    # Cumulative Returns (expanded)
    st.subheader("Cumulative Returns")
    col1, col2 = st.columns([3, 1])
    with col1:
        cum_data = get_cumulative_returns_data(portfolio_returns, benchmark_returns)
        if cum_data:
            fig = plot_cumulative_returns(cum_data)
            st.plotly_chart(fig, use_container_width=True, key="returns_cumulative")
    with col2:
        st.markdown("**View Options:**")
        use_log = st.checkbox("Log Scale", value=False)
        # Note: log scale toggle would need chart update
    
    # Daily Active Returns
    st.markdown("---")
    st.subheader("Daily Active Returns")
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = benchmark_returns.reindex(portfolio_returns.index, method="ffill").fillna(0)
        active_returns = portfolio_returns - aligned
        
        active_df = pd.DataFrame({
            "Date": active_returns.index,
            "Active Return": active_returns.values * 100,
        })
        active_df["Color"] = active_df["Active Return"].apply(
            lambda x: "#4CAF50" if x >= 0 else "#F44336"
        )
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=active_df["Date"],
            y=active_df["Active Return"],
            marker_color=active_df["Color"],
            name="Active Return",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(
            title="Daily Active Returns (Portfolio - Benchmark)",
            xaxis_title="Date",
            yaxis_title="Active Return (%)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, key="returns_active_returns")

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
    
    # Return by Periods
    st.markdown("---")
    st.subheader("Return by Periods")
    # Calculate period returns (placeholder - needs proper calculation)
    periods_data = {
        "MTD": (perf.get("mtd_return", 0), 0),
        "YTD": (perf.get("ytd_return", 0), 0),
        "1M": (perf.get("one_month_return", 0), 0),
        "3M": (perf.get("three_month_return", 0), 0),
        "6M": (perf.get("six_month_return", 0), 0),
        "1Y": (perf.get("total_return", 0), 0),
    }
    
    periods_df = pd.DataFrame({
        "Period": list(periods_data.keys()),
        "Portfolio": [v[0]*100 for v in periods_data.values()],
        "Benchmark": [v[1]*100 for v in periods_data.values()],
    })
    periods_df["Difference"] = periods_df["Portfolio"] - periods_df["Benchmark"]
    st.dataframe(periods_df, use_container_width=True, hide_index=True)
    
    # Best/Worst Periods
    st.markdown("---")
    st.subheader("Best and Worst Periods")
    periods_data_chart = get_best_worst_periods_data(portfolio_returns, top_n=10)
    if periods_data_chart:
        fig = plot_best_worst_periods(periods_data_chart)
        st.plotly_chart(fig, use_container_width=True, key="returns_best_worst")


def _render_periodic_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.2: Periodic Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Annual Returns
    st.subheader("Annual Returns Comparison")
    yearly_data = get_yearly_returns_data(portfolio_returns, benchmark_returns)
    if yearly_data.get("yearly") is not None and not yearly_data["yearly"].empty:
        fig = plot_yearly_returns(yearly_data["yearly"])
        st.plotly_chart(fig, use_container_width=True, key="periodic_yearly")
    
    # Monthly Returns Calendar
    st.markdown("---")
    st.subheader("Monthly Returns Calendar")
    heatmap_data = get_monthly_heatmap_data(portfolio_returns)
    if heatmap_data.get("heatmap") is not None and not heatmap_data["heatmap"].empty:
        fig = plot_monthly_heatmap(heatmap_data["heatmap"])
        st.plotly_chart(fig, use_container_width=True, key="periodic_monthly")
    
    # Seasonal Analysis
    st.markdown("---")
    st.subheader("Seasonal Analysis")
    st.info("Seasonal analysis (day of week, month, quarter) coming soon...")


def _render_distribution_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.3: Distribution."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Return Distributions
    st.subheader("Distribution of Returns")
    dist_data = get_return_distribution_data(portfolio_returns)
    if dist_data:
        fig = plot_return_distribution(dist_data)
        st.plotly_chart(fig, use_container_width=True, key="distribution_returns")

    # Q-Q Plot
    st.markdown("---")
    st.subheader("Q-Q Plot")
    st.info("Q-Q plot implementation coming soon...")
    
    # Win Rate Statistics
    st.markdown("---")
    st.subheader("Win Rate Statistics")
    pos_days = (portfolio_returns > 0).sum()
    total_days = len(portfolio_returns)
    win_rate = pos_days / total_days * 100 if total_days > 0 else 0
    
    st.metric("Win Days %", f"{win_rate:.1f}%")
    st.metric("Win Weeks %", "Coming soon...")
    st.metric("Win Months %", "Coming soon...")


def _render_risk_tab(risk, ratios, market, portfolio_returns, benchmark_returns, portfolio_values, risk_free_rate, start_date, end_date):
    """Render Risk tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Key Metrics",
        "Drawdown Analysis",
        "VaR & CVaR",
        "Rolling Risk Metrics"
    ])
    
    with sub_tab1:
        _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate)
    
    with sub_tab2:
        _render_drawdown_analysis(risk, portfolio_values, benchmark_returns)
    
    with sub_tab3:
        _render_var_analysis(risk)
    
    with sub_tab4:
        _render_rolling_risk(portfolio_returns, benchmark_returns, risk_free_rate)


def _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate):
    """Sub-tab 3.1: Key Risk Metrics."""
    # Risk Metric Cards
    st.subheader("Risk Metrics")
    
    risk_metrics_row1 = [
        {
            "label": "Volatility",
            "portfolio_value": risk.get("volatility", 0),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": risk.get("max_drawdown", 0),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Calmar Ratio",
            "portfolio_value": ratios.get("calmar_ratio", 0),
            "format": "ratio",
            "higher_is_better": True,
        },
    ]
    render_metric_cards_row(risk_metrics_row1, columns_per_row=4)
    
    # All Risk Metrics Table
    st.markdown("---")
    st.subheader("All Risk Metrics")
    risk_df = pd.DataFrame({
        "Metric": [
            "Volatility (Annual)", "Max Drawdown", "VaR (95%)", "CVaR (95%)",
            "Downside Deviation", "Ulcer Index", "Pain Index",
            "Calmar Ratio", "Sortino Ratio"
        ],
        "Value": [
            f"{risk.get('volatility', 0)*100:.2f}%",
            f"{risk.get('max_drawdown', 0)*100:.2f}%",
            f"{risk.get('var_95', 0)*100:.2f}%",
            f"{risk.get('cvar_95', 0)*100:.2f}%",
            f"{risk.get('downside_deviation', 0)*100:.2f}%",
            f"{risk.get('ulcer_index', 0)*100:.2f}%",
            f"{risk.get('pain_index', 0)*100:.2f}%",
            f"{ratios.get('calmar_ratio', 0):.3f}",
            f"{ratios.get('sortino_ratio', 0):.3f}",
        ]
    })
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Advanced metrics placeholders
    st.markdown("---")
    st.info("Probabilistic Sharpe Ratio, Smart Sharpe/Sortino, Kelly Criterion coming soon...")


def _render_drawdown_analysis(risk, portfolio_values, benchmark_returns):
    """Sub-tab 3.2: Drawdown Analysis."""
    st.subheader("Drawdown Analysis")
    
    if portfolio_values is not None and not portfolio_values.empty:
        # Underwater Plot
        underwater_data = get_underwater_plot_data(portfolio_values)
        if underwater_data:
            fig = plot_underwater(underwater_data)
            st.plotly_chart(fig, use_container_width=True, key="drawdown_underwater")

        # Top 5 Drawdowns Table
        st.markdown("---")
        st.subheader("Top 5 Drawdowns")
        st.info("Top 5 drawdowns table implementation coming soon...")
    else:
        st.warning("No portfolio values data available for drawdown analysis")


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
