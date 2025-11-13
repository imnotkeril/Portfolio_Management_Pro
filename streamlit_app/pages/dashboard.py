"""Dashboard page with market data and navigation."""

import streamlit as st

from services.data_service import DataService
from streamlit_app.components.market_index_card import render_market_index_card
from streamlit_app.components.market_indices_chart import (
    get_market_indices_data,
    plot_market_indices_comparison,
)
from streamlit_app.components.market_stats import render_market_stats
from streamlit_app.utils.market_data import get_index_data


def render_dashboard() -> None:
    """Render the dashboard page with market data and navigation."""
    st.title("Market Dashboard")
    st.markdown("---")

    # Navigation buttons section
    st.subheader("Quick Navigation")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("Portfolio List", use_container_width=True):
            st.switch_page("pages/portfolio_list.py")

    with col2:
        if st.button("Create Portfolio", use_container_width=True):
            st.switch_page("pages/create_portfolio.py")

    with col3:
        if st.button("Analysis", use_container_width=True):
            st.switch_page("pages/portfolio_analysis.py")

    with col4:
        if st.button("Optimization", use_container_width=True):
            st.switch_page("pages/portfolio_optimization.py")

    with col5:
        if st.button("Risk Analysis", use_container_width=True):
            st.switch_page("pages/risk_analysis.py")

    with col6:
        if st.button("Forecasting", use_container_width=True):
            st.switch_page("pages/forecasting.py")

    st.markdown("<br>", unsafe_allow_html=True)

    # Market indices section
    st.subheader("Market Indices")
    st.markdown("---")

    # Define indices
    indices = [
        {"name": "S&P 500", "symbol": "^GSPC"},
        {"name": "NASDAQ", "symbol": "^NDX"},
        {"name": "Dow Jones", "symbol": "^DJI"},
        {"name": "Russell 2000", "symbol": "^RUT"},
    ]

    # Initialize data service
    data_service = DataService()

    # Display index cards
    cols = st.columns(4)
    for i, index_info in enumerate(indices):
        with cols[i]:
            index_data = get_index_data(index_info["symbol"], data_service)
            render_market_index_card(
                name=index_info["name"],
                symbol=index_info["symbol"],
                current_price=index_data.get("price"),
                change=index_data.get("change"),
                change_pct=index_data.get("change_pct"),
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Indices comparison chart (always 1 year)
    st.subheader("Indices Comparison")
    
    period_days = 365  # Always 1 year

    # Fetch and display chart
    with st.spinner("Loading chart data..."):
        chart_data = get_market_indices_data(indices, period_days, data_service)

        if chart_data is not None and not chart_data.empty:
            fig = plot_market_indices_comparison(
                chart_data,
                title="Market Indices Comparison (1 Year)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load chart data. Please try again later.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Market statistics section
    st.subheader("Market Statistics")
    st.markdown("---")
    render_market_stats(data_service)


if __name__ == "__main__":
    render_dashboard()
