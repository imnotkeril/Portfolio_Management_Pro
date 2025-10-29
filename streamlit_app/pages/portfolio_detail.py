"""Portfolio detail page showing portfolio information and positions."""

import logging

import streamlit as st

from core.exceptions import PortfolioNotFoundError
from services.data_service import DataService
from services.portfolio_service import PortfolioService
from streamlit_app.components.position_table import render_position_table
from streamlit_app.utils.formatters import (
    format_currency,
    format_date,
    format_percentage,
)

logger = logging.getLogger(__name__)


def render_portfolio_detail() -> None:
    """Render the portfolio detail page."""
    # Get portfolio ID from session state or query params
    portfolio_id = st.session_state.get("selected_portfolio_id") or st.query_params.get("id")

    if not portfolio_id:
        st.error("No portfolio selected")
        st.info("Please select a portfolio from the list")
        if st.button("Go to Portfolio List"):
            st.switch_page("pages/portfolio_list.py")
        return

    # Initialize services
    portfolio_service = PortfolioService()
    data_service = DataService()

    try:
        # Fetch portfolio
        portfolio = portfolio_service.get_portfolio(portfolio_id)

        # Page header
        st.title(portfolio.name)

        if portfolio.description:
            st.write(portfolio.description)

        st.markdown("---")

        # Portfolio metrics
        try:
            metrics = portfolio_service.calculate_portfolio_metrics(portfolio_id)
            current_value = metrics.get("current_value", portfolio.starting_capital)
            weights = metrics.get("weights", {})
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            current_value = portfolio.starting_capital
            weights = {}

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Starting Capital", format_currency(portfolio.starting_capital))
        with col2:
            st.metric("Current Value", format_currency(current_value))
        with col3:
            total_return = (
                (current_value - portfolio.starting_capital) / portfolio.starting_capital
                if portfolio.starting_capital > 0
                else 0.0
            )
            st.metric(
                "Total Return",
                format_percentage(total_return),
                delta=None,
            )
        with col4:
            num_positions = len(portfolio.get_all_positions())
            st.metric("Positions", num_positions)

        st.markdown("---")

        # Actions
        st.subheader("Actions")
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)

        with action_col1:
            if st.button("Edit Portfolio"):
                st.session_state.edit_portfolio_id = portfolio_id
                st.info("Edit functionality coming soon")

        with action_col2:
            if st.button("Add Position"):
                st.session_state.show_add_position = True

        with action_col3:
            if st.button("Analyze Portfolio"):
                st.session_state.selected_portfolio_id = portfolio_id
                st.switch_page("pages/portfolio_analysis.py")

        with action_col4:
            if st.button("Clone Portfolio"):
                _handle_clone(portfolio_service, portfolio_id)

        st.markdown("---")

        # Positions section
        st.subheader("Positions")

        # Fetch current prices and prepare position data
        positions = portfolio.get_all_positions()
        position_data = []

        for pos in positions:
            try:
                current_price = data_service.fetch_current_price(pos.ticker)
                current_value = pos.shares * current_price
                weight = weights.get(pos.ticker, 0.0)
                purchase_value = (
                    pos.shares * pos.purchase_price
                    if pos.purchase_price
                    else None
                )
                gain_loss = (
                    current_value - purchase_value
                    if purchase_value
                    else None
                )

                position_data.append({
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "current_price": current_price,
                    "current_value": current_value,
                    "weight": weight,
                    "gain_loss": gain_loss,
                })
            except Exception as e:
                logger.warning(f"Error fetching price for {pos.ticker}: {e}")
                position_data.append({
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "current_price": None,
                    "current_value": None,
                    "weight": weights.get(pos.ticker, 0.0),
                    "gain_loss": None,
                })

        # Display positions table
        render_position_table(position_data, show_actions=True)

        # Handle add position
        if st.session_state.get("show_add_position"):
            _render_add_position_form(portfolio_service, portfolio_id)
            st.session_state.show_add_position = False

        st.markdown("---")

        # Portfolio details
        with st.expander("Portfolio Details"):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.write(f"**ID:** {portfolio.id}")
                st.write(f"**Base Currency:** {portfolio.base_currency}")
                st.write(
                    f"**Created:** {format_date(portfolio.created_date)}"
                    if hasattr(portfolio, "created_date")
                    else "**Created:** -"
                )
            with detail_col2:
                st.write(f"**Total Positions:** {num_positions}")
                st.write(
                    f"**Unrealized Gain/Loss:** "
                    f"{format_currency(sum(p.get('gain_loss', 0) or 0 for p in position_data))}"
                )

    except PortfolioNotFoundError:
        st.error(f"Portfolio {portfolio_id} not found")
        if st.button("Go to Portfolio List"):
            st.switch_page("pages/portfolio_list.py")
    except Exception as e:
        logger.error(f"Error loading portfolio detail: {e}", exc_info=True)
        st.error(f"Error loading portfolio detail: {e}")


def _handle_clone(portfolio_service: PortfolioService, portfolio_id: str) -> None:
    """Handle portfolio cloning."""
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    new_name = f"{portfolio.name} (Copy)"

    try:
        with st.spinner("Cloning portfolio..."):
            cloned = portfolio_service.clone_portfolio(portfolio_id, new_name)
        st.success(f"Portfolio cloned: {cloned.name}")
        st.info(f"New Portfolio ID: {cloned.id}")
        st.rerun()
    except Exception as e:
        logger.error(f"Error cloning portfolio: {e}", exc_info=True)
        st.error(f"Error cloning portfolio: {e}")


def _render_add_position_form(
    portfolio_service: PortfolioService, portfolio_id: str
) -> None:
    """Render form for adding a position."""
    with st.form("add_position_form"):
        st.subheader("Add Position")

        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", key="add_ticker")
        with col2:
            shares = st.number_input(
                "Shares",
                min_value=0.01,
                value=1.0,
                key="add_shares",
            )

        col3, col4 = st.columns(2)
        with col3:
            weight_target = st.number_input(
                "Weight Target (0.0-1.0)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                key="add_weight",
            )
        with col4:
            purchase_price = st.number_input(
                "Purchase Price",
                min_value=0.01,
                value=0.0,
                key="add_price",
            )

        submitted = st.form_submit_button("Add Position")

        if submitted:
            from services.schemas import AddPositionRequest

            try:
                request = AddPositionRequest(
                    ticker=ticker,
                    shares=shares,
                    weight_target=weight_target if weight_target > 0 else None,
                    purchase_price=purchase_price if purchase_price > 0 else None,
                )

                portfolio_service.add_position(portfolio_id, request)
                st.success(f"Position {ticker} added successfully")
                st.rerun()
            except Exception as e:
                logger.error(f"Error adding position: {e}", exc_info=True)
                st.error(f"Error adding position: {e}")


if __name__ == "__main__":
    render_portfolio_detail()
