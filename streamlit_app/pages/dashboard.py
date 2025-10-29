"""Dashboard page showing portfolio overview."""

import logging

import streamlit as st

from services.portfolio_service import PortfolioService
from streamlit_app.components.portfolio_card import render_portfolio_card

logger = logging.getLogger(__name__)


def render_dashboard() -> None:
    """Render the dashboard page."""
    st.title("Portfolio Dashboard")

    # Initialize services
    portfolio_service = PortfolioService()

    # Fetch all portfolios
    try:
        portfolios = portfolio_service.list_portfolios()

        if not portfolios:
            st.info("No portfolios yet. Create one to get started!")
            if st.button("Create Portfolio"):
                st.switch_page("pages/create_portfolio.py")
            return

        # Display summary stats
        total_portfolios = len(portfolios)

        # Calculate total value with prices
        total_value = 0.0
        for p in portfolios:
            try:
                # Use PortfolioService to calculate metrics with prices
                metrics = portfolio_service.calculate_portfolio_metrics(p.id)
                total_value += metrics.get("current_value", p.starting_capital)
            except Exception as e:
                logger.warning(
                    f"Error calculating value for portfolio {p.id}: {e}"
                )
                # Fallback to starting capital
                total_value += p.starting_capital

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolios", total_portfolios)
        col2.metric(
            "Total Value",
            f"${total_value:,.2f}" if total_value else "$0.00",
        )
        col3.metric(
            "Avg Portfolio Size",
            (
                f"${total_value / total_portfolios:,.2f}"
                if total_portfolios > 0
                else "$0.00"
            ),
        )

        st.markdown("---")

        # Display portfolio cards
        st.subheader("Your Portfolios")

        for portfolio in portfolios:
            # Calculate current metrics
            try:
                metrics = portfolio_service.calculate_portfolio_metrics(
                    portfolio.id
                )
                current_value = metrics.get(
                    "current_value", portfolio.starting_capital
                )
                num_positions = metrics.get("positions_count", 0)
                daily_change = None  # TODO: Calculate daily change
            except Exception as e:
                logger.warning(
                    f"Error calculating metrics for portfolio {portfolio.id}: {e}"
                )
                current_value = portfolio.starting_capital
                num_positions = (
                    len(portfolio.positions)
                    if hasattr(portfolio, "positions")
                    else 0
                )
                daily_change = None

            render_portfolio_card(
                portfolio_id=portfolio.id,
                name=portfolio.name,
                current_value=current_value,
                starting_capital=portfolio.starting_capital,
                num_positions=num_positions,
                daily_change=daily_change,
                created_date=(
                    str(portfolio.created_date)
                    if hasattr(portfolio, "created_date")
                    else None
                ),
            )

        # Handle delete action
        if "portfolio_to_delete" in st.session_state:
            portfolio_id = st.session_state.portfolio_to_delete
            del st.session_state.portfolio_to_delete

            try:
                portfolio_service.delete_portfolio(portfolio_id)
                st.success("Portfolio deleted successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting portfolio: {e}")

    except Exception as e:
        logger.error(f"Error loading dashboard: {e}", exc_info=True)
        st.error(f"Error loading dashboard: {e}")


if __name__ == "__main__":
    render_dashboard()
