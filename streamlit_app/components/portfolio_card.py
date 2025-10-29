"""Portfolio card component for displaying portfolio summary."""

from typing import Optional

import streamlit as st

from streamlit_app.utils.formatters import format_currency, format_percentage


def render_portfolio_card(
    portfolio_id: str,
    name: str,
    current_value: float,
    starting_capital: float,
    num_positions: int,
    daily_change: Optional[float] = None,
    created_date: Optional[str] = None,
) -> None:
    """
    Render a portfolio summary card.

    Args:
        portfolio_id: Portfolio ID
        name: Portfolio name
        current_value: Current portfolio value
        starting_capital: Starting capital
        num_positions: Number of positions
        daily_change: Daily change percentage (optional)
        created_date: Creation date (optional)
    """
    # Calculate total return
    total_return = (
        (current_value - starting_capital) / starting_capital
        if starting_capital > 0
        else 0.0
    )

    # Format daily change HTML
    daily_change_html = ""
    if daily_change is not None:
        daily_color = "#74F174" if daily_change >= 0 else "#FAA1A4"
        daily_value = format_percentage(daily_change)
        daily_change_html = (
            f'<p style="color: #D1D4DC; margin: 4px 0;">'
            f'Daily: <strong style="color: {daily_color}">{daily_value}</strong>'
            f'</p>'
        )

    # Card container
    return_color = "#74F174" if total_return >= 0 else "#FAA1A4"
    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 1px solid #2A2E39;
                border-radius: 8px;
                padding: 16px;
                background-color: #1A1E29;
                margin-bottom: 16px;
            ">
                <h3 style="color: #BF9FFB; margin-top: 0;">
                    {name}
                </h3>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p style="color: #D1D4DC; margin: 4px 0;">
                            Value: <strong style="color: #FFFFFF;">
                            {format_currency(current_value)}
                            </strong>
                        </p>
                        <p style="color: #D1D4DC; margin: 4px 0;">
                            Return: <strong style="color: {return_color}">
                            {format_percentage(total_return)}
                            </strong>
                        </p>
                    </div>
                    <div>
                        <p style="color: #D1D4DC; margin: 4px 0;">
                            Positions: <strong style="color: #FFFFFF;">
                            {num_positions}
                            </strong>
                        </p>
                        {daily_change_html}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("View Details", key=f"view_{portfolio_id}"):
                st.session_state.selected_portfolio_id = portfolio_id
                st.switch_page("pages/portfolio_detail.py")

        with col2:
            if st.button("Analyze", key=f"analyze_{portfolio_id}"):
                st.session_state.selected_portfolio_id = portfolio_id
                st.switch_page("pages/portfolio_analysis.py")

        with col3:
            if st.button("Delete", key=f"delete_{portfolio_id}"):
                st.session_state.portfolio_to_delete = portfolio_id
