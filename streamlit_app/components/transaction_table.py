"""Transaction table component for displaying transactions."""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd
import streamlit as st

from core.data_manager.transaction import Transaction
from streamlit_app.utils.chart_config import COLORS
from streamlit_app.utils.formatters import format_currency

logger = logging.getLogger(__name__)


def render_transaction_table(
    transactions: List[Transaction],
    portfolio_id: str,
    show_actions: bool = True,
) -> Optional[str]:
    """
    Render transactions table with edit/delete actions.

    Args:
        transactions: List of Transaction domain models
        portfolio_id: Portfolio ID
        show_actions: Whether to show action buttons

    Returns:
        Transaction ID to delete if delete button clicked, None otherwise
    """
    if not transactions:
        st.info("No transactions found. Add your first transaction to get started.")
        return None

    # Prepare data for table
    data = []
    for txn in transactions:
        # Color code transaction type
        type_color = {
            "BUY": COLORS["success"],
            "SELL": COLORS["danger"],
            "DEPOSIT": COLORS["info"],
            "WITHDRAWAL": COLORS["warning"],
        }.get(txn.transaction_type, "#D1D4DC")

        data.append(
            {
                "Date": txn.transaction_date.strftime("%Y-%m-%d"),
                "Type": txn.transaction_type,  # Plain text for dataframe
                "Ticker": txn.ticker,
                "Shares": f"{txn.shares:,.4f}",
                "Price": format_currency(txn.price),
                "Amount": format_currency(txn.amount),
                "Fees": format_currency(txn.fees or 0.0),
                "Notes": txn.notes or "",
                "_id": txn.id,  # Hidden column for ID
            }
        )

    df = pd.DataFrame(data)

    # Display table
    st.markdown("### Transaction History")
    st.dataframe(
        df.drop(columns=["_id"]),  # Hide ID column
        use_container_width=True,
        hide_index=True,
    )

    # Actions (if enabled)
    if show_actions and transactions:
        st.subheader("Actions")

        # Delete transaction
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create options with date and type for clarity
            transaction_options = [
                f"{txn.transaction_date.strftime('%Y-%m-%d')} - {txn.transaction_type} - {txn.ticker} ({txn.shares:.2f} @ {txn.price:.2f})"
                for txn in transactions
            ]
            selected_index = st.selectbox(
                "Select transaction to delete",
                options=range(len(transaction_options)),
                format_func=lambda i: transaction_options[i],
            )
        with col2:
            if st.button("Delete", type="secondary", use_container_width=True):
                selected_transaction = transactions[selected_index]
                return selected_transaction.id

        return None

    return None

