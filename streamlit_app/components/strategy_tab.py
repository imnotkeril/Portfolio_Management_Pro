"""Streamlit Strategies tab (Phase 5)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from services.portfolio_service import PortfolioService
from services.schemas import UpdateStrategyRequest
from services.strategy_service import StrategyService


def render_strategy_tab(portfolio_id: str) -> None:
    """Edit target weights, rebalance schedule, and preview trades."""
    portfolio_service: PortfolioService = st.session_state.portfolio_service
    strategy_service = StrategyService(portfolio_service=portfolio_service)

    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
    except Exception as exc:
        st.error(f"Failed to load portfolio: {exc}")
        return

    snap = strategy_service.get_strategy(portfolio_id)

    st.markdown(
        "Target weights and rebalance frequency drive scheduled maintenance "
        "for transaction-led portfolios (same engine as the Next.js Strategies tab)."
    )

    interval_options = {
        "Off": None,
        "Every 1 month": 1,
        "Every 3 months": 3,
        "Every 6 months": 6,
        "Every 12 months": 12,
    }
    current_label = next(
        (
            label
            for label, months in interval_options.items()
            if months == snap.rebalance_interval_months
        ),
        "Off",
    )
    interval_label = st.selectbox(
        "Rebalance frequency",
        list(interval_options.keys()),
        index=list(interval_options.keys()).index(current_label),
    )
    interval_months = interval_options[interval_label]

    positions = portfolio.get_all_positions()
    if not positions:
        st.info("No positions yet. Add holdings or transactions first.")
        return

    rows = []
    for pos in sorted(positions, key=lambda p: (p.ticker == "CASH", p.ticker)):
        wt = pos.weight_target if pos.weight_target and pos.weight_target > 0 else 0.0
        rows.append(
            {
                "Ticker": pos.ticker,
                "Target %": round(wt * 100.0, 2),
            }
        )
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Target %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0),
        },
    )

    total_pct = float(edited["Target %"].sum()) if not edited.empty else 0.0
    if abs(total_pct - 100.0) > 0.05:
        st.warning(f"Targets sum to {total_pct:.1f}% — must equal 100%.")
    else:
        st.caption(f"Targets sum to {total_pct:.1f}%.")

    col_save, col_preview = st.columns(2)
    with col_save:
        save = st.button(
            "Save strategy", type="primary", disabled=abs(total_pct - 100.0) > 0.05
        )
    with col_preview:
        preview = st.button("Preview rebalance")

    if save:
        targets = {
            str(row["Ticker"]).strip().upper(): float(row["Target %"]) / 100.0
            for _, row in edited.iterrows()
            if float(row["Target %"]) > 0
        }
        try:
            req = UpdateStrategyRequest(
                rebalance_interval_months=interval_months,
                targets=targets,
                replace_targets=True,
            )
            strategy_service.update_strategy(portfolio_id, req)
            st.success("Strategy saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if preview:
        try:
            plan = strategy_service.preview_rebalance(portfolio_id, date.today())
            st.info(plan.message)
            if plan.trades:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Ticker": t.ticker,
                                "Action": t.action,
                                "Shares": t.shares,
                                "Price": t.price,
                                "Target weight": f"{t.target_weight * 100:.1f}%",
                            }
                            for t in plan.trades
                        ]
                    ),
                    hide_index=True,
                )
            else:
                st.caption("No trades in preview.")
        except Exception as exc:
            st.error(str(exc))
