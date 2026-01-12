"""Portfolio list page with full CRUD operations."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from services.data_service import DataService
from streamlit_app.components.position_table import render_position_table
from streamlit_app.components.charts import (
    plot_asset_allocation,
    plot_sector_allocation,
)
from core.data_manager.ticker_validator import TickerValidator
from services.portfolio_service import PortfolioService
from services.schemas import (
    AddPositionRequest,
    UpdatePortfolioRequest,
    UpdatePositionRequest,
)
from streamlit_app.utils.formatters import format_currency

logger = logging.getLogger(__name__)


def render_portfolio_list() -> None:
    """Main function to render the portfolio list page."""
    st.title("Portfolio Management")

    # Initialize services
    portfolio_service = PortfolioService()
    data_service = DataService()

    # Store services in session state
    st.session_state.portfolio_service = portfolio_service
    st.session_state.data_service = data_service

    # Initialize session state for management operations
    if 'selected_portfolios' not in st.session_state:
        st.session_state.selected_portfolios = []

    if 'deleted_portfolios' not in st.session_state:
        st.session_state.deleted_portfolios = []

    if 'management_view' not in st.session_state:
        st.session_state.management_view = "list"

    # Main interface
    if st.session_state.management_view == "edit":
        render_portfolio_editor()
    elif st.session_state.management_view == "view":
        render_portfolio_view()
    else:
        render_portfolio_list_view()


def render_portfolio_list_view() -> None:
    """Render the main portfolio list view."""
    # Action bar
    render_action_bar()

    # Load portfolios
    portfolios_data = load_portfolios_with_cache()

    if not portfolios_data:
        render_empty_state()
        return

    # Search and filter
    filtered_portfolios = render_search_and_filter(portfolios_data)

    if not filtered_portfolios:
        st.info("No portfolios match your search criteria.")
        return

    # Bulk operations
    if st.session_state.selected_portfolios:
        render_bulk_operations()
        st.divider()

    # Portfolio table
    render_portfolio_table(filtered_portfolios)

    # Individual portfolio actions
    render_individual_actions(filtered_portfolios)

    # Undo section
    if st.session_state.deleted_portfolios:
        render_undo_section()


def render_action_bar() -> None:
    """Render the main action bar with primary actions."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        st.subheader("Your Portfolios")

    with col2:
        if st.button("Create New", type="primary", use_container_width=True):
            st.switch_page("pages/create_portfolio.py")

    with col3:
        if st.button("Refresh", use_container_width=True):
            clear_portfolio_cache()
            st.rerun()

    with col4:
        st.button("Export All", use_container_width=True, disabled=True,
                  help="Export functionality coming soon")


def load_portfolios_with_cache() -> List[Dict[str, Any]]:
    """Load portfolios with caching for performance."""
    cache_key = "portfolio_list"
    cache_time_key = "portfolio_list_time"

    # Check cache validity (5 minutes)
    if (cache_key in st.session_state and
            cache_time_key in st.session_state and
            datetime.now() - st.session_state[cache_time_key] < timedelta(minutes=5)):
        return st.session_state[cache_key]

    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service

        # Load fresh data
        portfolios = portfolio_service.list_portfolios()

        # Enrich with calculated metrics
        enriched_portfolios = []
        for portfolio in portfolios:
            try:
                # Calculate current value using same logic as create_portfolio
                # This ensures consistency across the application
                try:
                    from services.data_service import DataService
                    data_service = DataService()
                    positions = portfolio.get_all_positions()
                    tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
                    prices = data_service.get_latest_prices(tickers) if tickers else {}
                    
                    current_value = 0.0
                    for pos in positions:
                        if pos.ticker == "CASH":
                            current_value += pos.shares  # CASH shares = dollar amount
                        else:
                            price = prices.get(pos.ticker, pos.purchase_price or 0.0)
                            if price > 0:
                                current_value += pos.shares * price
                    
                    # Fallback to starting_capital if calculation fails
                    if current_value <= 0:
                        current_value = portfolio.starting_capital
                except Exception as e:
                    logger.warning(f"Error calculating current value for {portfolio.name}: {e}")
                    current_value = portfolio.starting_capital

                enriched_portfolios.append({
                    'id': portfolio.id,
                    'name': portfolio.name,
                    'description': portfolio.description or '',
                    'starting_capital': portfolio.starting_capital,
                    'current_value': current_value,
                    'asset_count': len(portfolio.get_all_positions()),
                    'portfolio_object': portfolio
                })
            except Exception as e:
                logger.warning(f"Error loading portfolio {portfolio.name}: {e}")
                # Add portfolio even if metrics calculation failed
                enriched_portfolios.append({
                    'id': portfolio.id,
                    'name': portfolio.name,
                    'description': portfolio.description or '',
                    'starting_capital': portfolio.starting_capital,
                    'current_value': portfolio.starting_capital,
                    'asset_count': len(portfolio.get_all_positions()),
                    'portfolio_object': portfolio
                })

        # Cache results
        st.session_state[cache_key] = enriched_portfolios
        st.session_state[cache_time_key] = datetime.now()

        return enriched_portfolios

    except Exception as e:
        st.error(f"Error loading portfolios: {str(e)}")
        logger.error(f"Error loading portfolios: {e}", exc_info=True)
        return []


def clear_portfolio_cache() -> None:
    """Clear portfolio cache to force refresh."""
    cache_keys = ["portfolio_list", "portfolio_list_time"]
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]


def render_empty_state() -> None:
    """Render empty state when no portfolios exist."""
    st.info("No portfolios found. Create your first portfolio to get started!")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Create Your First Portfolio", type="primary", use_container_width=True):
            st.switch_page("pages/create_portfolio.py")


def render_search_and_filter(portfolios: List[Dict]) -> List[Dict]:
    """Render search and filter controls."""
    with st.expander("Search & Filter", expanded=False):
        search_col, sort_col = st.columns(2)

        with search_col:
            search_term = st.text_input(
                "Search portfolios",
                placeholder="Search by name or description...",
                key="portfolio_search"
            )

        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                options=["Name", "Created Date", "Asset Count", "Value"],
                key="portfolio_sort"
            )

    # Apply filters
    filtered = portfolios.copy()

    # Search filter
    if search_term:
        search_lower = search_term.lower()
        filtered = [
            p for p in filtered
            if (search_lower in p.get('name', '').lower() or
                search_lower in p.get('description', '').lower())
        ]

    # Sort
    if sort_by == "Name":
        filtered.sort(key=lambda x: x.get('name', '').lower())
    elif sort_by == "Created Date":
        # Sort by name if no created_date
        filtered.sort(key=lambda x: x.get('name', '').lower())
    elif sort_by == "Asset Count":
        filtered.sort(key=lambda x: x.get('asset_count', 0), reverse=True)
    elif sort_by == "Value":
        filtered.sort(key=lambda x: x.get('current_value', 0), reverse=True)

    return filtered


def render_portfolio_table(portfolios: List[Dict]) -> None:
    """Render the main portfolio table with selection."""
    st.subheader(f"Portfolios ({len(portfolios)})")

    # Prepare table data
    table_data = []
    for portfolio in portfolios:
        table_data.append({
            'Select': False,
            'Name': portfolio.get('name', 'Unnamed'),
            'Description': (
                portfolio.get('description', '')[:50] +
                ('...' if len(portfolio.get('description', '')) > 50 else '')
            ),
            'Assets': portfolio.get('asset_count', 0),
            'Value': format_currency(portfolio.get('current_value', 0)),
            'Created': (
                portfolio.get('created_at', '')
                if portfolio.get('created_at')
                else 'N/A'
            ),
            'Last Updated': 'Recently',  # TODO: Calculate actual time
        })

    # Display editable table
    edited_df = st.data_editor(
        pd.DataFrame(table_data),
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', width=60),
            'Name': st.column_config.TextColumn('Name', width=150),
            'Description': st.column_config.TextColumn(
                'Description', width=200
            ),
            'Assets': st.column_config.NumberColumn('Assets', width=80),
            'Value': st.column_config.TextColumn('Value', width=120),
            'Created': st.column_config.TextColumn('Created', width=150),
            'Last Updated': st.column_config.TextColumn(
                'Last Updated', width=120
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="portfolio_table"
    )

    # Update selected portfolios
    selected_indices = [i for i, row in edited_df.iterrows() if row['Select']]
    st.session_state.selected_portfolios = [portfolios[i] for i in selected_indices]


def render_individual_actions(portfolios: List[Dict]) -> None:
    """Render individual action buttons for each portfolio."""
    st.subheader("Portfolio Actions")

    for i, portfolio in enumerate(portfolios):
        with st.container(border=True):
            # Portfolio info
            st.write(f"**{portfolio['name']}**")
            st.write(f"Assets: {portfolio.get('asset_count', 0)} | "
                     f"Value: {format_currency(portfolio.get('current_value', 0))}")

            # Action buttons
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

            with btn_col1:
                if st.button("Edit", key=f"edit_{i}", use_container_width=True):
                    edit_portfolio(portfolio)

            with btn_col2:
                if st.button("View", key=f"view_{i}", use_container_width=True):
                    st.session_state.view_portfolio_id = portfolio['id']
                    st.session_state.management_view = "view"
                    st.rerun()

            with btn_col3:
                if st.button("Analyze", key=f"analyze_{i}", use_container_width=True):
                    st.query_params["id"] = portfolio['id']
                    st.switch_page("pages/portfolio_analysis.py")

            with btn_col4:
                if st.button("Delete", key=f"delete_{i}", type="secondary", use_container_width=True):
                    delete_portfolio_confirmed(portfolio)


def render_bulk_operations() -> None:
    """Render bulk operations for selected portfolios."""
    st.subheader(f"Bulk Operations ({len(st.session_state.selected_portfolios)} selected)")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Update Prices", use_container_width=True, type="primary"):
            bulk_update_prices()

    with col2:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_portfolios = []
            st.rerun()

    with col3:
        if st.button("Delete Selected", use_container_width=True, type="secondary"):
            if st.session_state.selected_portfolios:
                bulk_delete_portfolios()


def edit_portfolio(portfolio_info: Dict) -> None:
    """Edit portfolio."""
    st.session_state.editing_portfolio = portfolio_info
    st.session_state.management_view = "edit"
    st.rerun()


def render_portfolio_editor() -> None:
    """Render the portfolio editing interface."""
    if 'editing_portfolio' not in st.session_state:
        st.error("No portfolio selected for editing.")
        if st.button("Back to List"):
            st.session_state.management_view = "list"
            st.rerun()
        return

    portfolio_info = st.session_state.editing_portfolio
    portfolio_service: PortfolioService = st.session_state.portfolio_service

    try:
        portfolio = portfolio_service.get_portfolio(portfolio_info['id'])
    except Exception as e:
        st.error(f"Failed to load portfolio for edit: {e}")
        if st.button("Back to List"):
            st.session_state.management_view = "list"
            st.rerun()
        return

    st.subheader(f"Edit Portfolio: {portfolio.name}")

    with st.form("edit_portfolio_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Name", value=portfolio.name)
            currency_options = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
            try:
                currency_index = currency_options.index(portfolio.base_currency)
            except Exception:
                currency_index = 0
            new_currency = st.selectbox("Currency", currency_options, index=currency_index)
        with col2:
            new_description = st.text_area("Description", value=portfolio.description or "")
            new_starting_capital = st.number_input(
                "Starting Capital",
                min_value=1.0,
                value=float(portfolio.starting_capital or 0.0),
                step=100.0,
            )

        colb1, colb2 = st.columns(2)
        with colb1:
            submitted = st.form_submit_button("Save", type="primary", use_container_width=True)
        with colb2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

    if 'cancel' in locals() and cancel:
        st.session_state.management_view = "list"
        st.rerun()

    if 'submitted' in locals() and submitted:
        save_portfolio_changes(
            portfolio,
            new_name=new_name,
            new_description=new_description,
            new_currency=new_currency,
            new_starting_capital=new_starting_capital,
        )

    # Tabs for different sections
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Positions", "Transactions", "Strategies"])

    with tab1:
        # Positions editor (inline editable grid)
        positions = portfolio.get_all_positions()
        
        # Fetch current prices
        data_service: DataService = st.session_state.data_service
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        current_prices = {}
        if tickers:
            try:
                current_prices = data_service.get_latest_prices(tickers)
            except Exception as e:
                logger.warning(f"Error fetching current prices: {e}")
                current_prices = {}
        
        pos_rows = []
        for pos in positions:
            current_price = current_prices.get(pos.ticker, 0.0) if pos.ticker != "CASH" else 1.0
            purchase_price = float(pos.purchase_price or 0.0)
            
            pos_rows.append({
                "Ticker": pos.ticker,
                "Shares": float(pos.shares or 0.0),
                "Current Price": current_price,
                "Purchase Price": purchase_price,
                "Weight Target": float(pos.weight_target * 100.0) if pos.weight_target else 0.0,
                "Remove": False,
            })
        import pandas as pd
        edited_df = st.data_editor(
            pd.DataFrame(pos_rows),
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "Shares": st.column_config.NumberColumn("Shares", step=0.01, format="%.4f"),
                "Current Price": st.column_config.NumberColumn("Current Price", step=0.01, format="$%.2f", disabled=True),
                "Purchase Price": st.column_config.NumberColumn("Purchase Price", step=0.01, format="$%.2f"),
                "Weight Target": st.column_config.NumberColumn("Weight Target (%)", step=0.1, format="%.1f"),
                "Remove": st.column_config.CheckboxColumn("Remove"),
            },
            hide_index=True,
            use_container_width=True,
            key="positions_editor",
        )

        colu1, colu2 = st.columns(2)
        with colu1:
            if st.button("Save Positions", type="primary", use_container_width=True):
                update_positions(portfolio.id, edited_df)
        with colu2:
            if st.button("Remove Selected", use_container_width=True):
                # Remove positions flagged in edited_df
                to_remove = [row["Ticker"] for _, row in edited_df.iterrows() if row.get("Remove")]
                if to_remove:
                    try:
                        portfolio_service: PortfolioService = st.session_state.portfolio_service
                        for tkr in to_remove:
                            portfolio_service.remove_position(portfolio.id, tkr)
                        st.success(f"Removed {len(to_remove)} positions")
                        clear_portfolio_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove positions: {e}")

        # Add position form
        st.markdown("---")
        st.subheader("Add Position")
        with st.form("add_position_form", clear_on_submit=True):
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            with ac1:
                add_ticker = st.text_input("Ticker", placeholder="AAPL")
            with ac2:
                add_shares = st.number_input("Shares", min_value=0.0, value=0.0, step=0.01)
            with ac3:
                add_weight = st.number_input("Weight Target (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            add_submit = st.form_submit_button("Add", type="primary")
        if add_submit and add_ticker:
            add_position_to_portfolio(
                portfolio.id,
                ticker=add_ticker,
                shares=float(add_shares),
                weight=float(add_weight) / 100.0 if add_weight > 0 else 0.0,
            )

    with tab2:
        # Transactions tab
        render_transactions_tab(portfolio.id)

    with tab3:
        # Strategies tab (placeholder for Phase 5)
        st.info("Strategy management - coming in Phase 5")
        st.markdown("""
        **Strategies can be applied to any portfolio mode:**
        - Buy-and-Hold portfolios can use strategies for backtesting
        - Transaction-based portfolios can also use strategies
        - Strategies generate simulated transactions for analysis
        """)


def render_portfolio_view() -> None:
    """Read-only view: positions table + two donuts (assets, sectors)."""
    portfolio_id = st.session_state.get("view_portfolio_id")
    if not portfolio_id:
        st.warning("No portfolio selected.")
        if st.button("Back"):
            st.session_state.management_view = "list"
            st.rerun()
        return

    portfolio_service: PortfolioService = st.session_state.portfolio_service
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
    except Exception as e:
        st.error(f"Failed to load portfolio: {e}")
        return

    st.subheader(f"Portfolio: {portfolio.name}")

    # Determine portfolio mode
    from services.transaction_service import TransactionService
    transaction_service = TransactionService()
    transactions = transaction_service.get_transactions(portfolio.id)
    
    portfolio_mode = "With Transactions" if transactions else "Buy-and-Hold"
    mode_color = "#50C878" if transactions else "#4A90E2"
    
    st.markdown(f"""
    <div style="padding: 10px; background-color: #1e1e1e; border-radius: 5px; margin-bottom: 20px;">
        <strong>Portfolio Mode:</strong> 
        <span style="color: {mode_color};">{portfolio_mode}</span>
        {f"({len(transactions)} transactions)" if transactions else "(Simple mode)"}
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Positions", "Transactions", "Strategies"])

    positions = portfolio.get_all_positions()

    with tab1:
        # Overview tab - existing view content
        _render_portfolio_overview(portfolio, positions, transaction_service)

    with tab2:
        # Positions tab
        _render_positions_tab(portfolio, positions)

    with tab3:
        # Transactions tab
        render_transactions_tab(portfolio.id)

    with tab4:
        # Strategies tab (placeholder for Phase 5)
        st.info("Strategy management - coming in Phase 5")
        st.markdown("""
        **Strategies can be applied to any portfolio mode:**
        - Buy-and-Hold portfolios can use strategies for backtesting
        - Transaction-based portfolios can also use strategies
        - Strategies generate simulated transactions for analysis
        """)


def _render_portfolio_overview(portfolio, positions, transaction_service):
    """Render overview tab with positions table and charts."""
    # Build detailed positions table similar to creation result, plus Sector
    try:
        data_service: DataService = st.session_state.data_service
        validator = TickerValidator()

        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        prices = {}
        company_names = {}
        sectors = {}
        if tickers:
            try:
                prices = data_service.get_latest_prices(tickers)
            except Exception:
                prices = {}
            for t in tickers:
                try:
                    info = data_service.get_ticker_info(t)
                    company_names[t] = info.name or t
                    sectors[t] = info.sector or "Other"
                except Exception:
                    company_names[t] = t
                    sectors[t] = "Other"

        # Calculate values first
        total_value = 0.0
        values_by_ticker = {}
        for pos in positions:
            if pos.ticker == "CASH":
                price = 1.0
                value = pos.shares * price
            else:
                price = prices.get(pos.ticker, pos.purchase_price or 0.0)
                value = pos.shares * price if price > 0 else 0.0
            values_by_ticker[pos.ticker] = value
            total_value += value

        table_rows = []
        for pos in positions:
            t = pos.ticker
            value = values_by_ticker.get(t, 0.0)
            price = 1.0 if t == "CASH" else prices.get(t, pos.purchase_price or 0.0)
            name = "Cash Position" if t == "CASH" else company_names.get(t, t)
            sector = "Cash" if t == "CASH" else sectors.get(t, "Other")
            shares_display = f"${pos.shares:,.2f}" if t == "CASH" else f"{pos.shares:,.2f}"
            weight = (value / total_value) if total_value > 0 else (pos.weight_target or 0.0)

            # Get purchase price for comparison
            purchase_price = pos.purchase_price or 0.0
            current_price = price
            
            table_rows.append({
                "Ticker": t,
                "Name": name,
                "Sector": sector,
                "Weight": f"{weight:.1%}",
                "Shares": shares_display,
                "Current Price": f"${current_price:,.2f}",
                "Purchase Price": f"${purchase_price:,.2f}" if purchase_price > 0 else "N/A",
                "Value": f"${value:,.2f}",
            })

        if table_rows:
            df = pd.DataFrame(table_rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No positions to display")
    except Exception as e:
        logger.warning(f"Error rendering positions table: {e}")
        render_position_table(positions)

    col1, col2 = st.columns(2)

    # Asset allocation donut
    with col1:
        st.markdown("**Asset Allocation**")
        weights = []
        for pos in positions:
            if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                weights.append(pos.weight_target)
            else:
                weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
        total_weight = sum(weights) or 1.0
        alloc = {}
        for pos, w in zip(positions, weights):
            pct = (w / total_weight * 100)
            alloc[pos.ticker] = alloc.get(pos.ticker, 0.0) + pct
        fig = plot_asset_allocation(alloc)
        st.plotly_chart(fig, use_container_width=True)

    # Sector allocation donut
    with col2:
        st.markdown("**Sector Allocation**")
        validator = TickerValidator()
        sector_map: dict[str, float] = {}
        for pos, w in zip(positions, weights):
            pct = (w / total_weight * 100)
            sector = "Cash" if pos.ticker == "CASH" else None
            if sector is None:
                try:
                    info = validator.get_ticker_info(pos.ticker)
                    sector = info.sector or "Other"
                except Exception:
                    sector = "Other"
            sector_map[sector] = sector_map.get(sector, 0.0) + pct
        fig = plot_sector_allocation(sector_map)
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Back to List"):
        st.session_state.management_view = "list"
        st.rerun()
        return


def _render_positions_tab(portfolio, positions):
    """Render positions tab in portfolio view."""
    st.markdown("### Current Positions")
    
    try:
        data_service: DataService = st.session_state.data_service
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        prices = {}
        company_names = {}
        sectors = {}
        if tickers:
            try:
                prices = data_service.get_latest_prices(tickers)
            except Exception:
                prices = {}
            for t in tickers:
                try:
                    info = data_service.get_ticker_info(t)
                    company_names[t] = info.name or t
                    sectors[t] = info.sector or "Other"
                except Exception:
                    company_names[t] = t
                    sectors[t] = "Other"

        # Calculate values
        total_value = 0.0
        values_by_ticker = {}
        for pos in positions:
            if pos.ticker == "CASH":
                price = 1.0
                value = pos.shares * price
            else:
                price = prices.get(pos.ticker, pos.purchase_price or 0.0)
                value = pos.shares * price if price > 0 else 0.0
            values_by_ticker[pos.ticker] = value
            total_value += value

        table_rows = []
        for pos in positions:
            t = pos.ticker
            value = values_by_ticker.get(t, 0.0)
            price = 1.0 if t == "CASH" else prices.get(t, pos.purchase_price or 0.0)
            name = "Cash Position" if t == "CASH" else company_names.get(t, t)
            sector = "Cash" if t == "CASH" else sectors.get(t, "Other")
            shares_display = f"${pos.shares:,.2f}" if t == "CASH" else f"{pos.shares:,.2f}"
            weight = (value / total_value) if total_value > 0 else (pos.weight_target or 0.0)
            purchase_price = pos.purchase_price or 0.0

            table_rows.append({
                "Ticker": t,
                "Name": name,
                "Sector": sector,
                "Weight": f"{weight:.1%}",
                "Shares": shares_display,
                "Current Price": f"${price:,.2f}",
                "Purchase Price": f"${purchase_price:,.2f}" if purchase_price > 0 else "N/A",
                "Value": f"${value:,.2f}",
            })

        if table_rows:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No positions to display")
    except Exception as e:
        logger.warning(f"Error rendering positions: {e}")
        render_position_table(positions)


def render_transactions_tab(portfolio_id: str) -> None:
    """Render transactions management tab."""
    from services.transaction_service import TransactionService
    from streamlit_app.components.transaction_form import render_transaction_form
    from streamlit_app.components.transaction_table import render_transaction_table
    from streamlit_app.utils.formatters import format_currency
    
    transaction_service = TransactionService()
    
    # Summary statistics
    transactions = transaction_service.get_transactions(portfolio_id)
    
    if transactions:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(transactions))
        with col2:
            first_date = min(txn.transaction_date for txn in transactions)
            st.metric("First Transaction", first_date.strftime("%Y-%m-%d"))
        with col3:
            last_date = max(txn.transaction_date for txn in transactions)
            st.metric("Last Transaction", last_date.strftime("%Y-%m-%d"))
        with col4:
            total_invested = sum(
                txn.amount for txn in transactions 
                if txn.transaction_type in ['BUY', 'DEPOSIT']
            )
            st.metric("Total Invested", format_currency(total_invested))
    else:
        st.info("No transactions yet. Add your first transaction to start tracking trades.")
    
    st.markdown("---")
    
    # Add transaction form
    with st.expander("Add Transaction", expanded=False):
        transaction_data = render_transaction_form(portfolio_id)
        if transaction_data:
            try:
                transaction_service.add_transaction(
                    portfolio_id=portfolio_id,
                    transaction_date=transaction_data['transaction_date'],
                    transaction_type=transaction_data['transaction_type'],
                    ticker=transaction_data['ticker'],
                    shares=transaction_data['shares'],
                    price=transaction_data['price'],
                    fees=transaction_data.get('fees', 0.0),
                    notes=transaction_data.get('notes'),
                )
                st.success("Transaction added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding transaction: {str(e)}")
                logger.exception("Error adding transaction")
    
    # Transaction table
    st.markdown("### Transaction History")
    transaction_id_to_delete = render_transaction_table(
        transactions, portfolio_id, show_actions=True
    )
    
    # Handle deletion
    if transaction_id_to_delete:
        try:
            deleted = transaction_service.delete_transaction(transaction_id_to_delete)
            if deleted:
                st.success("Transaction deleted successfully!")
                st.rerun()
            else:
                st.error("Transaction not found or could not be deleted.")
        except Exception as e:
            st.error(f"Error deleting transaction: {str(e)}")
            logger.exception("Error deleting transaction")
    
    # Import/Export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Import from CSV", use_container_width=True):
            st.info("CSV import functionality - coming in Phase 7")
    with col2:
        if transactions:
            # Export to CSV
            import io
            import csv
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Date', 'Type', 'Ticker', 'Shares', 'Price', 'Amount', 'Fees', 'Notes'])
            for txn in transactions:
                writer.writerow([
                    txn.transaction_date.strftime('%Y-%m-%d'),
                    txn.transaction_type,
                    txn.ticker,
                    txn.shares,
                    txn.price,
                    txn.amount,
                    txn.fees or 0.0,
                    txn.notes or '',
                ])
            csv_data = output.getvalue().encode('utf-8')
            st.download_button(
                "Export to CSV",
                csv_data,
                file_name=f"transactions_{portfolio_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def save_portfolio_changes(
    portfolio,
    new_name: str,
    new_description: str,
    new_currency: str,
    new_starting_capital: float
) -> None:
    """Save portfolio changes."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service

        request = UpdatePortfolioRequest(
            name=new_name,
            description=new_description,
            base_currency=new_currency,
            starting_capital=new_starting_capital
        )

        portfolio_service.update_portfolio(portfolio.id, request)

        st.success(f"Portfolio '{new_name}' updated successfully!")

        # Clear cache and return to list
        clear_portfolio_cache()
        st.session_state.management_view = "list"
        if 'editing_portfolio' in st.session_state:
            del st.session_state.editing_portfolio

        st.rerun()

    except Exception as e:
        st.error(f"Error saving portfolio changes: {str(e)}")
        logger.error(f"Error saving portfolio changes: {e}", exc_info=True)


def update_positions(portfolio_id: str, edited_positions: pd.DataFrame) -> None:
    """Update positions from edited dataframe."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        portfolio = portfolio_service.get_portfolio(portfolio_id)

        for _, row in edited_positions.iterrows():
            ticker = row['Ticker']
            # Current Price is read-only, so we don't update it
            request = UpdatePositionRequest(
                shares=float(row['Shares']),
                weight_target=float(row['Weight Target']) / 100.0 if row['Weight Target'] > 0 else None,
                purchase_price=float(row['Purchase Price']) if row['Purchase Price'] > 0 else None
            )
            portfolio_service.update_position(portfolio_id, ticker, request)

        st.success("Positions updated successfully!")
        clear_portfolio_cache()
        st.rerun()

    except Exception as e:
        st.error(f"Error updating positions: {str(e)}")
        logger.error(f"Error updating positions: {e}", exc_info=True)


def add_position_to_portfolio(portfolio_id: str, ticker: str, shares: float, weight: float) -> None:
    """Add position to portfolio."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        data_service: DataService = st.session_state.data_service

        # Validate ticker
        validation_results = data_service.validate_tickers([ticker.upper()])
        if not validation_results.get(ticker.upper(), False):
            st.error(f"Invalid ticker: {ticker}")
            return

        request = AddPositionRequest(
            ticker=ticker.upper(),
            shares=shares,
            weight_target=weight if weight > 0 else None
        )

        portfolio_service.add_position(portfolio_id, request)
        st.success(f"Position {ticker.upper()} added successfully!")
        clear_portfolio_cache()
        st.rerun()

    except Exception as e:
        st.error(f"Error adding position: {str(e)}")
        logger.error(f"Error adding position: {e}", exc_info=True)


def delete_portfolio_confirmed(portfolio_info: Dict) -> None:
    """Delete portfolio with confirmation."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        
        # Get portfolio ID directly from info to avoid loading relationships
        portfolio_id = portfolio_info['id']
        
        # Try to get portfolio name for undo, but don't fail if it doesn't work
        try:
            portfolio = portfolio_service.get_portfolio(portfolio_id)
            portfolio_name = portfolio.name
        except Exception:
            portfolio_name = portfolio_info.get('name', 'Unknown')
            portfolio = None

        # Store for undo functionality
        st.session_state.deleted_portfolios.append({
            'portfolio': portfolio,
            'portfolio_name': portfolio_name,
            'deleted_at': datetime.now(),
            'portfolio_id': portfolio_id
        })

        # Delete portfolio
        success = portfolio_service.delete_portfolio(portfolio_id)

        if success:
            st.success(f"Portfolio '{portfolio_name}' deleted successfully!")
            clear_portfolio_cache()
        else:
            st.error("Failed to delete portfolio.")

    except Exception as e:
        st.error(f"Error deleting portfolio: {str(e)}")
        logger.error(f"Error deleting portfolio: {e}", exc_info=True)


def render_undo_section() -> None:
    """Render undo section for deleted portfolios."""
    st.divider()
    st.subheader("Recently Deleted")

    for i, deleted_item in enumerate(st.session_state.deleted_portfolios):
        portfolio = deleted_item.get('portfolio')
        portfolio_name = deleted_item.get('portfolio_name', 'Unknown')
        deleted_at = deleted_item['deleted_at']

        with st.container(border=True):
            undo_col1, undo_col2, undo_col3 = st.columns([3, 2, 1])

            with undo_col1:
                st.write(f"**{portfolio_name}**")
                st.write(f"Deleted: {deleted_at.strftime('%Y-%m-%d %H:%M:%S')}")

            with undo_col2:
                if portfolio:
                    st.write(f"Assets: {len(portfolio.get_all_positions())}")
                    st.write(f"Starting Capital: {format_currency(portfolio.starting_capital)}")
                else:
                    st.write("Portfolio info unavailable")

            with undo_col3:
                if st.button("Restore", key=f"restore_{i}", use_container_width=True):
                    restore_deleted_portfolio(i)

    if len(st.session_state.deleted_portfolios) > 1:
        if st.button("Clear All Deleted", type="secondary"):
            st.session_state.deleted_portfolios = []
            st.rerun()


def restore_deleted_portfolio(index: int) -> None:
    """Restore a deleted portfolio."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        deleted_item = st.session_state.deleted_portfolios[index]
        portfolio = deleted_item.get('portfolio')
        portfolio_name = deleted_item.get('portfolio_name', 'Unknown')

        if not portfolio:
            st.error("Cannot restore: portfolio data not available. Portfolio was already deleted from database.")
            # Remove from deleted list
            st.session_state.deleted_portfolios.pop(index)
            return

        # Generate new name if original exists
        existing_portfolios = portfolio_service.list_portfolios()
        existing_names = [p.name for p in existing_portfolios]

        restore_name = portfolio.name
        if portfolio.name in existing_names:
            restore_name = f"{portfolio.name} (Restored)"

        # Recreate portfolio
        from services.schemas import CreatePortfolioRequest, PositionSchema

        positions = []
        for pos in portfolio.get_all_positions():
            positions.append(PositionSchema(
                ticker=pos.ticker,
                shares=pos.shares,
                weight_target=pos.weight_target,
                purchase_price=pos.purchase_price,
                purchase_date=pos.purchase_date
            ))

        request = CreatePortfolioRequest(
            name=restore_name,
            description=portfolio.description,
            starting_capital=portfolio.starting_capital,
            base_currency=portfolio.base_currency,
            positions=positions
        )

        portfolio_service.create_portfolio(request)

        # Remove from deleted list
        st.session_state.deleted_portfolios.pop(index)

        st.success(f"Portfolio '{restore_name}' restored successfully!")
        clear_portfolio_cache()

    except Exception as e:
        st.error(f"Error restoring portfolio: {str(e)}")
        logger.error(f"Error restoring portfolio: {e}", exc_info=True)


def bulk_update_prices() -> None:
    """Update prices for selected portfolios."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service

        selected_count = len(st.session_state.selected_portfolios)
        with st.spinner(f"Updating prices for {selected_count} portfolios..."):
            updated_count = 0

            for portfolio_info in st.session_state.selected_portfolios:
                try:
                    # Prices are updated on-demand when calculating metrics
                    portfolio_service.get_portfolio(portfolio_info['id'])
                    updated_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to update prices for {portfolio_info['name']}: {e}"
                    )

            st.success(f"Updated prices for {updated_count} portfolios!")
            clear_portfolio_cache()

    except Exception as e:
        st.error(f"Error updating prices: {str(e)}")
        logger.error(f"Error updating prices: {e}", exc_info=True)


def bulk_delete_portfolios() -> None:
    """Delete selected portfolios with confirmation."""
    try:
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        deleted_count = 0

        for portfolio_info in st.session_state.selected_portfolios:
            try:
                portfolio_id = portfolio_info['id']
                portfolio_name = portfolio_info.get('name', 'Unknown')
                
                # Try to get portfolio for undo, but don't fail if it doesn't work
                try:
                    portfolio = portfolio_service.get_portfolio(portfolio_id)
                except Exception:
                    portfolio = None

                # Store for undo
                st.session_state.deleted_portfolios.append({
                    'portfolio': portfolio,
                    'portfolio_name': portfolio_name,
                    'deleted_at': datetime.now(),
                    'portfolio_id': portfolio_id
                })

                # Delete
                success = portfolio_service.delete_portfolio(portfolio_id)
                if success:
                    deleted_count += 1

            except Exception as e:
                logger.warning(f"Failed to delete {portfolio_info['name']}: {e}")

        st.success(f"Deleted {deleted_count} portfolios successfully!")
        st.session_state.selected_portfolios = []
        clear_portfolio_cache()

    except Exception as e:
        st.error(f"Error deleting portfolios: {str(e)}")
        logger.error(f"Error deleting portfolios: {e}", exc_info=True)


if __name__ == "__main__":
    render_portfolio_list()
