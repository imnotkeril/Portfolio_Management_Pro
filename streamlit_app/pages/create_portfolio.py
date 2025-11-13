"""Create portfolio page with step-by-step portfolio creation and multiple input methods."""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.exceptions import ConflictError, ValidationError
from services.data_service import DataService
from services.portfolio_service import PortfolioService
from services.schemas import CreatePortfolioRequest, PositionSchema
from streamlit_app.utils.text_parser import parse_ticker_weights_text
from streamlit_app.utils.validators import (
    validate_portfolio_name,
    validate_ticker_format,
)

logger = logging.getLogger(__name__)


def render_create_portfolio() -> None:
    """Main function to render the create portfolio page."""
    st.title("Create New Portfolio")

    # Initialize services
    portfolio_service = PortfolioService()
    data_service = DataService()

    # Store services in session state
    st.session_state.portfolio_service = portfolio_service
    st.session_state.data_service = data_service

    # Show help section
    render_creation_help()

    # Start portfolio creation by default
    render_portfolio_creation()


def render_creation_help() -> None:
    """Render help section for portfolio creation."""
    with st.expander("How to Create a Portfolio", expanded=False):
        st.markdown("""
        ### Step-by-Step Portfolio Creation
        
        This process guides you through creating a portfolio in 5 steps:
        
        1. **Portfolio Information** - Name, description, currency, and initial investment
        2. **Choose Input Method** - Select how you want to add assets:
           - **Text Input** - Fast entry using natural language formats
           - **Upload File** - Import from CSV or Excel files
           - **Manual Entry** - Add each asset individually with full control
           - **Use Template** - Start with pre-built investment strategies
        3. **Add Assets** - Enter your portfolio assets based on selected method
        4. **Settings & Review** - Configure options and review your portfolio
        5. **Create** - Finalize and create your portfolio
        
        ### Supported Text Input Formats
        ```
        AAPL:40%, MSFT:30%, GOOGL:30%          # Colon with percentages
        AAPL 0.4, MSFT 0.3, GOOGL 0.3         # Space with decimals  
        AAPL 40, MSFT 30, GOOGL 30            # Numbers > 1 (auto %)
        AAPL, MSFT, GOOGL                     # Equal weights
        ```
        
        ### Important Notes
        - Weights are automatically normalized to sum to 100%
        - Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
        - Company information and prices are fetched automatically
        - All portfolios are validated before creation
        - You can allocate cash as a percentage of your portfolio
        """)


def render_portfolio_creation() -> None:
    """Render step-by-step portfolio creation."""
    st.subheader("Portfolio Creation")

    # Initialize creation state
    if 'creation_step' not in st.session_state:
        st.session_state.creation_step = 1

    if 'creation_data' not in st.session_state:
        st.session_state.creation_data = {}

    # Progress bar
    progress = (st.session_state.creation_step - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.creation_step} of 5")

    # Creation steps
    if st.session_state.creation_step == 1:
        render_step_1()
    elif st.session_state.creation_step == 2:
        render_step_2()
    elif st.session_state.creation_step == 3:
        render_step_3()
    elif st.session_state.creation_step == 4:
        render_step_4()
    elif st.session_state.creation_step == 5:
        render_step_5()


def render_step_1() -> None:
    """Step 1: Portfolio Information."""
    st.write("### Step 1: Portfolio Information")
    
    with st.expander("What information do I need?", expanded=False):
        st.markdown("""
        **Portfolio Name** - A unique name to identify your portfolio (required)
        - Must be unique - cannot match existing portfolio names
        - Use descriptive names like "Tech Growth Portfolio" or "Dividend Income"
        
        **Description** - Optional notes about your investment strategy
        - Helps you remember the purpose of this portfolio
        - Example: "Long-term growth focused on technology stocks"
        
        **Base Currency** - The currency for your portfolio
        - Currently supports: USD, EUR, GBP, JPY, CAD, AUD
        - All values and calculations will be in this currency
        
        **Initial Investment** - Starting capital amount
        - Used to calculate number of shares for each position
        - Can be adjusted later if needed
        - Minimum: $1.00
        """)

    col1, col2 = st.columns(2)

    with col1:
        portfolio_name = st.text_input(
            "Portfolio Name *",
            value=st.session_state.creation_data.get('name', ''),
            help="Enter a unique name for your portfolio",
            key='creation_name'
        )

        portfolio_description = st.text_area(
            "Description",
            value=st.session_state.creation_data.get('description', ''),
            help="Optional description of your investment strategy",
            key='creation_description'
        )

    with col2:
        currency = st.selectbox(
            "Base Currency",
            ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"],
            index=0,
            key='creation_currency'
        )

        initial_value = st.number_input(
            "Initial Investment ($)",
            min_value=1.0,
            value=st.session_state.creation_data.get('initial_value', 100000.0),
            step=1000.0,
            help="Used to calculate share quantities for each position",
            key='creation_initial_value'
        )

    # Validation - Level 1: UI validation
    name_valid, name_error = validate_portfolio_name(portfolio_name)

    if portfolio_name:
        # Check if portfolio name already exists - Level 2: Service validation
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        existing_portfolios = portfolio_service.list_portfolios()
        existing_names = [p.name for p in existing_portfolios]

        if portfolio_name in existing_names:
            st.error("A portfolio with this name already exists!")
            name_valid = False
        elif not name_valid:
            st.error(f"Portfolio name: {name_error}")
        else:
            st.success("Portfolio name is available")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Cancel", use_container_width=True):
            reset_creation()
            st.rerun()

    with col3:
        if st.button("Next Step →", use_container_width=True, disabled=not name_valid, type="primary"):
            st.session_state.creation_data.update({
                'name': portfolio_name,
                'description': portfolio_description,
                'currency': currency,
                'initial_value': initial_value
            })
            st.session_state.creation_step = 2
            st.rerun()


def render_step_2() -> None:
    """Step 2: Input Method Selection."""
    st.write("### Step 2: Choose Input Method")

    with st.expander("Which method should I choose?", expanded=False):
        st.markdown("""
        **Text Input** - Best for quick entry
        - Fastest way to create a portfolio
        - Supports multiple text formats
        - Good for portfolios with 5-20 assets
        
        **Upload File** - Best for existing data
        - Import from CSV or Excel spreadsheets
        - Perfect if you already have a portfolio list
        - Supports column mapping for flexibility
        
        **Manual Entry** - Best for precision
        - Add each asset one by one
        - Full control over each position
        - Real-time ticker validation
        
        **Use Template** - Best for beginners
        - Pre-built investment strategies
        - Factor-based portfolios (Value, Growth, Quality)
        - Classic strategies (60/40, All Weather)
        - Can be customized after selection
        """)

    method = st.radio(
        "How would you like to add assets?",
        ["Text Input", "Upload File", "Manual Entry", "Use Template"],
        index=0,
        help="Choose the most convenient method for your data",
        key='creation_input_method'
    )

    # Show preview of selected method
    if method == "Text Input":
        st.info("**Fastest method**: Enter ticker symbols with weights")
        st.code("AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%")
        st.markdown("**Supported formats**: `AAPL:30%`, `AAPL 0.3`, `AAPL 30`, `AAPL, MSFT` (equal)")

    elif method == "Upload File":
        st.info("**From spreadsheet**: Upload CSV or Excel files")
        st.markdown("**Required columns**: `ticker`, **Optional**: `weight`, `name`, `sector`")
        st.markdown("**Example**: ticker,weight -> AAPL,30 -> MSFT,25")

    elif method == "Manual Entry":
        st.info("**Full control**: Add each asset individually")
        st.markdown("**Best for**: Detailed portfolio construction with custom settings")

    elif method == "Use Template":
        st.info("**Quick start**: Begin with proven strategies")
        st.markdown("**Available**: Value Factor, Growth Factor, Quality Factor, 60/40 Portfolio, All Weather, Tech Focus")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 1
            st.rerun()

    with col3:
        if st.button("Next Step →", use_container_width=True, type="primary"):
            st.session_state.creation_data['input_method'] = method
            st.session_state.creation_step = 3
            st.rerun()


def render_step_3() -> None:
    """Step 3: Asset Input."""
    st.write("### Step 3: Add Your Assets")

    method = st.session_state.creation_data['input_method']

    if method == "Text Input":
        render_text_input()
    elif method == "Upload File":
        render_file_upload()
    elif method == "Manual Entry":
        render_manual_entry()
    elif method == "Use Template":
        render_template_selection()


def render_text_input() -> None:
    """Text input for assets."""
    st.markdown("**Enter your portfolio assets** (one of these formats):")
    
    with st.expander("How does text input work?", expanded=False):
        st.markdown("""
        **Supported Formats:**
        - `AAPL:40%, MSFT:30%, GOOGL:30%` - Colon with percentages
        - `AAPL 0.4, MSFT 0.3, GOOGL 0.3` - Space with decimals (0.0 to 1.0)
        - `AAPL 40, MSFT 30, GOOGL 30` - Numbers > 1 (automatically treated as percentages)
        - `AAPL, MSFT, GOOGL` - Equal weights (each asset gets equal allocation)
        
        **Tips:**
        - Weights don't need to sum to exactly 100% - they will be normalized automatically
        - Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
        - Invalid tickers will be highlighted and excluded
        - You can mix formats in the same input
        """)

    # Example tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Percentages", "Decimals", "Numbers", "Equal Weight"])

    with tab1:
        st.code("AAPL:40%, MSFT:30%, GOOGL:30%")
    with tab2:
        st.code("AAPL 0.4, MSFT 0.3, GOOGL 0.3")
    with tab3:
        st.code("AAPL 40, MSFT 30, GOOGL 30")
    with tab4:
        st.code("AAPL, MSFT, GOOGL")

    text_input = st.text_area(
        "Portfolio Assets:",
        value=st.session_state.creation_data.get('asset_text', ''),
        height=120,
        placeholder="Enter tickers and weights here...\nExample: AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%",
        key='creation_asset_text'
    )

    parsed_assets = []
    validation_passed = False

    if text_input.strip():
        try:
            # Parse text using utility
            parsed_assets = parse_ticker_weights_text(text_input)

            if parsed_assets:
                st.success(f"Parsed {len(parsed_assets)} assets successfully")

                # Validate tickers - Level 2: Service validation
                data_service: DataService = st.session_state.data_service
                tickers = [asset['ticker'] for asset in parsed_assets]

                validation_results = data_service.validate_tickers(tickers)
                valid_tickers = [t for t, valid in validation_results.items() if valid]
                invalid_tickers = [t for t, valid in validation_results.items() if not valid]

                if invalid_tickers:
                    st.warning(f"Unknown tickers: {', '.join(invalid_tickers)}")
                    # Filter out invalid tickers
                    parsed_assets = [asset for asset in parsed_assets if asset['ticker'] in valid_tickers]

                if parsed_assets:
                    # Show preview table
                    preview_data = []
                    for asset in parsed_assets:
                        preview_data.append({
                            'Ticker': asset['ticker'],
                            'Weight': f"{asset['weight']:.1%}",
                            'Status': 'Valid' if asset['ticker'] in valid_tickers else 'Invalid'
                        })

                    preview_df = pd.DataFrame(preview_data)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

                    # Check weight sum
                    total_weight = sum(asset['weight'] for asset in parsed_assets)
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Assets", len(parsed_assets))
                    with col2:
                        st.metric("Total Weight", f"{total_weight:.1%}")
                    with col3:
                        weight_status = "Perfect" if abs(total_weight - 1.0) < 0.001 else "Will normalize"
                        st.metric("Status", weight_status)

                    st.session_state.creation_data['parsed_assets'] = parsed_assets
                    validation_passed = True
                else:
                    st.error("No valid tickers found")
            else:
                st.error("Could not parse any assets from input")

        except Exception as e:
            st.error(f"Error parsing input: {str(e)}")
            logger.error(f"Text parsing error: {e}")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 2
            st.rerun()

    with col3:
        can_proceed = bool(text_input.strip() and validation_passed)
        if st.button("Next Step →", use_container_width=True, disabled=not can_proceed, type="primary"):
            st.session_state.creation_data['asset_text'] = text_input
            st.session_state.creation_step = 4
            st.rerun()


def render_file_upload() -> None:
    """File upload for assets."""
    st.markdown("**Upload a CSV or Excel file** with your portfolio data")
    
    with st.expander("What file format do I need?", expanded=False):
        st.markdown("""
        **File Requirements:**
        - Supported formats: CSV, Excel (.xlsx, .xls)
        - **Required column**: `ticker` (or any column name you can map)
        - **Optional columns**: `weight`, `name`, `sector`
        
        **Example CSV format:**
        ```
        ticker,weight
        AAPL,30
        MSFT,25
        GOOGL,20
        ```
        
        **Example with equal weights:**
        ```
        ticker
        AAPL
        MSFT
        GOOGL
        ```
        (Weights will be automatically set to equal)
        
        **Column Mapping:**
        - You can select which column contains tickers
        - You can select which column contains weights (or use equal weights)
        - Invalid tickers will be automatically excluded
        """)

    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain ticker symbols and optionally weights",
        key='creation_file_upload'
    )

    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"File loaded: {len(df)} rows, {len(df.columns)} columns")

            # Show file preview
            with st.expander("File Preview", expanded=True):
                st.dataframe(df.head(), hide_index=True, use_container_width=True)

            # Column mapping
            st.subheader("Column Mapping")

            col1, col2 = st.columns(2)

            with col1:
                ticker_col = st.selectbox(
                    "Ticker Column *",
                    df.columns.tolist(),
                    help="Column containing stock symbols",
                    key='creation_ticker_col'
                )

            with col2:
                weight_options = ["Auto (Equal Weight)"] + df.columns.tolist()
                weight_col = st.selectbox(
                    "Weight Column",
                    weight_options,
                    help="Column with weights/allocations (optional)",
                    key='creation_weight_col'
                )

            # Preview processed data
            if ticker_col:
                try:
                    # Extract and process data
                    tickers = df[ticker_col].astype(str).str.upper().str.strip().tolist()

                    # Remove empty/invalid tickers
                    tickers = [t for t in tickers if t and t != 'NAN' and len(t) > 0]

                    if weight_col != "Auto (Equal Weight)":
                        weights = df[weight_col].fillna(0).astype(float).tolist()[:len(tickers)]
                        # Normalize if weights are in percentage format (> 1)
                        # Check if any weight > 1, then assume percentages
                        if any(w > 1.0 for w in weights if w > 0):
                            weights = [w / 100.0 if w > 0 else 0.0 for w in weights]
                        # Normalize to sum to 1.0
                        total = sum(weights)
                        if total > 0:
                            weights = [w / total for w in weights]
                        else:
                            weights = [1.0 / len(tickers)] * len(tickers)
                    else:
                        # Equal weights
                        weights = [1.0 / len(tickers)] * len(tickers)

                    # Create preview
                    processed_data = []
                    for i, (ticker, weight) in enumerate(zip(tickers, weights)):
                        processed_data.append({
                            'Ticker': ticker,
                            'Weight': f"{weight:.1%}",
                            'Name': df.get('name', df.get('company_name', pd.Series([''] * len(df)))).iloc[
                                i] if i < len(df) else ''
                        })

                    processed_df = pd.DataFrame(processed_data)

                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_df, hide_index=True, use_container_width=True)

                    # Validate tickers
                    data_service: DataService = st.session_state.data_service
                    validation_results = data_service.validate_tickers(tickers)
                    valid_tickers = [t for t, valid in validation_results.items() if valid]
                    invalid_tickers = [t for t, valid in validation_results.items() if not valid]

                    if invalid_tickers:
                        st.warning(f"Unknown tickers will be excluded: {', '.join(invalid_tickers)}")

                    if valid_tickers:
                        st.success(f"{len(valid_tickers)} valid tickers found")

                        # Store file data
                        st.session_state.creation_data['file_data'] = {
                            'tickers': valid_tickers,
                            'weights': [weights[tickers.index(t)] for t in valid_tickers if t in tickers]
                        }
                    else:
                        st.error("No valid tickers found in file")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logger.error(f"File processing error: {e}")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            logger.error(f"File reading error: {e}")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 2
            st.rerun()

    with col3:
        can_proceed = uploaded_file and 'file_data' in st.session_state.creation_data
        if st.button("Next Step →", use_container_width=True, disabled=not can_proceed, type="primary"):
            st.session_state.creation_step = 4
            st.rerun()


def render_manual_entry() -> None:
    """Manual entry for assets."""
    st.markdown("**Add assets manually** for full control over your portfolio")
    
    with st.expander("How does manual entry work?", expanded=False):
        st.markdown("""
        **Manual Entry Process:**
        1. Enter ticker symbol (e.g., AAPL)
        2. Enter weight as percentage (e.g., 30 for 30%)
        3. Click "Validate Ticker" to check if ticker is valid
        4. Click "Add Asset" to add to portfolio
        5. Repeat for each asset
        
        **Features:**
        - Real-time ticker validation
        - See current price when validating
        - View total weight as you add assets
        - Remove assets from the table
        - Weights are automatically normalized to sum to 100%
        
        **Best for:**
        - Small portfolios (5-10 assets)
        - When you want to validate each ticker individually
        - Precise control over each position
        """)

    # Initialize manual assets
    if 'manual_assets' not in st.session_state.creation_data:
        st.session_state.creation_data['manual_assets'] = []

    # Asset entry form
    with st.form("creation_asset_entry", clear_on_submit=True):
        st.subheader("Add New Asset")

        col1, col2 = st.columns(2)

        with col1:
            ticker = st.text_input(
                "Ticker *",
                placeholder="e.g., AAPL",
                help="Stock symbol",
                key='creation_manual_ticker'
            )

        with col2:
            weight = st.number_input(
                "Weight (%)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1,
                help="Percentage allocation (will be normalized to sum to 100%)",
                key='creation_manual_weight'
            )

        col1, col2 = st.columns(2)

        with col1:
            add_button = st.form_submit_button("Add Asset", use_container_width=True, type="primary")

        with col2:
            validate_button = st.form_submit_button("Validate Ticker", use_container_width=True)

        # Validation logic
        if validate_button and ticker:
            # Validate ticker format - Level 1: UI validation
            ticker_valid, ticker_error = validate_ticker_format(ticker)
            if not ticker_valid:
                st.error(f"Ticker format: {ticker_error}")
            else:
                # Validate ticker existence - Level 2: Service validation
                data_service: DataService = st.session_state.data_service
                validation_results = data_service.validate_tickers([ticker.upper()])
                if validation_results.get(ticker.upper(), False):
                    try:
                        price = data_service.fetch_current_price(ticker.upper())
                        st.success(f"{ticker.upper()} is valid - Current price: ${price:.2f}")
                    except Exception:
                        st.success(f"{ticker.upper()} is valid - Price unavailable")
                else:
                    st.error(f"{ticker.upper()} is not a valid ticker")

        if add_button and ticker:
            # Validate ticker - Level 1 + Level 2
            ticker_valid, ticker_error = validate_ticker_format(ticker)
            if not ticker_valid:
                st.error(f"Ticker format: {ticker_error}")
            else:
                data_service: DataService = st.session_state.data_service
                validation_results = data_service.validate_tickers([ticker.upper()])
                
                if validation_results.get(ticker.upper(), False):
                    # Check for duplicates
                    existing_tickers = [asset['ticker'] for asset in st.session_state.creation_data['manual_assets']]

                    if ticker.upper() not in existing_tickers:
                        st.session_state.creation_data['manual_assets'].append({
                            'ticker': ticker.upper(),
                            'weight': weight / 100,
                            'name': ticker.upper()
                        })
                        st.success(f"Added {ticker.upper()}")
                        st.rerun()
                    else:
                        st.error(f"{ticker.upper()} already exists in portfolio")
                else:
                    st.error(f"{ticker.upper()} is not a valid ticker")

    # Show current assets
    if st.session_state.creation_data['manual_assets']:
        st.subheader("Current Assets")

        assets_data = []
        total_weight = 0

        for i, asset in enumerate(st.session_state.creation_data['manual_assets']):
            assets_data.append({
                'Ticker': asset['ticker'],
                'Weight': f"{asset['weight']:.1%}",
                'Remove': False
            })
            total_weight += asset['weight']

        # Asset management table
        edited_df = st.data_editor(
            pd.DataFrame(assets_data),
            column_config={
                'Ticker': st.column_config.TextColumn('Ticker', disabled=True),
                'Weight': st.column_config.TextColumn('Weight', disabled=True),
                'Remove': st.column_config.CheckboxColumn('Remove')
            },
            hide_index=True,
            use_container_width=True
        )

        # Handle removals
        if st.button("Remove Selected", use_container_width=True):
            indices_to_remove = [i for i, row in edited_df.iterrows() if row['Remove']]
            if indices_to_remove:
                for i in reversed(sorted(indices_to_remove)):
                    st.session_state.creation_data['manual_assets'].pop(i)
                st.rerun()

        # Portfolio metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Assets", len(st.session_state.creation_data['manual_assets']))
        with col2:
            st.metric("Total Weight", f"{total_weight:.1%}")
        with col3:
            weight_status = "Perfect" if abs(total_weight - 1.0) < 0.001 else "Will normalize"
            st.metric("Status", weight_status)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 2
            st.rerun()

    with col3:
        can_proceed = len(st.session_state.creation_data.get('manual_assets', [])) > 0
        if st.button("Next Step →", use_container_width=True, disabled=not can_proceed, type="primary"):
            st.session_state.creation_step = 4
            st.rerun()


def render_template_selection() -> None:
    """Template selection."""
    st.markdown("**Start with a proven strategy** and customize as needed")
    
    with st.expander("What are portfolio templates?", expanded=False):
        st.markdown("""
        **Portfolio Templates** are pre-built investment strategies based on:
        
        **Factor-Based Strategies:**
        - **Value Factor** - Undervalued companies with low P/E ratios
        - **Quality Factor** - High ROE companies with low debt
        - **Growth Factor** - Fast-growing companies with high revenue growth
        - **Low Volatility** - Stocks with beta < 0.8 for stability
        - **Dividend Factor** - High dividend yield stocks (3%+)
        
        **Classic Strategies:**
        - **60/40 Portfolio** - Classic balanced allocation (60% stocks, 40% bonds)
        - **All Weather Portfolio** - Multi-asset diversification across economic conditions
        - **Tech Focus** - Technology sector concentration
        
        **Customization:**
        - All templates can be customized after selection
        - You can modify weights, add/remove assets
        - Templates are starting points - make them your own!
        """)

    # Template definitions
    templates = {
        # Factor-based strategies
        "Value Factor": {
            "description": "Undervalued companies with low P/E and P/B ratios",
            "assets": "VTV 30%, IWD 25%, BRK-B 15%, JPM 10%, WMT 8%, CVX 7%, XOM 5%",
            "tags": ["Value", "Undervalued", "P/E < 15"]
        },
        "Quality Factor": {
            "description": "High ROE companies with low debt and stable profits",
            "assets": "QUAL 35%, MSFT 20%, AAPL 15%, JNJ 10%, PG 8%, V 7%, MA 5%",
            "tags": ["Quality", "ROE > 15%", "Low Debt"]
        },
        "Growth Factor": {
            "description": "Fast-growing companies with high revenue and EPS growth",
            "assets": "VUG 30%, IWF 25%, NVDA 15%, GOOGL 10%, AMZN 8%, TSLA 7%, META 5%",
            "tags": ["Growth", "Revenue > 10%", "EPS Growth"]
        },
        "Low Volatility": {
            "description": "Low volatility stocks with beta < 0.8",
            "assets": "USMV 40%, SPLV 30%, KO 8%, PG 7%, JNJ 6%, VZ 5%, WMT 4%",
            "tags": ["Low Vol", "Beta < 0.8", "Defensive"]
        },
        "Dividend Factor": {
            "description": "High dividend yield stocks with 3%+ yield",
            "assets": "VYM 25%, SCHD 25%, HDV 20%, T 8%, VZ 7%, XOM 6%, KO 5%, PFE 4%",
            "tags": ["Dividends", "Yield > 3%", "Income"]
        },
        # Core strategies
        "60/40 Portfolio": {
            "description": "Classic 60% stocks, 40% bonds allocation",
            "assets": "VTI 60%, BND 40%",
            "tags": ["Balanced", "Classic", "Moderate Risk"]
        },
        "All Weather Portfolio": {
            "description": "Multi-asset diversification across all economic conditions",
            "assets": "VTI 30%, BND 25%, GLD 15%, VNQ 15%, BTC-USD 10%, TIP 5%",
            "tags": ["Multi-Asset", "All Weather", "Diversified"]
        },
        "Tech Focus": {
            "description": "Technology sector concentration with growth leaders",
            "assets": "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 12%, META 10%, AMZN 8%, TSLA 5%, AMD 3%, CRM 2%",
            "tags": ["Technology", "Growth", "High Risk"]
        }
    }

    # Template selection
    selected_template = st.selectbox(
        "Choose a template:",
        list(templates.keys()),
        help="Select a base strategy to start with",
        key='creation_template_selection'
    )

    if selected_template:
        template_info = templates[selected_template]

        # Show template details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.info(f"**{selected_template}**: {template_info['description']}")
            st.code(template_info['assets'])

        with col2:
            st.markdown("**Tags:**")
            for tag in template_info['tags']:
                st.markdown(f"• {tag}")

        # Customization option
        customize = st.checkbox("Customize template", help="Modify the template allocation")

        if customize:
            custom_text = st.text_area(
                "Modify the template:",
                value=template_info['assets'],
                height=100,
                help="Edit the allocation to suit your needs",
                key='creation_custom_template'
            )
            final_template_text = custom_text
        else:
            final_template_text = template_info['assets']

        # Validate template
        if final_template_text:
            try:
                parsed_template = parse_ticker_weights_text(final_template_text)

                if parsed_template:
                    st.success(f"Template contains {len(parsed_template)} assets")

                    # Show parsed data
                    with st.expander("Template Preview", expanded=False):
                        template_data = []
                        for asset in parsed_template:
                            template_data.append({
                                'Ticker': asset['ticker'],
                                'Weight': f"{asset['weight']:.1%}"
                            })

                        st.dataframe(pd.DataFrame(template_data), hide_index=True, use_container_width=True)

                    st.session_state.creation_data['template_text'] = final_template_text
                    st.session_state.creation_data['template_name'] = selected_template
                    st.session_state.creation_data['description'] = template_info['description']
                else:
                    st.error("Could not parse template")

            except Exception as e:
                st.error(f"Error validating template: {str(e)}")
                logger.error(f"Template validation error: {e}")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 2
            st.rerun()

    with col3:
        can_proceed = selected_template and 'template_text' in st.session_state.creation_data
        if st.button("Next Step →", use_container_width=True, disabled=not can_proceed, type="primary"):
            st.session_state.creation_step = 4
            st.rerun()


def render_step_4() -> None:
    """Step 4: Options and Settings."""
    st.write("### Step 4: Portfolio Settings & Review")
    
    with st.expander("What settings should I configure?", expanded=False):
        st.markdown("""
        **Portfolio Options:**
        - **Fetch company information** - Automatically get company names, sectors, and market data
        - **Auto-normalize weights** - Automatically adjust weights to sum to 100%
        - **Update current prices** - Fetch latest market prices for all assets
        - **Calculate share quantities** - Calculate number of shares based on initial investment
        
        **Cash Management:**
        - **Planned Cash Allocation** - Percentage to intentionally keep in cash
        - Useful for maintaining liquidity or waiting for better entry points
        - Remaining percentage will be allocated to your assets
        - Example: 10% cash means 90% goes to your selected assets
        
        **Review:**
        - Check all your portfolio details before creation
        - Verify asset count and total weight
        - Make sure everything looks correct
        """)

    # Settings section
    st.subheader("Portfolio Options")

    col1, col2 = st.columns(2)

    with col1:
        fetch_info = st.checkbox(
            "Fetch company information",
            value=True,
            help="Automatically get company names, sectors, and market data"
        )

        auto_normalize = st.checkbox(
            "Auto-normalize weights",
            value=True,
            help="Automatically adjust weights to sum to 100%"
        )

    with col2:
        update_prices = st.checkbox(
            "Update current prices",
            value=True,
            help="Fetch latest market prices for all assets"
        )

        calculate_shares = st.checkbox(
            "Calculate share quantities",
            value=True,
            help="Calculate number of shares based on initial investment"
        )

    # Cash allocation section
    st.subheader("Cash Management")

    col1, col2 = st.columns(2)

    with col1:
        cash_allocation = st.slider(
            "Planned Cash Allocation (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Percentage to intentionally keep in cash"
        )

    with col2:
        if cash_allocation > 0:
            cash_amount = st.session_state.creation_data.get('initial_value', 100000) * (cash_allocation / 100)
            st.metric("Cash Amount", f"${cash_amount:,.0f}")
            st.info(f"Remaining for investments: {100 - cash_allocation}%")

    # Portfolio preview
    st.subheader("Portfolio Summary")

    # Basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Name", st.session_state.creation_data.get('name', 'N/A'))
        st.metric("Currency", st.session_state.creation_data.get('currency', 'USD'))

    with col2:
        st.metric("Initial Value", f"${st.session_state.creation_data.get('initial_value', 0):,.0f}")
        method_display = st.session_state.creation_data.get('input_method', 'N/A')
        st.metric("Method", method_display)

    with col3:
        # Count assets based on method
        asset_count = 0
        if 'parsed_assets' in st.session_state.creation_data:
            asset_count = len(st.session_state.creation_data['parsed_assets'])
        elif 'manual_assets' in st.session_state.creation_data:
            asset_count = len(st.session_state.creation_data['manual_assets'])
        elif 'file_data' in st.session_state.creation_data:
            asset_count = len(st.session_state.creation_data['file_data']['tickers'])
        elif 'template_text' in st.session_state.creation_data:
            try:
                parsed_template = parse_ticker_weights_text(
                    st.session_state.creation_data['template_text'])
                asset_count = len(parsed_template)
            except Exception:
                asset_count = 0

        st.metric("Assets", asset_count)
        description = st.session_state.creation_data.get('description', 'No description')
        st.metric("Description", description[:15] + "..." if len(description) > 15 else description)

    # Save settings
    st.session_state.creation_data['settings'] = {
        'fetch_info': fetch_info,
        'auto_normalize': auto_normalize,
        'update_prices': update_prices,
        'calculate_shares': calculate_shares,
        'cash_allocation': cash_allocation
    }

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state.creation_step = 3
            st.rerun()

    with col3:
        if st.button("Create Portfolio", use_container_width=True, type="primary"):
            st.session_state.creation_step = 5
            st.rerun()


def render_step_5() -> None:
    """Step 5: Portfolio Creation and Results."""
    st.write("### Step 5: Creating Portfolio...")

    # Show progress
    progress_bar = st.progress(0, "Initializing...")
    status_container = st.empty()

    try:
        # Create portfolio
        progress_bar.progress(0.2, "Creating portfolio structure...")
        status_container.info("Setting up portfolio...")

        result = create_portfolio_from_creation_data()

        if result['success']:
            progress_bar.progress(1.0, "Portfolio created successfully!")
            status_container.success("Portfolio creation completed!")

            st.success("Portfolio created successfully!")

            portfolio = result['portfolio']

            # Show results
            st.subheader("Portfolio Created")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Assets", len(portfolio.get_all_positions()))

            with col2:
                # Calculate total value from positions using creation prices
                # This ensures consistency with asset allocation table
                try:
                    portfolio_positions = portfolio.get_all_positions()
                    data_service = st.session_state.data_service
                    tickers = [pos.ticker for pos in portfolio_positions if pos.ticker != "CASH"]
                    prices = data_service.get_latest_prices(tickers) if tickers else {}
                    
                    total_value = 0.0
                    for pos in portfolio_positions:
                        if pos.ticker == "CASH":
                            total_value += pos.shares  # CASH shares = dollar amount
                        else:
                            price = prices.get(pos.ticker, pos.purchase_price or 0.0)
                            if price > 0:
                                total_value += pos.shares * price
                    
                    # Fallback to starting_capital if calculation fails
                    if total_value <= 0:
                        total_value = portfolio.starting_capital
                except Exception as e:
                    logger.warning(f"Error calculating total value: {e}")
                    total_value = portfolio.starting_capital
                st.metric("Total Value", f"${total_value:,.2f}")

            with col3:
                st.metric("Created", datetime.now().strftime("%H:%M:%S"))

            with col4:
                st.metric("Currency", portfolio.base_currency)

            # Asset Allocation Table
            st.subheader("Asset Allocation")

            try:
                # Get portfolio positions with prices and company names
                positions_data = []
                portfolio_positions = portfolio.get_all_positions()

                if portfolio_positions:
                    # Get data service from session state
                    data_service = st.session_state.data_service
                    # Get prices for all tickers (except CASH)
                    tickers = [
                        pos.ticker
                        for pos in portfolio_positions
                        if pos.ticker != "CASH"
                    ]
                    prices = {}
                    company_names = {}

                    if tickers:
                        try:
                            prices = data_service.get_latest_prices(tickers)
                            # Get company names (if available)
                            for ticker in tickers:
                                try:
                                    ticker_info = data_service.get_ticker_info(
                                        ticker
                                    )
                                    company_names[ticker] = (
                                        ticker_info.name
                                        if ticker_info.name
                                        else ticker
                                    )
                                except Exception:
                                    company_names[ticker] = ticker
                        except Exception as e:
                            logger.warning(f"Error fetching prices: {e}")
                    
                    # Calculate total value first
                    total_value = 0.0
                    position_values = {}
                    for pos in portfolio_positions:
                        ticker = pos.ticker
                        if ticker == "CASH":
                            price = 1.0
                            value = pos.shares * price
                        else:
                            price = prices.get(ticker, pos.purchase_price or 0.0)
                            value = pos.shares * price if price > 0 else 0.0
                        position_values[ticker] = value
                        total_value += value
                    
                    # Prepare table data with correct weights
                    for pos in portfolio_positions:
                        ticker = pos.ticker
                        value = position_values[ticker]
                        
                        if ticker == "CASH":
                            price = 1.0
                            company_name = "Cash Position"
                            shares_display = f"${pos.shares:,.2f}"
                        else:
                            price = prices.get(ticker, pos.purchase_price or 0.0)
                            company_name = company_names.get(ticker, ticker)
                            shares_display = f"{pos.shares:,.2f}"
                        
                        # Calculate weight based on actual value
                        weight = (value / total_value) if total_value > 0 else (pos.weight_target or 0.0)
                        
                        positions_data.append({
                            "Ticker": ticker,
                            "Name": company_name,
                            "Weight": f"{weight:.1%}",
                            "Shares": shares_display,
                            "Price": f"${price:,.2f}",
                            "Value": f"${value:,.2f}"
                        })
                    
                    # Display table
                    if positions_data:
                        df = pd.DataFrame(positions_data)
                        st.dataframe(
                            df,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No positions to display")
                else:
                    st.info("No positions in portfolio")
            except Exception as e:
                logger.error(f"Error displaying asset allocation: {e}")
                st.warning("Could not load asset allocation details")

            # Action buttons
            st.subheader("What's Next?")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Analyze Portfolio", use_container_width=True, type="primary"):
                    st.session_state.selected_portfolio_id = portfolio.id
                    reset_creation()
                    st.switch_page("pages/portfolio_analysis.py")

            with col2:
                if st.button("Create Another", use_container_width=True):
                    reset_creation()
                    st.rerun()

            with col3:
                if st.button("Go to Dashboard", use_container_width=True):
                    reset_creation()
                    st.switch_page("pages/dashboard.py")

        else:
            progress_bar.progress(0.0, "Error occurred")
            status_container.error("Portfolio creation failed")
            st.error(f"Portfolio creation failed: {result['error']}")

            # Error recovery options
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Try Again", use_container_width=True, type="primary"):
                    st.session_state.creation_step = 4
                    st.rerun()

            with col2:
                if st.button("Start Over", use_container_width=True):
                    reset_creation()
                    st.rerun()

    except Exception as e:
        progress_bar.progress(0.0, "Unexpected error")
        status_container.error("Unexpected error occurred")
        st.error(f"Unexpected error: {str(e)}")
        logger.error(f"Portfolio creation error: {e}", exc_info=True)

        if st.button("Start Over", use_container_width=True):
            reset_creation()
            st.rerun()


def create_portfolio_from_creation_data() -> Dict[str, Any]:
    """Create portfolio from creation data."""
    try:
        creation_data = st.session_state.creation_data
        portfolio_service: PortfolioService = st.session_state.portfolio_service
        data_service: DataService = st.session_state.data_service

        # Prepare positions based on input method
        positions: List[PositionSchema] = []

        if 'parsed_assets' in creation_data:
            # From text input
            for asset in creation_data['parsed_assets']:
                # Ensure weight is normalized (0.0 to 1.0)
                weight = asset['weight']
                if weight > 1.0:
                    weight = weight / 100.0
                weight = max(0.0, min(1.0, weight))  # Clamp to [0.0, 1.0]
                positions.append(PositionSchema(
                    ticker=asset['ticker'],
                    shares=None,  # Will be calculated
                    weight_target=weight
                ))

        elif 'manual_assets' in creation_data:
            # From manual entry
            for asset in creation_data['manual_assets']:
                # Ensure weight is normalized (0.0 to 1.0)
                weight = asset['weight']
                if weight > 1.0:
                    weight = weight / 100.0
                weight = max(0.0, min(1.0, weight))  # Clamp to [0.0, 1.0]
                positions.append(PositionSchema(
                    ticker=asset['ticker'],
                    shares=None,  # Will be calculated
                    weight_target=weight
                ))

        elif 'template_text' in creation_data:
            # From template
            parsed_template = parse_ticker_weights_text(creation_data['template_text'])
            for asset in parsed_template:
                # Ensure weight is normalized (0.0 to 1.0)
                weight = asset['weight']
                if weight > 1.0:
                    weight = weight / 100.0
                weight = max(0.0, min(1.0, weight))  # Clamp to [0.0, 1.0]
                positions.append(PositionSchema(
                    ticker=asset['ticker'],
                    shares=None,  # Will be calculated
                    weight_target=weight
                ))

        elif 'file_data' in creation_data:
            # From file upload
            file_data = creation_data['file_data']
            tickers = file_data['tickers']
            weights = file_data['weights']

            for ticker, weight in zip(tickers, weights):
                # Ensure weight is normalized (0.0 to 1.0)
                normalized_weight = weight
                if normalized_weight > 1.0:
                    normalized_weight = normalized_weight / 100.0
                normalized_weight = max(0.0, min(1.0, normalized_weight))  # Clamp
                positions.append(PositionSchema(
                    ticker=ticker,
                    shares=None,  # Will be calculated
                    weight_target=normalized_weight
                ))

        if not positions:
            return {'success': False, 'error': 'No asset data provided'}

        # Check for duplicate name - Level 2: Service validation
        existing_portfolios = portfolio_service.list_portfolios()
        existing_names = [p.name for p in existing_portfolios]

        if creation_data['name'] in existing_names:
            return {'success': False, 'error': f"Portfolio '{creation_data['name']}' already exists"}

        # Get initial value and calculate shares
        initial_value = creation_data.get('initial_value', 100000.0)
        settings = creation_data.get('settings', {})
        cash_allocation = settings.get('cash_allocation', 0)

        # Adjust asset weights if cash allocation is planned
        if cash_allocation > 0:
            # Scale down all asset weights to make room for cash
            cash_weight = cash_allocation / 100
            investment_weight = 1 - cash_weight

            # Create new positions with scaled weights
            scaled_positions = []
            for pos in positions:
                if pos.weight_target:
                    scaled_weight = pos.weight_target * investment_weight
                    scaled_positions.append(PositionSchema(
                        ticker=pos.ticker,
                        shares=pos.shares,
                        weight_target=scaled_weight,
                        purchase_price=pos.purchase_price,
                        purchase_date=pos.purchase_date
                    ))
                else:
                    scaled_positions.append(pos)
            positions = scaled_positions

        # If calculate_shares is enabled, fetch prices and calculate shares
        prices: Dict[str, float] = {}
        if settings.get('calculate_shares', True):
            tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
            try:
                prices = data_service.get_latest_prices(tickers)
                # Create new positions with calculated shares
                updated_positions = []
                for pos in positions:
                    if pos.ticker in prices and prices[pos.ticker] > 0:
                        # Calculate shares based on weight and initial value
                        calculated_shares = 0.0
                        if pos.weight_target:
                            position_value = initial_value * pos.weight_target
                            calculated_shares = max(0.01, position_value / prices[pos.ticker])
                        
                        # Create new PositionSchema with calculated shares
                        updated_positions.append(PositionSchema(
                            ticker=pos.ticker,
                            shares=calculated_shares,
                            weight_target=pos.weight_target,
                            purchase_price=pos.purchase_price,
                            purchase_date=pos.purchase_date
                        ))
                    else:
                        # Keep original position if price not available
                        # Use minimum shares if not calculated
                        if pos.shares is None:
                            updated_positions.append(PositionSchema(
                                ticker=pos.ticker,
                                shares=0.01,  # Minimum value
                                weight_target=pos.weight_target,
                                purchase_price=pos.purchase_price,
                                purchase_date=pos.purchase_date
                            ))
                        else:
                            updated_positions.append(pos)
                positions = updated_positions

                # Calculate total invested amount (actual shares * price)
                total_invested = 0.0
                for pos in positions:
                    if pos.ticker in prices and prices[pos.ticker] > 0:
                        # Calculate actual invested = shares * price
                        total_invested += pos.shares * prices[pos.ticker]
                    elif pos.ticker == "CASH":
                        # Skip CASH in invested calculation
                        continue

                # Calculate total cash (planned + remainder from rounding)
                # If cash_allocation > 0, include planned cash, otherwise just remainder
                if cash_allocation > 0:
                    planned_cash = initial_value * (cash_allocation / 100)
                    remainder_cash = max(0, initial_value - total_invested - planned_cash)
                    total_cash = planned_cash + remainder_cash
                else:
                    total_cash = max(0, initial_value - total_invested)

                # Add cash as a position if there's any cash (planned or remainder)
                if total_cash > 1.0:  # Only add if more than $1
                    cash_position = PositionSchema(
                        ticker="CASH",
                        shares=total_cash,  # Cash amount in shares field
                        weight_target=total_cash / initial_value
                    )
                    positions.append(cash_position)

                # Final normalization to ensure weights sum to exactly 1.0
                # (similar to reference implementation)
                total_weight = sum(
                    pos.weight_target for pos in positions if pos.weight_target
                )
                if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
                    # Normalize all weights
                    normalized_positions = []
                    for pos in positions:
                        normalized_weight = (
                            pos.weight_target / total_weight
                            if pos.weight_target
                            else None
                        )
                        normalized_positions.append(PositionSchema(
                            ticker=pos.ticker,
                            shares=pos.shares,
                            weight_target=normalized_weight,
                            purchase_price=pos.purchase_price,
                            purchase_date=pos.purchase_date
                        ))
                    positions = normalized_positions

            except Exception as e:
                logger.warning(f"Error fetching prices for share calculation: {e}")
                # If error, still add planned cash if allocation > 0
                if cash_allocation > 0:
                    planned_cash = initial_value * (cash_allocation / 100)
                    if planned_cash > 1.0:
                        cash_position = PositionSchema(
                            ticker="CASH",
                            shares=planned_cash,
                            weight_target=cash_allocation / 100
                        )
                        positions.append(cash_position)

        # Ensure all positions have shares (minimum 0.01 if not calculated)
        final_positions = []
        for pos in positions:
            if pos.shares is None or pos.shares <= 0:
                final_positions.append(PositionSchema(
                    ticker=pos.ticker,
                    shares=0.01,  # Minimum value
                    weight_target=pos.weight_target,
                    purchase_price=pos.purchase_price,
                    purchase_date=pos.purchase_date
                ))
            else:
                final_positions.append(pos)
        positions = final_positions

        # Create portfolio - Level 3: Model validation via Service
        request = CreatePortfolioRequest(
            name=creation_data['name'],
            description=creation_data.get('description', ''),
            starting_capital=initial_value,
            base_currency=creation_data.get('currency', 'USD'),
            positions=positions
        )

        portfolio = portfolio_service.create_portfolio(request)

        # Positions already have calculated shares in PositionSchema objects
        # They were used when creating portfolio, so no need to update again
        # The portfolio is created with correct shares already

        return {'success': True, 'portfolio': portfolio}

    except ConflictError as e:
        logger.error(f"Portfolio creation conflict: {e}")
        return {'success': False, 'error': str(e)}
    except ValidationError as e:
        logger.error(f"Portfolio creation validation error: {e}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Error creating portfolio from creation data: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def reset_creation() -> None:
    """Reset portfolio creation state."""
    st.session_state.creation_step = 1
    st.session_state.creation_data = {}


if __name__ == "__main__":
    render_create_portfolio()
