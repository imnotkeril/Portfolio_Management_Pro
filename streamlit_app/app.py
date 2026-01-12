"""Main Streamlit application entry point."""

import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from streamlit_app.utils.chart_config import COLORS

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Portfolio Management System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
from pathlib import Path

css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not load CSS file: {e}")
else:
    logger.warning("CSS file not found, using default styles")

# Sidebar navigation
with st.sidebar:
    st.title("Portfolio Manager")
    st.markdown("---")

    # Check if page was set in session state (e.g., from portfolio list)
    if "page" in st.session_state:
        default_page = st.session_state.page
        # Clear it after using
        del st.session_state.page
    else:
        default_page = "Dashboard"

    # Find index of default page
    page_options = [
        "Dashboard",
        "Create Portfolio",
        "Portfolio List",
        "Portfolio Analysis",
        "Portfolio Optimization",
        "Risk Analysis",
        "Forecasting",
    ]
    try:
        default_index = page_options.index(default_page)
    except ValueError:
        default_index = 0

    page = st.radio(
        "Navigation",
        page_options,
        index=default_index,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System status section
    st.subheader("System Status")

    # API Status with caching to avoid checking on every render
    if "api_status" not in st.session_state:
        st.session_state.api_status = None

    # Check API status (only once per session or on refresh)
    refresh_button = st.button(
        "Refresh Status", use_container_width=True
    )
    if st.session_state.api_status is None or refresh_button:
        try:
            from services.data_service import DataService
            import time

            start_time = time.time()
            data_service = DataService()
            test_result = data_service.validate_ticker("AAPL")
            response_time = (time.time() - start_time) * 1000  # ms

            if test_result:
                st.session_state.api_status = {
                    "status": "online",
                    "response_time": response_time
                }
            else:
                st.session_state.api_status = {
                    "status": "limited",
                    "response_time": response_time
                }
        except (ImportError, AttributeError) as e:
            logger.warning(f"API check failed due to import/attribute error: {e}")
            st.session_state.api_status = {
                "status": "offline",
                "error": "Service unavailable"
            }
        except Exception as e:
            logger.error(f"Unexpected error during API check: {e}", exc_info=True)
            st.session_state.api_status = {
                "status": "unknown",
                "error": "Unexpected error"
            }

    # Display status
    status = st.session_state.api_status
    if status:
        if status["status"] == "online":
            rt = status['response_time']
            st.success(f"API: Online ({rt:.0f}ms)")
        elif status["status"] == "limited":
            rt = status['response_time']
            st.warning(f"API: Limited ({rt:.0f}ms)")
        else:
            st.error("API: Offline")

    st.markdown("---")

    # About section
    st.subheader("About")
    st.markdown(f"""
    <div style="
        background-color: #1A1E29;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #2A2E39;
    ">
        <p style="color: #D1D4DC; font-size: 0.9em; margin: 0 0 8px 0;">
            <strong style="color: {COLORS['primary']};">Portfolio Management Terminal</strong>
        </p>
        <p style="color: {COLORS['secondary']}; font-size: 0.8em; margin: 0;
           line-height: 1.4;">
            Professional portfolio management terminal with:
        </p>
        <ul style="color: #D1D4DC; font-size: 0.75em; margin: 8px 0 0 0;
            padding-left: 20px; line-height: 1.6;">
            <li>70+ analytics metrics</li>
            <li>17 optimization methods</li>
            <li>Risk & scenario analysis</li>
            <li>Price forecasting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Footer
    st.caption("Version 1.0.0")
    st.caption("Data: Yahoo Finance")


# Route to selected page
if page == "Dashboard":
    from streamlit_app.pages.dashboard import render_dashboard

    render_dashboard()

elif page == "Create Portfolio":
    from streamlit_app.pages.create_portfolio import render_create_portfolio

    render_create_portfolio()

elif page == "Portfolio List":
    from streamlit_app.pages.portfolio_list import render_portfolio_list

    render_portfolio_list()

elif page == "Portfolio Analysis":
    from streamlit_app.pages.portfolio_analysis import show

    show()

elif page == "Portfolio Optimization":
    from streamlit_app.pages.portfolio_optimization import (
        render_optimization_page,
    )

    render_optimization_page()

elif page == "Risk Analysis":
    from streamlit_app.pages.risk_analysis import (
        render_risk_analysis_page,
    )

    render_risk_analysis_page()

elif page == "Forecasting":
    from streamlit_app.pages.forecasting import render_forecasting_page

    render_forecasting_page()
