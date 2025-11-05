"""Main Streamlit application entry point."""

import logging

import streamlit as st

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
with open("streamlit_app/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("📊 Portfolio Manager")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Create Portfolio",
            "Portfolio List",
            "Portfolio Analysis",
            "Portfolio Optimization",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick actions
    st.subheader("Quick Actions")
    if st.button("🆕 New Portfolio"):
        st.switch_page("pages/create_portfolio.py")

    st.markdown("---")

    # Info
    st.info(
        "Portfolio Management System\n\n"
        "Manage your investment portfolios with comprehensive analytics."
    )


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
