"""Components for individual asset analysis."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils.chart_config import COLORS
from streamlit_app.utils.formatters import format_currency, format_percentage

logger = logging.getLogger(__name__)


def calculate_asset_metrics(
    asset_prices: pd.Series,
    portfolio_returns: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    Calculate metrics for an individual asset.

    Args:
        asset_prices: Price series for the asset
        portfolio_returns: Portfolio returns for correlation

    Returns:
        Dictionary of asset metrics
    """
    # Calculate returns
    asset_returns = asset_prices.pct_change().dropna()

    if asset_returns.empty or len(asset_returns) < 2:
        return {}

    # Total return
    total_return = (asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1

    # Annualized return
    trading_days = len(asset_returns)
    years = trading_days / 252
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = None

    # Volatility
    volatility = asset_returns.std() * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + asset_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    if volatility > 0:
        sharpe_ratio = annualized_return / volatility
    else:
        sharpe_ratio = None

    # Beta and correlation with portfolio
    aligned_returns = pd.DataFrame(
        {"asset": asset_returns, "portfolio": portfolio_returns}
    ).dropna()

    if len(aligned_returns) > 10:
        covariance = aligned_returns["asset"].cov(aligned_returns["portfolio"])
        portfolio_variance = aligned_returns["portfolio"].var()

        if portfolio_variance > 0:
            beta = covariance / portfolio_variance
        else:
            beta = None

        correlation = aligned_returns["asset"].corr(aligned_returns["portfolio"])
    else:
        beta = None
        correlation = None

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "beta": beta,
        "correlation": correlation,
    }


def render_positions_overview_table(
    positions: List,
    prices_dict: Dict[str, float],
    total_value: float,
) -> None:
    """
    Render positions overview table.

    Args:
        positions: List of Position objects
        prices_dict: Dictionary of ticker -> current price
        total_value: Total portfolio value
    """
    st.subheader("Positions Overview")

    rows = []
    for position in positions:
        ticker = position.ticker
        shares = position.shares
        current_price = prices_dict.get(ticker, 0)
        value = shares * current_price
        weight = value / total_value if total_value > 0 else 0

        rows.append(
            {
                "Ticker": ticker,
                "Shares": f"{shares:,.2f}",
                "Price": format_currency(current_price),
                "Value": format_currency(value),
                "Weight": format_percentage(weight, decimals=2),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_asset_allocation_chart(
    positions: List,
    prices_dict: Dict[str, float],
) -> None:
    """
    Render asset allocation pie chart.

    Args:
        positions: List of Position objects
        prices_dict: Dictionary of ticker -> current price
    """
    st.subheader("Asset Allocation")

    # Prepare data
    data = []
    for position in positions:
        ticker = position.ticker
        shares = position.shares
        price = prices_dict.get(ticker, 0)
        value = shares * price

        if value > 0:
            data.append({"Ticker": ticker, "Value": value})

    if not data:
        st.warning("No allocation data available")
        return

    df = pd.DataFrame(data)

    # Create pie chart
    fig = px.pie(
        df,
        values="Value",
        names="Ticker",
        title="Portfolio Allocation",
        hole=0.4,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_top_bottom_performers(
    asset_returns_dict: Dict[str, float],
    top_n: int = 5,
) -> None:
    """
    Render top and bottom performers.

    Args:
        asset_returns_dict: Dictionary of ticker -> return
        top_n: Number of top/bottom performers to show
    """
    st.subheader("Top & Bottom Performers")

    if not asset_returns_dict:
        st.info("No performance data available")
        return

    # Sort by return
    sorted_returns = sorted(
        asset_returns_dict.items(), key=lambda x: x[1], reverse=True
    )

    # Top performers
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Performers**")
        top_performers = sorted_returns[:top_n]
        for ticker, ret in top_performers:
            st.metric(ticker, format_percentage(ret, decimals=2))

    with col2:
        st.markdown("**Bottom Performers**")
        bottom_performers = sorted_returns[-top_n:][::-1]
        for ticker, ret in bottom_performers:
            st.metric(ticker, format_percentage(ret, decimals=2))


def calculate_correlation_matrix(
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlation matrix between assets.

    Args:
        prices_df: DataFrame with prices (dates x tickers)

    Returns:
        Correlation matrix
    """
    # Calculate returns
    returns_df = prices_df.pct_change().dropna()

    # Calculate correlation
    correlation_matrix = returns_df.corr()

    return correlation_matrix


def render_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
) -> None:
    """
    Render correlation matrix heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
    """
    st.subheader("Correlation Matrix")

    if correlation_matrix.empty:
        st.info("Insufficient data for correlation analysis")
        return

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Asset Correlation Matrix",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_risk_contribution(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate risk contribution of each asset.

    Args:
        returns_df: DataFrame of asset returns
        weights: Array of asset weights

    Returns:
        DataFrame with risk contribution metrics
    """
    # Covariance matrix
    cov_matrix = returns_df.cov() * 252  # Annualized

    # Portfolio variance
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)

    # Marginal contribution to risk
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std

    # Component contribution to risk
    component_contrib = weights * marginal_contrib

    # Percentage contribution
    percentage_contrib = component_contrib / portfolio_std

    # Create DataFrame
    risk_contrib_df = pd.DataFrame(
        {
            "Ticker": returns_df.columns,
            "Weight": weights,
            "Marginal Risk": marginal_contrib,
            "Component Risk": component_contrib,
            "Risk Contribution (%)": percentage_contrib * 100,
        }
    )

    return risk_contrib_df


def render_risk_contribution_chart(
    risk_contrib_df: pd.DataFrame,
) -> None:
    """
    Render risk contribution bar chart.

    Args:
        risk_contrib_df: DataFrame with risk contribution data
    """
    st.subheader("Risk Contribution")

    if risk_contrib_df.empty:
        st.info("Insufficient data for risk contribution analysis")
        return

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=risk_contrib_df["Ticker"],
                y=risk_contrib_df["Risk Contribution (%)"],
                marker_color=COLORS["primary"],
                text=risk_contrib_df["Risk Contribution (%)"].round(2),
                texttemplate="%{text}%",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Risk Contribution by Asset",
        xaxis_title="Ticker",
        yaxis_title="Risk Contribution (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display table
    st.dataframe(
        risk_contrib_df.style.format(
            {
                "Weight": "{:.2%}",
                "Marginal Risk": "{:.4f}",
                "Component Risk": "{:.4f}",
                "Risk Contribution (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_individual_asset_metrics(
    ticker: str,
    metrics: Dict[str, Optional[float]],
) -> None:
    """
    Render metrics for individual asset.

    Args:
        ticker: Asset ticker
        metrics: Dictionary of metrics
    """
    st.markdown(f"### {ticker}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = metrics.get("total_return")
        if total_return is not None:
            st.metric("Total Return", format_percentage(total_return, decimals=2))

    with col2:
        ann_return = metrics.get("annualized_return")
        if ann_return is not None:
            formatted = format_percentage(ann_return, decimals=2)
            st.metric("Annualized Return", formatted)

    with col3:
        volatility = metrics.get("volatility")
        if volatility is not None:
            st.metric("Volatility", format_percentage(volatility, decimals=2))

    with col4:
        max_dd = metrics.get("max_drawdown")
        if max_dd is not None:
            st.metric("Max Drawdown", format_percentage(max_dd, decimals=2))

    col5, col6, col7 = st.columns(3)

    with col5:
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is not None:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col6:
        beta = metrics.get("beta")
        if beta is not None:
            st.metric("Beta", f"{beta:.2f}")

    with col7:
        correlation = metrics.get("correlation")
        if correlation is not None:
            st.metric("Correlation", f"{correlation:.2f}")

    st.markdown("---")


def render_assets_table_extended(
    asset_data: pd.DataFrame,
) -> None:
    """
    Render extended asset information table with sortable columns.
    
    Args:
        asset_data: DataFrame with columns: ticker, weight, name, sector, 
                    industry, currency, price, change_pct
    """
    if asset_data is None or asset_data.empty:
        st.info("No asset data available")
        return
    
    # Format the dataframe for display
    display_df = asset_data.copy()
    
    # Add row number
    display_df.insert(0, "#", range(1, len(display_df) + 1))
    
    # Format columns
    if "weight" in display_df.columns:
        display_df["weight"] = display_df["weight"].apply(lambda x: f"{x:.2f}%")
    
    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].apply(
            lambda x: f"${x:,.2f}" if x > 0 else "N/A"
        )
    
    if "change_pct" in display_df.columns:
        # Store original values for coloring
        change_values = display_df["change_pct"].copy()
        display_df["change_pct"] = display_df["change_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "0.00%"
        )
    
    # Rename columns for display
    column_mapping = {
        "#": "#",
        "ticker": "Ticker",
        "weight": "Weight %",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "currency": "Currency",
        "price": "Price",
        "change_pct": "Change %",
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Select columns to display
    display_columns = ["#", "Ticker", "Weight %", "Name", "Sector", 
                      "Industry", "Currency", "Price", "Change %"]
    display_df = display_df[[col for col in display_columns if col in display_df.columns]]
    
    # Apply styling
    def highlight_change(val):
        """Highlight positive/negative changes."""
        if isinstance(val, str) and "%" in val:
            try:
                num_val = float(val.replace("%", "").replace("+", ""))
                if num_val > 0:
                    return "color: #4CAF50;"  # Green
                elif num_val < 0:
                    return "color: #F44336;"  # Red
            except ValueError:
                pass
        return ""
    
    # Display with styling
    styled_df = display_df.style.applymap(
        highlight_change,
        subset=["Change %"] if "Change %" in display_df.columns else []
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, (len(display_df) + 1) * 35 + 3),
    )
