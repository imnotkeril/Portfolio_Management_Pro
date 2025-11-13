"""Portfolio Optimization page."""

import logging
from datetime import date, timedelta
from typing import Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.portfolio_service import PortfolioService
from services.optimization_service import OptimizationService
from services.analytics_service import AnalyticsService
from core.exceptions import CalculationError, InsufficientDataError
from streamlit_app.utils.chart_config import COLORS


logger = logging.getLogger(__name__)

# Method display names - only methods that are actual optimization algorithms
METHOD_NAMES = {
    "mean_variance": "Mean-Variance (Markowitz)",
    "black_litterman": "Black-Litterman",
    "risk_parity": "Risk Parity",
    "hrp": "Hierarchical Risk Parity (HRP)",
    "cvar_optimization": "CVaR Optimization",
    "mean_cvar": "Mean-CVaR",
    "robust": "Robust Optimization",
    "max_diversification": "Maximum Diversification",
    "min_correlation": "Minimum Correlation",
    "inverse_correlation": "Inverse Correlation Weighting",
}

# Methods available for selection (filtered list)
AVAILABLE_OPTIMIZATION_METHODS = [
    "mean_variance",
    "black_litterman",
    "risk_parity",
    "hrp",
    "cvar_optimization",
    "mean_cvar",
    "robust",
    "max_diversification",
    "min_correlation",
    "inverse_correlation",
]

# Objective function mapping
OBJECTIVE_METHOD_MAPPING = {
    "maximize_sharpe": {
        "methods": ["mean_variance", "black_litterman", "robust"],
        "display": "Maximize Sharpe Ratio",
        "default_for": ["mean_variance", "black_litterman", "robust"],
    },
    "minimize_volatility": {
        "methods": ["mean_variance", "black_litterman"],
        "display": "Minimize Volatility",
        "default_for": [],
    },
    "maximize_return": {
        "methods": ["mean_variance", "black_litterman"],
        "display": "Maximize Expected Return",
        "default_for": [],
    },
    "minimize_cvar": {
        "methods": ["cvar_optimization", "mean_cvar"],
        "display": "Minimize CVaR / Expected Shortfall",
        "default_for": ["cvar_optimization"],
    },
    # Note: risk_parity and hrp have fixed objectives (equal risk contribution)
    # They don't accept objective parameter
}

# Methods with fixed objectives (no choice)
FIXED_OBJECTIVE_METHODS = [
    "risk_parity",  # Always equal risk contribution
    "hrp",  # Always equal risk contribution (hierarchical)
    "max_diversification",
    "min_correlation",
    "inverse_correlation",
    "cvar_optimization",  # Always minimize CVaR
    "mean_cvar",  # Always maximize Return/CVaR
    "min_variance",  # Always minimize variance
    "max_return",  # Always maximize return
    "max_sharpe",  # Always maximize Sharpe
    "equal_weight",  # Always equal weights
    "market_cap",  # Always market cap weights
    "kelly_criterion",  # Always Kelly criterion
    "min_tracking_error",  # Always minimize tracking error
    "max_alpha",  # Always maximize alpha
]


def validate_weight_constraints(
    min_weight: float = None,
    max_weight: float = None,
    num_assets: int = 1,
) -> list:
    """
    Validate weight constraints against number of assets.
    
    Args:
        min_weight: Minimum weight per asset (0.0 to 1.0)
        max_weight: Maximum weight per asset (0.0 to 1.0)
        num_assets: Number of assets in portfolio
    
    Returns:
        List of warning messages (empty if no warnings)
    """
    warnings = []
    
    if min_weight is not None:
        if min_weight * num_assets > 1.0:
            max_allowed = 1.0 / num_assets
            warnings.append(
                f"**Warning:** Minimum weight ({min_weight:.1%}) is too high "
                f"for {num_assets} assets. Sum of minimum weights will be "
                f"{min_weight * num_assets:.1%} (>100%). "
                f"Recommended: min_weight ≤ {max_allowed:.2%}"
            )
    
    if max_weight is not None and min_weight is not None:
        if max_weight < min_weight:
            warnings.append(
                f"**Error:** Maximum weight ({max_weight:.1%}) is less than "
                f"minimum weight ({min_weight:.1%})"
            )
    
    return warnings


def get_available_objectives(method: str) -> list:
    """
    Get list of available objectives for a given method.
    
    Args:
        method: Optimization method name
    
    Returns:
        List of objective keys available for this method
    """
    if method in FIXED_OBJECTIVE_METHODS:
        return []
    
    available = []
    for obj_key, obj_data in OBJECTIVE_METHOD_MAPPING.items():
        if method in obj_data["methods"]:
            available.append(obj_key)
    
    return available


def get_default_objective(method: str) -> str:
    """
    Get default objective for a method.
    
    Args:
        method: Optimization method name
    
    Returns:
        Default objective key, or None if method has fixed objective
    """
    if method in FIXED_OBJECTIVE_METHODS:
        return None
    
    for obj_key, obj_data in OBJECTIVE_METHOD_MAPPING.items():
        if method in obj_data.get("default_for", []):
            return obj_key
    
    # If no default specified, use first available
    available = get_available_objectives(method)
    return available[0] if available else None


def _interpret_optimization_comparison(optimized_metrics: dict, current_metrics: dict) -> str:
    """Interpret comparison between optimized and current portfolio metrics."""
    if not optimized_metrics or not current_metrics:
        return ""
    
    parts = []
    parts.append("**Optimization Results Analysis:**")
    
    # Calculate improvements
    sharpe_diff = optimized_metrics.get("sharpe_ratio", 0) - current_metrics.get("sharpe_ratio", 0)
    return_diff = optimized_metrics.get("annualized_return", 0) - current_metrics.get("annualized_return", 0)
    vol_diff = optimized_metrics.get("volatility", 0) - current_metrics.get("volatility", 0)
    dd_diff = optimized_metrics.get("max_drawdown", 0) - current_metrics.get("max_drawdown", 0)
    
    # Sharpe ratio analysis
    if abs(sharpe_diff) < 0.1:
        parts.append(f"Sharpe ratio: Similar ({optimized_metrics.get('sharpe_ratio', 0):.2f} vs {current_metrics.get('sharpe_ratio', 0):.2f})")
    elif sharpe_diff > 0:
        parts.append(f"✓ Sharpe ratio improved by {sharpe_diff:.2f} ({optimized_metrics.get('sharpe_ratio', 0):.2f} vs {current_metrics.get('sharpe_ratio', 0):.2f})")
    else:
        parts.append(f"⚠ Sharpe ratio decreased by {abs(sharpe_diff):.2f} ({optimized_metrics.get('sharpe_ratio', 0):.2f} vs {current_metrics.get('sharpe_ratio', 0):.2f})")
    
    # Return analysis
    if abs(return_diff) < 0.01:
        parts.append(f"Annual return: Similar ({optimized_metrics.get('annualized_return', 0)*100:.2f}% vs {current_metrics.get('annualized_return', 0)*100:.2f}%)")
    elif return_diff > 0:
        parts.append(f"✓ Annual return increased by {return_diff*100:.2f}% ({optimized_metrics.get('annualized_return', 0)*100:.2f}% vs {current_metrics.get('annualized_return', 0)*100:.2f}%)")
    else:
        parts.append(f"Annual return decreased by {abs(return_diff)*100:.2f}% ({optimized_metrics.get('annualized_return', 0)*100:.2f}% vs {current_metrics.get('annualized_return', 0)*100:.2f}%)")
    
    # Volatility analysis
    if abs(vol_diff) < 0.01:
        parts.append(f"Volatility: Similar ({optimized_metrics.get('volatility', 0)*100:.2f}% vs {current_metrics.get('volatility', 0)*100:.2f}%)")
    elif vol_diff < 0:
        parts.append(f"✓ Volatility reduced by {abs(vol_diff)*100:.2f}% ({optimized_metrics.get('volatility', 0)*100:.2f}% vs {current_metrics.get('volatility', 0)*100:.2f}%)")
    else:
        parts.append(f"⚠ Volatility increased by {vol_diff*100:.2f}% ({optimized_metrics.get('volatility', 0)*100:.2f}% vs {current_metrics.get('volatility', 0)*100:.2f}%)")
    
    # Drawdown analysis
    if abs(dd_diff) < 0.01:
        parts.append(f"Max drawdown: Similar ({optimized_metrics.get('max_drawdown', 0)*100:.2f}% vs {current_metrics.get('max_drawdown', 0)*100:.2f}%)")
    elif dd_diff < 0:
        parts.append(f"✓ Max drawdown reduced by {abs(dd_diff)*100:.2f}% ({optimized_metrics.get('max_drawdown', 0)*100:.2f}% vs {current_metrics.get('max_drawdown', 0)*100:.2f}%)")
    else:
        parts.append(f"⚠ Max drawdown increased by {abs(dd_diff)*100:.2f}% ({optimized_metrics.get('max_drawdown', 0)*100:.2f}% vs {current_metrics.get('max_drawdown', 0)*100:.2f}%)")
    
    # Overall assessment
    improvements = sum([
        1 if sharpe_diff > 0.1 else 0,
        1 if return_diff > 0.01 else 0,
        1 if vol_diff < -0.01 else 0,
        1 if dd_diff < -0.01 else 0,
    ])
    
    if improvements >= 3:
        parts.append(f"\n✓ Optimization shows significant improvements across multiple metrics")
    elif improvements >= 2:
        parts.append(f"\nOptimization shows moderate improvements")
    elif improvements >= 1:
        parts.append(f"\nOptimization shows limited improvements")
    else:
        parts.append(f"\n⚠ Optimization did not improve metrics - consider adjusting constraints or method")
    
    return "\n".join(parts)


def _interpret_allocation_changes(current_weights: dict, optimal_weights: dict) -> str:
    """Interpret changes in portfolio allocation."""
    if not current_weights or not optimal_weights:
        return ""
    
    parts = []
    parts.append("**Allocation Changes Analysis:**")
    
    # Calculate differences
    changes = []
    for ticker in optimal_weights.keys():
        current_w = current_weights.get(ticker, 0.0)
        optimal_w = optimal_weights.get(ticker, 0.0)
        diff = optimal_w - current_w
        if abs(diff) > 0.01:  # Only significant changes (>1%)
            changes.append((ticker, current_w, optimal_w, diff))
    
    if not changes:
        parts.append("No significant weight changes required - portfolio is already close to optimal")
        return "\n".join(parts)
    
    # Sort by absolute difference
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Largest increases
    increases = [c for c in changes if c[3] > 0.01]
    if increases:
        top_increase = increases[0]
        parts.append(f"Largest increase: {top_increase[0]} ({top_increase[1]:.1%} → {top_increase[2]:.1%}, +{top_increase[3]:.1%})")
        if len(increases) > 1:
            parts.append(f"{len(increases)} asset(s) need weight increases")
    
    # Largest decreases
    decreases = [c for c in changes if c[3] < -0.01]
    if decreases:
        top_decrease = decreases[0]
        parts.append(f"Largest decrease: {top_decrease[0]} ({top_decrease[1]:.1%} → {top_decrease[2]:.1%}, {top_decrease[3]:.1%})")
        if len(decreases) > 1:
            parts.append(f"{len(decreases)} asset(s) need weight decreases")
    
    # Concentration analysis
    max_optimal_weight = max(optimal_weights.values()) if optimal_weights else 0
    if max_optimal_weight > 0.5:
        parts.append(f"⚠ High concentration: Max weight is {max_optimal_weight:.1%} - Consider diversification")
    elif max_optimal_weight > 0.3:
        parts.append(f"Moderate concentration: Max weight is {max_optimal_weight:.1%}")
    else:
        parts.append(f"✓ Well-diversified: Max weight is {max_optimal_weight:.1%}")
    
    # Total rebalancing needed
    total_change = sum(abs(c[3]) for c in changes)
    if total_change > 0.5:
        parts.append(f"⚠ Large rebalancing required: Total weight changes = {total_change:.1%}")
    elif total_change > 0.2:
        parts.append(f"Moderate rebalancing needed: Total weight changes = {total_change:.1%}")
    else:
        parts.append(f"✓ Small rebalancing: Total weight changes = {total_change:.1%}")
    
    return "\n".join(parts)


def _interpret_trade_list(trades: list) -> str:
    """Interpret trade list."""
    if not trades:
        return ""
    
    parts = []
    parts.append("**Trade List Analysis:**")
    
    # Count trades by type
    buys = [t for t in trades if t.get("action") == "BUY"]
    sells = [t for t in trades if t.get("action") == "SELL"]
    total_value = sum(t.get("value", 0) for t in trades)
    
    parts.append(f"Total trades: {len(trades)} ({len(buys)} buys, {len(sells)} sells)")
    parts.append(f"Total trade value: ${total_value:,.2f}")
    
    # Largest trades
    if trades:
        sorted_trades = sorted(trades, key=lambda x: abs(x.get("value", 0)), reverse=True)
        top_trade = sorted_trades[0]
        parts.append(f"Largest trade: {top_trade.get('ticker')} - {top_trade.get('action')} ${abs(top_trade.get('value', 0)):,.2f}")
    
    # Assessment
    if total_value < 1000:
        parts.append(f"✓ Small rebalancing - Low transaction costs expected")
    elif total_value < 10000:
        parts.append(f"Moderate rebalancing - Consider transaction costs")
    else:
        parts.append(f"⚠ Large rebalancing - Significant transaction costs may apply")
    
    # Weight changes
    if trades:
        avg_weight_change = np.mean([abs(t.get("optimal_weight", 0) - t.get("current_weight", 0)) for t in trades])
        if avg_weight_change < 0.05:
            parts.append(f"Average weight change: {avg_weight_change:.1%} - Minor adjustments")
        elif avg_weight_change < 0.15:
            parts.append(f"Average weight change: {avg_weight_change:.1%} - Moderate adjustments")
        else:
            parts.append(f"Average weight change: {avg_weight_change:.1%} - Major rebalancing")
    
    return "\n".join(parts)


def _interpret_returns_comparison(current_returns: pd.Series, optimized_returns: pd.Series) -> str:
    """Interpret returns comparison chart."""
    if current_returns is None or current_returns.empty or optimized_returns is None or optimized_returns.empty:
        return ""
    
    parts = []
    parts.append("**Returns Comparison Analysis:**")
    
    # Calculate cumulative returns
    current_cum = (1 + current_returns).cumprod() - 1
    optimized_cum = (1 + optimized_returns).cumprod() - 1
    
    if current_cum.empty or optimized_cum.empty:
        return ""
    
    current_final = current_cum.iloc[-1] if len(current_cum) > 0 else 0
    optimized_final = optimized_cum.iloc[-1] if len(optimized_cum) > 0 else 0
    
    diff = optimized_final - current_final
    
    parts.append(f"Current portfolio: {current_final*100:+.2f}% total return")
    parts.append(f"Optimized portfolio: {optimized_final*100:+.2f}% total return")
    
    if abs(diff) < 0.01:
        parts.append(f"Performance is similar ({diff*100:+.2f}% difference)")
    elif diff > 0:
        parts.append(f"✓ Optimized portfolio outperforms by {diff*100:.2f}%")
    else:
        parts.append(f"⚠ Optimized portfolio underperforms by {abs(diff)*100:.2f}%")
    
    # Period analysis (if we have enough data)
    if len(current_returns) > 20 and len(optimized_returns) > 20:
        # Check if optimized consistently outperforms
        aligned_idx = current_returns.index.intersection(optimized_returns.index)
        if len(aligned_idx) > 20:
            current_aligned = current_returns.loc[aligned_idx]
            optimized_aligned = optimized_returns.loc[aligned_idx]
            
            outperformance_days = (optimized_aligned > current_aligned).sum()
            total_days = len(aligned_idx)
            outperformance_pct = outperformance_days / total_days * 100
            
            if outperformance_pct > 60:
                parts.append(f"Optimized portfolio outperforms on {outperformance_pct:.1f}% of days")
            elif outperformance_pct < 40:
                parts.append(f"Optimized portfolio underperforms on {100-outperformance_pct:.1f}% of days")
            else:
                parts.append(f"Mixed performance: Optimized outperforms on {outperformance_pct:.1f}% of days")
    
    return "\n".join(parts)


def _interpret_drawdown_comparison(current_values: pd.Series, optimized_values: pd.Series) -> str:
    """Interpret drawdown comparison chart."""
    if current_values is None or current_values.empty or optimized_values is None or optimized_values.empty:
        return ""
    
    parts = []
    parts.append("**Drawdown Comparison Analysis:**")
    
    # Calculate drawdowns
    current_peak = current_values.expanding().max()
    current_dd = (current_values - current_peak) / current_peak * 100
    
    optimized_peak = optimized_values.expanding().max()
    optimized_dd = (optimized_values - optimized_peak) / optimized_peak * 100
    
    if current_dd.empty or optimized_dd.empty:
        return ""
    
    current_max_dd = current_dd.min()
    optimized_max_dd = optimized_dd.min()
    
    parts.append(f"Current portfolio max drawdown: {current_max_dd:.2f}%")
    parts.append(f"Optimized portfolio max drawdown: {optimized_max_dd:.2f}%")
    
    dd_diff = optimized_max_dd - current_max_dd
    
    if abs(dd_diff) < 1:
        parts.append(f"Drawdown levels are similar ({dd_diff:+.2f}% difference)")
    elif dd_diff < 0:
        parts.append(f"✓ Optimized portfolio has lower max drawdown by {abs(dd_diff):.2f}% - Better risk control")
    else:
        parts.append(f"⚠ Optimized portfolio has higher max drawdown by {dd_diff:.2f}% - Higher risk")
    
    # Recovery analysis
    if len(current_dd) > 0 and len(optimized_dd) > 0:
        # Count days in drawdown
        current_dd_days = (current_dd < -1).sum()  # More than 1% drawdown
        optimized_dd_days = (optimized_dd < -1).sum()
        
        if current_dd_days > 0 or optimized_dd_days > 0:
            if optimized_dd_days < current_dd_days:
                parts.append(f"Optimized portfolio spent fewer days in drawdown ({optimized_dd_days} vs {current_dd_days} days)")
            elif optimized_dd_days > current_dd_days:
                parts.append(f"Optimized portfolio spent more days in drawdown ({optimized_dd_days} vs {current_dd_days} days)")
            else:
                parts.append(f"Similar time in drawdown ({optimized_dd_days} days)")
    
    return "\n".join(parts)


def _interpret_efficient_frontier(frontier_data: dict, result, current_metrics: dict) -> str:
    """Interpret efficient frontier chart."""
    if not frontier_data or not result:
        return ""
    
    parts = []
    parts.append("**Efficient Frontier Analysis:**")
    
    # Get portfolio positions
    opt_vol = result.volatility
    opt_ret = result.expected_return
    opt_sharpe = result.sharpe_ratio
    
    tangency = frontier_data.get("tangency_portfolio")
    min_var = frontier_data.get("min_variance_portfolio")
    
    if opt_vol is None or opt_ret is None:
        return ""
    
    # Compare with tangency (max Sharpe)
    if tangency:
        tang_vol = tangency.get("volatility")
        tang_ret = tangency.get("expected_return")
        
        if tang_vol and tang_ret:
            # Distance from tangency
            vol_dist = abs(opt_vol - tang_vol) / tang_vol if tang_vol > 0 else 0
            ret_dist = abs(opt_ret - tang_ret) / abs(tang_ret) if tang_ret != 0 else 0
            
            if vol_dist < 0.05 and ret_dist < 0.05:
                parts.append(f"✓ Optimized portfolio is very close to Max Sharpe point (optimal)")
            elif vol_dist < 0.1 and ret_dist < 0.1:
                parts.append(f"Optimized portfolio is close to Max Sharpe point")
            else:
                parts.append(f"Optimized portfolio differs from Max Sharpe point - Consider adjusting constraints")
    
    # Compare with current portfolio
    if current_metrics:
        curr_vol = current_metrics.get("volatility", 0)
        curr_ret = current_metrics.get("annualized_return", 0)
        
        if curr_vol > 0:
            # Check if current is on frontier
            parts.append(f"Current portfolio: {curr_ret*100:.2f}% return, {curr_vol*100:.2f}% volatility")
            
            # Check if optimized is better positioned
            if opt_sharpe and opt_sharpe > 0:
                curr_sharpe = current_metrics.get("sharpe_ratio", 0)
                if opt_sharpe > curr_sharpe + 0.1:
                    parts.append(f"✓ Optimized portfolio has significantly better risk-adjusted return (Sharpe: {opt_sharpe:.2f} vs {curr_sharpe:.2f})")
                elif opt_sharpe > curr_sharpe:
                    parts.append(f"Optimized portfolio has better risk-adjusted return (Sharpe: {opt_sharpe:.2f} vs {curr_sharpe:.2f})")
    
    # Min variance comparison
    if min_var:
        min_var_vol = min_var.get("volatility")
        if min_var_vol and opt_vol:
            if opt_vol < min_var_vol * 1.1:
                parts.append(f"Optimized portfolio is close to minimum volatility ({opt_vol*100:.2f}% vs {min_var_vol*100:.2f}%)")
    
    return "\n".join(parts)


def _interpret_correlation_diversification(corr_matrix: pd.DataFrame, optimal_weights: dict, num_assets: int) -> str:
    """Interpret correlation and diversification analysis."""
    if corr_matrix is None or corr_matrix.empty or not optimal_weights:
        return ""
    
    parts = []
    parts.append("**Diversification Assessment:**")
    
    # Calculate average correlation
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    values = corr_matrix.values[mask]
    values = values[~np.isnan(values)]
    
    if len(values) > 0:
        avg_corr = np.mean(values)
        
        if avg_corr < 0.3:
            parts.append(f"✓ Low average correlation ({avg_corr:.2f}) - Excellent diversification")
        elif avg_corr < 0.5:
            parts.append(f"Moderate average correlation ({avg_corr:.2f}) - Good diversification")
        else:
            parts.append(f"⚠ High average correlation ({avg_corr:.2f}) - Limited diversification")
    
    # Weight concentration
    max_weight = max(optimal_weights.values()) if optimal_weights else 0
    if max_weight > 0.5:
        parts.append(f"⚠ High concentration: Max weight is {max_weight:.1%} - Portfolio is concentrated")
    elif max_weight > 0.3:
        parts.append(f"Moderate concentration: Max weight is {max_weight:.1%}")
    else:
        parts.append(f"✓ Well-diversified weights: Max weight is {max_weight:.1%}")
    
    # Number of assets
    if num_assets >= 10:
        parts.append(f"✓ Sufficient number of assets ({num_assets}) for diversification")
    elif num_assets >= 5:
        parts.append(f"Moderate number of assets ({num_assets}) - Consider adding more for better diversification")
    else:
        parts.append(f"⚠ Low number of assets ({num_assets}) - Limited diversification potential")
    
    return "\n".join(parts)


def _interpret_sensitivity_analysis(sensitivity_results: dict) -> str:
    """Interpret sensitivity analysis results."""
    if not sensitivity_results or not sensitivity_results.get("results"):
        return ""
    
    parts = []
    results = sensitivity_results.get("results", [])
    
    if not results:
        return ""
    
    # Convert to DataFrame for analysis
    sens_df = pd.DataFrame(results)
    variation_col = "variation"
    ticker_cols = [col for col in sens_df.columns if col != variation_col]
    
    if sens_df.empty or not ticker_cols:
        return ""
    
    parts.append("**Sensitivity Analysis:**")
    
    # Calculate sensitivity for each asset
    sensitivities = []
    for ticker in ticker_cols:
        weights = sens_df[ticker].values
        weight_range = weights.max() - weights.min()
        weight_std = weights.std()
        sensitivities.append({
            "ticker": ticker,
            "range": weight_range,
            "std": weight_std,
        })
    
    # Sort by sensitivity
    sensitivities.sort(key=lambda x: x["range"], reverse=True)
    
    # Most sensitive
    if sensitivities:
        most_sensitive = sensitivities[0]
        parts.append(f"Most sensitive asset: {most_sensitive['ticker']} (weight range: {most_sensitive['range']:.1%})")
        
        if most_sensitive["range"] > 0.2:
            parts.append(f"⚠ High sensitivity - Weight changes significantly with parameter variations")
        elif most_sensitive["range"] > 0.1:
            parts.append(f"Moderate sensitivity - Weight changes moderately")
        else:
            parts.append(f"✓ Low sensitivity - Weight is relatively stable")
    
    # Overall stability
    avg_range = np.mean([s["range"] for s in sensitivities])
    if avg_range < 0.05:
        parts.append(f"✓ Portfolio weights are stable (avg range: {avg_range:.1%}) - Optimization is robust")
    elif avg_range < 0.15:
        parts.append(f"Moderate stability (avg range: {avg_range:.1%}) - Some sensitivity to parameters")
    else:
        parts.append(f"⚠ High sensitivity (avg range: {avg_range:.1%}) - Weights change significantly with parameters")
    
    return "\n".join(parts)


def render_optimization_page() -> None:
    """Render portfolio optimization page."""
    st.title("Portfolio Optimization")
    st.markdown(
        "Optimize your portfolio weights using various optimization methods."
    )
    
    # Initialize services
    portfolio_service = PortfolioService()
    optimization_service = OptimizationService()
    
    # Get portfolios
    portfolios = portfolio_service.list_portfolios()
    
    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return
    
    # Portfolio selection
    portfolio_names = [p.name for p in portfolios]
    selected_name = st.selectbox(
        "Select Portfolio",
        portfolio_names,
        key="optimization_portfolio",
    )
    
    selected_portfolio = next(
        p for p in portfolios if p.name == selected_name
    )
    
    # Display current portfolio info
    with st.expander("Current Portfolio Information", expanded=False):
        positions = selected_portfolio.get_all_positions()
        
        if not positions:
            st.error("Portfolio has no positions.")
            return
        
        # Get current weights
        from services.data_service import DataService
        data_service = DataService()
        
        # Fetch current prices for all positions
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        prices = {}
        for ticker in tickers:
            current_price = data_service.fetch_current_price(ticker)
            if current_price:
                prices[ticker] = current_price
        
        # Add CASH price (always 1.0) if CASH position exists
        if any(pos.ticker == "CASH" for pos in positions):
            prices["CASH"] = 1.0
        
        # Calculate portfolio value
        current_value = selected_portfolio.calculate_current_value(prices)
        
        current_weights_data = []
        for pos in positions:
            if pos.ticker == "CASH":
                # Handle CASH position
                current_price = 1.0  # CASH is always 1.0
                pos_value = pos.shares  # CASH shares = dollar amount
            else:
                current_price = prices.get(pos.ticker)
                if not current_price:
                    continue
                pos_value = current_price * pos.shares
            
            weight = (pos_value / current_value) if current_value > 0 else 0.0
            current_weights_data.append({
                "Ticker": pos.ticker,
                "Shares": f"{pos.shares:,.2f}",
                "Price": f"${current_price:.2f}",
                "Value": f"${pos_value:,.2f}",
                "Weight": f"{weight:.2%}",
            })
        
        if current_weights_data:
            st.dataframe(
                pd.DataFrame(current_weights_data),
                use_container_width=True,
            )
    
    # Optimization parameters
    st.header("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        
        # Get saved dates if available, otherwise use defaults
        saved_start = st.session_state.get("optimization_start_date")
        saved_end = st.session_state.get("optimization_end_date")
        
        # Use saved dates as default if optimization was run,
        # otherwise use defaults
        start_date_value = saved_start if saved_start else default_start
        end_date_value = saved_end if saved_end else default_end
        
        start_date = st.date_input(
            "Start Date",
            value=start_date_value,
            max_value=date.today(),
            key="opt_start_date",
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=end_date_value,
            min_value=start_date,
            max_value=date.today(),
            key="opt_end_date",
        )
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    
    # Out-of-Sample Testing section (before benchmark)
    st.subheader("Out-of-Sample Testing")
    
    use_out_of_sample = st.checkbox(
        "Use Out-of-Sample Testing",
        value=False,
        key="opt_out_of_sample",
        help=(
            "If enabled, optimization is performed on the period BEFORE the specified dates, "
            "and results are validated on the specified period. This helps avoid "
            "overfitting and improves model reliability."
        ),
    )
    
    training_ratio = 0.3  # Default
    training_start = None
    
    if use_out_of_sample:
        training_window_options = {
            "30% (Recommended)": {
                "ratio": 0.3,
                "description": (
                    "Balance between data freshness and statistical reliability. "
                    "Suitable for most cases."
                ),
            },
            "50%": {
                "ratio": 0.5,
                "description": (
                    "More data for training. Suitable for stable markets "
                    "and long-term strategies."
                ),
            },
            "60%": {
                "ratio": 0.6,
                "description": (
                    "Maximum statistical reliability. Use for "
                    "stable assets and conservative portfolios."
                ),
            },
        }
        
        selected_window = st.selectbox(
            "Training Window Size",
            options=list(training_window_options.keys()),
            index=0,  # 30% by default
            key="opt_training_window",
        )
        
        training_ratio = training_window_options[selected_window]["ratio"]
        st.info(training_window_options[selected_window]["description"])
        
        # Calculate periods
        analysis_days = (end_date - start_date).days
        training_days = int(analysis_days * training_ratio)
        training_start = start_date - timedelta(days=training_days)
        
        # Show period information
        st.markdown("**Periods:**")
        st.write(
            f"📚 **Training**: "
            f"{training_start.strftime('%Y-%m-%d')} → "
            f"{start_date.strftime('%Y-%m-%d')} "
            f"({training_days} days)"
        )
        st.write(
            f"**Validation**: "
            f"{start_date.strftime('%Y-%m-%d')} → "
            f"{end_date.strftime('%Y-%m-%d')} "
            f"({analysis_days} days)"
        )
    else:
        st.caption(
            "Optimization will be performed on the specified period "
            f"({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')})"
        )
    
    st.divider()
    
    # Benchmark selection (always visible)
    st.markdown("**Benchmark (Optional, for visualization)**")
    presets = ["None", "SPY", "QQQ", "VTI", "DIA", "IWM"]
    benchmark_choice = st.selectbox(
        "Benchmark",
        options=presets,
        index=0,
        key="opt_benchmark_optional",
        help="Select benchmark for comparison on charts",
    )
    benchmark_ticker = None
    benchmark_for_viz = None
    if benchmark_choice != "None":
        benchmark_for_viz = benchmark_choice
    
    # Method selection - only show available optimization methods
    all_methods = optimization_service.get_available_methods()
    # Filter to only show methods from AVAILABLE_OPTIMIZATION_METHODS
    available_methods = [
        m for m in all_methods if m in AVAILABLE_OPTIMIZATION_METHODS
    ]
    
    if not available_methods:
        st.error("No optimization methods available.")
        return
    
    method_display = [
        METHOD_NAMES.get(m, m.replace("_", " ").title())
        for m in available_methods
    ]
    
    # Add "Not selected" option at the beginning
    method_options = ["-- Select Method --"] + available_methods
    method_display_options = ["-- Select Method --"] + method_display
    
    selected_method_choice = st.selectbox(
        "Optimization Method",
        range(len(method_options)),
        format_func=lambda x: method_display_options[x],
        key="opt_method",
    )
    
    selected_method = None
    if selected_method_choice > 0:  # 0 is "-- Select Method --"
        selected_method = method_options[selected_method_choice]
    
    # Method description - only show if method selected
    if selected_method:
        method_descriptions = {
            "mean_variance": {
                "short": "Markowitz mean-variance optimization",
                "detailed": """
**Mean-Variance Optimization (Markowitz Model)**

This is the foundational portfolio optimization method developed by Harry Markowitz in 1952, for which he won the Nobel Prize in Economics.

**How it works:**
- Maximizes expected return for a given level of risk (volatility)
- Uses historical returns to estimate expected returns and covariance matrix
- Finds the optimal portfolio on the efficient frontier
- Assumes returns are normally distributed

**Key Features:**
- Classic approach to portfolio theory
- Balances risk and return trade-off
- Works well when historical data is representative of future returns
- Sensitive to input parameters (expected returns, covariance)

**Best for:**
- Long-term strategic asset allocation
- When you have reliable return estimates
- Traditional portfolio construction

**Limitations:**
- Assumes normal distribution (may underestimate tail risk)
- Sensitive to estimation errors in expected returns
- May produce extreme weights if not constrained
"""
            },
            "black_litterman": {
                "short": "Black-Litterman model - combines market equilibrium with investor views",
                "detailed": """
**Black-Litterman Model**

An advanced optimization method that combines market equilibrium returns with your own views about asset performance.

**How it works:**
- Starts with market equilibrium (CAPM implied returns)
- Incorporates your views about specific assets or factors
- Uses Bayesian updating to blend market data with your opinions
- Produces more stable and intuitive portfolio weights

**Key Features:**
- Reduces extreme weights compared to pure Markowitz
- Incorporates investor expertise and market views
- More stable results (less sensitive to input errors)
- Handles uncertainty in views through confidence levels

**Best for:**
- When you have strong views about certain assets
- Combining quantitative analysis with qualitative insights
- Reducing estimation error sensitivity

**Limitations:**
- Requires defining views and confidence levels
- More complex to implement and understand
- Still assumes normal distribution
"""
            },
            "risk_parity": {
                "short": "Equal risk contribution from each asset",
                "detailed": """
**Risk Parity Optimization**

Allocates portfolio weights so that each asset contributes equally to portfolio risk, rather than equal dollar amounts.

**How it works:**
- Calculates risk contribution of each asset to total portfolio risk
- Adjusts weights until all assets have equal risk contribution
- Accounts for correlations between assets
- Typically results in more balanced allocations

**Key Features:**
- Equalizes risk, not capital allocation
- More stable than mean-variance (less sensitive to return estimates)
- Naturally diversifies across assets
- Often produces better risk-adjusted returns

**Best for:**
- Risk-focused portfolio construction
- When return estimates are uncertain
- Achieving balanced risk exposure
- Multi-asset portfolios

**Limitations:**
- May underweight high-return assets
- Doesn't explicitly maximize returns
- Can be conservative for return-seeking investors
"""
            },
            "hrp": {
                "short": "Hierarchical Risk Parity using clustering - more stable, less sensitive to estimation error",
                "detailed": """
**Hierarchical Risk Parity (HRP)**

A modern optimization method that uses machine learning clustering to build portfolios, making it more robust to estimation errors.

**How it works:**
- Uses hierarchical clustering to group similar assets
- Allocates weights based on correlation structure
- Reduces sensitivity to covariance matrix estimation errors
- More stable than traditional methods

**Key Features:**
- Very robust to input errors
- Uses clustering to understand asset relationships
- Produces well-diversified portfolios
- Less prone to extreme weights

**Best for:**
- When historical data may not be reliable
- Large portfolios with many assets
- Reducing estimation error impact
- Modern portfolio construction

**Limitations:**
- More computationally intensive
- Less intuitive than traditional methods
- May not maximize returns explicitly
"""
            },
            "cvar_optimization": {
                "short": "Minimize Conditional Value at Risk (tail risk) - focuses on extreme losses",
                "detailed": """
**CVaR Optimization (Conditional Value at Risk)**

Focuses on minimizing tail risk rather than overall volatility, protecting against extreme losses.

**How it works:**
- Minimizes Conditional Value at Risk (expected loss beyond VaR threshold)
- Focuses on worst-case scenarios (tail of distribution)
- Accounts for non-normal return distributions
- Protects against extreme market events

**Key Features:**
- Better handles fat tails and extreme events
- More relevant for risk management than variance
- Protects against worst-case losses
- Suitable for non-normal return distributions

**Best for:**
- Risk-averse investors
- Managing tail risk exposure
- Portfolios with non-normal returns
- Stress testing and risk management

**Limitations:**
- May sacrifice return for tail risk protection
- Requires defining confidence level (e.g., 95%, 99%)
- Can be more conservative than variance-based methods
"""
            },
            "mean_cvar": {
                "short": "Maximize Return / CVaR ratio - optimal trade-off between return and tail risk",
                "detailed": """
**Mean-CVaR Optimization**

Maximizes the ratio of expected return to Conditional Value at Risk, optimizing the risk-adjusted return considering tail risk.

**How it works:**
- Maximizes Return / CVaR ratio (similar to Sharpe ratio but using CVaR)
- Balances expected return with tail risk
- Accounts for extreme loss scenarios
- Finds optimal trade-off between return and downside risk

**Key Features:**
- Considers both return and tail risk
- Better than Sharpe ratio for non-normal distributions
- Protects against extreme losses while seeking returns
- More relevant risk measure than volatility

**Best for:**
- Balancing return and tail risk
- Non-normal return distributions
- Risk-adjusted return optimization
- Modern portfolio theory applications

**Limitations:**
- More complex than traditional Sharpe optimization
- Requires confidence level parameter
- May still be sensitive to return estimates
"""
            },
            "robust": {
                "short": "Robust optimization with uncertainty sets - accounts for parameter uncertainty",
                "detailed": """
**Robust Optimization**

Handles uncertainty in input parameters by optimizing over a range of possible scenarios rather than point estimates.

**How it works:**
- Defines uncertainty sets for expected returns and covariance
- Finds portfolio that performs well across all scenarios in the set
- Protects against parameter estimation errors
- More conservative but more reliable

**Key Features:**
- Handles parameter uncertainty explicitly
- More robust to estimation errors
- Performs well across multiple scenarios
- Reduces sensitivity to input assumptions

**Best for:**
- When parameter estimates are uncertain
- Robust portfolio construction
- Reducing estimation error impact
- Conservative optimization approach

**Limitations:**
- May be overly conservative
- Requires defining uncertainty sets
- More computationally complex
- May sacrifice return for robustness
"""
            },
            "max_diversification": {
                "short": "Maximize diversification ratio - maximum benefit from diversification",
                "detailed": """
**Maximum Diversification Optimization**

Maximizes the diversification ratio, which measures how much diversification reduces portfolio risk compared to weighted average asset risk.

**How it works:**
- Calculates diversification ratio = Portfolio Risk / Weighted Average Asset Risk
- Maximizes this ratio to get maximum diversification benefit
- Allocates more to less correlated assets
- Reduces portfolio risk through diversification

**Key Features:**
- Explicitly maximizes diversification benefits
- Reduces portfolio risk through asset selection
- Works well with many assets
- Less sensitive to return estimates

**Best for:**
- Maximizing diversification benefits
- Multi-asset portfolios
- When correlations are important
- Risk reduction focus

**Limitations:**
- Doesn't explicitly consider returns
- May underweight high-return assets
- Focuses on risk reduction, not return maximization
"""
            },
            "min_correlation": {
                "short": "Minimize average pairwise correlation - assets that move independently",
                "detailed": """
**Minimum Correlation Optimization**

Selects and weights assets to minimize the average pairwise correlation in the portfolio.

**How it works:**
- Calculates average correlation between all asset pairs
- Adjusts weights to minimize this average correlation
- Selects assets that move independently
- Reduces portfolio risk through low correlation

**Key Features:**
- Simple and intuitive approach
- Focuses on correlation structure
- Reduces portfolio volatility
- Works well with diverse asset classes

**Best for:**
- Multi-asset class portfolios
- When correlations are key concern
- Simple diversification strategy
- Reducing portfolio volatility

**Limitations:**
- Doesn't consider returns explicitly
- May ignore return potential
- Simple approach may miss optimization opportunities
"""
            },
            "inverse_correlation": {
                "short": "Inverse correlation weighting - analytical method based on correlation structure",
                "detailed": """
**Inverse Correlation Weighting**

An analytical method that weights assets inversely proportional to their correlations with the portfolio.

**How it works:**
- Uses inverse of correlation matrix to determine weights
- Assets with lower correlation to portfolio get higher weights
- Analytical solution (no optimization needed)
- Fast and computationally efficient

**Key Features:**
- Fast analytical solution
- Based on correlation structure
- Computationally efficient
- Easy to implement

**Best for:**
- Quick portfolio construction
- When speed is important
- Correlation-based allocation
- Simple analytical approach

**Limitations:**
- Doesn't consider returns
- May not be optimal for return maximization
- Simple approach may miss opportunities
"""
            },
        }
        
        method_info = method_descriptions.get(selected_method, {})
        if method_info:
            short_desc = method_info.get("short", "")
            detailed_desc = method_info.get("detailed", "")
            
            with st.expander(f"ℹ️ {short_desc}", expanded=False):
                st.markdown(detailed_desc)
    
    # Initialize variables (needed outside conditional block)
    constraints = {}
    method_params = {}
    selected_objective = None
    
    # Conditional sections: only show after method selection
    if selected_method:
        st.divider()
        
        # Get number of assets for validation
        positions = selected_portfolio.get_all_positions()
        num_assets = len([p for p in positions if p.ticker != "CASH"])
        
        # Constraints section
        st.subheader("Constraints")
        
        constraints = {}
        
        # Row 1: Long Only (left) + Max Cash Weight (right)
        col1, col2 = st.columns(2)
        with col1:
            long_only = st.checkbox(
                "Long Only (No Short Positions)",
                value=True,
                key="opt_long_only",
                help="If enabled, all weights must be >= 0 (no short positions). "
                     "Default: min_weight=0, max_weight=1",
            )
            constraints["long_only"] = long_only
        
        with col2:
            max_cash = st.slider(
                "Max Cash Weight %",
                min_value=0.0,
                max_value=100.0,
                value=10.0,  # Default 10%
                step=1.0,
                key="opt_max_cash",
                help="Maximum cash allocation to prevent 100% cash optimization. Default: 10%",
            ) / 100.0
            constraints["max_cash_weight"] = max_cash
        
        # Row 2: Min Weight (left) + Max Weight (right)
        col1, col2 = st.columns(2)
        with col1:
            use_min_weight = st.checkbox("Minimum Weight", key="opt_min_weight")
            if use_min_weight:
                # Calculate max allowed value based on number of assets
                max_allowed_pct = min(50.0, (1.0 / num_assets) * 100) if num_assets > 0 else 50.0
                min_weight = st.slider(
                    "Min Weight %",
                    min_value=0.0,
                    max_value=max_allowed_pct,
                    value=0.0,
                    step=0.5,
                    key="opt_min_weight_val",
                    help=f"Maximum allowed value: {max_allowed_pct:.1f}% (for {num_assets} assets)",
                ) / 100.0
                constraints["min_weight"] = min_weight
                
                # Validate
                warnings = validate_weight_constraints(
                    min_weight=min_weight,
                    num_assets=num_assets,
                )
                for warning in warnings:
                    st.warning(warning)
        
        with col2:
            use_max_weight = st.checkbox("Maximum Weight", key="opt_max_weight")
            if use_max_weight:
                max_weight = st.slider(
                    "Max Weight %",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=1.0,
                    key="opt_max_weight_val",
                ) / 100.0
                constraints["max_weight"] = max_weight
                
                # Validate if min_weight is also set
                if use_min_weight and "min_weight" in constraints:
                    warnings = validate_weight_constraints(
                        min_weight=constraints.get("min_weight"),
                        max_weight=max_weight,
                        num_assets=num_assets,
                    )
                    for warning in warnings:
                        st.warning(warning)
        
        # Set default values if not specified
        if not use_min_weight:
            # Default: min_weight = 0 for long_only, -1 for short allowed
            if long_only:
                constraints["min_weight"] = 0.0
            else:
                constraints["min_weight"] = -1.0
        
        if not use_max_weight:
            # Default: max_weight = 1.0
            constraints["max_weight"] = 1.0
        
        # Row 3: Min Expected Return (left) + Diversification (right)
        col1, col2 = st.columns(2)
        with col1:
            use_min_return = st.checkbox(
                "Minimum Expected Return",
                value=False,
                key="opt_min_return",
                help="Require minimum expected return to prevent 100% cash allocation",
            )
            
            if use_min_return:
                min_return = st.slider(
                    "Min Return % (annualized)",
                    min_value=0.0,
                    max_value=20.0,
                    value=3.0,
                    step=0.5,
                    key="opt_min_return_val",
                    help="Minimum expected return constraint (annualized)",
                ) / 100.0
                constraints["min_return"] = min_return
        
        with col2:
            use_div_reg = st.checkbox(
                "Enable Diversification Penalty",
                value=False,
                key="opt_div_reg",
                help=(
                    "Penalize concentration to encourage diversification. "
                    "Higher values = more diversification, lower = allow concentration"
                ),
            )
            
            if use_div_reg:
                lambda_div = st.slider(
                    "Diversification Strength",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="opt_lambda_div",
                    help=(
                        "Regularization strength (0.1-10.0). "
                        "Higher = more diversification, lower = allow concentration. "
                        "This adds a penalty for concentration: penalty = lambda × sum(weights²)"
                    ),
                )
                constraints["diversification_lambda"] = lambda_div
        
        st.divider()
        
        # Objective Function section
        st.subheader("Objective Function")
        
        available_objectives = get_available_objectives(selected_method)
        default_objective = get_default_objective(selected_method)
        
        if not available_objectives:
            # Method has fixed objective
            st.info(
                f"This method ({METHOD_NAMES.get(selected_method, selected_method)}) "
                "has a fixed optimization objective and does not support objective selection."
            )
            selected_objective = None
        else:
            # Create display list
            objective_display = [
                OBJECTIVE_METHOD_MAPPING[obj]["display"] for obj in available_objectives
            ]
            
            # Find default index
            default_idx = 0
            if default_objective and default_objective in available_objectives:
                default_idx = available_objectives.index(default_objective)
            
            selected_obj_idx = st.selectbox(
                "Select optimization objective",
                range(len(available_objectives)),
                index=default_idx,
                format_func=lambda x: objective_display[x],
                key="opt_objective",
                help="Determines what the selected method optimizes",
            )
            selected_objective = available_objectives[selected_obj_idx]
            
            # Show default objective info if using default
            if selected_objective == default_objective:
                st.caption(
                    f"Using default objective for method "
                    f"{METHOD_NAMES.get(selected_method, selected_method)}"
                )
        
        st.divider()
        
        # Method-specific parameters
        if selected_objective:
            method_params["objective"] = selected_objective
        
        if selected_method == "kelly_criterion":
            kelly_fraction = st.slider(
                "Kelly Fraction",
                min_value=0.25,
                max_value=1.0,
                value=0.5,
                step=0.25,
                key="opt_kelly_fraction",
                help="1.0 = Full Kelly, 0.5 = Half Kelly, 0.25 = Quarter Kelly",
            )
            constraints["kelly_fraction"] = kelly_fraction
        
        if selected_method in ["cvar_optimization", "mean_cvar"]:
            confidence_level = st.selectbox(
                "Confidence Level",
                options=[0.90, 0.95, 0.99],
                index=1,  # Default 0.95
                format_func=lambda x: f"{int(x * 100)}%",
                key="opt_confidence_level",
                help="Confidence level for CVaR calculation",
            )
            method_params["confidence_level"] = confidence_level
        
        if selected_method == "robust":
            uncertainty_returns = st.slider(
                "Uncertainty Radius (Returns)",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
                format="%.2f",
                key="opt_uncertainty_returns",
                help="Uncertainty radius for expected returns (as decimal, e.g., 0.10 = 10%)",
            )
            uncertainty_cov = st.slider(
                "Uncertainty Radius (Covariance)",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
                format="%.2f",
                key="opt_uncertainty_cov",
                help="Uncertainty radius for covariance matrix (as decimal, e.g., 0.10 = 10%)",
            )
            method_params["uncertainty_radius_returns"] = uncertainty_returns
            method_params["uncertainty_radius_cov"] = uncertainty_cov
        
        if selected_method == "black_litterman":
            st.markdown("**Black-Litterman Parameters**")
            st.info(
                "Black-Litterman requires market weights and investor views. "
                "Views can be added after optimization runs."
            )
            # Note: Full Black-Litterman UI would require views input,
            # which is complex. For now, use default market weights.
            # Views can be added in future enhancement.
    
    # Run optimization - only show if method selected
    if selected_method:
        st.header("Run Optimization")
        
        if st.button("Optimize Portfolio", type="primary", use_container_width=True):
            with st.spinner("Running optimization... This may take a moment."):
                try:
                    # Prepare method-specific parameters
                    optimize_kwargs = {
                        "portfolio_id": selected_portfolio.id,
                        "method": selected_method,
                        "start_date": start_date,
                        "end_date": end_date,
                        "constraints": constraints if constraints else None,
                        "benchmark_ticker": benchmark_ticker,
                    }
                    
                    # Add method-specific parameters
                    if method_params:
                        optimize_kwargs["method_params"] = method_params
                    
                    # Add out-of-sample parameters if enabled
                    if use_out_of_sample:
                        optimize_kwargs["out_of_sample"] = True
                        optimize_kwargs["training_ratio"] = training_ratio
                    
                    result = optimization_service.optimize_portfolio(**optimize_kwargs)
                    
                    # Store result in session state
                    st.session_state["optimization_result"] = result
                    st.session_state["optimization_portfolio_id"] = (
                        selected_portfolio.id
                    )
                    st.session_state["optimization_start_date"] = start_date
                    st.session_state["optimization_end_date"] = end_date
                    st.session_state["optimization_benchmark"] = benchmark_for_viz
                    st.session_state["optimization_method_used"] = selected_method
                    st.session_state["optimization_out_of_sample"] = use_out_of_sample
                    st.session_state["optimization_training_ratio"] = training_ratio
                    # Store the actual optimization period used
                    if use_out_of_sample:
                        analysis_days = (end_date - start_date).days
                        training_days = int(analysis_days * training_ratio)
                        optimization_start = start_date - timedelta(days=training_days)
                        optimization_end = start_date
                        st.session_state["optimization_period_start"] = optimization_start
                        st.session_state["optimization_period_end"] = optimization_end
                    else:
                        st.session_state["optimization_period_start"] = start_date
                        st.session_state["optimization_period_end"] = end_date
                    
                    st.success("Optimization completed successfully!")
                    st.rerun()
                
                except InsufficientDataError as e:
                    st.error(f"Insufficient data: {str(e)}")
                except CalculationError as e:
                    st.error(f"Calculation error: {str(e)}")
                except Exception as e:
                    logger.exception("Optimization failed")
                    st.error(f"Optimization failed: {str(e)}")
    
    # Display results if available
    if "optimization_result" in st.session_state:
        result = st.session_state["optimization_result"]
        portfolio_id = st.session_state.get("optimization_portfolio_id")
        # CRITICAL: Use CURRENT dates from date_input, not saved ones!
        # This ensures metrics are calculated for the period user selected
        saved_benchmark = st.session_state.get("optimization_benchmark")
        
        if result.success:
            # Get optimization period (training period if out-of-sample was used)
            opt_period_start = st.session_state.get(
                "optimization_period_start", start_date
            )
            opt_period_end = st.session_state.get(
                "optimization_period_end", end_date
            )
            
            _display_optimization_results(
                result,
                portfolio_id,
                portfolio_service,
                optimization_service,
                start_date,  # Use CURRENT start_date from date_input (for validation display)
                end_date,    # Use CURRENT end_date from date_input (for validation display)
                constraints,
                saved_benchmark,
                optimization_period_start=opt_period_start,  # Actual optimization period
                optimization_period_end=opt_period_end,      # Actual optimization period
            )
        else:
            st.error(f"Optimization failed: {result.message}")


def _display_optimization_results(
    result,
    portfolio_id: str,
    portfolio_service: PortfolioService,
    optimization_service: OptimizationService,
    start_date: date,
    end_date: date,
    constraints: Dict,
    benchmark_for_viz: str = None,
    optimization_period_start: date = None,
    optimization_period_end: date = None,
) -> None:
    """Display optimization results."""
    
    # Check if min_return constraint is violated
    min_return = constraints.get("min_return")
    if min_return is not None and result.expected_return is not None:
        if result.expected_return < min_return:
            st.warning(
                f"**Warning**: Expected return ({result.expected_return:.2%}) is below "
                f"minimum required return ({min_return:.2%}). "
                f"The optimizer could not find a solution that satisfies all constraints. "
                f"Try relaxing constraints (e.g., reduce min_return or increase max_cash_weight)."
            )
        else:
            # Show info about expected vs actual return
            st.info(
                f"**Note**: The 'Minimum Expected Return' constraint ({min_return:.2%}) applies to "
                f"**expected return** (based on historical average returns), not actual return. "
                f"Expected return: {result.expected_return:.2%}. "
                f"Actual return may differ due to market volatility."
            )
    st.header("Optimization Results")
    
    # Show date range info
    st.caption(
        f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to "
        f"{end_date.strftime('%Y-%m-%d')} "
        f"({(end_date - start_date).days} days)"
    )
    
    # Check if dates differ from optimization dates (if stored)
    saved_opt_start = st.session_state.get("optimization_start_date")
    saved_opt_end = st.session_state.get("optimization_end_date")
    if saved_opt_start and saved_opt_end:
        if saved_opt_start != start_date or saved_opt_end != end_date:
            st.info(
                f"**Note:** Metrics are calculated for the selected period "
                f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}), "
                f"which may differ from the optimization period "
                f"({saved_opt_start.strftime('%Y-%m-%d')} to {saved_opt_end.strftime('%Y-%m-%d')})."
            )
    
    # Comparison: Optimized vs Current Portfolio
    # Use EXACT same logic as portfolio_analysis.py overview tab
    st.subheader("Comparison: Optimized vs Current Portfolio")
    
    risk_free_rate = 0.0435
    
    # Get current portfolio returns for the SELECTED period
    current_returns = None
    try:
        analytics_service = AnalyticsService()
        current_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            start_date,
            end_date,
        )
        current_returns = current_analytics.get("portfolio_returns")
    except Exception as e:
        logger.warning(
            f"Could not calculate current portfolio returns: {e}"
        )
    
    # Calculate optimized portfolio returns for the SELECTED period
    optimized_returns = None
    try:
        # Get optimized portfolio historical returns (SAME logic as charts)
        from services.data_service import DataService
        data_service = DataService()
        
        optimal_weights_dict = result.get_weights_dict()
        tickers_list = result.tickers
        
        # Fetch historical prices and calculate returns
        asset_returns_df = pd.DataFrame()
        for ticker in tickers_list:
            if ticker == "CASH":
                dr = pd.bdate_range(
                    start=start_date, end=end_date, normalize=True
                )
                daily_return = (1 + risk_free_rate) ** (1.0 / 252) - 1
                cash_returns = pd.Series(
                    data=daily_return,
                    index=dr,
                    name=ticker,
                )
                asset_returns_df[ticker] = cash_returns
            else:
                try:
                    prices = data_service.fetch_historical_prices(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        save_to_db=False,
                    )
                    # Ensure prices is a DataFrame before checking .empty
                    if not isinstance(prices, pd.DataFrame):
                        logger.warning(
                            f"fetch_historical_prices returned non-DataFrame "
                            f"for {ticker}: {type(prices)}"
                        )
                        continue
                    if not prices.empty:
                        if "Date" in prices.columns:
                            prices.set_index("Date", inplace=True)
                        prices.index = pd.to_datetime(
                            prices.index, errors="coerce"
                        )
                        prices.index = prices.index.tz_localize(None)
                        asset_returns = (
                            prices["Adjusted_Close"].pct_change().dropna()
                        )
                        asset_returns_df[ticker] = asset_returns
                except Exception:
                    pass
        
        if not asset_returns_df.empty:
            # Align all returns to common index
            asset_returns_df = asset_returns_df.sort_index()
            common_index = asset_returns_df.index
            
            # Calculate optimized portfolio returns: weighted sum
            optimized_returns = pd.Series(0.0, index=common_index)
            for ticker, weight in optimal_weights_dict.items():
                if ticker in asset_returns_df.columns:
                    optimized_returns += (
                        asset_returns_df[ticker] * weight
                    )
            
            optimized_returns = optimized_returns.dropna()
            
            # CRITICAL: Align optimized returns with current portfolio
            # to ensure same date range (SAME as charts)
            if (current_returns is not None and
                    not current_returns.empty):
                aligned_index = current_returns.index.intersection(
                    optimized_returns.index
                )
                optimized_returns = optimized_returns.reindex(aligned_index)
                current_returns = current_returns.reindex(aligned_index)
    except Exception as e:
        logger.warning(
            f"Could not calculate optimized portfolio returns: {e}"
        )
    
    # Calculate metrics using EXACT same logic as portfolio_analysis.py
    # overview tab (_render_overview_tab function)
    from core.analytics_engine.performance import (
        calculate_annualized_return,
    )
    from core.analytics_engine.risk_metrics import (
        calculate_volatility,
        calculate_max_drawdown,
    )
    from core.analytics_engine.ratios import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )
    
    # Calculate optimized portfolio metrics
    optimized_metrics = {}
    if optimized_returns is not None and not optimized_returns.empty:
        optimized_metrics["total_return"] = float(
            (1 + optimized_returns).prod() - 1
        )
        optimized_metrics["annualized_return"] = float(
            calculate_annualized_return(optimized_returns)
        )
        vol_result = calculate_volatility(optimized_returns)
        if isinstance(vol_result, dict):
            optimized_metrics["volatility"] = float(
                vol_result.get("annual", 0.0)
            )
        else:
            optimized_metrics["volatility"] = float(vol_result)
        optimized_metrics["sharpe_ratio"] = float(
            calculate_sharpe_ratio(
                optimized_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        optimized_metrics["sortino_ratio"] = float(
            calculate_sortino_ratio(
                optimized_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        dd_result = calculate_max_drawdown(optimized_returns)
        optimized_metrics["max_drawdown"] = float(
            dd_result[0] if isinstance(dd_result, tuple) else dd_result
        )
    else:
        # Fallback to optimization result
        optimized_metrics = {
            "total_return": 0.0,
            "annualized_return": float(result.expected_return or 0.0),
            "volatility": float(result.volatility or 0.0),
            "sharpe_ratio": float(result.sharpe_ratio or 0.0),
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Calculate current portfolio metrics (same logic)
    current_metrics = {}
    if current_returns is not None and not current_returns.empty:
        current_metrics["total_return"] = float(
            (1 + current_returns).prod() - 1
        )
        current_metrics["annualized_return"] = float(
            calculate_annualized_return(current_returns)
        )
        vol_result = calculate_volatility(current_returns)
        if isinstance(vol_result, dict):
            current_metrics["volatility"] = float(
                vol_result.get("annual", 0.0)
            )
        else:
            current_metrics["volatility"] = float(vol_result)
        current_metrics["sharpe_ratio"] = float(
            calculate_sharpe_ratio(
                current_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        current_metrics["sortino_ratio"] = float(
            calculate_sortino_ratio(
                current_returns, risk_free_rate=risk_free_rate
            ) or 0
        )
        dd_result = calculate_max_drawdown(current_returns)
        current_metrics["max_drawdown"] = float(
            dd_result[0] if isinstance(dd_result, tuple) else dd_result
        )
    else:
        current_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Display metrics using same component as overview
    from streamlit_app.components.metric_card_comparison import (
        render_metric_cards_row,
    )
    
    # Row 1: Total Return, Sharpe Ratio, Volatility, Max Drawdown
    # (same format as overview, optimized vs current)
    metrics_row1 = [
        {
            "label": "Total Return",
            "portfolio_value": optimized_metrics.get("total_return", 0),
            "benchmark_value": current_metrics.get("total_return"),
            "format": "percent",
            "higher_is_better": True,
        },
        {
            "label": "Sharpe Ratio",
            "portfolio_value": optimized_metrics.get("sharpe_ratio", 0),
            "benchmark_value": current_metrics.get("sharpe_ratio"),
            "format": "ratio",
            "higher_is_better": True,
        },
        {
            "label": "Volatility",
            "portfolio_value": optimized_metrics.get("volatility", 0),
            "benchmark_value": current_metrics.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": optimized_metrics.get("max_drawdown", 0),
            "benchmark_value": current_metrics.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
        },
    ]
    render_metric_cards_row(metrics_row1, columns_per_row=4)
    
    # Interpretation: Optimization comparison
    interpretation = _interpret_optimization_comparison(optimized_metrics, current_metrics)
    if interpretation:
        st.info(interpretation)
    
    # Current vs Optimal weights comparison
    st.subheader("Current vs Optimal Allocation")
    
    # Get current weights
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    from services.data_service import DataService
    data_service = DataService()
    
    # Fetch current prices
    positions = portfolio.get_all_positions()
    tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
    prices = {}
    for ticker in tickers:
        current_price = data_service.fetch_current_price(ticker)
        if current_price:
            prices[ticker] = current_price
    
    # Add CASH price (always 1.0) if CASH position exists
    if any(pos.ticker == "CASH" for pos in positions):
        prices["CASH"] = 1.0
    
    # Calculate portfolio value
    current_value = portfolio.calculate_current_value(prices)
    
    current_weights = {}
    for pos in positions:
        if pos.ticker in result.tickers:
            if pos.ticker == "CASH":
                pos_value = pos.shares  # CASH shares = dollar amount
            else:
                current_price = prices.get(pos.ticker)
                if not current_price:
                    continue
                pos_value = current_price * pos.shares
            
            current_weights[pos.ticker] = (
                pos_value / current_value if current_value > 0 else 0.0
            )
    
    # Create comparison DataFrame
    comparison_data = []
    for ticker in result.tickers:
        current_w = current_weights.get(ticker, 0.0)
        optimal_w = result.get_weights_dict().get(ticker, 0.0)
        diff = optimal_w - current_w
        
        comparison_data.append({
            "Ticker": ticker,
            "Current Weight": f"{current_w:.2%}",
            "Optimal Weight": f"{optimal_w:.2%}",
            "Difference": f"{diff:+.2%}",
        })
    
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )
    
    # Visual comparison
    fig = go.Figure()
    
    tickers = result.tickers
    current_weights_array = [
        current_weights.get(t, 0.0) for t in tickers
    ]
    optimal_weights_array = [result.get_weights_dict()[t] for t in tickers]
    
    fig.add_trace(
        go.Bar(
            name="Current",
            x=tickers,
            y=[w * 100 for w in current_weights_array],
            marker_color=COLORS["primary"],  # Purple - current portfolio
        )
    )
    
    fig.add_trace(
        go.Bar(
            name="Optimal",
            x=tickers,
            y=[w * 100 for w in optimal_weights_array],
            marker_color=COLORS["secondary"],  # Blue - optimal portfolio
        )
    )
    
    fig.update_layout(
        title="Current vs Optimal Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        barmode="group",
        height=400,
        template="plotly_dark",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation: Allocation changes
    optimal_weights_dict = result.get_weights_dict()
    interpretation = _interpret_allocation_changes(current_weights, optimal_weights_dict)
    if interpretation:
        st.info(interpretation)
    
    # Trade list
    st.subheader("Trade List")
    
    try:
        trades = optimization_service.generate_trade_list(
            portfolio_id, result
        )
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Format columns
            trades_df["Weight Change"] = (
                trades_df["optimal_weight"]
                - trades_df["current_weight"]
            ).apply(lambda x: f"{x:+.2%}")
            
            trades_df["Shares"] = trades_df["shares"].apply(
                lambda x: f"{x:,.2f}"
            )
            trades_df["Value"] = trades_df["value"].apply(
                lambda x: f"${x:,.2f}"
            )
            
            display_df = trades_df[
                ["ticker", "action", "Shares", "Value", "Weight Change"]
            ].rename(
                columns={
                    "ticker": "Ticker",
                    "action": "Action",
                }
            )
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            total_trade_value = trades_df["value"].sum()
            st.info(f"Total Trade Value: ${total_trade_value:,.2f}")
            
            # Interpretation: Trade list
            interpretation = _interpret_trade_list(trades)
            if interpretation:
                st.info(interpretation)
        else:
            st.info("No trades required - portfolio is already optimal.")
    
    except Exception as e:
        logger.exception("Error generating trade list")
        st.warning(f"Could not generate trade list: {str(e)}")
    
    # Performance Charts Section
    st.subheader("Performance Charts")
    
    # Initialize variables for Efficient Frontier
    current_analytics = None
    benchmark_returns = None
    
    # Get historical data for charts
    try:
        analytics_service = AnalyticsService()
        
        # Get current portfolio returns (with optional benchmark)
        current_analytics = analytics_service.calculate_portfolio_metrics(
            portfolio_id,
            start_date,
            end_date,
            benchmark_ticker=benchmark_for_viz if benchmark_for_viz else None,
        )
        current_returns = current_analytics.get("portfolio_returns")
        current_values = current_analytics.get("portfolio_values")
        benchmark_returns = current_analytics.get("benchmark_returns")
        
        # Calculate optimized portfolio historical returns
        # Get asset returns for the period
        from services.data_service import DataService
        data_service = DataService()
        
        optimal_weights_dict = result.get_weights_dict()
        tickers_list = result.tickers
        
        # Fetch historical prices for all assets
        asset_returns_df = pd.DataFrame()
        for ticker in tickers_list:
            if ticker == "CASH":
                # CASH returns = risk-free rate / periods
                dr = pd.bdate_range(start=start_date, end=end_date, normalize=True)
                daily_return = (1 + 0.0435) ** (1.0 / 252) - 1
                cash_returns = pd.Series(
                    data=daily_return,
                    index=dr,
                    name=ticker,
                )
                asset_returns_df[ticker] = cash_returns
            else:
                try:
                    prices = data_service.fetch_historical_prices(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        save_to_db=False,
                    )
                    # Ensure prices is a DataFrame before checking .empty
                    if not isinstance(prices, pd.DataFrame):
                        logger.warning(
                            f"fetch_historical_prices returned non-DataFrame "
                            f"for {ticker}: {type(prices)}"
                        )
                        continue
                    if not prices.empty:
                        if "Date" in prices.columns:
                            prices.set_index("Date", inplace=True)
                        prices.index = pd.to_datetime(prices.index, errors="coerce")
                        prices.index = prices.index.tz_localize(None)
                        
                        asset_returns = prices["Adjusted_Close"].pct_change().dropna()
                        asset_returns_df[ticker] = asset_returns
                except Exception as e:
                    logger.warning(f"Could not fetch returns for {ticker}: {e}")
        
        # Align all returns to common index
        if not asset_returns_df.empty:
            asset_returns_df = asset_returns_df.sort_index()
            common_index = asset_returns_df.index
            
            # Calculate optimized portfolio returns: weighted sum
            optimized_returns = pd.Series(0.0, index=common_index)
            for ticker, weight in optimal_weights_dict.items():
                if ticker in asset_returns_df.columns:
                    optimized_returns += asset_returns_df[ticker] * weight
            
            optimized_returns = optimized_returns.dropna()
            
            # Calculate optimized portfolio values (cumulative)
            optimized_values = (1 + optimized_returns).cumprod()
            if current_values is not None and not current_values.empty:
                initial_value = float(current_values.iloc[0])
                optimized_values = optimized_values * initial_value
            else:
                optimized_values = optimized_values * 10000  # Default $10k
            
            # Align current and optimized returns
            if current_returns is not None and not current_returns.empty:
                aligned_index = current_returns.index.intersection(
                    optimized_returns.index
                )
                current_aligned = current_returns.reindex(aligned_index)
                optimized_aligned = optimized_returns.reindex(aligned_index)
                benchmark_aligned = None
                if benchmark_returns is not None and not benchmark_returns.empty:
                    benchmark_aligned = benchmark_returns.reindex(aligned_index)
            else:
                current_aligned = None
                optimized_aligned = optimized_returns
                benchmark_aligned = benchmark_returns if benchmark_returns is not None and not benchmark_returns.empty else None
            
            # 1. Returns Chart
            st.markdown("**Portfolio Returns Comparison**")
            fig_returns = go.Figure()
            
            if current_aligned is not None and not current_aligned.empty:
                current_cumulative = (1 + current_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=current_cumulative.index,
                        y=current_cumulative * 100,
                        mode="lines",
                        name="Current Portfolio",
                        line=dict(
                            color=COLORS["primary"], width=2
                        ),  # Purple
                    )
                )
            
            if optimized_aligned is not None and not optimized_aligned.empty:
                optimized_cumulative = (1 + optimized_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=optimized_cumulative.index,
                        y=optimized_cumulative * 100,
                        mode="lines",
                        name="Optimized Portfolio",
                        line=dict(
                            color=COLORS["success"], width=2
                        ),  # Green
                    )
                )
            
            if (
                benchmark_aligned is not None
                and not benchmark_aligned.empty
                and benchmark_for_viz
            ):
                benchmark_cumulative = (1 + benchmark_aligned).cumprod() - 1
                fig_returns.add_trace(
                    go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative * 100,
                        mode="lines",
                        name=f"Benchmark ({benchmark_for_viz})",
                        line=dict(
                            color=COLORS["secondary"], width=2
                        ),  # Blue
                    )
                )
            
            fig_returns.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Interpretation: Returns comparison
            if current_aligned is not None and not current_aligned.empty and optimized_aligned is not None and not optimized_aligned.empty:
                interpretation = _interpret_returns_comparison(current_aligned, optimized_aligned)
                if interpretation:
                    st.info(interpretation)
            
            # 2. Drawdown Chart
            st.markdown("**Drawdown Comparison**")
            fig_drawdown = go.Figure()
            
            # Drawdown: optimized - red fill
            if optimized_values is not None and not optimized_values.empty:
                optimized_peak = optimized_values.expanding().max()
                optimized_dd = (
                    (optimized_values - optimized_peak) / optimized_peak * 100
                )
                fig_drawdown.add_trace(
                    go.Scatter(
                        x=optimized_dd.index,
                        y=optimized_dd,
                        mode="lines",
                        name="Optimized Portfolio",
                        line=dict(
                            color=COLORS["danger"], width=2
                        ),  # Red
                        fill="tozeroy",
                        fillcolor="rgba(239, 85, 59, 0.3)",  # Red fill
                    )
                )
            
            # Drawdown: current - orange line (#FFCC80)
            if current_values is not None and not current_values.empty:
                current_peak = current_values.expanding().max()
                current_dd = (current_values - current_peak) / current_peak * 100
                fig_drawdown.add_trace(
                    go.Scatter(
                        x=current_dd.index,
                        y=current_dd,
                        mode="lines",
                        name="Current Portfolio",
                        line=dict(
                            color="#FFCC80", width=2  # Orange #FFCC80
                        ),
                    )
                )
            
            # Drawdown: benchmark - yellow line (#FFF59D)
            if (
                benchmark_returns is not None
                and not benchmark_returns.empty
                and benchmark_for_viz
            ):
                if benchmark_aligned is not None:
                    benchmark_values = (1 + benchmark_aligned).cumprod()
                    if current_values is not None and not current_values.empty:
                        initial_value = float(current_values.iloc[0])
                        benchmark_values = benchmark_values * initial_value
                    else:
                        benchmark_values = benchmark_values * 10000
                    
                    benchmark_peak = benchmark_values.expanding().max()
                    benchmark_dd = (
                        (benchmark_values - benchmark_peak) / benchmark_peak * 100
                    )
                    fig_drawdown.add_trace(
                        go.Scatter(
                            x=benchmark_dd.index,
                            y=benchmark_dd,
                            mode="lines",
                            name=f"Benchmark ({benchmark_for_viz})",
                            line=dict(
                                color="#FFF59D", width=2  # Yellow #FFF59D
                            ),
                        )
                    )
            
            fig_drawdown.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig_drawdown, use_container_width=True)
            
            # Interpretation: Drawdown comparison
            if current_values is not None and not current_values.empty and optimized_values is not None and not optimized_values.empty:
                interpretation = _interpret_drawdown_comparison(current_values, optimized_values)
                if interpretation:
                    st.info(interpretation)
            
    except Exception as e:
        logger.exception("Error generating performance charts")
        st.warning(f"Could not generate performance charts: {str(e)}")
    
    # Efficient Frontier
    st.subheader("Efficient Frontier")
    
    # Create a hash of constraints to detect changes and auto-update
    import hashlib
    constraints_str = str(sorted(constraints.items())) if constraints else "None"
    constraints_hash = hashlib.md5(constraints_str.encode()).hexdigest()[:8]
    
    # Store current constraints hash in session state
    last_constraints_hash = st.session_state.get("last_frontier_constraints_hash")
    constraints_changed = last_constraints_hash != constraints_hash
    
    # Auto-update frontier when constraints change
    if constraints_changed and last_constraints_hash is not None:
        # Clear any cached frontier data when constraints change
        if "frontier_data_cache" in st.session_state:
            del st.session_state["frontier_data_cache"]
        st.session_state["last_frontier_constraints_hash"] = constraints_hash
        # Show info message that frontier will be recalculated
        st.info("Constraints changed. Efficient Frontier will be recalculated with new constraints.")
    
    # Update hash in session state
    if last_constraints_hash is None:
        st.session_state["last_frontier_constraints_hash"] = constraints_hash
    
    show_frontier = st.checkbox("Show Efficient Frontier", value=True, key="show_frontier")
    
    if show_frontier:
        with st.spinner("Generating efficient frontier..."):
            try:
                # Use the same period as optimization for Efficient Frontier
                # This ensures Max Sharpe point matches optimized portfolio
                frontier_start = (
                    optimization_period_start
                    if optimization_period_start
                    else start_date
                )
                frontier_end = (
                    optimization_period_end
                    if optimization_period_end
                    else end_date
                )
                
                # Check if we're using out-of-sample testing
                # If so, show info about which period is used
                using_training_period = (
                    optimization_period_start 
                    and optimization_period_start < start_date
                )
                
                if using_training_period:
                    st.info(
                        f"**Efficient Frontier Period**: Training period "
                        f"({frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')})\n\n"
                        f"**Important**: Efficient Frontier shows **expected returns** (based on historical averages) "
                        f"for the training period. The statistics section below shows **actual returns** "
                        f"for the validation period ({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}). "
                        f"These values will differ because:\n"
                        f"- Expected returns are forward-looking estimates based on historical data\n"
                        f"- Actual returns reflect real market performance over the validation period\n"
                        f"- Market conditions may have changed between training and validation periods"
                    )
                
                # IMPORTANT: Always use CURRENT constraints for Efficient Frontier
                # This ensures the frontier line and all points (Max Sharpe, Min Volatility)
                # are calculated with the same constraints that the user has set
                # Note: constraints are passed from the current UI state, so they should
                # reflect any changes the user made to sliders/checkboxes
                frontier_data = None
                try:
                    frontier_data = optimization_service.generate_efficient_frontier(
                        portfolio_id=portfolio_id,
                        start_date=frontier_start,
                        end_date=frontier_end,
                        n_points=150,  # More points for smoother curve
                        constraints=constraints if constraints else None,
                    )
                except InsufficientDataError as e:
                    # If training period has no data, try validation period as fallback
                    if using_training_period:
                        st.warning(
                            f"**No data available for training period** "
                            f"({frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')}).\n\n"
                            f"**Trying validation period instead** "
                            f"({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')})..."
                        )
                        try:
                            frontier_data = optimization_service.generate_efficient_frontier(
                                portfolio_id=portfolio_id,
                                start_date=start_date,  # Use validation period
                                end_date=end_date,
                                n_points=150,
                                constraints=constraints if constraints else None,
                            )
                            st.info(
                                f"Using **validation period** for Efficient Frontier. "
                                f"Note: Max Sharpe point may not align with optimized portfolio "
                                f"(which was optimized on training period)."
                            )
                        except InsufficientDataError as e2:
                            st.warning(
                                f"**Insufficient data for Efficient Frontier**: {str(e2)}\n\n"
                                f"**Training period**: {frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')}\n"
                                f"**Validation period**: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}\n\n"
                                f"**Possible reasons**:\n"
                                f"- No price data available for these periods\n"
                                f"- Portfolio positions may not have existed during these periods\n"
                                f"- Try adjusting the date range or checking your portfolio positions"
                            )
                            return
                        except Exception as e2:
                            logger.exception("Error generating efficient frontier with validation period")
                            st.error(f"Error generating efficient frontier: {str(e2)}")
                            return
                    else:
                        st.warning(
                            f"**Insufficient data for Efficient Frontier**: {str(e)}\n\n"
                            f"**Period requested**: {frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')}\n\n"
                            f"**Possible reasons**:\n"
                            f"- No price data available for this period\n"
                            f"- Portfolio positions may not have existed during this period\n"
                            f"- Try adjusting the date range or checking your portfolio positions"
                        )
                        return
                except Exception as e:
                    logger.exception("Error generating efficient frontier")
                    st.error(f"Error generating efficient frontier: {str(e)}")
                    return
                
                if not frontier_data:
                    return
                
                # Plot frontier
                fig = go.Figure()
                
                # Get frontier points
                volatilities_list = frontier_data.get("volatilities", [])
                returns_list = frontier_data.get("returns", [])
                
                if not volatilities_list or not returns_list:
                    st.warning(
                        f"**No frontier data available** for period "
                        f"{frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')}.\n\n"
                        f"This may happen if:\n"
                        f"- There's insufficient data for this period\n"
                        f"- The optimization constraints are too restrictive\n"
                        f"- Try adjusting constraints or date range"
                    )
                    return
                
                # Convert to numpy arrays for easier manipulation
                vols = np.array(volatilities_list)
                rets = np.array(returns_list)
                
                # Filter only efficient part (upper part of curve)
                # Efficient frontier: for each volatility, take max return
                # Sort by volatility
                sorted_indices = np.argsort(vols)
                sorted_vols = vols[sorted_indices]
                sorted_rets = rets[sorted_indices]
                
                # Filter to keep only efficient part (non-decreasing returns)
                # Start from min volatility, keep points where return >= previous
                efficient_indices = [0]
                for i in range(1, len(sorted_vols)):
                    if sorted_rets[i] >= sorted_rets[efficient_indices[-1]]:
                        efficient_indices.append(i)
                    else:
                        # Check if this point has higher return than any previous
                        # at same or lower volatility (shouldn't happen, but safety)
                        prev_idx = efficient_indices[-1]
                        if sorted_rets[i] > sorted_rets[prev_idx]:
                            efficient_indices.append(i)
                
                efficient_vols = sorted_vols[efficient_indices]
                efficient_rets = sorted_rets[efficient_indices]
                
                # Convert to percentages for display
                x_plot = efficient_vols * 100
                y_plot = efficient_rets * 100
                
                # Add efficient frontier line with smooth interpolation
                fig.add_trace(
                    go.Scatter(
                        x=x_plot,
                        y=y_plot,
                        mode="lines",
                        name="Efficient Frontier",
                        line=dict(
                            color=COLORS["primary"],
                            width=3,
                            smoothing=1.0,  # Smooth the line
                        ),
                        showlegend=True,
                        hovertemplate=(
                            "Volatility: %{x:.2f}%<br>"
                            "Return: %{y:.2f}%<extra></extra>"
                        ),
                    )
                )
                
                # Get tangency (max Sharpe) and min variance portfolios
                tangency_portfolio = frontier_data.get("tangency_portfolio")
                min_var_portfolio = frontier_data.get("min_variance_portfolio")
                
                # Max Sharpe Ratio point (tangency portfolio)
                if tangency_portfolio:
                    tang_vol = tangency_portfolio.get("volatility")
                    tang_ret = tangency_portfolio.get("expected_return")
                    if tang_vol is not None and tang_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[float(tang_vol) * 100],
                                y=[float(tang_ret) * 100],
                                mode="markers",
                                name="Max Sharpe Ratio",
                                marker=dict(
                                    color="#FFD700",  # Gold
                                    size=16,
                                    symbol="triangle-up",
                                    line=dict(color="#000000", width=2),
                                ),
                                hovertemplate=(
                                    "Max Sharpe Ratio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Min Variance point
                if min_var_portfolio:
                    min_var_vol = min_var_portfolio.get("volatility")
                    min_var_ret = min_var_portfolio.get("expected_return")
                    if min_var_vol is not None and min_var_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[float(min_var_vol) * 100],
                                y=[float(min_var_ret) * 100],
                                mode="markers",
                                name="Min Volatility",
                                marker=dict(
                                    color="#4CAF50",  # Green
                                    size=16,
                                    symbol="triangle-down",
                                    line=dict(color="#000000", width=2),
                                ),
                                hovertemplate=(
                                    "Min Volatility<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Optimized portfolio - recalculate metrics for the SAME period as Efficient Frontier
                # This ensures consistency: Efficient Frontier shows expected returns for training period,
                # and Optimized Portfolio should also show expected returns for the same period
                if result and result.success:
                    # Use expected return and volatility from optimization result
                    # These are calculated for the training period (same as Efficient Frontier)
                    opt_vol = result.volatility
                    opt_ret = result.expected_return
                    
                    # Also get Sharpe ratio from result (calculated for training period)
                    opt_sharpe = result.sharpe_ratio
                    
                    if opt_vol is not None and opt_ret is not None and opt_vol > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=[float(opt_vol) * 100],
                                y=[float(opt_ret) * 100],
                                mode="markers",
                                name="Optimized Portfolio",
                                marker=dict(
                                    color=COLORS["success"],  # Green
                                    size=18,
                                    symbol="star",
                                    line=dict(color="#FFFFFF", width=2),
                                ),
                                hovertemplate=(
                                    "Optimized Portfolio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Expected Return: %{y:.2f}%<br>"
                                    f"Sharpe Ratio: {float(opt_sharpe) if opt_sharpe is not None else 0.0:.2f}<br>"
                                    f"<i>Period: {frontier_start.strftime('%Y-%m-%d')} → {frontier_end.strftime('%Y-%m-%d')}</i><br>"
                                    "<i>Note: Shows expected returns for training period (same as Efficient Frontier). "
                                    "Statistics section shows actual returns for validation period.</i>"
                                    "<extra></extra>"
                                ),
                            )
                        )
                
                # Current portfolio - use metrics from current_returns
                if current_returns is not None and not current_returns.empty:
                    curr_vol = current_metrics.get("volatility", 0.0)
                    curr_ret = current_metrics.get("annualized_return", 0.0)
                    if curr_vol > 0 and curr_ret is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[curr_vol * 100],
                                y=[curr_ret * 100],
                                mode="markers",
                                name="Current Portfolio",
                                marker=dict(
                                    color=COLORS["primary"],  # Purple
                                    size=18,
                                    symbol="star",
                                    line=dict(color="#FFFFFF", width=2),
                                ),
                                hovertemplate=(
                                    "Current Portfolio<br>"
                                    "Volatility: %{x:.2f}%<br>"
                                    "Return: %{y:.2f}%<extra></extra>"
                                ),
                            )
                        )
                
                # Benchmark
                if (
                    benchmark_returns is not None
                    and not benchmark_returns.empty
                    and benchmark_for_viz
                ):
                    try:
                        from core.analytics_engine.performance import (
                            calculate_annualized_return,
                        )
                        from core.analytics_engine.risk_metrics import (
                            calculate_volatility,
                        )
                        
                        bench_ret = calculate_annualized_return(benchmark_returns)
                        bench_vol_dict = calculate_volatility(benchmark_returns)
                        bench_vol = (
                            bench_vol_dict.get("annual", 0.0)
                            if isinstance(bench_vol_dict, dict)
                            else bench_vol_dict
                        )
                        
                        if bench_vol > 0 and bench_ret is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=[float(bench_vol) * 100],
                                    y=[float(bench_ret) * 100],
                                    mode="markers",
                                    name=f"Benchmark ({benchmark_for_viz})",
                                    marker=dict(
                                        color=COLORS["secondary"],  # Blue
                                        size=14,
                                        symbol="diamond",
                                        line=dict(color="#FFFFFF", width=2),
                                    ),
                                    hovertemplate=(
                                        f"Benchmark ({benchmark_for_viz})<br>"
                                        "Volatility: %{x:.2f}%<br>"
                                        "Return: %{y:.2f}%<extra></extra>"
                                    ),
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not add benchmark to frontier: {e}"
                        )
                
                # Update layout with better styling
                fig.update_layout(
                    title=dict(
                        text="Efficient Frontier",
                        font=dict(size=20),
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Volatility (Annualized, %)",
                            font=dict(size=14),
                        ),
                        gridcolor="rgba(128, 128, 128, 0.3)",
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Expected Return (Annualized, %)",
                            font=dict(size=14),
                        ),
                        gridcolor="rgba(128, 128, 128, 0.3)",
                    ),
                    height=600,
                    template="plotly_dark",
                    hovermode="closest",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0, 0, 0, 0.5)",
                        bordercolor="rgba(255, 255, 255, 0.3)",
                        borderwidth=1,
                    ),
                    margin=dict(l=60, r=20, t=60, b=60),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation: Efficient frontier
                interpretation = _interpret_efficient_frontier(frontier_data, result, current_metrics)
                if interpretation:
                    st.info(interpretation)
            
            except Exception as e:
                logger.exception("Error generating efficient frontier")
                st.warning(f"Could not generate efficient frontier: {str(e)}")
    
    # Correlation Analysis & Diversification
    st.markdown("---")
    st.subheader("Correlation Analysis & Diversification")
    try:
        # Get price data for correlation calculation
        from services.data_service import DataService
        data_service = DataService()

        # Fetch historical prices
        all_prices = []
        for ticker in result.tickers:
            if ticker == "CASH":
                continue
            try:
                prices = data_service.fetch_historical_prices(
                    ticker, start_date, end_date,
                    use_cache=True, save_to_db=False
                )
                # Ensure prices is a DataFrame - check immediately
                if not isinstance(prices, pd.DataFrame):
                    logger.error(
                        f"fetch_historical_prices returned non-DataFrame "
                        f"for {ticker}: {type(prices)}. "
                        f"Value type: {type(prices)}, "
                        f"Value repr: {repr(prices)[:200]}"
                    )
                    continue  # Skip this ticker
                
                # Now safe to check .empty
                if prices.empty:
                    logger.warning(
                        f"Empty DataFrame returned for {ticker}"
                    )
                    continue  # Skip this ticker
                
                # Ensure required columns exist
                if not hasattr(prices, 'columns'):
                    logger.error(
                        f"prices object has no 'columns' attribute for {ticker}: "
                        f"{type(prices)}"
                    )
                    continue  # Skip this ticker
                    
                required_cols = ["Date", "Adjusted_Close"]
                if not all(col in prices.columns for col in required_cols):
                    logger.warning(
                        f"Missing required columns for {ticker}: "
                        f"{required_cols}. Got: {list(prices.columns)}"
                    )
                    continue  # Skip this ticker
                    
                prices["Ticker"] = ticker
                all_prices.append(prices)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch {ticker}: {e}", exc_info=True
                )

        if not all_prices:
            st.warning(
                "No historical price data available for correlation "
                "analysis. Please ensure assets have price data for "
                "the selected date range."
            )
        elif len(all_prices) < 2:
            st.warning(
                "At least 2 assets are required for correlation analysis. "
                f"Only {len(all_prices)} asset(s) have price data."
            )
        else:
            # Combine and pivot
            try:
                # Ensure all items in all_prices are DataFrames
                if not all_prices:
                    st.warning("No price data to combine.")
                else:
                    # Verify all are DataFrames before concatenating
                    for i, df in enumerate(all_prices):
                        if not isinstance(df, pd.DataFrame):
                            logger.error(
                                f"Item {i} in all_prices is not a DataFrame: "
                                f"{type(df)}"
                            )
                            st.warning(
                                f"Data format error: expected DataFrame, "
                                f"got {type(df)}"
                            )
                            raise ValueError(
                                "Invalid data format in all_prices"
                            )

                    # Log all_prices before concatenation
                    logger.debug(
                        f"Concatenating {len(all_prices)} DataFrames. "
                        f"Types: {[type(df) for df in all_prices]}"
                    )
                    
                    combined = pd.concat(all_prices, ignore_index=True)

                    # Verify combined is a DataFrame
                    if not isinstance(combined, pd.DataFrame):
                        logger.error(
                            f"pd.concat returned non-DataFrame: "
                            f"{type(combined)}, value: {combined}"
                        )
                        st.warning(
                            "Combined data is not a DataFrame. "
                            f"Got {type(combined)}"
                        )
                        raise ValueError(
                            f"Combined data must be DataFrame, got "
                            f"{type(combined)}"
                        )
                    
                    # Check required columns before pivoting
                    required_cols = ["Date", "Ticker", "Adjusted_Close"]
                    missing_cols = [
                        col for col in required_cols
                        if col not in combined.columns
                    ]
                    if missing_cols:
                        st.warning(
                            f"Missing required columns for pivoting: "
                            f"{missing_cols}"
                        )
                        raise ValueError(
                            f"Missing columns: {missing_cols}"
                        )
                    
                    # Log combined data info before pivoting
                    logger.debug(
                        f"Combined data type: {type(combined)}, "
                        f"shape: {combined.shape if hasattr(combined, 'shape') else 'N/A'}, "
                        f"columns: {list(combined.columns) if hasattr(combined, 'columns') else 'N/A'}"
                    )
                    
                    # Initialize price_data to None to ensure it's defined
                    price_data = None
                    try:
                        price_data = combined.pivot_table(
                            index="Date",
                            columns="Ticker",
                            values="Adjusted_Close",
                            aggfunc="first",
                        )
                        
                        # Log result type immediately after pivot
                        logger.debug(
                            f"pivot_table result type: {type(price_data)}, "
                            f"is DataFrame: {isinstance(price_data, pd.DataFrame)}"
                        )
                        
                    except Exception as pivot_err:
                        logger.error(
                            f"Error in pivot_table: {pivot_err}",
                            exc_info=True
                        )
                        st.warning(
                            f"Error creating pivot table: {pivot_err}"
                        )
                        raise

                    # Check if price_data is defined and is a DataFrame
                    if price_data is None:
                        logger.error("price_data is None after pivot_table")
                        st.warning("Price data is None after pivoting.")
                        raise ValueError("price_data is None after pivot_table")
                    
                    if not isinstance(price_data, pd.DataFrame):
                        logger.error(
                            f"pivot_table returned non-DataFrame: "
                            f"{type(price_data)}, value: {price_data}"
                        )
                        st.warning(
                            "Price data format error after pivoting. "
                            f"Expected DataFrame but got {type(price_data)}."
                        )
                        raise ValueError(
                            f"pivot_table returned {type(price_data)}, "
                            f"expected DataFrame"
                        )
                    
                    # Now safe to check .empty
                    if price_data.empty:
                        st.warning(
                            "Price data is empty after pivoting. "
                            "Check date alignment."
                        )
                    else:
                        # Calculate returns
                        returns_df = price_data.pct_change().dropna()

                        if returns_df.empty or len(returns_df) < 10:
                            st.warning(
                                "Insufficient return data for correlation "
                                "analysis. Need at least 10 days of data."
                            )
                        elif len(returns_df.columns) < 2:
                            st.warning(
                                "At least 2 assets are required for "
                                "correlation analysis."
                            )
                        else:
                            # Calculate correlation matrix
                            corr_matrix = returns_df.corr()

                            # Correlation Warnings
                            st.markdown("#### Correlation Warnings")
                            warnings = []

                            # Check for high correlations
                            high_corr_pairs = []
                            for i, ticker1 in enumerate(corr_matrix.columns):
                                for ticker2 in corr_matrix.columns[i+1:]:
                                    corr_val = corr_matrix.loc[ticker1, ticker2]
                                    if (
                                        not np.isnan(corr_val)
                                        and abs(corr_val) > 0.8
                                    ):
                                        high_corr_pairs.append(
                                            (ticker1, ticker2, corr_val)
                                        )

                            if high_corr_pairs:
                                warnings.append({
                                    "type": "warning",
                                    "message": (
                                        f"Found {len(high_corr_pairs)} "
                                        f"pair(s) with correlation > 0.8. "
                                        f"High correlation reduces "
                                        f"diversification."
                                    ),
                                })

                                # Show pairs
                                with st.expander(
                                    "High Correlation Pairs",
                                    expanded=False
                                ):
                                    for (
                                        ticker1,
                                        ticker2,
                                        corr_val,
                                    ) in high_corr_pairs[:10]:
                                        st.write(
                                            f"**{ticker1} - {ticker2}**: "
                                            f"{corr_val:.3f}"
                                        )
                            else:
                                warnings.append({
                                    "type": "success",
                                    "message": (
                                        "✓ No pairs with correlation > 0.8 "
                                        "found. Good diversification "
                                        "potential."
                                    ),
                                })

                            # Check average correlation
                            upper_triangle = corr_matrix.where(
                                np.triu(
                                    np.ones(corr_matrix.shape), k=1
                                ).astype(bool)
                            ).stack()
                            avg_corr = float(upper_triangle.mean())

                            if avg_corr > 0.5:
                                warnings.append({
                                    "type": "warning",
                                    "message": (
                                        f"Average correlation "
                                        f"({avg_corr:.2f}) is high. "
                                        f"Consider adding assets with "
                                        f"lower correlation."
                                    ),
                                })
                            elif avg_corr < 0.3:
                                warnings.append({
                                    "type": "success",
                                    "message": (
                                        f"✓ Low average correlation "
                                        f"({avg_corr:.2f}). Excellent "
                                        f"diversification."
                                    ),
                                })

                            # Display warnings
                            for warning in warnings:
                                if warning["type"] == "warning":
                                    st.warning(warning["message"])
                                else:
                                    st.success(warning["message"])

                            # Diversification Checklist
                            st.markdown("---")
                            st.markdown("#### Diversification Checklist")

                            checklist_items = []

                            # Check 1: Number of assets
                            num_assets = len(result.tickers)
                            if num_assets >= 10:
                                checklist_items.append({
                                    "item": (
                                        f"Portfolio has {num_assets} assets "
                                        f"(≥10)"
                                    ),
                                    "status": "✓",
                                    "color": "green",
                                })
                            elif num_assets >= 5:
                                checklist_items.append({
                                    "item": (
                                        f"Portfolio has {num_assets} assets "
                                        f"(5-9)"
                                    ),
                                    "status": "⚠",
                                    "color": "orange",
                                })
                            else:
                                checklist_items.append({
                                    "item": (
                                        f"Portfolio has {num_assets} assets "
                                        f"(<5)"
                                    ),
                                    "status": "✗",
                                    "color": "red",
                                })

                            # Check 2: Average correlation
                            if avg_corr < 0.3:
                                checklist_items.append({
                                    "item": (
                                        f"Average correlation < 0.3 "
                                        f"({avg_corr:.2f})"
                                    ),
                                    "status": "✓",
                                    "color": "green",
                                })
                            elif avg_corr < 0.5:
                                checklist_items.append({
                                    "item": (
                                        f"Average correlation 0.3-0.5 "
                                        f"({avg_corr:.2f})"
                                    ),
                                    "status": "⚠",
                                    "color": "orange",
                                })
                            else:
                                checklist_items.append({
                                    "item": (
                                        f"Average correlation > 0.5 "
                                        f"({avg_corr:.2f})"
                                    ),
                                    "status": "✗",
                                    "color": "red",
                                })

                            # Check 3: High correlation pairs
                            if len(high_corr_pairs) == 0:
                                checklist_items.append({
                                    "item": (
                                        "No pairs with correlation > 0.8"
                                    ),
                                    "status": "✓",
                                    "color": "green",
                                })
                            elif len(high_corr_pairs) <= 2:
                                checklist_items.append({
                                    "item": (
                                        f"{len(high_corr_pairs)} pair(s) "
                                        f"with correlation > 0.8"
                                    ),
                                    "status": "⚠",
                                    "color": "orange",
                                })
                            else:
                                checklist_items.append({
                                    "item": (
                                        f"{len(high_corr_pairs)} pairs "
                                        f"with correlation > 0.8"
                                    ),
                                    "status": "✗",
                                    "color": "red",
                                })

                            # Check 4: Weight concentration
                            optimal_weights_dict = result.get_weights_dict()
                            max_weight = max(optimal_weights_dict.values())
                            if max_weight < 0.3:
                                checklist_items.append({
                                    "item": (
                                        f"Max weight < 30% "
                                        f"({max_weight:.1%})"
                                    ),
                                    "status": "✓",
                                    "color": "green",
                                })
                            elif max_weight < 0.5:
                                checklist_items.append({
                                    "item": (
                                        f"Max weight 30-50% "
                                        f"({max_weight:.1%})"
                                    ),
                                    "status": "⚠",
                                    "color": "orange",
                                })
                            else:
                                checklist_items.append({
                                    "item": (
                                        f"Max weight > 50% "
                                        f"({max_weight:.1%})"
                                    ),
                                    "status": "✗",
                                    "color": "red",
                                })

                            # Check 5: Negative correlations (hedging)
                            negative_corrs = [
                                corr for corr in upper_triangle.values
                                if corr < 0
                            ]
                            if len(negative_corrs) > 0:
                                checklist_items.append({
                                    "item": (
                                        f"Found {len(negative_corrs)} "
                                        f"negative correlation(s) (hedging)"
                                    ),
                                    "status": "✓",
                                    "color": "green",
                                })
                            else:
                                checklist_items.append({
                                    "item": (
                                        "No negative correlations "
                                        "(no natural hedging)"
                                    ),
                                    "status": "⚠",
                                    "color": "orange",
                                })

                            # Display checklist
                            for item in checklist_items:
                                color_emoji = {
                                    "green": "✓",
                                    "orange": "⚠",
                                    "red": "✗",
                                }
                                st.markdown(
                                    f"{color_emoji.get(item['color'], '•')} "
                                    f"**{item['item']}**"
                                )

                            # Risk Parity with Correlations Note
                            if result.method == "Risk Parity":
                                st.markdown("---")
                                st.info(
                                    "**Risk Parity Method:** This "
                                    "optimization method already accounts "
                                    "for correlations through the "
                                    "covariance matrix. Each asset "
                                    "contributes equally to portfolio risk, "
                                    "considering both individual volatility "
                                    "and correlations with other assets."
                                )
                            
                            # Interpretation: Correlation and diversification
                            optimal_weights_dict = result.get_weights_dict()
                            num_assets = len(result.tickers)
                            interpretation = _interpret_correlation_diversification(
                                corr_matrix, optimal_weights_dict, num_assets
                            )
                            if interpretation:
                                st.info(interpretation)
            except Exception as pivot_error:
                logger.warning(
                    f"Error processing price data: {pivot_error}"
                )
                st.warning(
                    f"Error processing price data for correlation: "
                    f"{pivot_error}"
                )
    except Exception as e:
        logger.exception("Error in correlation analysis")
        st.warning(f"Could not calculate correlation analysis: {str(e)}")

    # Sensitivity Analysis
    st.markdown("---")
    st.subheader("Sensitivity Analysis")
    
    perform_sensitivity = st.checkbox(
        "Perform Sensitivity Analysis",
        value=False,
        key="opt_sensitivity",
        help="Analyze how optimal weights change with parameter variations",
    )
    
    if perform_sensitivity:
        sensitivity_col1, sensitivity_col2 = st.columns(2)
        
        with sensitivity_col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                options=["returns", "covariance"],
                index=0,
                key="opt_sensitivity_type",
                help="Parameter to vary for sensitivity analysis",
            )
        
        with sensitivity_col2:
            variation_range = st.slider(
                "Variation Range",
                min_value=0.05,
                max_value=0.50,
                value=0.10,
                step=0.05,
                format="%.2f",
                key="opt_sensitivity_range",
                help="Range of variation to test (as decimal, e.g., 0.10 = 10%)",
            )
        
        num_points = st.slider(
            "Number of Points",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="opt_sensitivity_points",
            help="Number of variation points to test",
        )
        
        if st.button("Run Sensitivity Analysis", key="opt_run_sensitivity"):
            with st.spinner("Running sensitivity analysis..."):
                try:
                    # Get method name from session state (original method used)
                    original_method = st.session_state.get(
                        "optimization_method_used"
                    )
                    if not original_method:
                        st.error(
                            "Cannot determine optimization method. "
                            "Please run optimization first."
                        )
                        return
                    
                    sensitivity_results = (
                        optimization_service.perform_sensitivity_analysis(
                            portfolio_id=portfolio_id,
                            method=original_method,
                            start_date=start_date,
                            end_date=end_date,
                            base_constraints=constraints if constraints else None,
                            analysis_type=analysis_type,
                            variation_range=variation_range,
                            num_points=num_points,
                        )
                    )
                    
                    st.session_state["sensitivity_results"] = (
                        sensitivity_results
                    )
                    st.success("Sensitivity analysis completed!")
                    st.rerun()
                except Exception as e:
                    logger.exception("Sensitivity analysis failed")
                    st.error(f"Sensitivity analysis failed: {str(e)}")
        
        # Display sensitivity results if available
        if "sensitivity_results" in st.session_state:
            sens_results = st.session_state["sensitivity_results"]
            
            # Convert to DataFrame for visualization
            sens_df = pd.DataFrame(sens_results["results"])
            
            if not sens_df.empty:
                st.markdown("**Sensitivity Analysis Results**")
                
                # Prepare data for heatmap
                variation_col = "variation"
                ticker_cols = [
                    col for col in sens_df.columns if col != variation_col
                ]
                
                # Calculate sensitivity metrics
                sensitivity_info = []
                for ticker in ticker_cols:
                    weights = sens_df[ticker].values
                    weight_range = weights.max() - weights.min()
                    weight_std = weights.std()
                    sensitivity_info.append({
                        "ticker": ticker,
                        "range": weight_range,
                        "std": weight_std,
                        "min": weights.min(),
                        "max": weights.max(),
                    })
                
                # Check if weights are changing
                total_range = sum(info["range"] for info in sensitivity_info)
                if total_range < 0.01:  # Less than 1% total variation
                    st.info(
                        "ℹ️ **Interpretation:** Portfolio weights are practically unchanged "
                        "when the parameter varies. This may mean:\n"
                        "- Optimization is stable within the given constraints\n"
                        "- Constraints (min/max weights) are sufficiently tight\n"
                        "- Variation parameter does not affect the optimal solution\n\n"
                        "**Recommendation:** Try increasing the variation range "
                        "or relaxing constraints for more detailed analysis."
                    )
                
                # Create heatmap of weight changes
                fig = go.Figure()
                
                # Create heatmap with custom colorscale from palette
                # Custom colorscale: red (low) -> white -> green (high)
                custom_colorscale = [
                    [0.0, COLORS["danger"]],   # Red (low weight)
                    [0.5, "#FFFFFF"],          # White (middle)
                    [1.0, COLORS["success"]], # Green (high weight)
                ]
                
                z_data = sens_df[ticker_cols].T.values
                fig.add_trace(
                    go.Heatmap(
                        z=z_data,
                        x=sens_df[variation_col].values * 100,  # Convert to %
                        y=ticker_cols,
                        colorscale=custom_colorscale,
                        zmin=0.0,  # Minimum weight
                        zmax=1.0,  # Maximum weight
                        colorbar=dict(title="Weight"),
                        hovertemplate=(
                            "Variation: %{x:.1f}%<br>"
                            "Ticker: %{y}<br>"
                            "Weight: %{z:.2%}<extra></extra>"
                        ),
                    )
                )
                
                fig.update_layout(
                    title="Weight Sensitivity Heatmap",
                    xaxis_title="Parameter Variation (%)",
                    yaxis_title="Asset",
                    height=400,
                    template="plotly_dark",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.caption(
                    "**How to read heatmap:** "
                    "Each horizontal bar shows the asset weight at different "
                    "parameter variation values. "
                    "Red = low weight, Green = high weight. "
                    "If a bar is the same color across its width - weight does not change."
                )
                
                # Line chart for individual assets
                st.markdown("**Weight Changes by Asset**")
                
                # Sort tickers by sensitivity (most sensitive first)
                sorted_tickers = sorted(
                    sensitivity_info,
                    key=lambda x: x["range"],
                    reverse=True,
                )
                ticker_options = [info["ticker"] for info in sorted_tickers]
                
                selected_tickers = st.multiselect(
                    "Select Assets to Display",
                    options=ticker_options,
                    default=ticker_options[:min(6, len(ticker_options))],
                    key="opt_sensitivity_tickers",
                    help=(
                        "Select assets to display. "
                        "Assets are sorted by sensitivity "
                        "(most sensitive at top)."
                    ),
                )
                
                if selected_tickers:
                    fig_lines = go.Figure()
                    
                    # Use colors from unified palette
                    colors = [
                        COLORS["primary"],    # Purple
                        COLORS["secondary"],  # Blue
                        COLORS["success"],    # Green
                        COLORS["warning"],    # Orange
                        COLORS["additional"], # Yellow
                        COLORS["danger"],     # Red
                    ]
                    
                    for idx, ticker in enumerate(selected_tickers):
                        weights = sens_df[ticker].values * 100
                        color = colors[idx % len(colors)]
                        
                        # Calculate if line is flat
                        weight_range = weights.max() - weights.min()
                        is_flat = weight_range < 0.1  # Less than 0.1% variation
                        
                        fig_lines.add_trace(
                            go.Scatter(
                                x=sens_df[variation_col].values * 100,
                                y=weights,
                                mode="lines+markers",
                                name=f"{ticker} (Δ={weight_range:.2f}%)",
                                line=dict(width=3, color=color),
                                marker=dict(size=6, color=color),
                                hovertemplate=(
                                    f"<b>{ticker}</b><br>"
                                    "Variation: %{x:.1f}%<br>"
                                    "Weight: %{y:.2f}%<br>"
                                    f"Range: {weight_range:.2f}%<extra></extra>"
                                ),
                            )
                        )
                    
                    # Add horizontal reference lines for constraints if available
                    if constraints:
                        min_weight = constraints.get("min_weight")
                        max_weight = constraints.get("max_weight")
                        
                        if min_weight is not None:
                            fig_lines.add_hline(
                                y=min_weight * 100,
                                line_dash="dash",
                                line_color="rgba(255, 255, 255, 0.3)",
                                annotation_text=f"Min: {min_weight:.1%}",
                                annotation_position="right",
                            )
                        if max_weight is not None:
                            fig_lines.add_hline(
                                y=max_weight * 100,
                                line_dash="dash",
                                line_color="rgba(255, 255, 255, 0.3)",
                                annotation_text=f"Max: {max_weight:.1%}",
                                annotation_position="right",
                            )
                    
                    fig_lines.update_layout(
                        title="Weight Sensitivity by Asset",
                        xaxis_title="Parameter Variation (%)",
                        yaxis_title="Weight (%)",
                        height=400,
                        template="plotly_dark",
                        hovermode="x unified",
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02,
                        ),
                    )
                    
                    st.plotly_chart(fig_lines, use_container_width=True)
                    
                    # Add explanation
                    st.caption(
                        "**How to read chart:** "
                        "Horizontal lines mean the asset weight does not change "
                        "when the parameter varies (stable optimization). "
                        "Sloped lines show weight sensitivity to the parameter. "
                        "Number in parentheses (Δ=...) shows the weight change range."
                    )
                else:
                    st.info("Select assets to display on the chart.")
                
                # Summary statistics
                st.markdown("**Sensitivity Summary**")
                summary_data = []
                for info in sorted_tickers:
                    summary_data.append({
                        "Ticker": info["ticker"],
                        "Min Weight": f"{info['min']:.2%}",
                        "Max Weight": f"{info['max']:.2%}",
                        "Range": f"{info['range']:.2%}",
                        "Std Dev": f"{info['std']:.2%}",
                        "Sensitivity": (
                            "High" if info["range"] > 0.05
                            else "Medium" if info["range"] > 0.01
                            else "Low"
                        ),
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )
                
                st.caption(
                    "**Interpretation:** "
                    "**Range** shows how much the asset weight changes when the parameter varies. "
                    "**High sensitivity** (>5%) = weight changes significantly, "
                    "**Low sensitivity** (<1%) = weight practically does not change. "
                    "Assets are sorted by sensitivity (most sensitive at top)."
                )
                
                # Interpretation: Sensitivity analysis
                interpretation = _interpret_sensitivity_analysis(sens_results)
                if interpretation:
                    st.info(interpretation)
    
    # Apply optimization
    st.subheader("Apply Optimization")
    
    st.warning(
        "⚠️ Applying optimization will update your portfolio weights. "
        "This action cannot be undone."
    )
    
    if st.button("Apply Optimization to Portfolio", type="primary"):
        # TODO: Implement apply optimization
        st.info("Apply optimization feature will be implemented soon.")


if __name__ == "__main__":
    render_optimization_page()

