"""Forecasting page for price and returns prediction."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.forecasting_service import ForecastingService
from services.portfolio_service import PortfolioService
from services.data_service import DataService
from core.exceptions import CalculationError, InsufficientDataError
from streamlit_app.utils.chart_config import COLORS
from streamlit_app.components.forecast_charts import (
    plot_forecast_comparison,
    plot_individual_forecast,
    plot_forecast_quality,
    plot_residuals,
)

logger = logging.getLogger(__name__)


def render_forecasting_page() -> None:
    """Render forecasting page."""
    st.title("Price & Returns Forecasting")
    st.markdown(
        "Forecast future prices and returns using various forecasting models. "
        "Compare multiple methods and evaluate forecast quality."
    )

    # Initialize services
    forecasting_service = ForecastingService()
    portfolio_service = PortfolioService()
    data_service = DataService()

    # Asset/Portfolio selection
    st.subheader("Asset Selection")

    forecast_type = st.radio(
        "Forecast Type",
        ["Single Asset", "Portfolio"],
        horizontal=True,
        key="forecast_type",
    )

    selected_ticker = None
    selected_portfolio_id = None

    if forecast_type == "Single Asset":
        # Get available tickers from portfolios
        portfolios = portfolio_service.list_portfolios()
        all_tickers = set()
        for portfolio in portfolios:
            positions = portfolio.get_all_positions()
            for pos in positions:
                if pos.ticker != "CASH":
                    all_tickers.add(pos.ticker)

        if not all_tickers:
            st.warning("No tickers found in portfolios. Please create a portfolio first.")
            return

        selected_ticker = st.selectbox(
            "Select Ticker",
            sorted(list(all_tickers)),
            key="forecast_ticker",
        )
    else:
        # Portfolio selection
        portfolios = portfolio_service.list_portfolios()
        if not portfolios:
            st.warning("No portfolios found. Please create a portfolio first.")
            return

        portfolio_names = [p.name for p in portfolios]
        selected_name = st.selectbox(
            "Select Portfolio",
            portfolio_names,
            key="forecast_portfolio",
        )

        selected_portfolio = next(
            p for p in portfolios if p.name == selected_name
        )
        selected_portfolio_id = selected_portfolio.id

    # Forecasting parameters
    st.subheader("Forecasting Parameters")

    col1, col2 = st.columns(2)

    with col1:
        default_end = date.today()
        default_start = default_end - timedelta(days=365)

        start_date = st.date_input(
            "Training Start Date",
            value=default_start,
            max_value=date.today(),
            key="forecast_start_date",
        )

    with col2:
        end_date = st.date_input(
            "Training End Date",
            value=default_end,
            min_value=start_date,
            max_value=date.today(),
            key="forecast_end_date",
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # Out-of-Sample Testing
    st.subheader("Out-of-Sample Testing")

    use_out_of_sample = st.checkbox(
        "Use Out-of-Sample Testing",
        value=True,
        key="forecast_out_of_sample",
        help=(
            "If enabled, model is trained on period BEFORE validation dates, "
            "and forecast quality is evaluated on validation period. "
            "This prevents overfitting and shows real forecast accuracy."
        ),
    )

    training_ratio = 0.3
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
            index=0,
            key="forecast_training_window",
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
            f"Training: "
            f"{training_start.strftime('%Y-%m-%d')} → "
            f"{start_date.strftime('%Y-%m-%d')} "
            f"({training_days} days)"
        )
        st.write(
            f"Validation: "
            f"{start_date.strftime('%Y-%m-%d')} → "
            f"{end_date.strftime('%Y-%m-%d')} "
            f"({analysis_days} days)"
        )

    # Forecast horizon
    st.subheader("Forecast Horizon")

    forecast_horizon_options = {
        "1 Day": 1,
        "1 Week (5 days)": 5,
        "2 Weeks (10 days)": 10,
        "1 Month (21 days)": 21,
        "3 Months (63 days)": 63,
        "6 Months (126 days)": 126,
        "1 Year (252 days)": 252,
        "Custom": None,
    }

    horizon_choice = st.selectbox(
        "Forecast Horizon",
        options=list(forecast_horizon_options.keys()),
        index=3,  # 1 Month by default
        key="forecast_horizon_choice",
    )

    if horizon_choice == "Custom":
        forecast_days = st.number_input(
            "Days Ahead",
            min_value=1,
            max_value=1000,
            value=21,
            key="forecast_custom_days",
        )
    else:
        forecast_days = forecast_horizon_options[horizon_choice]

    st.divider()

    # Method selection
    st.subheader("Forecasting Methods")

    method_categories = st.tabs([
        "Classical",
        "Machine Learning",
        "Deep Learning",
        "Simple",
        "Ensemble",
    ])

    selected_methods = []
    method_params = {}

    # Classical tab
    with method_categories[0]:
        st.markdown("**Classical Time Series Models**")

        use_arima = st.checkbox(
            "ARIMA",
            value=False,
            key="method_arima",
            help="AutoRegressive Integrated Moving Average - for trend forecasting",
        )

        if use_arima:
            selected_methods.append("arima")
            with st.expander("ARIMA Parameters", expanded=False):
                auto_arima = st.checkbox(
                    "Auto ARIMA (auto-select best parameters)",
                    value=True,
                    key="arima_auto",
                )
                if not auto_arima:
                    arima_p = st.slider("p (AR order)", 0, 5, 1, key="arima_p")
                    arima_d = st.slider("d (Differencing)", 0, 2, 1, key="arima_d")
                    arima_q = st.slider("q (MA order)", 0, 5, 1, key="arima_q")
                    method_params["arima"] = {
                        "auto": False,
                        "p": arima_p,
                        "d": arima_d,
                        "q": arima_q,
                    }
                else:
                    method_params["arima"] = {"auto": True}

        use_garch = st.checkbox(
            "GARCH",
            value=False,
            key="method_garch",
            help="Generalized Autoregressive Conditional Heteroskedasticity - for volatility forecasting",
        )

        if use_garch:
            selected_methods.append("garch")
            with st.expander("GARCH Parameters", expanded=False):
                garch_p = st.slider("GARCH p (lag order)", 1, 3, 1, key="garch_p")
                garch_q = st.slider("GARCH q (lag order)", 1, 3, 1, key="garch_q")
                method_params["garch"] = {
                    "p": garch_p,
                    "q": garch_q,
                }

        use_arima_garch = st.checkbox(
            "ARIMA-GARCH",
            value=False,
            key="method_arima_garch",
            help="Combined model: ARIMA for mean returns + GARCH for volatility",
        )

        if use_arima_garch:
            selected_methods.append("arima_garch")
            with st.expander("ARIMA-GARCH Parameters", expanded=False):
                arima_garch_auto = st.checkbox(
                    "Auto ARIMA (auto-select best parameters)",
                    value=True,
                    key="arima_garch_auto",
                )
                if not arima_garch_auto:
                    arima_garch_p = st.slider("ARIMA p (AR order)", 0, 5, 1, key="arima_garch_p")
                    arima_garch_d = st.slider("ARIMA d (Differencing)", 0, 2, 1, key="arima_garch_d")
                    arima_garch_q = st.slider("ARIMA q (MA order)", 0, 5, 1, key="arima_garch_q")
                    arima_garch_garch_p = st.slider("GARCH p (lag order)", 1, 3, 1, key="arima_garch_garch_p")
                    arima_garch_garch_q = st.slider("GARCH q (lag order)", 1, 3, 1, key="arima_garch_garch_q")
                    method_params["arima_garch"] = {
                        "auto_arima": False,
                        "arima_p": arima_garch_p,
                        "arima_d": arima_garch_d,
                        "arima_q": arima_garch_q,
                        "garch_p": arima_garch_garch_p,
                        "garch_q": arima_garch_garch_q,
                    }
                else:
                    arima_garch_garch_p = st.slider("GARCH p (lag order)", 1, 3, 1, key="arima_garch_garch_p_auto")
                    arima_garch_garch_q = st.slider("GARCH q (lag order)", 1, 3, 1, key="arima_garch_garch_q_auto")
                    method_params["arima_garch"] = {
                        "auto_arima": True,
                        "garch_p": arima_garch_garch_p,
                        "garch_q": arima_garch_garch_q,
                    }

    # Machine Learning tab
    with method_categories[1]:
        st.markdown("**Machine Learning Models**")

        use_xgboost = st.checkbox(
            "XGBoost",
            value=False,
            key="method_xgboost",
            help="Extreme Gradient Boosting - high accuracy, requires features",
        )

        if use_xgboost:
            selected_methods.append("xgboost")
            with st.expander("XGBoost Parameters", expanded=False):
                xgb_n_estimators = st.slider(
                    "Number of Trees", 50, 500, 100, key="xgb_n_estimators"
                )
                xgb_max_depth = st.slider("Max Depth", 3, 10, 6, key="xgb_max_depth")
                xgb_learning_rate = st.slider(
                    "Learning Rate", 0.01, 0.3, 0.1, key="xgb_learning_rate"
                )
                use_technical_features = st.checkbox(
                    "Include Technical Indicators", True, key="xgb_technical"
                )
                method_params["xgboost"] = {
                    "n_estimators": xgb_n_estimators,
                    "max_depth": xgb_max_depth,
                    "learning_rate": xgb_learning_rate,
                    "use_technical_features": use_technical_features,
                }

        use_random_forest = st.checkbox(
            "Random Forest",
            value=False,
            key="method_random_forest",
            help="Ensemble of decision trees - robust to overfitting",
        )

        if use_random_forest:
            selected_methods.append("random_forest")
            with st.expander("Random Forest Parameters", expanded=False):
                rf_n_estimators = st.slider(
                    "Number of Trees", 50, 300, 100, key="rf_n_estimators"
                )
                rf_max_depth = st.slider("Max Depth", 5, 20, 10, key="rf_max_depth")
                method_params["random_forest"] = {
                    "n_estimators": rf_n_estimators,
                    "max_depth": rf_max_depth,
                }

        use_svm = st.checkbox(
            "SVM/SVR",
            value=False,
            key="method_svm",
            help="Support Vector Regression - good for non-linear patterns",
        )

        if use_svm:
            selected_methods.append("svm")
            with st.expander("SVM Parameters", expanded=False):
                svm_c = st.slider("C (Regularization)", 0.1, 100.0, 1.0, key="svm_c")
                svm_epsilon = st.slider("Epsilon (Tolerance)", 0.001, 0.1, 0.01, key="svm_epsilon")
                svm_kernel = st.selectbox(
                    "Kernel",
                    ["rbf", "linear", "poly", "sigmoid"],
                    index=0,
                    key="svm_kernel",
                )
                method_params["svm"] = {
                    "C": svm_c,
                    "epsilon": svm_epsilon,
                    "kernel": svm_kernel,
                }

    # Deep Learning tab
    with method_categories[2]:
        st.markdown("**Deep Learning Models**")
        st.warning("Deep Learning models require more computation time and data.")
        
        use_ssa_maemd_tcn = st.checkbox(
            "SSA-MAEMD-TCN (Hybrid with Decomposition)",
            value=False,
            key="method_ssa_maemd_tcn",
            help="Hybrid model: SSA for denoising + EMD for decomposition + TCN for forecasting",
        )

        if use_ssa_maemd_tcn:
            selected_methods.append("ssa_maemd_tcn")
            st.info(
                "SSA-MAEMD-TCN: Advanced hybrid model combining decomposition "
                "and deep learning. Requires PyEMD library (pip install PyEMD)."
            )

        use_lstm = st.checkbox(
            "LSTM",
            value=False,
            key="method_lstm",
            help="Long Short-Term Memory - captures long-term dependencies",
        )

        if use_lstm:
            selected_methods.append("lstm")
            
        use_tcn = st.checkbox(
            "TCN",
            value=False,
            key="method_tcn",
            help="Temporal Convolutional Network - efficient for time series",
        )

        if use_tcn:
            selected_methods.append("tcn")

    # Simple tab
    with method_categories[3]:
        st.markdown("**Simple & Fast Models**")

        use_prophet = st.checkbox(
            "Prophet (Meta/Facebook)",
            value=False,  # Default disabled
            key="method_prophet",
            help="Fast, simple forecasting - good for seasonal patterns",
        )

        if use_prophet:
            selected_methods.append("prophet")
            with st.expander("Prophet Parameters", expanded=False):
                prophet_growth = st.selectbox(
                    "Growth Model",
                    ["linear", "logistic"],
                    key="prophet_growth",
                )
                prophet_seasonality = st.checkbox(
                    "Include Seasonality", True, key="prophet_seasonality"
                )
                prophet_holidays = st.checkbox(
                    "Include Holidays (US)", False, key="prophet_holidays"
                )
                method_params["prophet"] = {
                    "growth": prophet_growth,
                    "seasonality": prophet_seasonality,
                    "holidays": prophet_holidays,
                }

    # Ensemble tab
    with method_categories[4]:
        st.markdown("**Ensemble & Hybrid Models**")
        st.info(
            "Ensemble models combine predictions from multiple methods "
            "for improved accuracy and robustness."
        )

        use_ensemble = st.checkbox(
            "Optimized Ensemble",
            value=False,
            key="method_ensemble",
            help="Weighted combination of selected models (best MAPE)",
        )

        if use_ensemble:
            st.markdown("**Select models to include in ensemble:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical Models:**")
                ensemble_arima = st.checkbox("Include ARIMA", True, key="ensemble_arima")
                ensemble_garch = st.checkbox("Include GARCH", False, key="ensemble_garch")
                ensemble_arima_garch = st.checkbox("Include ARIMA-GARCH", False, key="ensemble_arima_garch")
                ensemble_prophet = st.checkbox("Include Prophet", True, key="ensemble_prophet")
            
            with col2:
                st.markdown("**ML & Deep Learning Models:**")
                ensemble_xgboost = st.checkbox("Include XGBoost", True, key="ensemble_xgboost")
                ensemble_random_forest = st.checkbox("Include Random Forest", False, key="ensemble_random_forest")
                ensemble_svm = st.checkbox("Include SVM/SVR", False, key="ensemble_svm")
                ensemble_lstm = st.checkbox("Include LSTM", False, key="ensemble_lstm")
                ensemble_tcn = st.checkbox("Include TCN", False, key="ensemble_tcn")

    # Run forecast button
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_forecast_button = st.button(
            "Run Forecasts",
            type="primary",
            use_container_width=True,
            key="run_forecast",
        )

    # Results section
    if run_forecast_button:
        # Clear saved results when starting new forecast
        if "saved_forecast_results" in st.session_state:
            del st.session_state["saved_forecast_results"]
    
    # Check if we have saved forecast results and display them first (only if no new forecast is running)
    if "saved_forecast_results" in st.session_state and st.session_state["saved_forecast_results"] and not run_forecast_button:
        _display_forecast_results(
            st.session_state["saved_forecast_results"],
            st.session_state.get("saved_results_start"),
            st.session_state.get("saved_results_end"),
            st.session_state.get("saved_results_out_of_sample", False),
            ticker=st.session_state.get("saved_results_ticker"),
            portfolio_id=st.session_state.get("saved_results_portfolio_id"),
            training_start=st.session_state.get("saved_results_training_start"),
            forecast_end=st.session_state.get("saved_results_forecast_end"),
            chart_suffix="saved",
        )
        st.divider()
    
    # Results section
    if run_forecast_button:
        if not selected_methods:
            st.error("Please select at least one forecasting method.")
            st.stop()

        if forecast_type == "Single Asset" and not selected_ticker:
            st.error("Please select a ticker.")
            st.stop()

        if forecast_type == "Portfolio" and not selected_portfolio_id:
            st.error("Please select a portfolio.")
            st.stop()

        # Cancel any previous forecast run by setting a new run ID
        import time
        current_run_id = time.time()
        st.session_state["forecast_run_id"] = current_run_id
        st.session_state["forecast_cancelled"] = False

        st.success(f"Running forecasts with {len(selected_methods)} method(s)...")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        forecasts = {}

        try:
            for idx, method in enumerate(selected_methods):
                # Check if this run was cancelled (new run started)
                if st.session_state.get("forecast_run_id") != current_run_id:
                    st.warning("Previous forecast run was cancelled. Starting new run...")
                    break
                
                if st.session_state.get("forecast_cancelled", False):
                    st.warning("Forecast run was cancelled.")
                    break
                
                status_text.text(f"Running {method}...")
                progress_bar.progress((idx + 1) / len(selected_methods))

                # Check again before starting forecast (in case it was cancelled during previous check)
                if st.session_state.get("forecast_run_id") != current_run_id:
                    st.warning(f"Forecast run cancelled before starting {method}.")
                    break
                
                try:
                    if forecast_type == "Single Asset":
                        result = forecasting_service.forecast_asset(
                            ticker=selected_ticker,
                            start_date=start_date,
                            end_date=end_date,
                            horizon=forecast_days,
                            method=method,
                            method_params=method_params.get(method),
                            out_of_sample=use_out_of_sample,
                            training_ratio=training_ratio,
                        )
                    else:
                        result = forecasting_service.forecast_portfolio(
                            portfolio_id=selected_portfolio_id,
                            start_date=start_date,
                            end_date=end_date,
                            horizon=forecast_days,
                            method=method,
                            method_params=method_params.get(method),
                            out_of_sample=use_out_of_sample,
                            training_ratio=training_ratio,
                        )

                    # Check again after forecast completes
                    if st.session_state.get("forecast_run_id") != current_run_id:
                        st.warning(f"Forecast run cancelled after {method} completed. Results discarded.")
                        break

                    if result is not None:
                        forecasts[method] = result
                    else:
                        forecasts[method] = {
                            "success": False,
                            "message": "Forecast returned None",
                        }

                except Exception as e:
                    # Check if cancelled during exception
                    if st.session_state.get("forecast_run_id") != current_run_id:
                        st.warning(f"Forecast run cancelled during {method} execution.")
                        break
                    
                    logger.error(f"Error running {method} forecast: {e}", exc_info=True)
                    st.warning(f"Failed to run {method}: {str(e)}")
                    forecasts[method] = {
                        "success": False,
                        "message": str(e),
                    }

            # Final check before creating ensemble - if new run started, cancel this one
            if st.session_state.get("forecast_run_id") != current_run_id:
                st.warning("Forecast run was cancelled. Skipping ensemble creation and clearing results.")
                forecasts = {}  # Clear results to prevent showing old data
            
            # Create ensemble if requested (only if run wasn't cancelled)
            if use_ensemble and st.session_state.get("forecast_run_id") == current_run_id:
                try:
                    ensemble_methods = []
                    
                    # Classical models
                    if ensemble_arima and "arima" in forecasts:
                        ensemble_methods.append("arima")
                    if ensemble_garch and "garch" in forecasts:
                        ensemble_methods.append("garch")
                    if ensemble_arima_garch and "arima_garch" in forecasts:
                        ensemble_methods.append("arima_garch")
                    elif ensemble_arima_garch and "arima-garch" in forecasts:
                        ensemble_methods.append("arima-garch")
                    if ensemble_prophet and "prophet" in forecasts:
                        ensemble_methods.append("prophet")
                    
                    # ML models
                    if ensemble_xgboost and "xgboost" in forecasts:
                        ensemble_methods.append("xgboost")
                    if ensemble_random_forest and "random_forest" in forecasts:
                        ensemble_methods.append("random_forest")
                    elif ensemble_random_forest and "randomforest" in forecasts:
                        ensemble_methods.append("randomforest")
                    if ensemble_svm and "svm" in forecasts:
                        ensemble_methods.append("svm")
                    elif ensemble_svm and "svr" in forecasts:
                        ensemble_methods.append("svr")
                    
                    # Deep learning models
                    if ensemble_lstm and "lstm" in forecasts:
                        ensemble_methods.append("lstm")
                    if ensemble_tcn and "tcn" in forecasts:
                        ensemble_methods.append("tcn")
                    # SSA-MAEMD-TCN can be included automatically if available
                    if "ssa_maemd_tcn" in forecasts:
                        ensemble_methods.append("ssa_maemd_tcn")
                    elif "ssa-maemd-tcn" in forecasts:
                        ensemble_methods.append("ssa-maemd-tcn")

                    if len(ensemble_methods) >= 2:
                        ensemble_forecasts = {
                            m: forecasts[m] for m in ensemble_methods
                        }
                        ensemble_result = forecasting_service.create_ensemble(
                            ensemble_forecasts,
                            method="weighted_average",
                        )
                        forecasts["ensemble"] = ensemble_result
                    elif len(ensemble_methods) == 1:
                        st.warning(
                            f"Ensemble requires at least 2 models, "
                            f"but only {ensemble_methods[0]} is selected and available."
                        )
                    else:
                        st.warning(
                            "No selected models are available for ensemble. "
                            "Make sure selected models ran successfully."
                        )
                except Exception as e:
                    logger.error(f"Error creating ensemble: {e}", exc_info=True)
                    st.warning(f"Failed to create ensemble: {str(e)}")

            progress_bar.empty()
            status_text.empty()

            # Calculate forecast end date
            # For out-of-sample: forecast should extend from end_date (validation end) to end_date + horizon
            # For regular: forecast extends from last historical date to last historical date + horizon
            forecast_end_date = None
            if use_out_of_sample:
                # Forecast period starts after validation period ends
                forecast_end_date = end_date + timedelta(days=forecast_days)
            else:
                # Get the last forecast date from any successful forecast
                if forecasts:
                    for method, forecast_data in forecasts.items():
                        if forecast_data and isinstance(forecast_data, dict) and forecast_data.get("success", False):
                            if "forecast_dates" in forecast_data and forecast_data["forecast_dates"]:
                                forecast_dates = pd.to_datetime(forecast_data["forecast_dates"])
                                if len(forecast_dates) > 0:
                                    forecast_end_date = forecast_dates[-1]
                                    break
            
            # Save forecasts to session state (use different keys to avoid conflicts with widget keys)
            # Use completely different key names that don't contain widget key substrings
            st.session_state["saved_forecast_results"] = forecasts
            st.session_state["saved_results_start"] = start_date
            st.session_state["saved_results_end"] = end_date
            st.session_state["saved_results_training_start"] = training_start if use_out_of_sample else None
            st.session_state["saved_results_forecast_end"] = forecast_end_date
            st.session_state["saved_results_out_of_sample"] = use_out_of_sample
            st.session_state["saved_results_ticker"] = selected_ticker if forecast_type == "Single Asset" else None
            st.session_state["saved_results_portfolio_id"] = selected_portfolio_id if forecast_type == "Portfolio" else None

            # Display results
            if forecasts:
                _display_forecast_results(
                    forecasts,
                    start_date,
                    end_date,
                    use_out_of_sample,
                    ticker=selected_ticker if forecast_type == "Single Asset" else None,
                    portfolio_id=selected_portfolio_id if forecast_type == "Portfolio" else None,
                    training_start=training_start if use_out_of_sample else None,
                    forecast_end=forecast_end_date,
                )

        except Exception as e:
            logger.error(f"Error running forecasts: {e}", exc_info=True)
            st.error(f"Error running forecasts: {str(e)}")


def _display_forecast_results(
    forecasts: Dict[str, Dict],
    start_date: date,
    end_date: date,
    use_out_of_sample: bool,
    ticker: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    training_start: Optional[date] = None,
    forecast_end: Optional[pd.Timestamp] = None,
    chart_suffix: Optional[str] = None,
) -> None:
    """Display forecast results."""
    # Generate unique suffix for chart keys if not provided
    if chart_suffix is None:
        import time
        chart_suffix = str(int(time.time() * 1000))  # Use timestamp for uniqueness
    
    st.divider()
    st.subheader("Forecast Results")

    results_tabs = st.tabs([
        "Forecasts Comparison",
        "Individual Forecasts",
        "Forecast Quality",
        "Detailed Analysis",
    ])

    # Filter successful forecasts
    successful_forecasts = {
        k: v for k, v in forecasts.items() 
        if v is not None and isinstance(v, dict) and v.get("success", False)
    }
    
    # Save successful forecasts to session state for persistence
    st.session_state["successful_forecasts"] = successful_forecasts

    if not successful_forecasts:
        st.error("No successful forecasts to display.")
        return

    # Fetch historical data for comparison
    data_service = DataService()
    historical_dates = None
    historical_values = None

    try:
        if ticker:
            # Fetch historical prices for single asset
            # Use training_start if available, otherwise start_date
            chart_start_date = training_start if (training_start and use_out_of_sample) else start_date
            hist_data = data_service.fetch_historical_prices(
                ticker=ticker,
                start_date=chart_start_date,  # Start from training period
                end_date=end_date,  # End at validation end (includes validation period)
                use_cache=True,
                save_to_db=False,
            )
            if not hist_data.empty and "Adjusted_Close" in hist_data.columns:
                if "Date" in hist_data.columns:
                    hist_data.set_index("Date", inplace=True)
                    hist_data.index = pd.to_datetime(hist_data.index, errors="coerce")
                    hist_data.index = hist_data.index.tz_localize(None)
                historical_values = hist_data["Adjusted_Close"].sort_index().values
                historical_dates = hist_data.index.sort_values()
        elif portfolio_id:
            # Fetch historical portfolio values
            try:
                portfolio_service = PortfolioService()
                portfolio = portfolio_service.get_portfolio(portfolio_id)
                positions = portfolio.get_all_positions()
                tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
                
                if tickers:
                    # Fetch historical prices for all tickers
                    # Use training_start if available, otherwise start_date
                    chart_start_date = training_start if (training_start and use_out_of_sample) else start_date
                    hist_price_data = data_service.fetch_bulk_prices(
                        tickers=tickers,
                        start_date=chart_start_date,  # Start from training period
                        end_date=end_date,  # End at validation end (includes validation period)
                        use_cache=True,
                        save_to_db=False,
                    )
                    
                    if not hist_price_data.empty:
                        # Convert to pivot format
                        if "Ticker" in hist_price_data.columns and "Adjusted_Close" in hist_price_data.columns:
                            if "Date" in hist_price_data.columns:
                                hist_price_data["Date"] = pd.to_datetime(hist_price_data["Date"], errors="coerce")
                                hist_price_data["Date"] = hist_price_data["Date"].dt.tz_localize(None)
                                hist_pivot = hist_price_data.pivot_table(
                                    index="Date",
                                    columns="Ticker",
                                    values="Adjusted_Close",
                                    aggfunc="last",
                                )
                                hist_price_data = hist_pivot
                        
                        # Calculate portfolio values over time
                        from services.forecasting_service import ForecastingService
                        forecasting_service = ForecastingService()
                        portfolio_prices = forecasting_service.calculate_portfolio_prices(
                            hist_price_data, positions
                        )
                        
                        if len(portfolio_prices) > 0:
                            historical_values = portfolio_prices.sort_index().values
                            historical_dates = portfolio_prices.index.sort_values()
            except Exception as e:
                logger.warning(f"Could not fetch portfolio historical data: {e}")
    except Exception as e:
        logger.warning(f"Could not fetch historical data: {e}")

    # Get training_start and forecast_end from parameters or session_state
    if training_start is None and "saved_results_training_start" in st.session_state:
        training_start = st.session_state["saved_results_training_start"]
    if forecast_end is None and "saved_results_forecast_end" in st.session_state:
        forecast_end = st.session_state["saved_results_forecast_end"]
    
    # Comparison tab
    with results_tabs[0]:
        st.subheader("All Forecasts Comparison")

        if historical_dates is not None and historical_values is not None:
            validation_start = pd.Timestamp(start_date) if use_out_of_sample else None
            validation_end = pd.Timestamp(end_date) if use_out_of_sample else None
            training_start_ts = pd.Timestamp(training_start) if training_start else None

            # Use successful_forecasts from session_state if available
            if "successful_forecasts" in st.session_state and st.session_state["successful_forecasts"]:
                forecasts_to_plot = st.session_state["successful_forecasts"]
            else:
                forecasts_to_plot = successful_forecasts
            
            # Get forecast_end from session_state if available
            forecast_end_ts = None
            if "saved_results_forecast_end" in st.session_state and st.session_state["saved_results_forecast_end"]:
                forecast_end_ts = pd.Timestamp(st.session_state["saved_results_forecast_end"])
            elif forecast_end:
                forecast_end_ts = pd.Timestamp(forecast_end)
                
            fig = plot_forecast_comparison(
                historical_dates=historical_dates,
                historical_values=historical_values,
                forecasts=forecasts_to_plot,
                validation_start=validation_start,
                validation_end=validation_end,
                training_start=training_start_ts,
                forecast_end=forecast_end_ts,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"forecast_comparison_chart_with_history_{chart_suffix}")
        else:
            # Fallback: just show forecasts without historical
            fig = go.Figure()

            for method_name, forecast_data in successful_forecasts.items():
                if not isinstance(forecast_data, dict):
                    continue
                
                try:
                    # Safely convert forecast dates
                    forecast_dates_raw = forecast_data.get("forecast_dates")
                    if forecast_dates_raw is None:
                        logger.warning(f"Method {method_name}: forecast_dates is None")
                        continue
                    
                    forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
                    if len(forecast_dates) == 0 or forecast_dates.isna().all():
                        logger.warning(f"Method {method_name}: invalid forecast_dates")
                        continue
                    
                    # Normalize timezone
                    if hasattr(forecast_dates, "tz") and forecast_dates.tz is not None:
                        forecast_dates = forecast_dates.tz_localize(None)
                    
                    # Safely convert forecast values
                    forecast_values_raw = forecast_data.get("forecast_values")
                    if forecast_values_raw is None:
                        logger.warning(f"Method {method_name}: forecast_values is None")
                        continue
                    
                    if isinstance(forecast_values_raw, (list, tuple)):
                        forecast_values = np.array(forecast_values_raw, dtype=float)
                    elif isinstance(forecast_values_raw, np.ndarray):
                        forecast_values = forecast_values_raw.astype(float)
                    else:
                        forecast_values = np.array([forecast_values_raw], dtype=float)
                    
                    # Filter invalid values
                    valid_mask = np.isfinite(forecast_values)
                    if not np.any(valid_mask):
                        logger.warning(f"Method {method_name}: no valid forecast_values")
                        continue
                    
                    forecast_values = forecast_values[valid_mask]
                    min_len = min(len(forecast_dates), len(valid_mask))
                    forecast_dates = forecast_dates[:min_len][valid_mask[:min_len]]
                    
                    if len(forecast_dates) == 0 or len(forecast_values) == 0:
                        logger.warning(f"Method {method_name}: empty forecast data after filtering")
                        continue

                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name=f"{method_name} Forecast",
                        line=dict(dash="dash", width=2),
                    ))
                except Exception as e:
                    logger.warning(f"Error plotting {method_name}: {e}", exc_info=True)
                    continue

            fig.update_layout(
                title="Price Forecasts Comparison",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                template="plotly_dark",
            )

            st.plotly_chart(fig, use_container_width=True, key=f"forecast_comparison_chart_no_history_{chart_suffix}")

        # Comparison table
        comparison_data = []
        for method_name, forecast_data in successful_forecasts.items():
            if not isinstance(forecast_data, dict):
                continue
            
            # Safely extract values
            def safe_get_value(data, key, default=0, format_func=lambda x: f"{x:.2f}"):
                """Safely get and format value from dict."""
                value = data.get(key, default)
                try:
                    value_float = float(value) if value is not None else default
                    if not np.isfinite(value_float):
                        return format_func(default)
                    return format_func(value_float)
                except (ValueError, TypeError):
                    return format_func(default)
            
            final_value = safe_get_value(forecast_data, 'final_value', 0, lambda x: f"${x:.2f}")
            change_pct = safe_get_value(forecast_data, 'change_pct', 0, lambda x: f"{x:.2f}%")
            
            # Handle validation_metrics
            validation_metrics = forecast_data.get("validation_metrics")
            if validation_metrics and isinstance(validation_metrics, dict):
                mape = safe_get_value(validation_metrics, 'mape', np.nan, lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A")
                rmse = safe_get_value(validation_metrics, 'rmse', np.nan, lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
            else:
                mape = "N/A"
                rmse = "N/A"
            
            comparison_data.append({
                "Method": method_name,
                "Forecast Value": final_value,
                "Change %": change_pct,
                "MAPE": mape,
                "RMSE": rmse,
            })

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Individual forecasts tab
    with results_tabs[1]:
        st.subheader("Individual Method Forecasts")
        
        with st.expander("ℹ️ Что такое Individual Forecasts?", expanded=False):
            st.markdown("""
            **Individual Forecasts** позволяет детально изучить прогноз одного конкретного метода.
            
            **Что вы увидите:**
            - **График прогноза** с историческими данными
            - **Confidence Intervals (95% CI)** - заштрихованная область, показывающая диапазон возможных значений
              - Верхняя граница (Upper 95% CI) - максимальное ожидаемое значение
              - Нижняя граница (Lower 95% CI) - минимальное ожидаемое значение
              - Чем шире интервал, тем больше неопределенность прогноза
            - **Метрики качества** (MAPE, RMSE) - показывают точность прогноза
            - **Residuals Analysis** - анализ ошибок модели
            """)

        # Use successful_forecasts from session_state if available, otherwise use passed parameter
        if "successful_forecasts" in st.session_state and st.session_state["successful_forecasts"]:
            available_forecasts = st.session_state["successful_forecasts"]
        else:
            available_forecasts = successful_forecasts
        
        # Initialize session state for method selector if not exists
        available_methods = list(available_forecasts.keys())
        if not available_methods:
            st.error("No successful forecasts available.")
            return
            
        # Use unique key with chart_suffix to prevent conflicts
        selector_key = f"individual_method_selector_{chart_suffix}"
        if selector_key not in st.session_state:
            st.session_state[selector_key] = available_methods[0]
        
        # Update if current selection is not in available options
        if st.session_state[selector_key] not in available_methods:
            st.session_state[selector_key] = available_methods[0]
        
        # Get current index
        current_index = 0
        if st.session_state[selector_key] in available_methods:
            current_index = available_methods.index(st.session_state[selector_key])
        
        method_selector = st.selectbox(
            "Select Method to View",
            options=available_methods,
            index=current_index,
            key=selector_key,  # Unique key with suffix
        )

        selected_forecast = available_forecasts[method_selector]

        # Display forecast chart with historical data
        if historical_dates is not None and historical_values is not None:
            validation_start = pd.Timestamp(start_date) if use_out_of_sample else None
            validation_end = pd.Timestamp(end_date) if use_out_of_sample else None
            training_start_ts = pd.Timestamp(training_start) if training_start else None
            
            # Get forecast_end from parameter or session_state
            forecast_end_ts = None
            if forecast_end:
                forecast_end_ts = pd.Timestamp(forecast_end)
            elif "saved_results_forecast_end" in st.session_state and st.session_state["saved_results_forecast_end"]:
                forecast_end_ts = pd.Timestamp(st.session_state["saved_results_forecast_end"])

            fig = plot_individual_forecast(
                historical_dates=historical_dates,
                historical_values=historical_values,
                forecast_data=selected_forecast,
                method_name=method_selector,
                validation_start=validation_start,
                validation_end=validation_end,
                training_start=training_start_ts,
                forecast_end=forecast_end_ts,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"forecast_individual_chart_with_history_{chart_suffix}")
        else:
            # Fallback without historical
            from streamlit_app.utils.chart_config import get_method_color
            
            # Safely convert forecast dates
            try:
                forecast_dates_raw = selected_forecast.get("forecast_dates")
                if forecast_dates_raw is None:
                    forecast_dates = pd.DatetimeIndex([])
                else:
                    forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
                    if hasattr(forecast_dates, "tz") and forecast_dates.tz is not None:
                        forecast_dates = forecast_dates.tz_localize(None)
            except Exception as e:
                logger.warning(f"Error parsing forecast_dates for {method_selector}: {e}")
                forecast_dates = pd.DatetimeIndex([])
            
            # Safely convert forecast values
            try:
                forecast_values_raw = selected_forecast.get("forecast_values")
                if forecast_values_raw is None:
                    forecast_values = np.array([])
                elif isinstance(forecast_values_raw, (list, tuple)):
                    forecast_values = np.array(forecast_values_raw, dtype=float)
                elif isinstance(forecast_values_raw, np.ndarray):
                    forecast_values = forecast_values_raw.astype(float)
                else:
                    forecast_values = np.array([forecast_values_raw], dtype=float)
                
                # Filter invalid values
                valid_mask = np.isfinite(forecast_values)
                if np.any(valid_mask):
                    forecast_values = forecast_values[valid_mask]
                    if len(forecast_dates) > 0:
                        min_len = min(len(forecast_dates), len(valid_mask))
                        forecast_dates = forecast_dates[:min_len][valid_mask[:min_len]]
                else:
                    forecast_values = np.array([])
                    forecast_dates = pd.DatetimeIndex([])
            except Exception as e:
                logger.warning(f"Error parsing forecast_values for {method_selector}: {e}")
                forecast_values = np.array([])
            
            method_color = get_method_color(method_selector)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name=f"{method_selector} Forecast",
                line=dict(color=method_color, width=2),  # Solid line with method color
            ))

            # Add confidence intervals if available
            if "confidence_intervals" in selected_forecast and selected_forecast["confidence_intervals"]:
                ci = selected_forecast["confidence_intervals"]
                if isinstance(ci, dict):
                    # Try different key formats
                    upper_key = None
                    lower_key = None
                    for key in ci.keys():
                        if "upper" in key.lower() and "95" in key:
                            upper_key = key
                        if "lower" in key.lower() and "95" in key:
                            lower_key = key
                    
                    if upper_key and lower_key:
                        try:
                            upper_95 = ci[upper_key]
                            lower_95 = ci[lower_key]
                            
                            # Convert to arrays if needed
                            if isinstance(upper_95, (list, tuple)):
                                upper_95 = np.array(upper_95, dtype=float)
                            elif not isinstance(upper_95, np.ndarray):
                                upper_95 = np.array([upper_95], dtype=float)
                            
                            if isinstance(lower_95, (list, tuple)):
                                lower_95 = np.array(lower_95, dtype=float)
                            elif not isinstance(lower_95, np.ndarray):
                                lower_95 = np.array([lower_95], dtype=float)
                            
                            # Align lengths
                            min_len = min(len(forecast_dates), len(upper_95), len(lower_95))
                            if min_len > 0:
                                fig.add_trace(go.Scatter(
                                    x=forecast_dates[:min_len],
                                    y=upper_95[:min_len],
                                    name="Upper 95% CI",
                                    line=dict(color="rgba(0,0,0,0)", width=0),
                                    showlegend=False,
                                ))
                                fig.add_trace(go.Scatter(
                                    x=forecast_dates[:min_len],
                                    y=lower_95[:min_len],
                                    name="Lower 95% CI",
                                    fill="tonexty",
                                    fillcolor="rgba(191, 159, 251, 0.2)",
                                    line=dict(color="rgba(0,0,0,0)", width=0),
                                ))
                        except Exception as e:
                            logger.warning(f"Error plotting confidence intervals for {method_selector}: {e}")

            fig.update_layout(
                title=f"{method_selector} Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
            )

            st.plotly_chart(fig, use_container_width=True, key=f"forecast_individual_chart_{chart_suffix}")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_value = selected_forecast.get('final_value', 0)
            try:
                final_value = float(final_value) if final_value is not None else 0.0
                if not np.isfinite(final_value):
                    final_value = 0.0
            except (ValueError, TypeError):
                final_value = 0.0
            st.metric("Final Forecast", f"${final_value:.2f}")
        with col2:
            change_pct = selected_forecast.get('change_pct', 0)
            try:
                change_pct = float(change_pct) if change_pct is not None else 0.0
                if not np.isfinite(change_pct):
                    change_pct = 0.0
            except (ValueError, TypeError):
                change_pct = 0.0
            st.metric("Expected Change", f"{change_pct:.2f}%")
        with col3:
            validation_metrics = selected_forecast.get("validation_metrics")
            if validation_metrics and isinstance(validation_metrics, dict):
                mape = validation_metrics.get("mape", np.nan)
                try:
                    mape = float(mape) if mape is not None else np.nan
                    if not np.isfinite(mape):
                        mape = np.nan
                except (ValueError, TypeError):
                    mape = np.nan
                st.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
            else:
                st.metric("MAPE", "N/A")
        with col4:
            validation_metrics = selected_forecast.get("validation_metrics")
            if validation_metrics and isinstance(validation_metrics, dict):
                rmse = validation_metrics.get("rmse", np.nan)
                try:
                    rmse = float(rmse) if rmse is not None else np.nan
                    if not np.isfinite(rmse):
                        rmse = np.nan
                except (ValueError, TypeError):
                    rmse = np.nan
                st.metric("RMSE", f"${rmse:.2f}" if not np.isnan(rmse) else "N/A")
            else:
                st.metric("RMSE", "N/A")

        # Residuals plot if available
        if "residuals" in selected_forecast and selected_forecast["residuals"] is not None:
            st.subheader("Residuals Analysis")
            
            with st.expander("ℹ️ Что такое Residuals Analysis?", expanded=False):
                st.markdown("""
                **Residuals (остатки)** - это разница между фактическими и предсказанными значениями.
                
                **Хорошие residuals:**
                - Случайно распределены вокруг нуля
                - Нет явных трендов или паттернов
                - Постоянная дисперсия (не увеличивается со временем)
                
                **Плохие residuals:**
                - Имеют тренд (растут/падают)
                - Показывают сезонность или циклы
                - Увеличивающаяся дисперсия (гетероскедастичность)
                
                Если residuals показывают паттерны, модель не уловила все закономерности в данных.
                """)
            
            residuals_raw = selected_forecast.get("residuals")
            logger.debug(f"Residuals for {method_selector}: type={type(residuals_raw)}, is None={residuals_raw is None}")
            
            if residuals_raw is not None:
                try:
                    # residuals_raw может быть списком из to_dict()
                    if isinstance(residuals_raw, list):
                        if len(residuals_raw) > 0:
                            residuals = np.array(residuals_raw, dtype=float)
                            logger.debug(f"Converted residuals from list: length={len(residuals)}")
                        else:
                            logger.warning(f"Residuals list is empty for {method_selector}")
                            st.info("Residuals data is empty for this forecast method.")
                            residuals = None
                    else:
                        residuals = np.array(residuals_raw, dtype=float)
                        logger.debug(f"Converted residuals from {type(residuals_raw)}: length={len(residuals)}")
                    
                    if residuals is not None and len(residuals) > 0:
                        method_name = method_selector
                        fig_residuals = plot_residuals(residuals, method_name)
                        st.plotly_chart(fig_residuals, use_container_width=True, key=f"residuals_chart_{method_selector}")
                    else:
                        st.info("No valid residuals data available for this forecast method.")
                except Exception as e:
                    logger.error(f"Error plotting residuals for {method_selector}: {e}", exc_info=True)
                    logger.error(f"Residuals type: {type(residuals_raw)}, value sample: {residuals_raw[:5] if hasattr(residuals_raw, '__getitem__') and len(residuals_raw) > 0 else 'N/A'}")
                    st.warning(f"Could not plot residuals: {e}")
            else:
                logger.debug(f"No residuals key found or residuals is None for {method_selector}")
                st.info("No residuals data available for this forecast method.")

    # Forecast quality tab
    with results_tabs[2]:
        st.subheader("Forecast Quality Metrics")

        if not use_out_of_sample:
            st.warning(
                "Out-of-sample testing is disabled. "
                "Enable it to see forecast quality metrics."
            )
        else:
            # Use successful_forecasts from session_state if available
            if "successful_forecasts" in st.session_state and st.session_state["successful_forecasts"]:
                forecasts_to_use = st.session_state["successful_forecasts"]
            else:
                forecasts_to_use = successful_forecasts
                
            quality_data = []
            for method_name, forecast_data in forecasts_to_use.items():
                if not isinstance(forecast_data, dict):
                    logger.warning(f"Method {method_name}: forecast_data is not a dict (type={type(forecast_data)})")
                    continue
                
                metrics = forecast_data.get("validation_metrics")
                logger.debug(f"Method {method_name}: validation_metrics type={type(metrics)}, value={metrics}")
                
                if metrics and isinstance(metrics, dict):
                    # Extract numeric values, handling NaN and invalid types
                    def safe_get_metric(metrics_dict, key, default=np.nan):
                        """Safely get metric value, handling various types."""
                        value = metrics_dict.get(key, default)
                        if value is None:
                            return np.nan
                        try:
                            value_float = float(value)
                            return value_float if np.isfinite(value_float) else np.nan
                        except (ValueError, TypeError):
                            return np.nan
                    
                    mape = safe_get_metric(metrics, 'mape')
                    rmse = safe_get_metric(metrics, 'rmse')
                    mae = safe_get_metric(metrics, 'mae')
                    direction_accuracy = safe_get_metric(metrics, 'direction_accuracy')
                    r_squared = safe_get_metric(metrics, 'r_squared')
                    
                    quality_data.append({
                        "Method": method_name,
                        "MAPE": f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
                        "RMSE": f"${rmse:.2f}" if not np.isnan(rmse) else "N/A",
                        "MAE": f"${mae:.2f}" if not np.isnan(mae) else "N/A",
                        "Direction Accuracy": f"{direction_accuracy:.1f}%" if not np.isnan(direction_accuracy) else "N/A",
                        "R²": f"{r_squared:.3f}" if not np.isnan(r_squared) else "N/A",
                    })
                else:
                    logger.warning(f"Method {method_name}: validation_metrics is missing or not a dict (type={type(metrics)})")

            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                
                # Display all metrics in a table format
                st.markdown("### Metrics by Method")
                st.dataframe(quality_df, use_container_width=True, hide_index=True)

                # Visualize all metrics - show all graphs one under another
                st.markdown("### Visualizations")
                metrics_to_plot = ["MAPE", "RMSE", "MAE", "Direction Accuracy", "R²"]
                
                for metric in metrics_to_plot:
                    fig_quality = plot_forecast_quality(forecasts_to_use, metric=metric)
                    st.plotly_chart(fig_quality, use_container_width=True, key=f"quality_chart_{metric}")

                # Overall ranking based on all metrics
                st.markdown("### Overall Method Ranking")
                st.markdown("Ranking based on all quality metrics (lower is better for MAPE, RMSE, MAE; higher is better for Direction Accuracy and R²):")
                
                # Calculate composite score
                # Normalize each metric and combine
                numeric_df = quality_df.copy()
                
                # Safely convert string metrics to numeric, handling "N/A"
                def safe_convert_to_float(series, remove_chars=""):
                    """Safely convert series to float, handling N/A values."""
                    result = []
                    for val in series:
                        if val == "N/A" or pd.isna(val):
                            result.append(np.nan)
                        else:
                            try:
                                cleaned = str(val)
                                for char in remove_chars:
                                    cleaned = cleaned.replace(char, "")
                                result.append(float(cleaned))
                            except (ValueError, TypeError):
                                result.append(np.nan)
                    return pd.Series(result)
                
                numeric_df["MAPE_num"] = safe_convert_to_float(numeric_df["MAPE"], "%")
                numeric_df["RMSE_num"] = safe_convert_to_float(numeric_df["RMSE"], "$")
                numeric_df["MAE_num"] = safe_convert_to_float(numeric_df["MAE"], "$")
                numeric_df["Direction_num"] = safe_convert_to_float(numeric_df["Direction Accuracy"], "%")
                numeric_df["R2_num"] = safe_convert_to_float(numeric_df["R²"], "")
                
                # Normalize to 0-1 scale (lower is better for MAPE, RMSE, MAE; higher is better for Direction and R²)
                # For MAPE, RMSE, MAE: score = 1 - (value - min) / (max - min)
                # For Direction: score = (value - min) / (max - min)
                # For R²: handle negative values - normalize to 0-1, but penalize negative values
                
                # Filter out NaN values before calculating min/max
                mape_valid = numeric_df["MAPE_num"].dropna()
                rmse_valid = numeric_df["RMSE_num"].dropna()
                mae_valid = numeric_df["MAE_num"].dropna()
                dir_valid = numeric_df["Direction_num"].dropna()
                r2_valid = numeric_df["R2_num"].dropna()
                
                if len(mape_valid) == 0:
                    max_mape = min_mape = 0
                else:
                    max_mape = float(mape_valid.max())
                    min_mape = float(mape_valid.min())
                
                if len(rmse_valid) == 0:
                    max_rmse = min_rmse = 0
                else:
                    max_rmse = float(rmse_valid.max())
                    min_rmse = float(rmse_valid.min())
                
                if len(mae_valid) == 0:
                    max_mae = min_mae = 0
                else:
                    max_mae = float(mae_valid.max())
                    min_mae = float(mae_valid.min())
                
                if len(dir_valid) == 0:
                    max_dir = min_dir = 0
                else:
                    max_dir = float(dir_valid.max())
                    min_dir = float(dir_valid.min())
                
                if len(r2_valid) == 0:
                    max_r2 = min_r2 = 0
                else:
                    max_r2 = float(r2_valid.max())
                    min_r2 = float(r2_valid.min())
                
                # Normalize metrics with safe division
                mape_range = max_mape - min_mape if max_mape != min_mape and max_mape > 0 else 1.0
                rmse_range = max_rmse - min_rmse if max_rmse != min_rmse and max_rmse > 0 else 1.0
                mae_range = max_mae - min_mae if max_mae != min_mae and max_mae > 0 else 1.0
                dir_range = max_dir - min_dir if max_dir != min_dir and max_dir > 0 else 1.0
                r2_range = max_r2 - min_r2 if max_r2 != min_r2 else 1.0
                
                # Calculate normalized scores (0-1 scale, higher is better)
                # Handle NaN values by setting score to 0.5 (neutral)
                mape_score = np.where(
                    np.isfinite(numeric_df["MAPE_num"]),
                    1 - (numeric_df["MAPE_num"] - min_mape) / mape_range,
                    0.5
                )
                rmse_score = np.where(
                    np.isfinite(numeric_df["RMSE_num"]),
                    1 - (numeric_df["RMSE_num"] - min_rmse) / rmse_range,
                    0.5
                )
                mae_score = np.where(
                    np.isfinite(numeric_df["MAE_num"]),
                    1 - (numeric_df["MAE_num"] - min_mae) / mae_range,
                    0.5
                )
                dir_score = np.where(
                    np.isfinite(numeric_df["Direction_num"]),
                    (numeric_df["Direction_num"] - min_dir) / dir_range,
                    0.5
                )
                # For R²: if negative, penalize; if positive, reward
                r2_score = np.where(
                    np.isfinite(numeric_df["R2_num"]),
                    (numeric_df["R2_num"] - min_r2) / r2_range if r2_range > 0 else 0.5,
                    0.5
                )
                # Clamp R² score to 0-1 range
                r2_score = np.clip(r2_score, 0, 1)
                
                # Calculate composite score (weighted average)
                # Convert to pandas Series for easier handling
                mape_score_series = pd.Series(mape_score, index=numeric_df.index)
                rmse_score_series = pd.Series(rmse_score, index=numeric_df.index)
                mae_score_series = pd.Series(mae_score, index=numeric_df.index)
                dir_score_series = pd.Series(dir_score, index=numeric_df.index)
                r2_score_series = pd.Series(r2_score, index=numeric_df.index)
                
                numeric_df["score"] = (
                    mape_score_series * 0.3 +  # 30% weight for MAPE
                    rmse_score_series * 0.25 +  # 25% weight for RMSE
                    mae_score_series * 0.15 +  # 15% weight for MAE
                    dir_score_series * 0.2 +  # 20% weight for Direction Accuracy
                    r2_score_series * 0.1  # 10% weight for R²
                )
                
                # Ensure score is finite
                numeric_df["score"] = numeric_df["score"].replace([np.inf, -np.inf], 0.5)
                numeric_df["score"] = numeric_df["score"].fillna(0.5)
                
                # Sort by composite score (higher is better)
                ranked_df = numeric_df.sort_values("score", ascending=False)
                
                # Display ranking
                for idx, (_, row) in enumerate(ranked_df.iterrows(), 1):
                    try:
                        score = row.get("score", 0.5)
                        if not np.isfinite(score):
                            score = 0.5
                        score_pct = float(score) * 100
                        method_name = str(row.get('Method', 'Unknown'))
                        st.markdown(f"{idx}. **{method_name}** - Overall Score: {score_pct:.1f}%")
                    except Exception as e:
                        logger.warning(f"Error displaying ranking for row {idx}: {e}")
                        continue
            else:
                st.info(
                    "No validation metrics available. "
                    "This usually means:\n"
                    "- Out-of-sample testing is enabled but validation data is not available\n"
                    "- Forecasts were generated without validation period\n"
                    "- Validation metrics could not be calculated"
                )

    # Detailed analysis tab
    with results_tabs[3]:
        st.subheader("Detailed Forecast Analysis")

        # Use successful_forecasts from session_state if available, otherwise use passed parameter
        if "successful_forecasts" in st.session_state and st.session_state["successful_forecasts"]:
            available_forecasts = st.session_state["successful_forecasts"]
        else:
            available_forecasts = successful_forecasts
        
        # Initialize session state for detail method selector if not exists
        available_methods = list(available_forecasts.keys())
        if not available_methods:
            st.error("No successful forecasts available.")
            return
            
        # Use unique key with chart_suffix to prevent conflicts
        detail_selector_key = f"detail_method_selector_{chart_suffix}"
        if detail_selector_key not in st.session_state:
            st.session_state[detail_selector_key] = available_methods[0]
        
        # Update if current selection is not in available options
        if st.session_state[detail_selector_key] not in available_methods:
            st.session_state[detail_selector_key] = available_methods[0]
        
        # Get current index
        current_index = 0
        if st.session_state[detail_selector_key] in available_methods:
            current_index = available_methods.index(st.session_state[detail_selector_key])
        
        detail_method = st.selectbox(
            "Select Method to View",
            options=available_methods,
            index=current_index,
            key=detail_selector_key,  # Unique key with suffix
        )

        selected_detail = available_forecasts[detail_method]

        # Display model info in readable format
        if "model_info" in selected_detail and selected_detail["model_info"]:
            st.markdown("### Model Information")
            model_info = selected_detail["model_info"]
            
            col1, col2 = st.columns(2)
            with col1:
                # Safely display model info
                if "order" in model_info and model_info["order"] is not None:
                    try:
                        order_str = str(model_info["order"])
                        st.metric("ARIMA Order", order_str)
                    except Exception:
                        pass
                if "auto" in model_info:
                    try:
                        auto_val = model_info["auto"]
                        st.metric("Auto Parameters", "Yes" if auto_val else "No")
                    except Exception:
                        pass
                if "use_returns" in model_info:
                    try:
                        use_returns_val = model_info["use_returns"]
                        st.metric("Used Returns", "Yes" if use_returns_val else "No")
                    except Exception:
                        pass
            with col2:
                if "aic" in model_info and model_info["aic"] is not None:
                    try:
                        aic = float(model_info["aic"])
                        if np.isfinite(aic):
                            st.metric("AIC", f"{aic:.2f}")
                    except (ValueError, TypeError):
                        pass
                if "bic" in model_info and model_info["bic"] is not None:
                    try:
                        bic = float(model_info["bic"])
                        if np.isfinite(bic):
                            st.metric("BIC", f"{bic:.2f}")
                    except (ValueError, TypeError):
                        pass
                if "training_time" in model_info and model_info["training_time"] is not None:
                    try:
                        training_time = float(model_info["training_time"])
                        if np.isfinite(training_time) and training_time >= 0:
                            st.metric("Training Time", f"{training_time:.2f}s")
                    except (ValueError, TypeError):
                        pass
        
        # Forecast summary
        st.markdown("### Forecast Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_dates_detail = selected_detail.get('forecast_dates', [])
            horizon_days = len(forecast_dates_detail) if forecast_dates_detail else 0
            st.metric("Forecast Horizon", f"{horizon_days} days")
        with col2:
            final_value_detail = selected_detail.get('final_value', 0)
            try:
                final_value_detail = float(final_value_detail) if final_value_detail is not None else 0.0
                if not np.isfinite(final_value_detail):
                    final_value_detail = 0.0
            except (ValueError, TypeError):
                final_value_detail = 0.0
            st.metric("Final Value", f"${final_value_detail:.2f}")
        with col3:
            change_pct_detail = selected_detail.get('change_pct', 0)
            try:
                change_pct_detail = float(change_pct_detail) if change_pct_detail is not None else 0.0
                if not np.isfinite(change_pct_detail):
                    change_pct_detail = 0.0
            except (ValueError, TypeError):
                change_pct_detail = 0.0
            st.metric("Change %", f"{change_pct_detail:.2f}%")
        
        # Validation metrics if available
        if "validation_metrics" in selected_detail and selected_detail["validation_metrics"]:
            st.markdown("### Validation Metrics")
            metrics = selected_detail["validation_metrics"]
            if isinstance(metrics, dict):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                def safe_metric_display(key, default=0, format_str="{:.2f}%", prefix=""):
                    """Safely get and display metric."""
                    value = metrics.get(key, default)
                    try:
                        value_float = float(value) if value is not None else default
                        if not np.isfinite(value_float):
                            return f"{prefix}N/A"
                        return f"{prefix}{format_str.format(value_float)}"
                    except (ValueError, TypeError):
                        return f"{prefix}N/A"
                
                with col1:
                    mape_display = safe_metric_display('mape', np.nan, "{:.2f}%")
                    st.metric("MAPE", mape_display)
                with col2:
                    rmse_display = safe_metric_display('rmse', np.nan, "{:.2f}", "$")
                    st.metric("RMSE", rmse_display)
                with col3:
                    mae_display = safe_metric_display('mae', np.nan, "{:.2f}", "$")
                    st.metric("MAE", mae_display)
                with col4:
                    dir_display = safe_metric_display('direction_accuracy', np.nan, "{:.1f}%")
                    st.metric("Direction Accuracy", dir_display)
                with col5:
                    r2_display = safe_metric_display('r_squared', np.nan, "{:.3f}")
                    st.metric("R²", r2_display)

        # Display full forecast data (collapsed by default)
        with st.expander("Raw Forecast Data (JSON)", expanded=False):
            st.json(selected_detail)

