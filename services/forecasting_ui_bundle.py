"""
JSON bundles for Next.js forecasting page (Streamlit forecasting.py parity).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from core.exceptions import ValidationError
from services.data_service import DataService
from services.forecasting_service import ForecastingService
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


def _pivot_bulk_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty:
        return price_data
    if "Ticker" in price_data.columns and "Adjusted_Close" in price_data.columns:
        if "Date" in price_data.columns:
            price_data = price_data.copy()
            price_data["Date"] = pd.to_datetime(price_data["Date"], errors="coerce")
            price_data["Date"] = price_data["Date"].dt.tz_localize(None)
            return price_data.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="last",
            )
    return price_data


def fetch_historical_asset_series(
    data_service: DataService,
    ticker: str,
    chart_start: date,
    end_date: date,
) -> list[dict[str, Any]]:
    hist = data_service.fetch_historical_prices(
        ticker=ticker,
        start_date=chart_start,
        end_date=end_date,
        use_cache=True,
        save_to_db=False,
    )
    if hist.empty or "Adjusted_Close" not in hist.columns:
        return []
    if "Date" in hist.columns:
        hist = hist.copy()
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.set_index("Date")
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    hist.index = hist.index.tz_localize(None)
    s = hist["Adjusted_Close"].sort_index().dropna()
    return [{"date": str(i)[:10], "value": float(v)} for i, v in s.items()]


def fetch_historical_portfolio_series(
    data_service: DataService,
    forecasting_service: ForecastingService,
    portfolio_service: PortfolioService,
    portfolio_id: str,
    chart_start: date,
    end_date: date,
) -> list[dict[str, Any]]:
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    if not portfolio:
        return []
    positions = portfolio.get_all_positions()
    tickers = [p.ticker for p in positions if p.ticker != "CASH"]
    if not tickers:
        return []
    bulk = data_service.fetch_bulk_prices(
        tickers=tickers,
        start_date=chart_start,
        end_date=end_date,
        use_cache=True,
        save_to_db=False,
    )
    if bulk.empty:
        return []
    pivot = _pivot_bulk_prices(bulk)
    if pivot.empty:
        return []
    series = forecasting_service.calculate_portfolio_prices(pivot, positions)
    if len(series) == 0:
        return []
    series = series.sort_index().dropna()
    return [{"date": str(i)[:10], "value": float(v)} for i, v in series.items()]


def merge_comparison_chart_data(
    historical: list[dict[str, Any]],
    forecasts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in historical:
        d = row["date"]
        merged[d] = {"date": d, "historical": row.get("value")}
    for method, fd in forecasts.items():
        if not isinstance(fd, dict) or not fd.get("success"):
            continue
        dates = fd.get("forecast_dates") or []
        vals = fd.get("forecast_values") or []
        for d_raw, v in zip(dates, vals):
            d = str(d_raw)[:10] if d_raw is not None else ""
            if not d:
                continue
            if d not in merged:
                merged[d] = {"date": d}
            try:
                merged[d][method] = float(v) if v is not None else None
            except (TypeError, ValueError):
                merged[d][method] = None
    return sorted(merged.values(), key=lambda x: x["date"])


def build_forecast_batch_bundle(
    forecasting_service: ForecastingService,
    data_service: DataService,
    portfolio_service: PortfolioService,
    *,
    scope: str,
    ticker: str | None,
    portfolio_id: str | None,
    start_date: date,
    end_date: date,
    horizon: int,
    methods: list[str],
    method_params: dict[str, dict[str, Any]] | None = None,
    out_of_sample: bool,
    training_ratio: float,
    create_ensemble: bool,
) -> dict[str, Any]:
    if start_date >= end_date:
        raise ValidationError("Start date must be before end date")
    if not methods:
        raise ValidationError("Select at least one forecasting method")

    training_start: date | None = None
    if out_of_sample:
        analysis_days = (end_date - start_date).days
        training_days = int(analysis_days * training_ratio)
        training_start = start_date - timedelta(days=training_days)

    chart_start = (
        training_start if (training_start is not None and out_of_sample) else start_date
    )
    mp = method_params or {}

    if scope == "asset":
        if not ticker:
            raise ValidationError("Ticker is required for single-asset forecast")
        forecasts = forecasting_service.run_multiple_forecasts(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            methods=methods,
            method_params=mp,
            out_of_sample=out_of_sample,
            training_ratio=training_ratio,
        )
        historical = fetch_historical_asset_series(
            data_service, ticker, chart_start, end_date
        )
    elif scope == "portfolio":
        if not portfolio_id:
            raise ValidationError("Portfolio is required")
        forecasts = forecasting_service.run_multiple_forecasts_portfolio(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            methods=methods,
            method_params=mp,
            out_of_sample=out_of_sample,
            training_ratio=training_ratio,
        )
        historical = fetch_historical_portfolio_series(
            data_service,
            forecasting_service,
            portfolio_service,
            portfolio_id,
            chart_start,
            end_date,
        )
    else:
        raise ValidationError("scope must be 'asset' or 'portfolio'")

    if create_ensemble:
        ok_only = {k: v for k, v in forecasts.items() if v.get("success")}
        if len(ok_only) >= 2:
            try:
                last_hist: float | None = None
                if historical:
                    v = historical[-1].get("value")
                    if v is not None:
                        last_hist = float(v)
                forecasts["ensemble"] = forecasting_service.create_ensemble(
                    ok_only,
                    method="weighted_average",
                    last_historical_price=last_hist,
                )
            except Exception as exc:
                logger.warning("Ensemble failed: %s", exc)
                forecasts["ensemble"] = {"success": False, "message": str(exc)}
        else:
            forecasts["ensemble"] = {
                "success": False,
                "message": "Need at least 2 successful model runs for ensemble",
            }

    comparison_chart = merge_comparison_chart_data(historical, forecasts)

    forecast_end: str | None = None
    if out_of_sample:
        forecast_end = (end_date + timedelta(days=horizon)).isoformat()
    else:
        for fd in forecasts.values():
            if isinstance(fd, dict) and fd.get("success") and fd.get("forecast_dates"):
                fds = fd["forecast_dates"]
                if fds:
                    forecast_end = str(fds[-1])[:10]
                    break

    return {
        "forecasts": forecasts,
        "historical": historical,
        "comparison_chart": comparison_chart,
        "meta": {
            "training_start": training_start.isoformat() if training_start else None,
            "validation_start": start_date.isoformat() if out_of_sample else None,
            "validation_end": end_date.isoformat() if out_of_sample else None,
            "forecast_end": forecast_end,
            "out_of_sample": out_of_sample,
            "horizon": horizon,
        },
    }
