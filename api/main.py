"""FastAPI entrypoint for Next.js frontend."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.routers import auth as auth_router
from core.analytics_engine.chart_data import (
    get_asset_impact_on_return_data,
    get_asset_impact_on_risk_data,
    get_asset_metrics_data,
    get_average_correlation_to_portfolio_data,
    get_correlation_matrix_data,
    get_correlation_statistics_data,
    get_correlation_with_benchmark_data,
    get_detailed_asset_analysis_data,
    get_diversification_coefficient_data,
    get_risk_vs_weight_comparison_data,
)
from core.data_manager.transaction import Transaction
from core.exceptions import (
    CalculationError,
    ConflictError,
    DataFetchError,
    InsufficientDataError,
    PortfolioNotFoundError,
    ValidationError,
)
from core.scenario_engine.custom_scenarios import (
    create_custom_scenario,
    validate_scenario,
)
from core.scenario_engine.historical_scenarios import get_scenario_by_name
from core.scenario_engine.scenario_chain import create_scenario_chain
from database.session import ensure_database_schema
from models.user import User
from services.analytics_service import AnalyticsService
from services.data_service import DataService
from services.forecasting_service import ForecastingService
from services.forecasting_ui_bundle import build_forecast_batch_bundle
from services.optimization_service import OptimizationService
from services.optimization_ui_bundle import (
    build_optimization_full_bundle,
    portfolio_snapshot_rows,
)
from services.portfolio_service import PortfolioService
from services.risk_service import RiskService
from services.risk_ui_bundle import (
    build_monte_carlo_display_bundle,
    build_stress_historical_display_bundle,
    build_var_full_bundle,
    serialize_scenario_catalog,
)
from services.schemas import (
    AddPositionRequest,
    CreatePortfolioRequest,
    UpdatePortfolioRequest,
    UpdatePositionRequest,
)
from services.transaction_service import TransactionService


class AddTransactionRequest(BaseModel):
    """API payload for adding a transaction."""

    transaction_date: date
    transaction_type: str = Field(
        ...,
        examples=["BUY", "SELL", "DEPOSIT", "WITHDRAWAL", "DIVIDEND", "SPLIT"],
    )
    ticker: str
    shares: float
    price: float
    fees: float = 0.0
    notes: str | None = None
    reinvest: bool | None = None
    split_ratio: float | None = None
    currency: str = "USD"


class DividendSyncRequest(BaseModel):
    """Sync dividends from market data."""

    tickers: list[str]
    start_date: date
    end_date: date
    reinvest: bool = False


class RiskVarRequest(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    confidence_level: float = 0.95
    include_monte_carlo: bool = True
    num_simulations: int = 10000
    time_horizon: int = 1


class RiskMonteCarloRequest(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    time_horizon: int = 252
    num_simulations: int = 10000
    initial_value: float = 1.0
    model: str = "gbm"


class RiskVarFullRequest(RiskVarRequest):
    rolling_window: int = 63


class RiskMonteCarloFullRequest(RiskMonteCarloRequest):
    include_sample_paths: bool = False


class StressTestRequest(BaseModel):
    portfolio_id: str
    scenario_names: list[str]


class CustomScenarioApiRequest(BaseModel):
    portfolio_id: str
    name: str
    description: str = ""
    market_impact_pct: float
    asset_impacts: dict[str, float] = {}


class ScenarioChainApiRequest(BaseModel):
    portfolio_id: str
    name: str
    description: str = ""
    scenario_keys: list[str]


class AnalyticsRequest(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    benchmark_ticker: str | None = None
    comparison_type: str | None = None
    comparison_value: str | None = None


class AssetsAnalyticsRequest(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    benchmark_ticker: str | None = None


class OptimizationRequest(BaseModel):
    portfolio_id: str
    method: str
    start_date: date
    end_date: date
    constraints: dict[str, Any] | None = None
    benchmark_ticker: str | None = None
    method_params: dict[str, Any] | None = None
    out_of_sample: bool = False
    training_ratio: float = 0.3


class OptimizationFullRequest(OptimizationRequest):
    """Optimization + charts, frontier, trades (Streamlit-equivalent bundle)."""

    benchmark_for_charts: str | None = None
    include_efficient_frontier: bool = True
    frontier_n_points: int = 150
    include_sensitivity: bool = False
    sensitivity_analysis_type: str = "returns"
    notebook_split: bool = False
    notebook_train_fraction: float = 0.7


class ForecastAssetRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    horizon: int
    method: str
    method_params: dict[str, Any] | None = None
    out_of_sample: bool = False
    training_ratio: float = 0.3


class ForecastPortfolioRequest(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    horizon: int
    method: str
    method_params: dict[str, Any] | None = None
    out_of_sample: bool = False
    training_ratio: float = 0.3


class ForecastBatchRequest(BaseModel):
    """Run multiple methods in one request (Streamlit forecasting page parity)."""

    scope: str  # "asset" | "portfolio"
    ticker: str | None = None
    portfolio_id: str | None = None
    start_date: date
    end_date: date
    horizon: int
    methods: list[str]
    method_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    out_of_sample: bool = True
    training_ratio: float = 0.3
    create_ensemble: bool = False


def _serialize_position(position: Any) -> dict[str, Any]:
    return {
        "ticker": position.ticker,
        "shares": position.shares,
        "weight_target": position.weight_target,
        "purchase_price": position.purchase_price,
        "purchase_date": (
            position.purchase_date.isoformat() if position.purchase_date else None
        ),
    }


def _serialize_portfolio(portfolio: Any) -> dict[str, Any]:
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "starting_capital": portfolio.starting_capital,
        "base_currency": portfolio.base_currency,
        "cost_basis_method": getattr(portfolio, "cost_basis_method", "fifo"),
        "positions": [_serialize_position(p) for p in portfolio.get_all_positions()],
    }


def _serialize_transaction(tx: Transaction) -> dict[str, Any]:
    return {
        "id": tx.id,
        "transaction_date": tx.transaction_date.isoformat(),
        "transaction_type": tx.transaction_type,
        "ticker": tx.ticker,
        "shares": tx.shares,
        "price": tx.price,
        "amount": tx.amount,
        "fees": tx.fees,
        "notes": tx.notes,
        "reinvest": tx.reinvest,
        "split_ratio": tx.split_ratio,
        "currency": tx.currency,
    }


def _serialize_holding(h: Any) -> dict[str, Any]:
    return {
        "ticker": h.ticker,
        "quantity": h.quantity,
        "avg_cost": h.avg_cost,
        "market_price": h.market_price,
        "market_value": h.market_value,
        "cost_basis": h.cost_basis,
        "unrealized_pnl": h.unrealized_pnl,
    }


def _serialize_pnl(summary: Any) -> dict[str, Any]:
    return {
        "realized_pnl": summary.realized_pnl,
        "unrealized_pnl": summary.unrealized_pnl,
        "dividend_income": summary.dividend_income,
        "cost_basis": summary.cost_basis,
        "market_value": summary.market_value,
        "total_return_twr": summary.total_return_twr,
        "total_return_mwr": summary.total_return_mwr,
        "cash_balance": summary.cash_balance,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)[:10]
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, pd.DataFrame):
        df = value.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        return df.to_dict(orient="records")
    if isinstance(value, pd.Series):
        s = value.copy()
        if isinstance(s.index, pd.DatetimeIndex):
            return [{"x": str(idx), "y": _to_jsonable(val)} for idx, val in s.items()]
        return {str(k): _to_jsonable(v) for k, v in s.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, PortfolioNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(
        exc, (ValidationError, ValueError, CalculationError, InsufficientDataError)
    ):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, ConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, DataFetchError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _verify_portfolio_access(portfolio_id: str, user: User) -> None:
    """Ensure portfolio exists and belongs to the user (404 if not)."""
    portfolio_service.get_portfolio(portfolio_id, user.id)


ensure_database_schema()
app = FastAPI(title="WMC Portfolio API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router.router)

portfolio_service = PortfolioService()
transaction_service = TransactionService()
data_service = DataService()
analytics_service = AnalyticsService()
optimization_service = OptimizationService()
risk_service = RiskService()
forecasting_service = ForecastingService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard/indices")
def dashboard_indices() -> list[dict[str, Any]]:
    indices = [
        {"name": "S&P 500", "symbol": "^GSPC"},
        {"name": "NASDAQ", "symbol": "^NDX"},
        {"name": "Dow Jones", "symbol": "^DJI"},
        {"name": "Russell 2000", "symbol": "^RUT"},
    ]
    result: list[dict[str, Any]] = []
    for item in indices:
        try:
            prices = data_service.fetch_historical_prices(
                ticker=item["symbol"],
                start_date=date.today().replace(year=date.today().year - 1),
                end_date=date.today(),
                use_cache=True,
                save_to_db=True,
            )
            if prices.empty:
                result.append(
                    {
                        **item,
                        "price": None,
                        "change": None,
                        "change_pct": None,
                        "series": [],
                    }
                )
                continue
            close_col = "Close" if "Close" in prices.columns else "Adjusted_Close"
            latest = float(prices[close_col].iloc[-1])
            prev = float(prices[close_col].iloc[-2]) if len(prices) > 1 else latest
            change = latest - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            series = []
            if "Date" in prices.columns:
                series = [
                    {"x": str(d), "y": float(v)}
                    for d, v in zip(prices["Date"], prices[close_col])
                ]
            result.append(
                {
                    **item,
                    "price": latest,
                    "change": change,
                    "change_pct": change_pct,
                    "series": series,
                }
            )
        except Exception:
            result.append(
                {
                    **item,
                    "price": None,
                    "change": None,
                    "change_pct": None,
                    "series": [],
                }
            )
    return result


@app.post("/validate-tickers")
def validate_tickers(tickers: list[str]) -> dict[str, bool]:
    return data_service.validate_tickers(tickers)


@app.get("/ticker-price/{ticker}")
def ticker_price(ticker: str) -> dict[str, Any]:
    try:
        validation = data_service.validate_tickers([ticker.upper()])
        is_valid = validation.get(ticker.upper(), False)
        if not is_valid:
            return {"ticker": ticker.upper(), "valid": False, "price": None}
        price = data_service.fetch_current_price(ticker.upper())
        return {"ticker": ticker.upper(), "valid": True, "price": price}
    except Exception:
        return {"ticker": ticker.upper(), "valid": False, "price": None}


@app.get("/portfolios")
def list_portfolios(
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    return [
        _serialize_portfolio(p)
        for p in portfolio_service.list_portfolios(current_user.id)
    ]


@app.get("/portfolios/{portfolio_id}")
def get_portfolio(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.get_portfolio(portfolio_id, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/portfolios")
def create_portfolio(
    payload: CreatePortfolioRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.create_portfolio(payload, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.patch("/portfolios/{portfolio_id}")
def update_portfolio(
    portfolio_id: str,
    payload: UpdatePortfolioRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.update_portfolio(portfolio_id, payload, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.delete("/portfolios/{portfolio_id}")
def delete_portfolio(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, bool]:
    try:
        return {
            "deleted": portfolio_service.delete_portfolio(portfolio_id, current_user.id)
        }
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/portfolios/{portfolio_id}/positions")
def add_position(
    portfolio_id: str,
    payload: AddPositionRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.add_position(portfolio_id, payload, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.patch("/portfolios/{portfolio_id}/positions/{ticker}")
def update_position(
    portfolio_id: str,
    ticker: str,
    payload: UpdatePositionRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.update_position(
                portfolio_id, ticker, payload, current_user.id
            )
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.delete("/portfolios/{portfolio_id}/positions/{ticker}")
def remove_position(
    portfolio_id: str,
    ticker: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_portfolio(
            portfolio_service.remove_position(portfolio_id, ticker, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.get("/portfolios/{portfolio_id}/transactions")
def get_transactions(
    portfolio_id: str,
    from_date: date | None = None,
    to_date: date | None = None,
    transaction_type: str | None = None,
    ticker: str | None = None,
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    return [
        _serialize_transaction(tx)
        for tx in transaction_service.get_transactions(
            portfolio_id,
            start_date=from_date,
            end_date=to_date,
            transaction_type=transaction_type,
            ticker=ticker,
            user_id=current_user.id,
        )
    ]


@app.post("/portfolios/{portfolio_id}/transactions")
def add_transaction(
    portfolio_id: str,
    payload: AddTransactionRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        tx = transaction_service.add_transaction(
            portfolio_id=portfolio_id,
            transaction_date=payload.transaction_date,
            transaction_type=payload.transaction_type,
            ticker=payload.ticker,
            shares=payload.shares,
            price=payload.price,
            fees=payload.fees,
            notes=payload.notes,
            user_id=current_user.id,
            reinvest=payload.reinvest,
            split_ratio=payload.split_ratio,
            currency=payload.currency,
        )
        return _serialize_transaction(tx)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.delete("/portfolios/{portfolio_id}/transactions/{transaction_id}")
def delete_portfolio_transaction(
    portfolio_id: str,
    transaction_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, bool]:
    try:
        _verify_portfolio_access(portfolio_id, current_user)
        return {
            "deleted": transaction_service.delete_transaction(
                transaction_id,
                current_user.id,
                portfolio_id=portfolio_id,
            )
        }
    except Exception as exc:
        _handle_error(exc)
        raise


@app.delete("/transactions/{transaction_id}")
def delete_transaction_legacy(
    transaction_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, bool]:
    """Deprecated: prefer DELETE /portfolios/{id}/transactions/{tx_id}."""
    return {
        "deleted": transaction_service.delete_transaction(
            transaction_id, current_user.id
        )
    }


@app.get("/portfolios/{portfolio_id}/holdings")
def get_holdings(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    try:
        rows = transaction_service.get_holdings(portfolio_id, current_user.id)
        return [_serialize_holding(h) for h in rows]
    except Exception as exc:
        _handle_error(exc)
        raise


@app.get("/portfolios/{portfolio_id}/pnl")
def get_pnl(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return _serialize_pnl(
            transaction_service.get_pnl(portfolio_id, current_user.id)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.get("/portfolios/{portfolio_id}/dividends")
def get_dividends(
    portfolio_id: str,
    from_date: date | None = None,
    to_date: date | None = None,
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    try:
        txs = transaction_service.get_dividends(
            portfolio_id,
            start_date=from_date,
            end_date=to_date,
            user_id=current_user.id,
        )
        return [_serialize_transaction(tx) for tx in txs]
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/portfolios/{portfolio_id}/dividends/sync")
def sync_dividends(
    portfolio_id: str,
    payload: DividendSyncRequest,
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    try:
        created = transaction_service.sync_dividends(
            portfolio_id,
            payload.tickers,
            payload.start_date,
            payload.end_date,
            current_user.id,
            payload.reinvest,
        )
        return [_serialize_transaction(tx) for tx in created]
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/analytics/calculate")
def calculate_analytics(
    payload: AnalyticsRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        result = analytics_service.calculate_portfolio_metrics(
            portfolio_id=payload.portfolio_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            benchmark_ticker=payload.benchmark_ticker,
            comparison_type=payload.comparison_type,
            comparison_value=payload.comparison_value,
        )
        return _to_jsonable(result)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/analytics/assets")
def calculate_asset_analytics(
    payload: AssetsAnalyticsRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Compute per-asset data: correlations, metrics, returns, impact analysis."""
    try:
        portfolio = portfolio_service.get_portfolio(
            payload.portfolio_id, current_user.id
        )
        positions = portfolio.get_all_positions()
        tickers = [p.ticker for p in positions]

        if not tickers:
            raise ValidationError("Portfolio has no positions")

        price_data = analytics_service._fetch_portfolio_prices(
            tickers, payload.start_date, payload.end_date
        )

        benchmark_returns: pd.Series | None = None
        if payload.benchmark_ticker:
            try:
                bm_prices = analytics_service._fetch_portfolio_prices(
                    [payload.benchmark_ticker], payload.start_date, payload.end_date
                )
                if (
                    not bm_prices.empty
                    and payload.benchmark_ticker in bm_prices.columns
                ):
                    bm_series = (
                        bm_prices[payload.benchmark_ticker].sort_index().ffill().bfill()
                    )
                    benchmark_returns = bm_series.pct_change().dropna()
            except Exception:
                pass

        portfolio_returns = analytics_service._calculate_portfolio_returns(
            price_data, positions
        )

        result: dict[str, Any] = {}

        # 1. Asset metrics (name, sector, industry, etc.)
        am = get_asset_metrics_data(positions, price_data)
        result["asset_metrics"] = am

        # 2. Impact on return
        result["impact_on_return"] = get_asset_impact_on_return_data(
            positions, price_data, payload.start_date, payload.end_date
        )

        # 3. Impact on risk
        result["impact_on_risk"] = get_asset_impact_on_risk_data(positions, price_data)

        # 4. Risk vs weight comparison
        result["risk_vs_weight"] = get_risk_vs_weight_comparison_data(
            positions, price_data, payload.start_date, payload.end_date
        )

        # 5. Diversification
        result["diversification"] = get_diversification_coefficient_data(
            positions, price_data
        )

        # 6. Correlation matrix
        corr_data = get_correlation_matrix_data(
            positions, price_data, benchmark_returns
        )
        if corr_data and corr_data.get("correlation_matrix") is not None:
            corr_matrix = corr_data["correlation_matrix"]
            result["correlations"] = {
                "matrix": {
                    str(r): {
                        str(c): float(corr_matrix.loc[r, c])
                        for c in corr_matrix.columns
                    }
                    for r in corr_matrix.index
                },
                "tickers": corr_data["tickers"],
            }
            # 7. Correlation statistics
            result["correlation_stats"] = get_correlation_statistics_data(corr_matrix)
        else:
            result["correlations"] = None
            result["correlation_stats"] = None

        # 8. Benchmark correlations
        if benchmark_returns is not None and not benchmark_returns.empty:
            result["benchmark_correlations"] = get_correlation_with_benchmark_data(
                positions, price_data, benchmark_returns
            )
        else:
            result["benchmark_correlations"] = None

        # 9. Average correlation to portfolio
        result["avg_correlation_to_portfolio"] = (
            get_average_correlation_to_portfolio_data(positions, price_data)
        )

        # 10. Individual asset returns (daily, as { dates: [...], TICKER: [...] })
        returns_df = price_data.pct_change().dropna()
        asset_tickers = [t for t in tickers if t in returns_df.columns and t != "CASH"]
        if not returns_df.empty and asset_tickers:
            dates_list = [str(d)[:10] for d in returns_df.index]
            asset_returns_dict: dict[str, Any] = {"dates": dates_list}
            for t in asset_tickers:
                asset_returns_dict[t] = returns_df[t].tolist()
            result["asset_returns"] = asset_returns_dict
        else:
            result["asset_returns"] = None

        # 11. Per-asset detailed metrics
        per_asset: dict[str, Any] = {}
        for t in asset_tickers:
            detail = get_detailed_asset_analysis_data(
                t, positions, price_data, portfolio_returns, benchmark_returns
            )
            if detail:
                per_asset[t] = {
                    "metrics": detail["metrics"],
                    "portfolio_metrics": detail["portfolio_metrics"],
                    "other_correlations": detail.get("other_correlations", {}),
                }
        result["per_asset"] = per_asset

        return _to_jsonable(result)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/optimization/run")
def optimize(
    payload: OptimizationRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        result = optimization_service.optimize_portfolio(
            portfolio_id=payload.portfolio_id,
            method=payload.method,
            start_date=payload.start_date,
            end_date=payload.end_date,
            constraints=payload.constraints,
            benchmark_ticker=payload.benchmark_ticker,
            method_params=payload.method_params,
            out_of_sample=payload.out_of_sample,
            training_ratio=payload.training_ratio,
        )
        return _to_jsonable(
            result.to_dict() if hasattr(result, "to_dict") else result.__dict__
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.get("/portfolios/{portfolio_id}/optimization-snapshot")
def optimization_snapshot(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Current holdings with live prices (Streamlit expander)."""
    try:
        _verify_portfolio_access(portfolio_id, current_user)
        ds = optimization_service._data_service  # noqa: SLF001
        rows = portfolio_snapshot_rows(portfolio_service, ds, portfolio_id)
        return _to_jsonable(rows)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/optimization/full")
def optimization_full(
    payload: OptimizationFullRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        bundle = build_optimization_full_bundle(
            optimization_service,
            portfolio_service,
            portfolio_id=payload.portfolio_id,
            method=payload.method,
            start_date=payload.start_date,
            end_date=payload.end_date,
            constraints=payload.constraints,
            benchmark_ticker=payload.benchmark_ticker,
            method_params=payload.method_params,
            out_of_sample=payload.out_of_sample,
            training_ratio=payload.training_ratio,
            benchmark_for_charts=payload.benchmark_for_charts,
            include_efficient_frontier=payload.include_efficient_frontier,
            frontier_n_points=payload.frontier_n_points,
            include_sensitivity=payload.include_sensitivity,
            sensitivity_analysis_type=payload.sensitivity_analysis_type,
            notebook_split=payload.notebook_split,
            notebook_train_fraction=payload.notebook_train_fraction,
        )
        return _to_jsonable(bundle)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/var")
def risk_var(
    payload: RiskVarRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        return _to_jsonable(
            risk_service.calculate_var_analysis(
                portfolio_id=payload.portfolio_id,
                start_date=payload.start_date,
                end_date=payload.end_date,
                confidence_level=payload.confidence_level,
                include_monte_carlo=payload.include_monte_carlo,
                num_simulations=payload.num_simulations,
                time_horizon=payload.time_horizon,
            )
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/var/full")
def risk_var_full(
    payload: RiskVarFullRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        bundle = build_var_full_bundle(
            risk_service,
            portfolio_service,
            portfolio_id=payload.portfolio_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            confidence_level=payload.confidence_level,
            time_horizon=payload.time_horizon,
            rolling_window=payload.rolling_window,
            include_monte_carlo=payload.include_monte_carlo,
            num_simulations=payload.num_simulations,
        )
        return _to_jsonable(bundle)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/monte-carlo")
def risk_monte_carlo(
    payload: RiskMonteCarloRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        return _to_jsonable(
            risk_service.run_monte_carlo_simulation(
                portfolio_id=payload.portfolio_id,
                start_date=payload.start_date,
                end_date=payload.end_date,
                time_horizon=payload.time_horizon,
                num_simulations=payload.num_simulations,
                initial_value=payload.initial_value,
                model=payload.model,
            )
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/monte-carlo/full")
def risk_monte_carlo_full(
    payload: RiskMonteCarloFullRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        bundle = build_monte_carlo_display_bundle(
            risk_service,
            portfolio_id=payload.portfolio_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            time_horizon=payload.time_horizon,
            num_simulations=payload.num_simulations,
            initial_value=payload.initial_value,
            model=payload.model,
            include_sample_paths=payload.include_sample_paths,
        )
        return _to_jsonable(bundle)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.get("/risk/scenarios")
def risk_scenarios_catalog() -> list[dict[str, Any]]:
    try:
        return _to_jsonable(serialize_scenario_catalog(risk_service))
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/stress-tests")
def stress_tests(
    payload: StressTestRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        return _to_jsonable(
            risk_service.run_stress_test(payload.portfolio_id, payload.scenario_names)
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/stress-tests/full")
def stress_tests_full(
    payload: StressTestRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Historical scenarios bundle: recovery paths, breakdowns, timelines (Streamlit parity)."""
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        bundle = build_stress_historical_display_bundle(
            risk_service,
            portfolio_service,
            portfolio_id=payload.portfolio_id,
            scenario_keys=payload.scenario_names,
        )
        return _to_jsonable(bundle)
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/custom-scenario")
def custom_scenario(
    payload: CustomScenarioApiRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        scenario = create_custom_scenario(
            name=payload.name,
            description=payload.description,
            market_impact_pct=payload.market_impact_pct,
            asset_impacts=payload.asset_impacts or None,
        )
        ok, msg = validate_scenario(scenario)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return _to_jsonable(
            risk_service.run_custom_scenario(payload.portfolio_id, scenario)
        )
    except HTTPException:
        raise
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/risk/scenario-chain")
def scenario_chain(
    payload: ScenarioChainApiRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        selected = []
        for key in payload.scenario_keys:
            sc = get_scenario_by_name(key)
            if sc:
                selected.append(sc)
        if not selected:
            raise HTTPException(status_code=400, detail="No valid scenario keys")
        chain = create_scenario_chain(
            name=payload.name,
            description=payload.description,
            scenarios=selected,
        )
        return _to_jsonable(
            risk_service.run_scenario_chain(payload.portfolio_id, chain)
        )
    except HTTPException:
        raise
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/forecasting/asset")
def forecast_asset(payload: ForecastAssetRequest) -> Any:
    try:
        return _to_jsonable(
            forecasting_service.forecast_asset(
                ticker=payload.ticker,
                start_date=payload.start_date,
                end_date=payload.end_date,
                horizon=payload.horizon,
                method=payload.method,
                method_params=payload.method_params,
                out_of_sample=payload.out_of_sample,
                training_ratio=payload.training_ratio,
            )
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/forecasting/portfolio")
def forecast_portfolio(
    payload: ForecastPortfolioRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        _verify_portfolio_access(payload.portfolio_id, current_user)
        return _to_jsonable(
            forecasting_service.forecast_portfolio(
                portfolio_id=payload.portfolio_id,
                start_date=payload.start_date,
                end_date=payload.end_date,
                horizon=payload.horizon,
                method=payload.method,
                method_params=payload.method_params,
                out_of_sample=payload.out_of_sample,
                training_ratio=payload.training_ratio,
            )
        )
    except Exception as exc:
        _handle_error(exc)
        raise


@app.post("/forecasting/batch")
def forecast_batch(
    payload: ForecastBatchRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        if payload.scope not in ("asset", "portfolio"):
            raise HTTPException(
                status_code=400, detail="scope must be 'asset' or 'portfolio'"
            )
        if payload.scope == "portfolio" and payload.portfolio_id:
            _verify_portfolio_access(payload.portfolio_id, current_user)
        bundle = build_forecast_batch_bundle(
            forecasting_service,
            data_service,
            portfolio_service,
            scope=payload.scope,
            ticker=payload.ticker,
            portfolio_id=payload.portfolio_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            horizon=payload.horizon,
            methods=payload.methods,
            method_params=payload.method_params,
            out_of_sample=payload.out_of_sample,
            training_ratio=payload.training_ratio,
            create_ensemble=payload.create_ensemble,
        )
        return _to_jsonable(bundle)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_error(exc)
        raise
