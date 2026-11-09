"""Portfolio rebalance strategy: target weights and schedule (Phase 5).

Maps to ``weight_target`` on positions and ``rebalance_interval_months`` on the
portfolio — see docs/production/phases/phase-5-strategies.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from core.exceptions import ValidationError
from services.portfolio_service import PortfolioService
from services.rebalance_service import (
    RebalancePlan,
    RebalanceService,
    normalize_target_weights,
)
from services.schemas import (
    AddPositionRequest,
    UpdatePortfolioRequest,
    UpdatePositionRequest,
    UpdateStrategyRequest,
)

logger = logging.getLogger(__name__)

WEIGHT_SUM_TOLERANCE = 0.0001


def extract_target_weights(portfolio) -> dict[str, float]:
    """Read rebalance targets from portfolio positions (same rules as live rebalance)."""
    weights: dict[str, float] = {}
    for pos in portfolio.get_all_positions():
        if pos.weight_target is not None and pos.weight_target > 0:
            weights[pos.ticker] = float(pos.weight_target)
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    cash_w = weights.get("CASH", 0.0)
    if cash_w > 0 and abs(total - 1.0) > 1e-4:
        stock_sum = sum(w for t, w in weights.items() if t != "CASH")
        if stock_sum > 0:
            scale = (1.0 - cash_w) / stock_sum
            return {
                t: (cash_w if t == "CASH" else w * scale) for t, w in weights.items()
            }
    if abs(total - 1.0) > 1e-4:
        return {t: w / total for t, w in weights.items()}
    return weights


@dataclass
class StrategySnapshot:
    """Serializable strategy view for API and UI."""

    portfolio_id: str
    rebalance_interval_months: int | None
    targets: dict[str, float]
    targets_normalized: dict[str, float]
    total_weight: float
    is_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "rebalance_interval_months": self.rebalance_interval_months,
            "targets": self.targets,
            "targets_normalized": self.targets_normalized,
            "total_weight": self.total_weight,
            "is_active": self.is_active,
        }


def serialize_rebalance_plan(plan: RebalancePlan) -> dict[str, Any]:
    """JSON-friendly rebalance preview (shared by strategy and rebalance routes)."""
    return {
        "scheduled_date": plan.scheduled_date.isoformat(),
        "execution_date": plan.execution_date.isoformat(),
        "as_of_date": plan.execution_date.isoformat(),
        "cash_target_pct": plan.cash_target_pct,
        "projected_cash_pct": plan.projected_cash_pct,
        "complete": plan.complete,
        "message": plan.message,
        "trades": [
            {
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "fees": t.fees,
                "current_shares": t.current_shares,
                "target_shares": t.target_shares,
                "current_weight": t.current_weight,
                "target_weight": t.target_weight,
            }
            for t in plan.trades
        ],
        "trade_count": len(plan.trades),
    }


class StrategyService:
    """CRUD for portfolio rebalance strategy (weights + schedule)."""

    def __init__(
        self,
        portfolio_service: PortfolioService | None = None,
        rebalance_service: RebalanceService | None = None,
    ) -> None:
        self._portfolio_service = portfolio_service or PortfolioService()
        self._rebalance_service = rebalance_service or RebalanceService()

    def get_strategy(
        self, portfolio_id: str, user_id: str | None = None
    ) -> StrategySnapshot:
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        targets = extract_target_weights(portfolio)
        raw: dict[str, float] = {}
        for pos in portfolio.get_all_positions():
            if pos.weight_target is not None and pos.weight_target > 0:
                raw[pos.ticker] = float(pos.weight_target)
        total = sum(raw.values()) if raw else 0.0
        interval = getattr(portfolio, "rebalance_interval_months", None)
        is_active = interval is not None and bool(targets)
        return StrategySnapshot(
            portfolio_id=portfolio_id,
            rebalance_interval_months=interval,
            targets=raw,
            targets_normalized=targets,
            total_weight=total,
            is_active=is_active,
        )

    def update_strategy(
        self,
        portfolio_id: str,
        request: UpdateStrategyRequest,
        user_id: str | None = None,
    ):
        """Update rebalance schedule and/or target weights from API payload."""
        if "rebalance_interval_months" in request.model_fields_set:
            req = UpdatePortfolioRequest(
                rebalance_interval_months=request.rebalance_interval_months
            )
            self._portfolio_service.update_portfolio(portfolio_id, req, user_id)

        if request.targets is not None:
            if not request.targets:
                raise ValidationError("Strategy targets cannot be empty")
            raw_total = sum(
                float(v) for v in request.targets.values() if float(v) > 1e-15
            )
            if abs(raw_total - 1.0) > WEIGHT_SUM_TOLERANCE:
                raise ValidationError(
                    f"Strategy weights must sum to 1.0, got {raw_total:.6f}"
                )
            normalized = normalize_target_weights(request.targets)
            if not normalized:
                raise ValidationError("Strategy targets cannot be empty")
            self._apply_targets(
                portfolio_id,
                normalized,
                request.replace_targets,
                user_id,
            )

        logger.info("Updated strategy for portfolio %s", portfolio_id)
        return self._portfolio_service.get_portfolio(portfolio_id, user_id)

    def _apply_targets(
        self,
        portfolio_id: str,
        normalized: dict[str, float],
        replace_targets: bool,
        user_id: str | None,
    ) -> None:
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        tickers_in_request = set(normalized.keys())

        for ticker, weight in normalized.items():
            pos = portfolio.get_position(ticker)
            if pos is not None:
                self._portfolio_service.update_position(
                    portfolio_id,
                    ticker,
                    UpdatePositionRequest(weight_target=weight),
                    user_id,
                )
            elif ticker == "CASH":
                cash_shares = max(
                    portfolio.starting_capital,
                    1.0,
                )
                self._portfolio_service.add_position(
                    portfolio_id,
                    AddPositionRequest(
                        ticker="CASH",
                        shares=cash_shares,
                        weight_target=weight,
                        purchase_price=1.0,
                    ),
                    user_id,
                )
            else:
                raise ValidationError(
                    f"Cannot set target weight for {ticker}: no position in portfolio"
                )

        if replace_targets:
            portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
            for pos in portfolio.get_all_positions():
                if (
                    pos.ticker not in tickers_in_request
                    and pos.weight_target is not None
                ):
                    self._portfolio_service.update_position(
                        portfolio_id,
                        pos.ticker,
                        UpdatePositionRequest(weight_target=0.0),
                        user_id,
                    )

    def preview_rebalance(
        self,
        portfolio_id: str,
        as_of_date: date | None = None,
        user_id: str | None = None,
    ) -> RebalancePlan:
        scheduled = as_of_date or date.today()
        return self._rebalance_service.preview(portfolio_id, scheduled, user_id)

    def build_policy_summary(
        self,
        portfolio_id: str,
        *,
        optimized_weights: dict[str, float] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Labels for Optimization workbench: live strategy vs optimized run."""
        snap = self.get_strategy(portfolio_id, user_id)
        interval = snap.rebalance_interval_months
        interval_label = f"Every {interval} month(s)" if interval else "Rebalancing off"
        current_targets = snap.targets_normalized
        opt_norm = (
            normalize_target_weights(optimized_weights) if optimized_weights else {}
        )
        return {
            "rebalance_interval_months": interval,
            "rebalance_interval_label": interval_label,
            "current_targets": current_targets,
            "optimized_targets": opt_norm,
            "strategy_active": snap.is_active,
            "current_label": (
                "Current portfolio uses saved target weights and your rebalance "
                f"schedule ({interval_label})."
                if current_targets
                else "Current portfolio has no saved target weights; live book "
                "may infer targets from the ledger."
            ),
            "optimized_label": (
                "Optimized track uses weights from this optimization run with the "
                f"same cash-flow replay and schedule ({interval_label})."
                if opt_norm
                else "Optimized track uses weights from this optimization run."
            ),
        }
