"""Sync ORM positions cache from transaction ledger."""

import logging
from datetime import date

from core.data_manager.portfolio import Portfolio
from core.data_manager.portfolio_repository import PortfolioRepository
from core.data_manager.transaction import Transaction
from services.cost_basis import CostBasisCalculator

logger = logging.getLogger(__name__)


def sync_positions_from_transactions(
    portfolio: Portfolio,
    transactions: list[Transaction],
    repository: PortfolioRepository | None = None,
    user_id: str | None = None,
) -> Portfolio:
    """
    Rebuild domain positions from transactions and persist (variant A cache).

    CASH balance is stored as a CASH position when non-zero.
    """
    if portfolio.id is None:
        raise ValueError("Portfolio must have an id to sync positions")

    summary = CostBasisCalculator(method=portfolio.cost_basis_method).summarize(
        transactions
    )

    rebuilt = Portfolio(
        name=portfolio.name,
        starting_capital=portfolio.starting_capital,
        description=portfolio.description,
        base_currency=portfolio.base_currency,
        portfolio_id=portfolio.id,
        cost_basis_method=portfolio.cost_basis_method,
    )

    for ticker, leg in summary.holdings.items():
        if leg.quantity <= 1e-9:
            continue
        avg = leg.total_cost / leg.quantity if leg.quantity > 0 else 0.0
        rebuilt.add_position(
            ticker=ticker,
            shares=leg.quantity,
            purchase_price=avg if avg > 0 else None,
        )

    if summary.cash_balance > 1e-9:
        rebuilt.add_position(
            ticker="CASH",
            shares=summary.cash_balance,
            purchase_price=1.0,
            purchase_date=date.today(),
        )

    repo = repository or PortfolioRepository()
    uid = user_id
    if uid is None:
        raise ValueError("user_id is required to sync positions")
    saved = repo.save(rebuilt, uid)
    logger.info(
        "Synced %d positions for portfolio %s",
        len(saved.get_all_positions()),
        portfolio.id,
    )
    return saved
