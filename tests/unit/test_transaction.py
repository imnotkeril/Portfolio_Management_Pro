"""Unit tests for transaction domain model and repository."""

import pytest
from datetime import date
from unittest.mock import patch

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from core.exceptions import ValidationError
from database.session import Base, get_db_session
from models.portfolio import Portfolio
from models.position import Position  # Import Position for SQLAlchemy relationship resolution
from models.transaction import Transaction as TransactionORM


@pytest.fixture(scope="function")
def test_db():
    """Create test database and cleanup after test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from contextlib import contextmanager

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    SessionLocal = sessionmaker(bind=test_engine)

    # Mock get_db_session to use test database
    @contextmanager
    def mock_get_db_session():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Patch get_db_session in the repository module
    with patch("core.data_manager.transaction_repository.get_db_session", mock_get_db_session):
        yield SessionLocal

    Base.metadata.drop_all(test_engine)


@pytest.fixture
def sample_portfolio(test_db):
    """Create a sample portfolio for testing."""
    # Ensure all models are imported for SQLAlchemy
    from models.position import Position  # noqa: F401
    from models.transaction import Transaction as TransactionORM  # noqa: F401
    
    session = test_db()
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
        description="Test",
    )
    session.add(portfolio)
    session.commit()
    session.refresh(portfolio)
    return portfolio


class TestTransactionDomain:
    """Test transaction domain model."""

    def test_create_valid_buy_transaction(self) -> None:
        """Test creating a valid BUY transaction."""
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )
        assert txn.transaction_type == "BUY"
        assert txn.ticker == "AAPL"
        assert txn.shares == 10.0
        assert txn.price == 150.0
        assert txn.amount == 1500.0
        assert txn.fees == 0.0

    def test_create_valid_sell_transaction(self) -> None:
        """Test creating a valid SELL transaction."""
        txn = Transaction(
            transaction_date=date(2024, 1, 20),
            transaction_type="SELL",
            ticker="AAPL",
            shares=5.0,
            price=155.0,
            fees=5.0,
        )
        assert txn.transaction_type == "SELL"
        assert txn.amount == 775.0
        assert txn.fees == 5.0

    def test_create_deposit_transaction(self) -> None:
        """Test creating a DEPOSIT transaction."""
        txn = Transaction(
            transaction_date=date(2024, 1, 1),
            transaction_type="DEPOSIT",
            ticker="CASH",
            shares=10000.0,
            price=1.0,
        )
        assert txn.transaction_type == "DEPOSIT"
        assert txn.ticker == "CASH"
        assert txn.amount == 10000.0

    def test_create_withdrawal_transaction(self) -> None:
        """Test creating a WITHDRAWAL transaction."""
        txn = Transaction(
            transaction_date=date(2024, 1, 10),
            transaction_type="WITHDRAWAL",
            ticker="CASH",
            shares=5000.0,
            price=1.0,
        )
        assert txn.transaction_type == "WITHDRAWAL"
        assert txn.ticker == "CASH"

    def test_invalid_transaction_type(self) -> None:
        """Test that invalid transaction type raises error."""
        with pytest.raises(ValidationError, match="Invalid transaction type"):
            Transaction(
                transaction_date=date(2024, 1, 15),
                transaction_type="INVALID",
                ticker="AAPL",
                shares=10.0,
                price=150.0,
            )

    def test_negative_shares(self) -> None:
        """Test that negative shares raises error."""
        with pytest.raises(ValidationError, match="Shares must be greater"):
            Transaction(
                transaction_date=date(2024, 1, 15),
                transaction_type="BUY",
                ticker="AAPL",
                shares=-10.0,
                price=150.0,
            )

    def test_zero_shares(self) -> None:
        """Test that zero shares raises error."""
        with pytest.raises(ValidationError, match="Shares must be greater"):
            Transaction(
                transaction_date=date(2024, 1, 15),
                transaction_type="BUY",
                ticker="AAPL",
                shares=0.0,
                price=150.0,
            )

    def test_negative_price(self) -> None:
        """Test that negative price raises error."""
        with pytest.raises(ValidationError, match="Price must be greater"):
            Transaction(
                transaction_date=date(2024, 1, 15),
                transaction_type="BUY",
                ticker="AAPL",
                shares=10.0,
                price=-150.0,
            )

    def test_deposit_without_cash_ticker(self) -> None:
        """Test that DEPOSIT without CASH ticker raises error."""
        with pytest.raises(
            ValidationError, match="DEPOSIT/WITHDRAWAL transactions"
        ):
            Transaction(
                transaction_date=date(2024, 1, 15),
                transaction_type="DEPOSIT",
                ticker="AAPL",
                shares=1000.0,
                price=1.0,
            )

    def test_custom_amount(self) -> None:
        """Test creating transaction with custom amount."""
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
            amount=1600.0,  # Custom amount (includes fees)
        )
        assert txn.amount == 1600.0

    def test_transaction_with_notes(self) -> None:
        """Test creating transaction with notes."""
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
            notes="Initial purchase",
        )
        assert txn.notes == "Initial purchase"

    def test_ticker_normalization(self) -> None:
        """Test that ticker is normalized to uppercase."""
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="aapl",
            shares=10.0,
            price=150.0,
        )
        assert txn.ticker == "AAPL"

    def test_transaction_type_normalization(self) -> None:
        """Test that transaction type is normalized to uppercase."""
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="buy",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )
        assert txn.transaction_type == "BUY"


class TestTransactionRepository:
    """Test transaction repository."""

    def test_save_new_transaction(self, test_db, sample_portfolio) -> None:
        """Test saving a new transaction."""
        repository = TransactionRepository()
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )

        saved = repository.save(txn, sample_portfolio.id)

        assert saved.id is not None
        assert saved.transaction_type == "BUY"
        assert saved.ticker == "AAPL"

    def test_find_by_portfolio(self, test_db, sample_portfolio) -> None:
        """Test finding transactions by portfolio."""
        repository = TransactionRepository()

        # Create multiple transactions
        txn1 = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )
        txn2 = Transaction(
            transaction_date=date(2024, 1, 20),
            transaction_type="SELL",
            ticker="AAPL",
            shares=5.0,
            price=155.0,
        )

        repository.save(txn1, sample_portfolio.id)
        repository.save(txn2, sample_portfolio.id)

        transactions = repository.find_by_portfolio(sample_portfolio.id)

        assert len(transactions) == 2
        assert transactions[0].transaction_date == date(2024, 1, 15)
        assert transactions[1].transaction_date == date(2024, 1, 20)

    def test_find_by_portfolio_with_date_filter(
        self, test_db, sample_portfolio
    ) -> None:
        """Test finding transactions with date filter."""
        repository = TransactionRepository()

        txn1 = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )
        txn2 = Transaction(
            transaction_date=date(2024, 2, 20),
            transaction_type="SELL",
            ticker="AAPL",
            shares=5.0,
            price=155.0,
        )

        repository.save(txn1, sample_portfolio.id)
        repository.save(txn2, sample_portfolio.id)

        # Filter by start date
        transactions = repository.find_by_portfolio(
            sample_portfolio.id, start_date=date(2024, 2, 1)
        )
        assert len(transactions) == 1
        assert transactions[0].transaction_date == date(2024, 2, 20)

        # Filter by end date
        transactions = repository.find_by_portfolio(
            sample_portfolio.id, end_date=date(2024, 1, 31)
        )
        assert len(transactions) == 1
        assert transactions[0].transaction_date == date(2024, 1, 15)

        # Filter by date range
        transactions = repository.find_by_portfolio(
            sample_portfolio.id,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert len(transactions) == 1

    def test_update_transaction(self, test_db, sample_portfolio) -> None:
        """Test updating an existing transaction."""
        repository = TransactionRepository()
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )

        saved = repository.save(txn, sample_portfolio.id)
        saved.shares = 15.0
        saved.price = 160.0

        updated = repository.save(saved, sample_portfolio.id)

        assert updated.shares == 15.0
        assert updated.price == 160.0
        assert updated.id == saved.id

    def test_delete_transaction(self, test_db, sample_portfolio) -> None:
        """Test deleting a transaction."""
        repository = TransactionRepository()
        txn = Transaction(
            transaction_date=date(2024, 1, 15),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10.0,
            price=150.0,
        )

        saved = repository.save(txn, sample_portfolio.id)
        result = repository.delete(saved.id)

        assert result is True

        # Verify deleted
        transactions = repository.find_by_portfolio(sample_portfolio.id)
        assert len(transactions) == 0

    def test_delete_nonexistent_transaction(self, test_db) -> None:
        """Test deleting a non-existent transaction."""
        repository = TransactionRepository()
        result = repository.delete("nonexistent-id")
        assert result is False

