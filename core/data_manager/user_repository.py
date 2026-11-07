"""Repository for user persistence."""

from typing import Optional

from sqlalchemy.orm import Session

from core.auth.constants import SYSTEM_USER_EMAIL, SYSTEM_USER_ID
from core.auth.password import hash_password
from core.data_manager.subscription_repository import SubscriptionRepository
from database.session import get_db_session
from models.user import User as UserORM


class UserRepository:
    """Repository for user CRUD operations."""

    def create(self, email: str, password: str) -> UserORM:
        """Create a new user with hashed password."""
        normalized = email.strip().lower()
        with get_db_session() as session:
            user = UserORM(
                email=normalized,
                password_hash=hash_password(password),
            )
            session.add(user)
            session.flush()
            session.refresh(user)
            session.expunge(user)
        # Commit the user (close the transaction above) BEFORE creating the
        # free subscription: ensure_free() opens its own session/transaction,
        # which on a real RDBMS cannot see an uncommitted user and would fail
        # the subscriptions.user_id foreign key.
        SubscriptionRepository().ensure_free(user.id)
        return user

    def find_by_email(self, email: str) -> Optional[UserORM]:
        """Find user by email (case-insensitive)."""
        normalized = email.strip().lower()
        with get_db_session() as session:
            user = session.query(UserORM).filter(UserORM.email == normalized).first()
            if user:
                session.expunge(user)
            return user

    def find_by_id(self, user_id: str) -> Optional[UserORM]:
        """Find user by primary key."""
        with get_db_session() as session:
            user = session.query(UserORM).filter(UserORM.id == user_id).first()
            if user:
                session.expunge(user)
            return user

    def delete(self, user_id: str) -> bool:
        """Delete a user and everything it owns. Returns False if not found.

        Children are removed explicitly (not only via ``ON DELETE CASCADE``) so
        the result is the same on databases that do not enforce foreign keys.
        """
        from models.portfolio import Portfolio
        from models.position import Position
        from models.subscription import Subscription
        from models.transaction import Transaction

        with get_db_session() as session:
            portfolio_ids = [
                pid
                for (pid,) in session.query(Portfolio.id).filter(
                    Portfolio.user_id == user_id
                )
            ]
            if portfolio_ids:
                session.query(Transaction).filter(
                    Transaction.portfolio_id.in_(portfolio_ids)
                ).delete(synchronize_session=False)
                session.query(Position).filter(
                    Position.portfolio_id.in_(portfolio_ids)
                ).delete(synchronize_session=False)
                session.query(Portfolio).filter(Portfolio.id.in_(portfolio_ids)).delete(
                    synchronize_session=False
                )
            session.query(Subscription).filter(Subscription.user_id == user_id).delete(
                synchronize_session=False
            )
            removed = (
                session.query(UserORM)
                .filter(UserORM.id == user_id)
                .delete(synchronize_session=False)
            )
            return bool(removed)

    def ensure_system_user(self) -> UserORM:
        """Create migration/system user if missing (AUTH_DISABLED / backfill)."""
        with get_db_session() as session:
            existing = (
                session.query(UserORM).filter(UserORM.id == SYSTEM_USER_ID).first()
            )
            if existing:
                session.expunge(existing)
                SubscriptionRepository().ensure_free(existing.id)
                return existing

            user = UserORM(
                id=SYSTEM_USER_ID,
                email=SYSTEM_USER_EMAIL,
                password_hash=hash_password("not-used-migration-user"),
                is_active=True,
            )
            session.add(user)
            session.flush()
            session.refresh(user)
            session.expunge(user)
            SubscriptionRepository().ensure_free(user.id)
            return user

    @staticmethod
    def get_in_session(session: Session, user_id: str) -> Optional[UserORM]:
        """Load user within an existing session."""
        return session.query(UserORM).filter(UserORM.id == user_id).first()
