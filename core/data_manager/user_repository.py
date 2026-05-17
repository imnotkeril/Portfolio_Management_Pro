"""Repository for user persistence."""

from typing import Optional

from sqlalchemy.orm import Session

from core.auth.constants import SYSTEM_USER_EMAIL, SYSTEM_USER_ID
from core.auth.password import hash_password
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

    def ensure_system_user(self) -> UserORM:
        """Create migration/system user if missing (AUTH_DISABLED / backfill)."""
        with get_db_session() as session:
            existing = (
                session.query(UserORM).filter(UserORM.id == SYSTEM_USER_ID).first()
            )
            if existing:
                session.expunge(existing)
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
            return user

    @staticmethod
    def get_in_session(session: Session, user_id: str) -> Optional[UserORM]:
        """Load user within an existing session."""
        return session.query(UserORM).filter(UserORM.id == user_id).first()
