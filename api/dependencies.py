"""FastAPI dependencies for authentication."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from config.settings import settings
from core.auth.tokens import decode_access_token
from core.data_manager.subscription_repository import SubscriptionRepository
from core.data_manager.user_repository import UserRepository
from models.user import User
from services.billing.plans import is_pro_subscription

_bearer_scheme = HTTPBearer(auto_error=False)
_user_repository = UserRepository()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """
    Resolve the authenticated user from Bearer JWT.

    When ``AUTH_DISABLED=true``, returns the system migration user (Streamlit dev).
    """
    if settings.auth_disabled:
        return _user_repository.ensure_system_user()

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _user_repository.find_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return user


_subscription_repository = SubscriptionRepository()


def billing_enforcement_active() -> bool:
    """Skip tier checks for Streamlit dev or explicit opt-out."""
    if settings.auth_disabled:
        return False
    return settings.billing_enforcement


async def require_pro(
    current_user: User = Depends(get_current_user),
) -> User:
    """Require active Pro subscription for paid API features."""
    if not billing_enforcement_active():
        return current_user
    sub = _subscription_repository.ensure_free(current_user.id)
    if not is_pro_subscription(sub):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pro subscription required",
        )
    return current_user
