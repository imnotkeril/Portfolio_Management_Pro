"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError

from api.dependencies import get_current_user
from api.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut
from core.auth.password import verify_password
from core.auth.tokens import create_access_token
from core.data_manager.user_repository import UserRepository
from models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])
_user_repository = UserRepository()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> User:
    """Register a new user account."""
    if _user_repository.find_by_email(payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    try:
        return _user_repository.create(payload.email, payload.password)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc


@router.post("/login", response_model=TokenResponse)
def login_form(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Login with OAuth2 password form (username field = email)."""
    return _authenticate(form.username, form.password)


@router.post("/login/json", response_model=TokenResponse)
def login_json(payload: LoginRequest) -> TokenResponse:
    """Login with JSON body (convenient for frontend)."""
    return _authenticate(payload.email, payload.password)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the authenticated user profile."""
    return current_user


def _authenticate(email: str, password: str) -> TokenResponse:
    user = _user_repository.find_by_email(email)
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    token = create_access_token(subject=user.id)
    return TokenResponse(access_token=token)
