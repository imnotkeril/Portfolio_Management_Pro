"""Auth request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """User registration payload."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """JSON login payload (alternative to OAuth2 form)."""

    email: EmailStr
    password: str


class DeleteAccountRequest(BaseModel):
    """Password confirmation required to delete the account."""

    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    """Public user profile."""

    id: str
    email: EmailStr
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}
