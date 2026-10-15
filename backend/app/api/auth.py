"""
AI Mango Orchard Health & Pest Intelligence — Authentication Endpoint
Handles user registration, login, JWT token issuance, and profile retrieval.

Accounts live in the local SQLite store. If Supabase is configured, login is
first attempted against Supabase Auth and the user is mirrored locally; the
backend always issues its own JWT so every endpoint validates tokens the same way.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.database import get_supabase_client
from app.core.deps import get_current_user, get_store
from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.store import Store

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalise_email(value: str) -> str:
    value = value.strip().lower()
    if "@" not in value or value.startswith("@") or value.endswith("@") or len(value) > 254:
        raise ValueError("Enter a valid email address.")
    return value


class UserLogin(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)

    _email = field_validator("email")(_normalise_email)


class UserRegister(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=120)

    _email = field_validator("email")(_normalise_email)


class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None
    role: str = "grower"


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


def _profile(user: Dict[str, Any]) -> UserProfile:
    return UserProfile(user_id=user["id"], email=user["email"], full_name=user.get("full_name"), role=user["role"])


def _auth_response(user: Dict[str, Any]) -> AuthResponse:
    return AuthResponse(access_token=create_access_token(user["id"]), user=_profile(user))


def _try_supabase_login(credentials: UserLogin, store: Store) -> Optional[Dict[str, Any]]:
    supabase = get_supabase_client()
    if supabase is None:
        return None
    try:
        res = supabase.auth.sign_in_with_password({"email": credentials.email, "password": credentials.password})
    except Exception as e:
        logger.info("Supabase login rejected for %s: %s", credentials.email, e)
        return None
    if not res.user:
        return None
    user = store.get_user_by_email(credentials.email)
    if user is None:
        user = store.create_user(
            email=credentials.email,
            hashed_password=None,
            full_name=(res.user.user_metadata or {}).get("full_name"),
        )
    return user


@router.post("/login", response_model=AuthResponse)
def login(credentials: UserLogin, store: Store = Depends(get_store)):
    """Authenticate with email and password."""
    user = store.get_user_by_email(credentials.email)
    if user and verify_password(credentials.password, user["hashed_password"]):
        return _auth_response(user)

    supabase_user = _try_supabase_login(credentials, store)
    if supabase_user:
        return _auth_response(supabase_user)

    raise HTTPException(status_code=401, detail="Invalid email or password.")


@router.post("/register", response_model=AuthResponse)
def register(data: UserRegister, store: Store = Depends(get_store)):
    """Register a new grower account. Self-registration always creates the 'grower' role."""
    if store.get_user_by_email(data.email):
        raise HTTPException(status_code=400, detail="Account with this email already exists.")
    user = store.create_user(
        email=data.email,
        hashed_password=get_password_hash(data.password),
        full_name=data.full_name or None,
    )
    return _auth_response(user)


@router.get("/me", response_model=UserProfile)
def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    """Validates the JWT bearer token and returns the grower profile."""
    return _profile(user)
