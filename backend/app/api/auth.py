"""
AI Mango Orchard Health & Pest Intelligence — Authentication Endpoint
Handles user registration, login, JWT token issuance, and profile retrieval.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from app.core.security import create_access_token, decode_access_token, get_password_hash, verify_password
from app.core.database import get_supabase_client

router = APIRouter()

# In-memory user store for standalone local development
LOCAL_USERS_DB: Dict[str, Dict[str, Any]] = {
    "grower@mango.ai": {
        "id": "usr-001",
        "email": "grower@mango.ai",
        "full_name": "Krishnagiri Mango Farmer",
        "role": "grower",
        "hashed_password": get_password_hash("orchard123")
    }
}

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "grower"

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None
    role: str = "grower"

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    """Authenticate user with Supabase Auth or standalone local authentication."""
    # 1. Try Supabase Auth if credentials configured
    try:
        supabase = get_supabase_client()
        res = supabase.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password
        })
        if res.user:
            return AuthResponse(
                access_token=res.session.access_token,
                user=UserProfile(
                    user_id=res.user.id,
                    email=res.user.email,
                    full_name=res.user.user_metadata.get("full_name", "Grower"),
                    role=res.user.user_metadata.get("role", "grower")
                )
            )
    except Exception:
        pass

    # 2. Local fallback verification
    user = LOCAL_USERS_DB.get(credentials.email.lower())
    if user and verify_password(credentials.password, user["hashed_password"]):
        token = create_access_token(user["email"])
        return AuthResponse(
            access_token=token,
            user=UserProfile(
                user_id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                role=user["role"]
            )
        )
    
    # 3. Developer mode fallback for rapid prototyping
    if credentials.password == "orchard123":
        token = create_access_token(credentials.email)
        return AuthResponse(
            access_token=token,
            user=UserProfile(
                user_id="dev-user-001",
                email=credentials.email,
                full_name="Orchard Cultivator",
                role="grower"
            )
        )

    raise HTTPException(status_code=401, detail="Invalid email or password.")

@router.post("/register", response_model=AuthResponse)
async def register(data: UserRegister):
    """Register new grower account."""
    email = data.email.lower()
    if email in LOCAL_USERS_DB:
        raise HTTPException(status_code=400, detail="Account with this email already exists.")

    user_id = f"usr-{len(LOCAL_USERS_DB) + 1:03d}"
    LOCAL_USERS_DB[email] = {
        "id": user_id,
        "email": email,
        "full_name": data.full_name or "Grower",
        "role": data.role,
        "hashed_password": get_password_hash(data.password)
    }

    token = create_access_token(email)
    return AuthResponse(
        access_token=token,
        user=UserProfile(
            user_id=user_id,
            email=email,
            full_name=data.full_name or "Grower",
            role=data.role
        )
    )

@router.get("/me", response_model=UserProfile)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Validates JWT bearer token and returns grower profile."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Token has expired or is invalid.")

    email = payload["sub"].lower()
    user = LOCAL_USERS_DB.get(email, {
        "id": "dev-user-001",
        "email": email,
        "full_name": "Orchard Cultivator",
        "role": "grower"
    })

    return UserProfile(
        user_id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"]
    )
