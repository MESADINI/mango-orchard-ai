"""Shared FastAPI dependencies: persistence store and authenticated user."""

from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.store import Store


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(settings.DATABASE_PATH)


def get_current_user(
    authorization: Optional[str] = Header(None),
    store: Store = Depends(get_store),
) -> Dict[str, Any]:
    """Validates the `Authorization: Bearer <jwt>` header and returns the stored user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")

    payload = decode_access_token(authorization[len("Bearer "):].strip())
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Token has expired or is invalid.")

    user = store.get_user(payload["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="User account no longer exists.")
    return user
