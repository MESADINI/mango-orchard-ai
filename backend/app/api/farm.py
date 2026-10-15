"""
AI Mango Orchard Health & Pest Intelligence — Farm & Tree Management Endpoints
Farms are owned by the authenticated user.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import get_current_user, get_store
from app.core.store import Store

router = APIRouter()


class FarmCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    cultivar: str = Field(default="Alphonso", max_length=120)
    acreage: float = Field(default=5.0, ge=0)
    tree_count: int = Field(default=0, ge=0)


class FarmUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    cultivar: Optional[str] = Field(default=None, max_length=120)
    acreage: Optional[float] = Field(default=None, ge=0)
    tree_count: Optional[int] = Field(default=None, ge=0)


class TreeCreate(BaseModel):
    tag_code: str = Field(min_length=1, max_length=50)
    row_number: int = Field(ge=0)
    tree_number: int = Field(ge=0)
    variety: str = Field(default="Alphonso", max_length=100)
    planted_year: Optional[int] = Field(default=None, ge=1800, le=2100)


class TreeItem(BaseModel):
    id: str
    farm_id: str
    tag_code: str
    row_number: int
    tree_number: int
    variety: str
    planted_year: Optional[int] = None


class FarmItem(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    cultivar: str
    acreage: float
    tree_count: int


def _require_farm(store: Store, user: Dict[str, Any], farm_id: str) -> Dict[str, Any]:
    farm = store.get_farm(user["id"], farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found.")
    return farm


@router.get("/", response_model=List[FarmItem])
def list_farms(user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Returns the current user's registered mango orchards."""
    return [FarmItem(**f) for f in store.list_farms(user["id"])]


@router.post("/", response_model=FarmItem, status_code=201)
def create_farm(data: FarmCreate, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Registers a new mango orchard."""
    return FarmItem(**store.create_farm(user["id"], data.model_dump()))


@router.get("/{farm_id}", response_model=FarmItem)
def get_farm(farm_id: str, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Retrieves orchard details by ID."""
    return FarmItem(**_require_farm(store, user, farm_id))


@router.put("/{farm_id}", response_model=FarmItem)
def update_farm(farm_id: str, data: FarmUpdate, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Updates orchard metadata (name, coordinates, cultivar, acreage, tree count)."""
    _require_farm(store, user, farm_id)
    return FarmItem(**store.update_farm(user["id"], farm_id, data.model_dump(exclude_none=True)))


@router.get("/{farm_id}/trees", response_model=List[TreeItem])
def list_farm_trees(farm_id: str, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Lists tagged mango trees in an orchard block."""
    _require_farm(store, user, farm_id)
    return [TreeItem(**t) for t in store.list_trees(farm_id)]


@router.post("/{farm_id}/trees", response_model=TreeItem, status_code=201)
def add_tree_to_farm(farm_id: str, data: TreeCreate, user=Depends(get_current_user),
                     store: Store = Depends(get_store)):
    """Registers a tagged tree in an orchard."""
    _require_farm(store, user, farm_id)
    return TreeItem(**store.add_tree(farm_id, data.model_dump()))
