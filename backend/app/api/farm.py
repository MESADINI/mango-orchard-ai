"""
AI Mango Orchard Health & Pest Intelligence — Farm & Tree Management Endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid

router = APIRouter()

class FarmCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    cultivar: str = "Alphonso"
    acreage: float = 5.0
    tree_count: int = 150

class TreeCreate(BaseModel):
    tag_code: str
    row_number: int
    tree_number: int
    variety: Optional[str] = "Alphonso"
    planted_year: Optional[int] = 2018

class TreeItem(BaseModel):
    id: str
    farm_id: str
    tag_code: str
    row_number: int
    tree_number: int
    variety: str
    planted_year: int

class FarmItem(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    cultivar: str
    acreage: float
    tree_count: int

# In-memory storage for local farm management
FARMS_DB: Dict[str, Dict[str, Any]] = {
    "farm-001": {
        "id": "farm-001",
        "name": "Krishnagiri Mango Valley",
        "latitude": 12.5266,
        "longitude": 78.2146,
        "cultivar": "Alphonso & Banganapalli",
        "acreage": 12.5,
        "tree_count": 350
    },
    "farm-002": {
        "id": "farm-002",
        "name": "Malihabad Dussehri Estate",
        "latitude": 26.9200,
        "longitude": 80.7100,
        "cultivar": "Dussehri",
        "acreage": 8.0,
        "tree_count": 220
    }
}

TREES_DB: Dict[str, List[Dict[str, Any]]] = {
    "farm-001": [
        {
            "id": "tree-001",
            "farm_id": "farm-001",
            "tag_code": "KG-R01-T01",
            "row_number": 1,
            "tree_number": 1,
            "variety": "Alphonso",
            "planted_year": 2016
        },
        {
            "id": "tree-002",
            "farm_id": "farm-001",
            "tag_code": "KG-R01-T02",
            "row_number": 1,
            "tree_number": 2,
            "variety": "Banganapalli",
            "planted_year": 2017
        }
    ]
}

@router.get("/", response_model=List[FarmItem])
async def list_farms():
    """Returns the list of registered mango orchards."""
    return [FarmItem(**f) for f in FARMS_DB.values()]

@router.post("/", response_model=FarmItem)
async def create_farm(data: FarmCreate):
    """Registers a new mango orchard."""
    farm_id = f"farm-{len(FARMS_DB) + 1:03d}"
    farm_record = {
        "id": farm_id,
        "name": data.name,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "cultivar": data.cultivar,
        "acreage": data.acreage,
        "tree_count": data.tree_count
    }
    FARMS_DB[farm_id] = farm_record
    TREES_DB[farm_id] = []
    return FarmItem(**farm_record)

@router.get("/{farm_id}", response_model=FarmItem)
async def get_farm(farm_id: str):
    """Retrieves orchard details by ID."""
    farm = FARMS_DB.get(farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found.")
    return FarmItem(**farm)

@router.get("/{farm_id}/trees", response_model=List[TreeItem])
async def list_farm_trees(farm_id: str):
    """Lists tagged mango trees in an orchard block."""
    if farm_id not in FARMS_DB:
        raise HTTPException(status_code=404, detail="Farm not found.")
    trees = TREES_DB.get(farm_id, [])
    return [TreeItem(**t) for t in trees]

@router.post("/{farm_id}/trees", response_model=TreeItem)
async def add_tree_to_farm(farm_id: str, data: TreeCreate):
    """Registers a geotagged tree in an orchard."""
    if farm_id not in FARMS_DB:
        raise HTTPException(status_code=404, detail="Farm not found.")
    
    tree_id = f"tree-{uuid.uuid4().hex[:6]}"
    tree_record = {
        "id": tree_id,
        "farm_id": farm_id,
        "tag_code": data.tag_code,
        "row_number": data.row_number,
        "tree_number": data.tree_number,
        "variety": data.variety or "Alphonso",
        "planted_year": data.planted_year or 2018
    }
    TREES_DB.setdefault(farm_id, []).append(tree_record)
    FARMS_DB[farm_id]["tree_count"] = len(TREES_DB[farm_id])
    return TreeItem(**tree_record)
