"""Seeds a demo grower account and orchard into an empty database for local evaluation."""

import logging

from app.core.config import settings
from app.core.security import get_password_hash
from app.core.store import Store

logger = logging.getLogger(__name__)

DEMO_FARM = {
    "name": "Krishnagiri Mango Valley",
    "latitude": 12.5266,
    "longitude": 78.2147,
    "cultivar": "Alphonso & Banganapalli",
    "acreage": 12.5,
    "tree_count": 2,
}

DEMO_TREES = [
    {"tag_code": "KG-R01-T01", "row_number": 1, "tree_number": 1, "variety": "Alphonso", "planted_year": 2016},
    {"tag_code": "KG-R01-T02", "row_number": 1, "tree_number": 2, "variety": "Banganapalli", "planted_year": 2017},
]


def seed_demo_data(store: Store) -> None:
    if not settings.DEMO_USER_EMAIL or store.count_users() > 0:
        return
    user = store.create_user(
        email=settings.DEMO_USER_EMAIL,
        hashed_password=get_password_hash(settings.DEMO_USER_PASSWORD),
        full_name="Krishnagiri Mango Farmer",
    )
    farm = store.create_farm(user["id"], DEMO_FARM)
    for tree in DEMO_TREES:
        store.add_tree(farm["id"], tree)
    logger.info("Seeded demo account %s with orchard '%s'", user["email"], farm["name"])
