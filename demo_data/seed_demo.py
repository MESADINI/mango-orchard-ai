"""
Creates the demo account, farm and trees idempotently (safe to run any number of times, with or without the app
running). Uses the demo credentials only.
    backend\\venv\\Scripts\\python.exe demo_data\\seed_demo.py                  # create whatever is missing
    backend\\venv\\Scripts\\python.exe demo_data\\seed_demo.py --clear-history  # also delete the demo user's scans
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.security import get_password_hash, verify_password  # noqa: E402
from app.core.seed import DEMO_FARM, DEMO_TREES  # noqa: E402
from app.core.store import Store  # noqa: E402

DEMO_EMAIL = "grower@mango.ai"
DEMO_PASSWORD = "orchard123"
EXTRA_TREES = [
    {"tag_code": "KG-R02-T01", "row_number": 2, "tree_number": 1, "variety": "Alphonso", "planted_year": 2015},
    {"tag_code": "KG-R02-T02", "row_number": 2, "tree_number": 2, "variety": "Totapuri", "planted_year": 2018},
]


def seed(store: Store, clear_history: bool = False) -> dict:
    user = store.get_user_by_email(DEMO_EMAIL)
    if user is None:
        user = store.create_user(DEMO_EMAIL, get_password_hash(DEMO_PASSWORD), "Krishnagiri Mango Farmer")
    elif not user.get("hashed_password") or not verify_password(DEMO_PASSWORD, user["hashed_password"]):
        with store._connect() as conn:          # restore the documented demo password
            conn.execute("UPDATE users SET hashed_password = ? WHERE id = ?",
                         (get_password_hash(DEMO_PASSWORD), user["id"]))
    farm = next((f for f in store.list_farms(user["id"]) if f["name"] == DEMO_FARM["name"]), None)
    if farm is None:
        farm = store.create_farm(user["id"], DEMO_FARM)
    have = {t["tag_code"] for t in store.list_trees(farm["id"])}
    for tree in DEMO_TREES + EXTRA_TREES:
        if tree["tag_code"] not in have:
            store.add_tree(farm["id"], tree)
    trees = store.list_trees(farm["id"])
    store.update_farm(user["id"], farm["id"], {"tree_count": len(trees)})
    if clear_history:
        with store._connect() as conn:
            conn.execute("DELETE FROM scans WHERE user_id = ?", (user["id"],))
    return {"user": user["email"], "farm": farm["name"], "trees": [t["tag_code"] for t in trees],
            "scans": len(store.list_scans(user["id"], limit=1000))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clear-history", action="store_true", help="delete the demo user's previous scans")
    args = ap.parse_args()
    out = seed(Store(settings.DATABASE_PATH), args.clear_history)
    print(f"Demo account {out['user']} (password {DEMO_PASSWORD}); farm '{out['farm']}' with trees "
          f"{', '.join(out['trees'])}; {out['scans']} scan(s) in history.")
