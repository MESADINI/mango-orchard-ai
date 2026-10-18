"""demo_data/seed_demo.py is idempotent and always restores the documented demo login."""

import importlib.util
from pathlib import Path

from app.core.security import get_password_hash, verify_password
from app.core.store import Store

SPEC = importlib.util.spec_from_file_location(
    "seed_demo", Path(__file__).resolve().parents[2] / "demo_data" / "seed_demo.py")
seed_demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed_demo)


def test_seed_is_idempotent_and_restores_password(tmp_path):
    store = Store(tmp_path / "demo.db")
    first = seed_demo.seed(store)
    assert first["user"] == "grower@mango.ai" and len(first["trees"]) == 4
    second = seed_demo.seed(store)
    assert second == first and store.count_users() == 1

    user = store.get_user_by_email("grower@mango.ai")
    with store._connect() as conn:
        conn.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (get_password_hash("other"), user["id"]))
    seed_demo.seed(store)
    assert verify_password("orchard123", store.get_user_by_email("grower@mango.ai")["hashed_password"])


def test_clear_history_removes_only_demo_scans(tmp_path):
    store = Store(tmp_path / "demo.db")
    seed_demo.seed(store)
    demo = store.get_user_by_email("grower@mango.ai")
    other = store.create_user("other@example.org", None, "Other")
    for uid, sid in ((demo["id"], "s1"), (other["id"], "s2")):
        store.save_scan(uid, {"scan_id": sid, "created_at": "2026-01-01T00:00:00Z"})
    assert seed_demo.seed(store, clear_history=True)["scans"] == 0
    assert len(store.list_scans(other["id"])) == 1
