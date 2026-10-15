"""
Local SQLite persistence for users, farms, trees and scan results.

Uses only the Python standard library so the backend works out of the box without
Supabase. Each call opens its own connection, which keeps it safe to use from
FastAPI's threadpool.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'grower',
    hashed_password TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS farms (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    cultivar TEXT NOT NULL,
    acreage REAL NOT NULL,
    tree_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trees (
    id TEXT PRIMARY KEY,
    farm_id TEXT NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    tag_code TEXT NOT NULL,
    row_number INTEGER NOT NULL,
    tree_number INTEGER NOT NULL,
    variety TEXT NOT NULL,
    planted_year INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    farm_id TEXT,
    tree_id TEXT,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_user_created ON scans(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_farms_user ON farms(user_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- Users ----------------
    def create_user(self, email: str, hashed_password: Optional[str], full_name: Optional[str],
                    role: str = "grower") -> Dict[str, Any]:
        user = {
            "id": f"usr-{uuid.uuid4().hex[:12]}",
            "email": email.lower(),
            "full_name": full_name,
            "role": role,
            "hashed_password": hashed_password,
            "created_at": utcnow(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, full_name, role, hashed_password, created_at) "
                "VALUES (:id, :email, :full_name, :role, :hashed_password, :created_at)",
                user,
            )
        return user

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def count_users(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # ---------------- Farms ----------------
    def create_farm(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        farm = {"id": f"farm-{uuid.uuid4().hex[:10]}", "user_id": user_id, "created_at": utcnow(), **data}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO farms (id, user_id, name, latitude, longitude, cultivar, acreage, tree_count, created_at) "
                "VALUES (:id, :user_id, :name, :latitude, :longitude, :cultivar, :acreage, :tree_count, :created_at)",
                farm,
            )
        return farm

    def list_farms(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM farms WHERE user_id = ? ORDER BY created_at, id", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_farm(self, user_id: str, farm_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM farms WHERE id = ? AND user_id = ?", (farm_id, user_id)).fetchone()
        return dict(row) if row else None

    def update_farm(self, user_id: str, farm_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fields = ["name", "latitude", "longitude", "cultivar", "acreage", "tree_count"]
        updates = {k: data[k] for k in fields if k in data}
        if updates:
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE farms SET {assignments} WHERE id = :farm_id AND user_id = :user_id",
                    {**updates, "farm_id": farm_id, "user_id": user_id},
                )
        return self.get_farm(user_id, farm_id)

    # ---------------- Trees ----------------
    def add_tree(self, farm_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        tree = {"id": f"tree-{uuid.uuid4().hex[:10]}", "farm_id": farm_id, "created_at": utcnow(), **data}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trees (id, farm_id, tag_code, row_number, tree_number, variety, planted_year, created_at) "
                "VALUES (:id, :farm_id, :tag_code, :row_number, :tree_number, :variety, :planted_year, :created_at)",
                tree,
            )
        return tree

    def list_trees(self, farm_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trees WHERE farm_id = ? ORDER BY row_number, tree_number", (farm_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- Scans ----------------
    def save_scan(self, user_id: str, scan: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scans (id, user_id, farm_id, tree_id, created_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (scan["scan_id"], user_id, scan.get("farm_id"), scan.get("tree_id"),
                 scan["created_at"], json.dumps(scan)),
            )

    def get_scan(self, user_id: str, scan_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM scans WHERE id = ? AND user_id = ?", (scan_id, user_id)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_scans(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM scans WHERE user_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]
