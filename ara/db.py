# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ARA's persistent store — the one place that remembers across runs.

A small SQLite database in the per-OS user data dir. Three concerns, cleanly separated:
machine calibration (per machine + engine), the model catalog (metadata), and
characterizations (a model's fitted safe-context ceiling on this machine + engine).

This is ARA's job, not the engine's: an engine measures; ARA discovers, catalogs, and
remembers — once, for whichever engine is in play.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import platformdirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_key       TEXT NOT NULL,
    engine            TEXT NOT NULL,
    fixed_overhead_gb REAL,
    calibrated_at     TEXT,
    PRIMARY KEY (machine_key, engine)
);

CREATE TABLE IF NOT EXISTS models (
    model_id     TEXT PRIMARY KEY,
    modality     TEXT,
    params       INTEGER,
    quant        TEXT,
    n_layers     INTEGER,
    hidden_size  INTEGER,
    kv_heads     INTEGER,
    head_dim     INTEGER,
    weights_gb   REAL,
    max_context  INTEGER,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS characterizations (
    machine_key   TEXT NOT NULL,
    engine        TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    safe_context  INTEGER,
    decode_context INTEGER,
    points_json   TEXT,
    measured_at   TEXT,
    PRIMARY KEY (machine_key, engine, model_id)
);
"""


def _db_path() -> Path:
    """Where the store lives — ``ARA_DB_PATH`` if set (tests), else the OS data dir."""
    override = os.environ.get("ARA_DB_PATH")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("ara")) / "ara.db"


def connect() -> sqlite3.Connection:
    """Open (creating if needed) the store, with the schema applied. Rows come back as dicts."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(characterizations)")}
    if "decode_context" not in cols:
        con.execute("ALTER TABLE characterizations ADD COLUMN decode_context INTEGER")
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- machine profiles (per machine + engine) ---
def upsert_machine(con: sqlite3.Connection, machine_key: str, engine: str, *,
                   fixed_overhead_gb: float, calibrated_at: str) -> None:
    con.execute(
        "INSERT INTO machines (machine_key, engine, fixed_overhead_gb, calibrated_at) "
        "VALUES (?,?,?,?) ON CONFLICT(machine_key, engine) DO UPDATE SET "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, calibrated_at=excluded.calibrated_at",
        (machine_key, engine, fixed_overhead_gb, calibrated_at))
    con.commit()


def get_machine(con: sqlite3.Connection, machine_key: str, engine: str) -> dict | None:
    row = con.execute("SELECT * FROM machines WHERE machine_key=? AND engine=?",
                      (machine_key, engine)).fetchone()
    return dict(row) if row else None


# --- model catalog ---
_MODEL_COLS = ("modality", "params", "quant", "n_layers", "hidden_size",
               "kv_heads", "head_dim", "weights_gb", "max_context")


def upsert_model(con: sqlite3.Connection, model_id: str, **fields) -> None:
    row = {"model_id": model_id, **{c: fields.get(c) for c in _MODEL_COLS}, "updated_at": _now()}
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{c}=excluded.{c}" for c in (*_MODEL_COLS, "updated_at"))
    con.execute(
        f"INSERT INTO models ({', '.join(row)}) VALUES ({placeholders}) "
        f"ON CONFLICT(model_id) DO UPDATE SET {updates}", row)
    con.commit()


def get_model(con: sqlite3.Connection, model_id: str) -> dict | None:
    row = con.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
    return dict(row) if row else None


def list_models(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM models ORDER BY model_id")]


# --- characterizations (fitted ceilings, per machine + engine + model) ---
def save_characterization(con: sqlite3.Connection, machine_key: str, engine: str,
                          model_id: str, *, safe_context: int | None,
                          points: list, measured_at: str | None = None,
                          decode_context: int | None = None) -> None:
    con.execute(
        "INSERT INTO characterizations "
        "(machine_key, engine, model_id, safe_context, decode_context, points_json, measured_at) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(machine_key, engine, model_id) DO UPDATE SET "
        "safe_context=excluded.safe_context, decode_context=excluded.decode_context, "
        "points_json=excluded.points_json, measured_at=excluded.measured_at",
        (machine_key, engine, model_id, safe_context, decode_context,
         json.dumps(points), measured_at or _now()))
    con.commit()


def get_characterization(con: sqlite3.Connection, machine_key: str, engine: str,
                         model_id: str) -> dict | None:
    row = con.execute(
        "SELECT * FROM characterizations WHERE machine_key=? AND engine=? AND model_id=?",
        (machine_key, engine, model_id)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["points"] = json.loads(d["points_json"]) if d["points_json"] else []
    return d


def list_characterizations(con: sqlite3.Connection, machine_key: str,
                           engine: str) -> list[dict]:
    """Every model characterized on this machine + engine, newest fields parsed."""
    rows = con.execute(
        "SELECT * FROM characterizations WHERE machine_key=? AND engine=? ORDER BY model_id",
        (machine_key, engine)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["points"] = json.loads(d["points_json"]) if d["points_json"] else []
        out.append(d)
    return out
