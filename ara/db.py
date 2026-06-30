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
CREATE TABLE IF NOT EXISTS calibrations (
    machine_key       TEXT NOT NULL,
    engine            TEXT NOT NULL,
    fixed_overhead_gb REAL,
    calibrated_at     TEXT,
    wall_gb           REAL,
    safe_budget_gb    REAL,
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

CREATE TABLE IF NOT EXISTS profiles (
    machine_key  TEXT NOT NULL,
    captured_at  TEXT NOT NULL,
    profile_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    machine_key  TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    use_case     TEXT NOT NULL,
    engine_key   TEXT,
    backend      TEXT,
    base_model   TEXT,
    quant        TEXT,
    benchmark_id TEXT,
    tier         TEXT NOT NULL DEFAULT 'measured',
    score        REAL NOT NULL,
    max_score    REAL,
    sample_size  INTEGER,
    source       TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    PRIMARY KEY (machine_key, model_id, use_case)
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,        -- characterize | run | serve | benchmark
    args_json   TEXT NOT NULL,
    status      TEXT NOT NULL,        -- queued | running | done | failed
    result_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
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
    # Measured wall + safe budget joined the calibration store later — add to old DBs.
    cal_cols = {r["name"] for r in con.execute("PRAGMA table_info(calibrations)")}
    if "wall_gb" not in cal_cols:
        con.execute("ALTER TABLE calibrations ADD COLUMN wall_gb REAL")
    if "safe_budget_gb" not in cal_cols:
        con.execute("ALTER TABLE calibrations ADD COLUMN safe_budget_gb REAL")
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- per-engine calibration ---
def upsert_calibration(con: sqlite3.Connection, machine_key: str, engine: str, *,
                   fixed_overhead_gb: float, calibrated_at: str,
                   wall_gb: float | None = None, safe_budget_gb: float | None = None) -> None:
    con.execute(
        "INSERT INTO calibrations "
        "(machine_key, engine, fixed_overhead_gb, calibrated_at, wall_gb, safe_budget_gb) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(machine_key, engine) DO UPDATE SET "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, calibrated_at=excluded.calibrated_at, "
        "wall_gb=excluded.wall_gb, safe_budget_gb=excluded.safe_budget_gb",
        (machine_key, engine, fixed_overhead_gb, calibrated_at, wall_gb, safe_budget_gb))
    con.commit()


def get_calibration(con: sqlite3.Connection, machine_key: str, engine: str) -> dict | None:
    row = con.execute("SELECT * FROM calibrations WHERE machine_key=? AND engine=?",
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


# --- system profiles (the persisted Machine snapshot; Spec 2026-06-23-capability-pipeline) ---
def save_profile(con: sqlite3.Connection, machine_key: str, profile_json: str,
                 captured_at: str | None = None) -> None:
    """Append a profile capture. History is kept — one row per capture."""
    con.execute(
        "INSERT INTO profiles (machine_key, captured_at, profile_json) VALUES (?,?,?)",
        (machine_key, captured_at or _now(), profile_json))
    con.commit()


def get_latest_profile(con: sqlite3.Connection, machine_key: str) -> dict | None:
    """The most recent profile capture for this machine, or None."""
    row = con.execute(
        "SELECT * FROM profiles WHERE machine_key=? ORDER BY captured_at DESC, rowid DESC LIMIT 1",
        (machine_key,)).fetchone()
    return dict(row) if row else None


def list_profiles(con: sqlite3.Connection, machine_key: str) -> list[dict]:
    """Every profile capture for this machine, newest first."""
    rows = con.execute(
        "SELECT * FROM profiles WHERE machine_key=? ORDER BY captured_at DESC, rowid DESC",
        (machine_key,)).fetchall()
    return [dict(r) for r in rows]


# --- benchmark results (scored model × use-case outcomes, per machine) ---
def save_benchmark_result(con: sqlite3.Connection, machine_key: str, model_id: str,
                          use_case: str, *, score: float, source: str,
                          engine_key: str | None = None, backend: str | None = None,
                          base_model: str | None = None, quant: str | None = None,
                          benchmark_id: str | None = None, max_score: float | None = None,
                          sample_size: int | None = None, tier: str = "measured") -> None:
    con.execute(
        "INSERT INTO benchmark_results "
        "(machine_key, model_id, use_case, engine_key, backend, base_model, quant, "
        "benchmark_id, tier, score, max_score, sample_size, source, measured_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(machine_key, model_id, use_case) DO UPDATE SET "
        "engine_key=excluded.engine_key, backend=excluded.backend, "
        "base_model=excluded.base_model, quant=excluded.quant, "
        "benchmark_id=excluded.benchmark_id, tier=excluded.tier, score=excluded.score, "
        "max_score=excluded.max_score, sample_size=excluded.sample_size, "
        "source=excluded.source, measured_at=excluded.measured_at",
        (machine_key, model_id, use_case, engine_key, backend, base_model, quant,
         benchmark_id, tier, score, max_score, sample_size, source, _now()))
    con.commit()


def get_benchmark_result(con: sqlite3.Connection, machine_key: str, model_id: str,
                         use_case: str) -> dict | None:
    row = con.execute(
        "SELECT * FROM benchmark_results WHERE machine_key=? AND model_id=? AND use_case=?",
        (machine_key, model_id, use_case)).fetchone()
    return dict(row) if row else None


def list_benchmark_results(con: sqlite3.Connection, machine_key: str) -> list[dict]:
    """All benchmark results for a machine, ordered by model_id then use_case."""
    rows = con.execute(
        "SELECT * FROM benchmark_results WHERE machine_key=? ORDER BY model_id, use_case",
        (machine_key,)).fetchall()
    return [dict(r) for r in rows]


# --- node job store (the `ara node` daemon's async jobs) ---
def create_job(con: sqlite3.Connection, job_id: str, kind: str, args_json: str,
               *, created_at: str | None = None) -> None:
    """Record a new job in the ``queued`` state. ``created_at`` defaults to now (tests pin it)."""
    con.execute(
        "INSERT INTO jobs (id, kind, args_json, status, created_at) VALUES (?,?,?,?,?)",
        (job_id, kind, args_json, "queued", created_at or _now()))
    con.commit()


def get_job(con: sqlite3.Connection, job_id: str) -> dict | None:
    """The job row as a dict, or None if there's no such id."""
    row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def update_job(con: sqlite3.Connection, job_id: str, **fields) -> None:
    """Patch a job's mutable fields (status/result_json/error/started_at/finished_at). Unknown ids
    and fields are no-ops — only recognised columns are written, so a partial update never clobbers
    the fields it doesn't mention."""
    cols = [k for k in ("status", "result_json", "error", "started_at", "finished_at")
            if k in fields]
    if not cols:
        return
    con.execute(f"UPDATE jobs SET {', '.join(f'{c}=?' for c in cols)} WHERE id = ?",
                (*(fields[c] for c in cols), job_id))
    con.commit()


def list_jobs(con: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    """All jobs, newest first (id breaks ties so same-timestamp order is stable)."""
    sql = "SELECT * FROM jobs ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in con.execute(sql).fetchall()]
