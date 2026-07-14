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
import tempfile
import time
from contextlib import contextmanager
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
    refused_n    INTEGER,
    errored_n    INTEGER,
    source       TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    PRIMARY KEY (machine_key, model_id, use_case)
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
    # Per-run honesty counts joined benchmark_results later — add to old DBs (Rule #3). NULL is
    # legacy/unknown (an un-annotated run); 0 is a measured clean run.
    bench_cols = {r["name"] for r in con.execute("PRAGMA table_info(benchmark_results)")}
    if "refused_n" not in bench_cols:
        con.execute("ALTER TABLE benchmark_results ADD COLUMN refused_n INTEGER")
    if "errored_n" not in bench_cols:
        con.execute("ALTER TABLE benchmark_results ADD COLUMN errored_n INTEGER")
    # One-time data fix (user_version 0→1): wmx calibrations stored before 2026-07-02 carry
    # decimal-GB walls (~7.4% high vs ARA's binary-GiB contract — the apple boundary now
    # converts). A float can't reveal its own units, so honest re-measurement beats arithmetic
    # repair: drop the rows and the next run re-calibrates. Slug 2026-07-02-analytic-units-gib.
    if con.execute("PRAGMA user_version").fetchone()[0] < 1:
        con.execute("DELETE FROM calibrations WHERE engine='wmx'")
        con.execute("PRAGMA user_version = 1")
        con.commit()
    # One-time rekey (user_version 1→2): legacy byte-exact machine_keys → the versioned
    # GiB-rounded format, rescuing measurements orphaned by reboot RAM drift (Rule #1 data-loss).
    # Idempotent + data-preserving. Slug 2026-07-04-machine-key-stabilization.
    if con.execute("PRAGMA user_version").fetchone()[0] < 2:
        _rekey_legacy(con)
        con.execute("PRAGMA user_version = 2")
        con.commit()
    if con.execute("PRAGMA user_version").fetchone()[0] < 3:
        _backup_before_engine_identity_v3(con, path)
        try:
            con.execute("BEGIN")
            _migrate_engine_identity_v3(con)
            con.execute("PRAGMA user_version = 3")
            con.commit()
        except Exception:
            con.rollback()
            con.close()
            raise
    return con


# Tables keyed on machine_key that have a composite PRIMARY KEY (so a rekey can collide): the
# remaining machine_key columns plus the timestamp used to pick the survivor on a merge.
_REKEY_PK_TABLES = (
    ("calibrations", ("engine",), "calibrated_at"),
    ("characterizations", ("engine", "model_id"), "measured_at"),
    ("benchmark_results", ("model_id", "use_case"), "measured_at"),
)

_ENGINE_REKEY_TABLES = (
    ("calibrations", ("machine_key", "engine"), "calibrated_at"),
    ("characterizations", ("machine_key", "engine", "model_id"), "measured_at"),
)


def _backup_before_engine_identity_v3(con: sqlite3.Connection, path: Path) -> None:
    """Keep one byte-independent SQLite backup of the pre-v3 evidence store."""
    backup_path = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    expected_version = con.execute("PRAGMA user_version").fetchone()[0]

    def valid(candidate: Path) -> bool:
        try:
            check = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
            try:
                return (check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                        and check.execute("PRAGMA user_version").fetchone()[0]
                        == expected_version)
            finally:
                check.close()
        except sqlite3.Error:
            return False

    now = time.time()
    for orphan in backup_path.parent.glob(backup_path.name + ".*.tmp"):
        try:
            if now - orphan.stat().st_mtime > 86_400:
                orphan.unlink()
        except FileNotFoundError:
            pass
    if valid(backup_path):
        return
    fd, temp_name = tempfile.mkstemp(
        prefix=backup_path.name + ".", suffix=".tmp", dir=backup_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        backup = sqlite3.connect(temp_path)
        try:
            con.backup(backup)
        finally:
            backup.close()
        if not valid(temp_path):
            raise sqlite3.DatabaseError("pre-v3 backup validation failed")
        os.replace(temp_path, backup_path)
        if not valid(backup_path):
            raise sqlite3.DatabaseError("published pre-v3 backup validation failed")
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _canonicalize_engine_pk_table(con: sqlite3.Connection, table: str,
                                  key_cols: tuple[str, ...], timestamp: str) -> None:
    """Canonicalize an engine-bearing PK while retaining one complete newest row per key."""
    from ara.engine_identity import canonical_engine

    rows = [dict(r) for r in con.execute(f"SELECT rowid, * FROM {table}")]  # noqa: S608
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(canonical_engine(row[col]) if col == "engine" else row[col]
                    for col in key_cols)
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        if len(group) == 1 and group[0]["engine"] == key[key_cols.index("engine")]:
            continue
        winner = max(group, key=lambda row: ((row[timestamp] or ""), row["rowid"]))
        cols = [col for col in winner if col != "rowid"]
        for row in group:
            con.execute(f"DELETE FROM {table} WHERE rowid=?", (row["rowid"],))  # noqa: S608
        values = [key[key_cols.index(col)] if col in key_cols else winner[col] for col in cols]
        con.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            values,
        )  # noqa: S608


def _migrate_engine_identity_v3(con: sqlite3.Connection) -> None:
    """Rewrite legacy engine identities in persisted and serialized evidence."""
    from ara.engine_identity import canonical_engine

    for table, key_cols, timestamp in _ENGINE_REKEY_TABLES:
        _canonicalize_engine_pk_table(con, table, key_cols, timestamp)
    benchmark_cols = {row["name"] for row in con.execute("PRAGMA table_info(benchmark_results)")}
    if "engine_key" in benchmark_cols:
        con.execute("UPDATE benchmark_results SET engine_key='mlx' WHERE engine_key IN ('wmx','wmx-suite')")
        con.execute("UPDATE benchmark_results SET engine_key='cuda' WHERE engine_key IN ('wcx','wcx-suite')")
    for row in con.execute("SELECT rowid, profile_json FROM profiles").fetchall():
        profile_json = json.loads(row["profile_json"])
        changed = False
        for section in ("machine", "projection"):
            record = profile_json.get(section)
            if isinstance(record, dict) and "engine" in record:
                canonical = canonical_engine(record["engine"])
                if canonical != record["engine"]:
                    record["engine"] = canonical
                    changed = True
        if changed:
            con.execute("UPDATE profiles SET profile_json=? WHERE rowid=?",
                        (json.dumps(profile_json, separators=(",", ":")), row["rowid"]))


def _rekey_pk_table(con: sqlite3.Connection, table: str, pk_rest: tuple[str, ...],
                    ts: str) -> int:
    """Rewrite legacy machine_keys in one composite-PK *table* to their versioned form, merging
    collisions (two legacy keys collapsing to one) by keeping the row with the newest *ts*. Returns
    the number of legacy rows rekeyed."""
    from ara import profile
    rows = [dict(r) for r in con.execute(f"SELECT rowid, * FROM {table}")]  # noqa: S608 — fixed table
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        new_key = profile.rekey_legacy_key(r["machine_key"]) or r["machine_key"]
        groups.setdefault((new_key, *(r[c] for c in pk_rest)), []).append(r)
    rekeyed = 0
    for (new_key, *_), grp in groups.items():
        legacy_n = sum(1 for r in grp if profile.rekey_legacy_key(r["machine_key"]))
        if not legacy_n:                       # pure non-legacy group — leave untouched
            continue
        winner = max(grp, key=lambda r: ((r[ts] or ""), r["rowid"]))
        cols = [c for c in winner if c != "rowid"]
        for r in grp:                          # clear the whole colliding group, then reinsert one
            con.execute(f"DELETE FROM {table} WHERE rowid=?", (r["rowid"],))  # noqa: S608
        vals = [new_key if c == "machine_key" else winner[c] for c in cols]
        con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    vals)                       # noqa: S608 — cols are DB-derived identifiers
        rekeyed += legacy_n
    return rekeyed


def _rekey_legacy(con: sqlite3.Connection) -> int:
    """Migrate every legacy (byte-exact) machine_key to the versioned GiB-rounded form across all
    four keyed tables. Idempotent (no legacy keys → 0, no writes) and data-preserving. Returns the
    number of DB rows rekeyed. See ``profile.rekey_legacy_key`` and Spec
    2026-07-04-machine-key-stabilization."""
    from ara import profile
    n = 0
    for r in con.execute("SELECT rowid, machine_key FROM profiles").fetchall():
        new_key = profile.rekey_legacy_key(r["machine_key"])   # append-only: no PK, no collision
        if new_key:
            con.execute("UPDATE profiles SET machine_key=? WHERE rowid=?", (new_key, r["rowid"]))
            n += 1
    for table, pk_rest, ts in _REKEY_PK_TABLES:
        n += _rekey_pk_table(con, table, pk_rest, ts)
    con.commit()
    return n


@contextmanager
def connected():
    """Open the store, yield the connection, and close it on exit — even on error.

    The write helpers (``upsert_*``/``save_*``) commit as they go, so this contract is purely
    about releasing the handle: SQLite connections are a finite OS resource, and every
    ``con = connect()`` caller that returned without closing leaked one. Use as::

        with db.connected() as con:
            ...
    """
    con = connect()
    try:
        yield con
    finally:
        con.close()


@contextmanager
def connected_readonly():
    """Open the existing store query-only, without creating or migrating it."""
    uri = _db_path().resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    try:
        yield con
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- per-engine calibration ---
def upsert_calibration(con: sqlite3.Connection, machine_key: str, engine: str, *,
                   fixed_overhead_gb: float, calibrated_at: str,
                   wall_gb: float | None = None, safe_budget_gb: float | None = None) -> None:
    from ara.engine_identity import canonical_engine
    con.execute(
        "INSERT INTO calibrations "
        "(machine_key, engine, fixed_overhead_gb, calibrated_at, wall_gb, safe_budget_gb) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(machine_key, engine) DO UPDATE SET "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, calibrated_at=excluded.calibrated_at, "
        "wall_gb=excluded.wall_gb, safe_budget_gb=excluded.safe_budget_gb",
        (machine_key, canonical_engine(engine), fixed_overhead_gb, calibrated_at,
         wall_gb, safe_budget_gb))
    con.commit()


def get_calibration(con: sqlite3.Connection, machine_key: str, engine: str) -> dict | None:
    from ara.engine_identity import canonical_engine
    row = con.execute("SELECT * FROM calibrations WHERE machine_key=? AND engine=?",
                      (machine_key, canonical_engine(engine))).fetchone()
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
    from ara.engine_identity import canonical_engine
    con.execute(
        "INSERT INTO characterizations "
        "(machine_key, engine, model_id, safe_context, decode_context, points_json, measured_at) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(machine_key, engine, model_id) DO UPDATE SET "
        "safe_context=excluded.safe_context, decode_context=excluded.decode_context, "
        "points_json=excluded.points_json, measured_at=excluded.measured_at",
        (machine_key, canonical_engine(engine), model_id, safe_context, decode_context,
         json.dumps(points), measured_at or _now()))
    con.commit()


def get_characterization(con: sqlite3.Connection, machine_key: str, engine: str,
                         model_id: str) -> dict | None:
    from ara.engine_identity import canonical_engine
    row = con.execute(
        "SELECT * FROM characterizations WHERE machine_key=? AND engine=? AND model_id=?",
        (machine_key, canonical_engine(engine), model_id)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["points"] = json.loads(d["points_json"]) if d["points_json"] else []
    return d


def list_characterizations(con: sqlite3.Connection, machine_key: str,
                           engine: str | None = None) -> list[dict]:
    """Every model characterized on this machine, newest fields parsed. Scoped to one ``engine``
    when given, else across every engine (ordered by model then engine so it's stable)."""
    if engine is None:
        rows = con.execute(
            "SELECT * FROM characterizations WHERE machine_key=? ORDER BY model_id, engine",
            (machine_key,)).fetchall()
    else:
        from ara.engine_identity import canonical_engine
        rows = con.execute(
            "SELECT * FROM characterizations WHERE machine_key=? AND engine=? ORDER BY model_id",
            (machine_key, canonical_engine(engine))).fetchall()
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
                          sample_size: int | None = None, tier: str = "measured",
                          refused_n: int | None = None, errored_n: int | None = None) -> None:
    from ara.engine_identity import canonical_engine
    con.execute(
        "INSERT INTO benchmark_results "
        "(machine_key, model_id, use_case, engine_key, backend, base_model, quant, "
        "benchmark_id, tier, score, max_score, sample_size, refused_n, errored_n, "
        "source, measured_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(machine_key, model_id, use_case) DO UPDATE SET "
        "engine_key=excluded.engine_key, backend=excluded.backend, "
        "base_model=excluded.base_model, quant=excluded.quant, "
        "benchmark_id=excluded.benchmark_id, tier=excluded.tier, score=excluded.score, "
        "max_score=excluded.max_score, sample_size=excluded.sample_size, "
        "refused_n=excluded.refused_n, errored_n=excluded.errored_n, "
        "source=excluded.source, measured_at=excluded.measured_at",
        (machine_key, model_id, use_case, canonical_engine(engine_key), backend, base_model, quant,
         benchmark_id, tier, score, max_score, sample_size, refused_n, errored_n,
         source, _now()))
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
