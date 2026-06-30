# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""SQLite store for model metadata, probe runs, raw measurements, and fitted ceilings."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "suite.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    hf_id           TEXT PRIMARY KEY,
    weights_gb      REAL,
    n_layers        INTEGER,
    growing_layers  INTEGER,
    kv_heads        INTEGER,
    head_dim        INTEGER,
    hidden_size     INTEGER,
    max_context     INTEGER,
    cache_type      TEXT,
    can_quantize_kv INTEGER,
    layer_types     TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS probe_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    hf_id              TEXT NOT NULL,
    kv_bits            INTEGER,          -- NULL = fp16
    kv_group_size      INTEGER,
    quantized_kv_start INTEGER,
    mlx_version        TEXT,
    wall_gb            REAL,
    safe_threshold_gb  REAL,
    created_at         TEXT,
    FOREIGN KEY (hf_id) REFERENCES models(hf_id)
);

CREATE TABLE IF NOT EXISTS measurements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    context       INTEGER NOT NULL,
    mlx_peak_gb   REAL,
    mlx_true_gb   REAL,
    os_wired_gb   REAL,
    status        TEXT,                  -- ok | skipped | error
    note          TEXT,
    FOREIGN KEY (run_id) REFERENCES probe_runs(id)
);

CREATE TABLE IF NOT EXISTS generation_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    hf_id          TEXT NOT NULL,
    prompt_tokens  INTEGER,
    gen_tokens     INTEGER,
    prompt_tps     REAL,
    gen_tps        REAL,
    peak_gb        REAL,             -- mlx self-reported (undercounts wired; for reference only)
    max_kv_size    INTEGER,          -- the context cap the launcher set for this run
    created_at     TEXT,
    FOREIGN KEY (hf_id) REFERENCES models(hf_id)
);

CREATE TABLE IF NOT EXISTS fits (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL,
    model_base_gb    REAL,               -- DELTA intercept: model's own wired footprint at c->0 (invariant)
    slope_gb_per_k   REAL,               -- GB per 1000 tokens
    r2               REAL,
    ref_baseline_gb  REAL,               -- live system baseline used for the ceiling calc
    threshold_gb     REAL,
    safe_ceiling_ctx INTEGER,            -- context where (ref_baseline + model_base + slope*c) hits threshold
    hard_wall_ctx    INTEGER,            -- ... hits the measured wall (17.18 GB on the testbed)
    n_points         INTEGER,
    created_at       TEXT,
    FOREIGN KEY (run_id) REFERENCES probe_runs(id)
);

CREATE TABLE IF NOT EXISTS kokoro_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    voice       TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_measurements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    text_length    INTEGER NOT NULL,
    audio_duration REAL NOT NULL,
    compute_time   REAL NOT NULL,
    rtf            REAL NOT NULL,
    cps            REAL NOT NULL,
    peak_gb        REAL,
    os_wired_gb    REAL,
    FOREIGN KEY (run_id) REFERENCES kokoro_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kokoro_ttfa_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    voice       TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_ttfa_measurements (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL,
    text_length          INTEGER NOT NULL,
    ttfa_sec             REAL NOT NULL,
    total_sec            REAL NOT NULL,
    speedup_ratio        REAL NOT NULL,
    first_chunk_duration REAL NOT NULL,
    peak_gb              REAL,
    FOREIGN KEY (run_id) REFERENCES kokoro_ttfa_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kokoro_batch_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    voice       TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_batch_measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    batch_size  INTEGER NOT NULL,
    total_time  REAL NOT NULL,
    cps         REAL NOT NULL,
    peak_gb     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES kokoro_batch_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kokoro_voice_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_voice_measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    cond_type   TEXT NOT NULL,          -- static_baseline | warm_switch | cold_load
    voice_from  TEXT NOT NULL,
    voice_to    TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES kokoro_voice_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kokoro_cache_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_cache_measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    cache_size  INTEGER NOT NULL,
    os_wired_gb REAL NOT NULL,
    peak_gb     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES kokoro_cache_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kokoro_baseline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS kokoro_baseline_measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    baseline_gb REAL NOT NULL,
    active_gb   REAL NOT NULL,
    overhead_gb REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES kokoro_baseline_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS system_profiles (
    device_name       TEXT NOT NULL,
    total_ram_bytes   INTEGER NOT NULL,
    macos_major       INTEGER NOT NULL,
    resident_factor   REAL NOT NULL,
    fixed_overhead_gb REAL NOT NULL,
    model_id          TEXT,
    n_points          INTEGER,
    mlx_version       TEXT,
    calibrated_at     TEXT,
    PRIMARY KEY (device_name, total_ram_bytes, macos_major)
);

CREATE TABLE IF NOT EXISTS embeddings_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS embeddings_measurements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    batch_size     INTEGER NOT NULL,
    seq_len        INTEGER NOT NULL,
    os_wired_gb    REAL,
    peak_gb        REAL,
    throughput_tps REAL NOT NULL,
    latency_ms     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES embeddings_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embedding_profiles (
    device_name       TEXT NOT NULL,
    total_ram_bytes   INTEGER NOT NULL,
    macos_major       INTEGER NOT NULL,
    mlx_version       TEXT NOT NULL,
    model_id          TEXT NOT NULL,
    coef_intercept_gb REAL NOT NULL,
    coef_linear       REAL NOT NULL,
    coef_quad         REAL NOT NULL,
    n_points          INTEGER NOT NULL,
    created_at        TEXT,
    PRIMARY KEY (device_name, total_ram_bytes, macos_major, mlx_version, model_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    # Additive column migrations for benchmark tables created before a column existed.
    # Check-then-add (not try/except) so genuine DB errors are not silently swallowed.
    cols = {row[1] for row in con.execute("PRAGMA table_info(kokoro_measurements)")}
    if "os_wired_gb" not in cols:
        con.execute("ALTER TABLE kokoro_measurements ADD COLUMN os_wired_gb REAL")
    return con


def upsert_model(con: sqlite3.Connection, info: dict) -> None:
    cols = ["hf_id", "weights_gb", "n_layers", "growing_layers", "kv_heads", "head_dim",
            "hidden_size", "max_context", "cache_type", "can_quantize_kv", "layer_types"]
    row = {k: info.get(k) for k in cols}
    row["can_quantize_kv"] = int(bool(row["can_quantize_kv"]))
    if isinstance(row["layer_types"], dict):
        row["layer_types"] = json.dumps(row["layer_types"])
    row["updated_at"] = _now()
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "hf_id")
    con.execute(
        f"INSERT INTO models ({', '.join(row)}) VALUES ({placeholders}) "
        f"ON CONFLICT(hf_id) DO UPDATE SET {updates}",
        row,
    )
    con.commit()


def upsert_profile(con: sqlite3.Connection, key: tuple[str, int, int], *,
                   resident_factor: float, fixed_overhead_gb: float,
                   model_id: str | None, n_points: int | None,
                   mlx_version: str | None) -> None:
    device_name, total_ram_bytes, macos_major = key
    con.execute(
        "INSERT INTO system_profiles (device_name, total_ram_bytes, macos_major, "
        "resident_factor, fixed_overhead_gb, model_id, n_points, mlx_version, calibrated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(device_name, total_ram_bytes, macos_major) DO UPDATE SET "
        "resident_factor=excluded.resident_factor, "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, model_id=excluded.model_id, "
        "n_points=excluded.n_points, mlx_version=excluded.mlx_version, "
        "calibrated_at=excluded.calibrated_at",
        (device_name, total_ram_bytes, macos_major, resident_factor, fixed_overhead_gb,
         model_id, n_points, mlx_version, _now()),
    )
    con.commit()


def get_profile(con: sqlite3.Connection, key: tuple[str, int, int]) -> dict | None:
    device_name, total_ram_bytes, macos_major = key
    row = con.execute(
        "SELECT device_name, total_ram_bytes, macos_major, resident_factor, "
        "fixed_overhead_gb, model_id, n_points, mlx_version, calibrated_at "
        "FROM system_profiles WHERE device_name=? AND total_ram_bytes=? AND macos_major=?",
        (device_name, total_ram_bytes, macos_major),
    ).fetchone()
    return dict(row) if row is not None else None


def start_run(con: sqlite3.Connection, hf_id: str, *, kv_bits, kv_group_size,
              quantized_kv_start, mlx_version, wall_gb, safe_threshold_gb) -> int:
    cur = con.execute(
        "INSERT INTO probe_runs (hf_id, kv_bits, kv_group_size, quantized_kv_start, "
        "mlx_version, wall_gb, safe_threshold_gb, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (hf_id, kv_bits, kv_group_size, quantized_kv_start, mlx_version,
         wall_gb, safe_threshold_gb, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_measurement(con: sqlite3.Connection, run_id: int, context: int, *,
                    mlx_peak_gb=None, mlx_true_gb=None, os_wired_gb=None,
                    status="ok", note=None) -> None:
    con.execute(
        "INSERT INTO measurements (run_id, context, mlx_peak_gb, mlx_true_gb, "
        "os_wired_gb, status, note) VALUES (?,?,?,?,?,?,?)",
        (run_id, context, mlx_peak_gb, mlx_true_gb, os_wired_gb, status, note),
    )
    con.commit()


def log_generation(con: sqlite3.Connection, hf_id: str, *, prompt_tokens=None,
                   gen_tokens=None, prompt_tps=None, gen_tps=None, peak_gb=None,
                   max_kv_size=None) -> None:
    con.execute(
        "INSERT INTO generation_log (hf_id, prompt_tokens, gen_tokens, prompt_tps, "
        "gen_tps, peak_gb, max_kv_size, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (hf_id, prompt_tokens, gen_tokens, prompt_tps, gen_tps, peak_gb, max_kv_size, _now()),
    )
    con.commit()


def gen_speeds(con: sqlite3.Connection) -> dict[str, list[float]]:
    """Map of hf_id -> list of recorded generation tokens-per-sec (for median in `list`)."""
    rows = con.execute(
        "SELECT hf_id, gen_tps FROM generation_log WHERE gen_tps IS NOT NULL"
    ).fetchall()
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["hf_id"], []).append(float(r["gen_tps"]))
    return out


def latest_fit(con: sqlite3.Connection, hf_id: str) -> dict | None:
    """Most recent fitted curve for a model, or None if never characterized."""
    row = con.execute(
        "SELECT f.*, r.created_at AS characterized_at, r.kv_bits AS fit_kv_bits "
        "FROM fits f JOIN probe_runs r ON f.run_id = r.id "
        "WHERE r.hf_id = ? ORDER BY f.id DESC LIMIT 1",
        (hf_id,),
    ).fetchone()
    return dict(row) if row else None


def latest_fits(con: sqlite3.Connection) -> list[dict]:
    """Most recent fitted curve for every characterized model."""
    rows = con.execute(
        "WITH ranked AS ("
        "  SELECT m.hf_id, m.cache_type, f.model_base_gb, f.slope_gb_per_k, "
        "         f.ref_baseline_gb, f.safe_ceiling_ctx, f.hard_wall_ctx, f.r2, "
        "         r.created_at AS characterized_at, "
        "         ROW_NUMBER() OVER (PARTITION BY m.hf_id ORDER BY f.id DESC) AS rn "
        "  FROM fits f JOIN probe_runs r ON f.run_id=r.id "
        "  JOIN models m ON r.hf_id=m.hf_id"
        ") SELECT * FROM ranked WHERE rn=1 ORDER BY hf_id"
    ).fetchall()
    return [dict(row) for row in rows]


def save_fit(con: sqlite3.Connection, run_id: int, fit: dict) -> None:
    fit = {**fit, "run_id": run_id, "created_at": _now()}
    keys = ["run_id", "model_base_gb", "slope_gb_per_k", "r2", "ref_baseline_gb",
            "threshold_gb", "safe_ceiling_ctx", "hard_wall_ctx", "n_points", "created_at"]
    con.execute(
        f"INSERT INTO fits ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
        tuple(fit.get(k) for k in keys),
    )
    con.commit()


def get_model(con: sqlite3.Connection, hf_id: str) -> dict | None:
    """Get metadata for a specific model."""
    row = con.execute("SELECT * FROM models WHERE hf_id = ?", (hf_id,)).fetchone()
    return dict(row) if row else None


def get_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """Get all measurements for a specific run ordered by context."""
    rows = con.execute(
        "SELECT * FROM measurements WHERE run_id = ? ORDER BY context ASC",
        (run_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_model_runs_and_fits(con: sqlite3.Connection, hf_id: str) -> list[dict]:
    """Get all runs and fits for a specific model."""
    rows = con.execute(
        "SELECT r.id AS run_id, r.kv_bits, r.kv_group_size, r.quantized_kv_start, "
        "       r.mlx_version, r.wall_gb, r.safe_threshold_gb, r.created_at AS run_created_at, "
        "       f.model_base_gb, f.slope_gb_per_k, f.r2, f.ref_baseline_gb, "
        "       f.safe_ceiling_ctx, f.hard_wall_ctx, f.created_at AS fit_created_at "
        "FROM probe_runs r "
        "LEFT JOIN fits f ON f.run_id = r.id "
        "WHERE r.hf_id = ? "
        "ORDER BY r.id DESC",
        (hf_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def start_kokoro_run(con: sqlite3.Connection, model_id: str, voice: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_runs (model_id, voice, mlx_version, created_at) VALUES (?, ?, ?, ?)",
        (model_id, voice, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_measurement(
    con: sqlite3.Connection,
    run_id: int,
    text_length: int,
    audio_duration: float,
    compute_time: float,
    rtf: float,
    cps: float,
    peak_gb: float | None,
    os_wired_gb: float | None = None,
) -> None:
    con.execute(
        "INSERT INTO kokoro_measurements (run_id, text_length, audio_duration, compute_time, rtf, cps, peak_gb, os_wired_gb) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, text_length, audio_duration, compute_time, rtf, cps, peak_gb, os_wired_gb),
    )
    con.commit()


def get_all_kokoro_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, text_length, audio_duration, compute_time, rtf, cps, peak_gb, os_wired_gb "
        "FROM kokoro_measurements WHERE run_id = ? ORDER BY text_length ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def start_kokoro_ttfa_run(con: sqlite3.Connection, model_id: str, voice: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_ttfa_runs (model_id, voice, mlx_version, created_at) VALUES (?, ?, ?, ?)",
        (model_id, voice, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_ttfa_measurement(
    con: sqlite3.Connection,
    run_id: int,
    text_length: int,
    ttfa_sec: float,
    total_sec: float,
    speedup_ratio: float,
    first_chunk_duration: float,
    peak_gb: float | None,
) -> None:
    con.execute(
        "INSERT INTO kokoro_ttfa_measurements (run_id, text_length, ttfa_sec, total_sec, speedup_ratio, first_chunk_duration, peak_gb) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, text_length, ttfa_sec, total_sec, speedup_ratio, first_chunk_duration, peak_gb),
    )
    con.commit()


def get_all_kokoro_ttfa_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_ttfa_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_ttfa_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, text_length, ttfa_sec, total_sec, speedup_ratio, first_chunk_duration, peak_gb "
        "FROM kokoro_ttfa_measurements WHERE run_id = ? ORDER BY text_length ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_ttfa_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_ttfa_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def start_kokoro_batch_run(con: sqlite3.Connection, model_id: str, voice: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_batch_runs (model_id, voice, mlx_version, created_at) VALUES (?, ?, ?, ?)",
        (model_id, voice, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_batch_measurement(
    con: sqlite3.Connection,
    run_id: int,
    batch_size: int,
    total_time: float,
    cps: float,
    peak_gb: float,
) -> None:
    con.execute(
        "INSERT INTO kokoro_batch_measurements (run_id, batch_size, total_time, cps, peak_gb) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, batch_size, total_time, cps, peak_gb),
    )
    con.commit()


def get_all_kokoro_batch_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_batch_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_batch_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, batch_size, total_time, cps, peak_gb "
        "FROM kokoro_batch_measurements WHERE run_id = ? ORDER BY batch_size ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_batch_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, voice, mlx_version, created_at FROM kokoro_batch_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# --- Kokoro Voice ---
def start_kokoro_voice_run(con: sqlite3.Connection, model_id: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_voice_runs (model_id, mlx_version, created_at) VALUES (?, ?, ?)",
        (model_id, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_voice_measurement(
    con: sqlite3.Connection,
    run_id: int,
    cond_type: str,
    voice_from: str,
    voice_to: str,
    duration_ms: float,
) -> None:
    con.execute(
        "INSERT INTO kokoro_voice_measurements (run_id, cond_type, voice_from, voice_to, duration_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, cond_type, voice_from, voice_to, duration_ms),
    )
    con.commit()


def get_all_kokoro_voice_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_voice_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_voice_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, cond_type, voice_from, voice_to, duration_ms "
        "FROM kokoro_voice_measurements WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_voice_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_voice_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# --- Kokoro Cache ---
def start_kokoro_cache_run(con: sqlite3.Connection, model_id: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_cache_runs (model_id, mlx_version, created_at) VALUES (?, ?, ?)",
        (model_id, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_cache_measurement(
    con: sqlite3.Connection,
    run_id: int,
    cache_size: int,
    os_wired_gb: float,
    peak_gb: float,
) -> None:
    con.execute(
        "INSERT INTO kokoro_cache_measurements (run_id, cache_size, os_wired_gb, peak_gb) "
        "VALUES (?, ?, ?, ?)",
        (run_id, cache_size, os_wired_gb, peak_gb),
    )
    con.commit()


def get_all_kokoro_cache_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_cache_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_cache_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, cache_size, os_wired_gb, peak_gb "
        "FROM kokoro_cache_measurements WHERE run_id = ? ORDER BY cache_size ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_cache_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_cache_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# --- Kokoro Baseline ---
def start_kokoro_baseline_run(con: sqlite3.Connection, model_id: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO kokoro_baseline_runs (model_id, mlx_version, created_at) VALUES (?, ?, ?)",
        (model_id, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_kokoro_baseline_measurement(
    con: sqlite3.Connection,
    run_id: int,
    baseline_gb: float,
    active_gb: float,
    overhead_gb: float,
) -> None:
    con.execute(
        "INSERT INTO kokoro_baseline_measurements (run_id, baseline_gb, active_gb, overhead_gb) "
        "VALUES (?, ?, ?, ?)",
        (run_id, baseline_gb, active_gb, overhead_gb),
    )
    con.commit()


def get_all_kokoro_baseline_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_baseline_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_kokoro_baseline_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, baseline_gb, active_gb, overhead_gb "
        "FROM kokoro_baseline_measurements WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_kokoro_baseline_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM kokoro_baseline_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_latest_kokoro_baseline(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT m.id, m.run_id, m.baseline_gb, m.active_gb, m.overhead_gb "
        "FROM kokoro_baseline_measurements m JOIN kokoro_baseline_runs r ON m.run_id = r.id "
        "ORDER BY r.id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# --- Embeddings benchmark (2D batch x seq memory surface) ---
def start_embeddings_run(con: sqlite3.Connection, model_id: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO embeddings_runs (model_id, mlx_version, created_at) VALUES (?, ?, ?)",
        (model_id, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_embeddings_measurement(
    con: sqlite3.Connection,
    run_id: int,
    batch_size: int,
    seq_len: int,
    os_wired_gb: float,
    peak_gb: float,
    throughput_tps: float,
    latency_ms: float,
) -> None:
    con.execute(
        "INSERT INTO embeddings_measurements "
        "(run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms),
    )
    con.commit()


def get_all_embeddings_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM embeddings_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_embeddings_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms "
        "FROM embeddings_measurements WHERE run_id = ? ORDER BY batch_size ASC, seq_len ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_embeddings_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM embeddings_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# --- Embedding calibration profiles (per machine + MLX + model) ---
def upsert_embedding_profile(
    con: sqlite3.Connection,
    key: tuple[str, int, int, str, str],
    *,
    coef_intercept_gb: float,
    coef_linear: float,
    coef_quad: float,
    n_points: int,
) -> None:
    dev, ram, osv, mlxv, model_id = key
    con.execute(
        "INSERT INTO embedding_profiles "
        "(device_name, total_ram_bytes, macos_major, mlx_version, model_id, "
        " coef_intercept_gb, coef_linear, coef_quad, n_points, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(device_name, total_ram_bytes, macos_major, mlx_version, model_id) "
        "DO UPDATE SET coef_intercept_gb=excluded.coef_intercept_gb, "
        "coef_linear=excluded.coef_linear, coef_quad=excluded.coef_quad, "
        "n_points=excluded.n_points, created_at=excluded.created_at",
        (dev, ram, osv, mlxv, model_id, coef_intercept_gb, coef_linear, coef_quad,
         n_points, _now()),
    )
    con.commit()


def get_embedding_profile(
    con: sqlite3.Connection, key: tuple[str, int, int, str, str]
) -> dict | None:
    dev, ram, osv, mlxv, model_id = key
    row = con.execute(
        "SELECT device_name, total_ram_bytes, macos_major, mlx_version, model_id, "
        "coef_intercept_gb, coef_linear, coef_quad, n_points, created_at "
        "FROM embedding_profiles WHERE device_name=? AND total_ram_bytes=? AND "
        "macos_major=? AND mlx_version=? AND model_id=?",
        (dev, ram, osv, mlxv, model_id),
    ).fetchone()
    return dict(row) if row else None

