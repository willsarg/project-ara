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

from ara.measurement_authority import (
    LEGACY_UNIT_UNKNOWN_AUTHORITY_KEY,
    LEGACY_UNIT_UNKNOWN_ENVIRONMENT_KEY,
    UNSCOPED_AUTHORITY_KEY,
    UNSCOPED_ENVIRONMENT_KEY,
)

_MODEL_ARTIFACTS_DDL = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id          TEXT PRIMARY KEY,
    logical_model_id     TEXT NOT NULL,
    artifact_source      TEXT NOT NULL,
    resolved_revision    TEXT,
    manifest_hash        TEXT,
    size_bytes           INTEGER,
    format               TEXT,
    quantization         TEXT,
    artifact_confidence  TEXT NOT NULL,
    facts_json           TEXT,
    evidence_json        TEXT,
    observed_at          TEXT
);
"""

_CALIBRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    machine_key       TEXT NOT NULL,
    runtime           TEXT NOT NULL,
    backend           TEXT NOT NULL,
    config_key        TEXT NOT NULL,
    legacy_engine     TEXT,
    fixed_overhead_gb REAL,
    calibrated_at     TEXT,
    wall_gb           REAL,
    safe_budget_gb    REAL,
    evidence_json     TEXT,
    environment_key   TEXT NOT NULL,
    authority_key     TEXT NOT NULL,
    memory_unit       TEXT NOT NULL,
    wall_bytes        INTEGER,
    safe_budget_bytes INTEGER,
    authority_evidence_json TEXT,
    PRIMARY KEY (machine_key, runtime, backend, config_key, authority_key)
);
"""


_CHARACTERIZATIONS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    machine_key          TEXT NOT NULL,
    runtime              TEXT NOT NULL,
    backend              TEXT NOT NULL,
    artifact_id          TEXT NOT NULL,
    config_key           TEXT NOT NULL,
    logical_model_id     TEXT NOT NULL,
    legacy_engine        TEXT,
    safe_context         INTEGER,
    decode_context       INTEGER,
    config_json          TEXT,
    points_json          TEXT,
    evidence_json        TEXT,
    artifact_confidence  TEXT NOT NULL,
    reusable             INTEGER NOT NULL DEFAULT 0,
    measured_at          TEXT,
    environment_key      TEXT NOT NULL,
    authority_key        TEXT NOT NULL,
    memory_unit          TEXT NOT NULL,
    PRIMARY KEY (machine_key, runtime, backend, artifact_id, config_key, authority_key)
);
"""


_CALIBRATIONS_V5_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    machine_key       TEXT NOT NULL,
    runtime           TEXT NOT NULL,
    backend           TEXT NOT NULL,
    config_key        TEXT NOT NULL,
    legacy_engine     TEXT,
    fixed_overhead_gb REAL,
    calibrated_at     TEXT,
    wall_gb           REAL,
    safe_budget_gb    REAL,
    evidence_json     TEXT,
    PRIMARY KEY (machine_key, runtime, backend, config_key)
);
"""


_CHARACTERIZATIONS_V5_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    machine_key          TEXT NOT NULL,
    runtime              TEXT NOT NULL,
    backend              TEXT NOT NULL,
    artifact_id          TEXT NOT NULL,
    config_key           TEXT NOT NULL,
    logical_model_id     TEXT NOT NULL,
    legacy_engine        TEXT,
    safe_context         INTEGER,
    decode_context       INTEGER,
    config_json          TEXT,
    points_json          TEXT,
    evidence_json        TEXT,
    artifact_confidence  TEXT NOT NULL,
    reusable             INTEGER NOT NULL DEFAULT 0,
    measured_at          TEXT,
    PRIMARY KEY (machine_key, runtime, backend, artifact_id, config_key)
);
"""


_BENCHMARK_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    machine_key  TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    use_case     TEXT NOT NULL,
    evidence_key TEXT NOT NULL,
    runtime      TEXT NOT NULL,
    placement    TEXT NOT NULL,
    config_key   TEXT NOT NULL,
    request_policy_key TEXT NOT NULL,
    engine_key   TEXT,
    backend      TEXT NOT NULL,
    base_model   TEXT,
    quant        TEXT,
    benchmark_id TEXT,
    methodology_id TEXT,
    tier         TEXT NOT NULL DEFAULT 'measured',
    score        REAL NOT NULL,
    max_score    REAL,
    sample_size  INTEGER,
    refused_n    INTEGER,
    errored_n    INTEGER,
    probe_context INTEGER,
    generation_cap INTEGER,
    repeat_count INTEGER,
    total_generations INTEGER,
    run_scores_json TEXT,
    artifact_id  TEXT,
    canonical_model_id TEXT,
    target_json TEXT,
    request_policy_json TEXT,
    runtime_metrics_json TEXT,
    source       TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    PRIMARY KEY (machine_key, model_id, use_case, evidence_key)
);
"""


SCHEMA = _CALIBRATIONS_DDL.format(table="calibrations") + """
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

CREATE TABLE IF NOT EXISTS profiles (
    machine_key  TEXT NOT NULL,
    captured_at  TEXT NOT NULL,
    profile_json TEXT NOT NULL
);
""" + _MODEL_ARTIFACTS_DDL + _CHARACTERIZATIONS_DDL.format(
    table="characterizations") + _BENCHMARK_RESULTS_DDL.format(table="benchmark_results")


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
    if "config_json" not in cols:
        con.execute("ALTER TABLE characterizations ADD COLUMN config_json TEXT")
    if "artifact_id" not in cols:
        con.execute("ALTER TABLE characterizations ADD COLUMN artifact_id TEXT")
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
    for column, column_type in (
        ("probe_context", "INTEGER"),
        ("generation_cap", "INTEGER"),
        ("repeat_count", "INTEGER"),
        ("total_generations", "INTEGER"),
        ("run_scores_json", "TEXT"),
        ("artifact_id", "TEXT"),
        ("canonical_model_id", "TEXT"),
        ("methodology_id", "TEXT"),
        ("target_json", "TEXT"),
        ("request_policy_json", "TEXT"),
        ("runtime_metrics_json", "TEXT"),
    ):
        if column not in bench_cols:
            con.execute(f"ALTER TABLE benchmark_results ADD COLUMN {column} {column_type}")  # noqa: S608
    # One-time data fix (user_version 0→1): wmx calibrations stored before 2026-07-02 carry
    # decimal-GB walls (~7.4% high vs ARA's binary-GiB contract — the apple boundary now
    # converts). A float can't reveal its own units, so honest re-measurement beats arithmetic
    # repair: drop the rows and the next run re-calibrates. Slug 2026-07-02-analytic-units-gib.
    if con.execute("PRAGMA user_version").fetchone()[0] < 1:
        engine_column = "engine" if "engine" in cal_cols else "legacy_engine"
        con.execute(f"DELETE FROM calibrations WHERE {engine_column}='wmx'")  # noqa: S608
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
    if con.execute("PRAGMA user_version").fetchone()[0] < 4:
        rebuild = _benchmark_cells_v4_needed(con)
        if rebuild:
            _backup_before_target_cells_v4(con, path)
        con.commit()
        con.execute("PRAGMA foreign_keys = OFF")
        try:
            con.execute("BEGIN IMMEDIATE")
            if rebuild:
                _migrate_benchmark_cells_v4(con)
            con.execute("PRAGMA user_version = 4")
            con.commit()
        except Exception:
            con.rollback()
            con.execute("PRAGMA foreign_keys = ON")
            con.close()
            raise
        con.execute("PRAGMA foreign_keys = ON")
    if con.execute("PRAGMA user_version").fetchone()[0] < 5:
        rebuild = _target_schema_v5_needed(con)
        if rebuild:
            _backup_before_target_schema_v5(con, path)
        con.commit()
        con.execute("PRAGMA foreign_keys = OFF")
        try:
            con.execute("BEGIN IMMEDIATE")
            if rebuild:
                _migrate_target_schema_v5(con)
            con.execute("PRAGMA user_version = 5")
            con.commit()
        except Exception:
            con.rollback()
            con.execute("PRAGMA foreign_keys = ON")
            con.close()
            raise
        con.execute("PRAGMA foreign_keys = ON")
    if con.execute("PRAGMA user_version").fetchone()[0] < 6:
        rebuild = _measurement_authority_v6_needed(con)
        if rebuild:
            _backup_before_measurement_authority_v6(con, path)
        con.commit()
        con.execute("PRAGMA foreign_keys = OFF")
        try:
            con.execute("BEGIN IMMEDIATE")
            if rebuild:
                _migrate_measurement_authority_v6(con)
            con.execute("PRAGMA user_version = 6")
            con.commit()
        except Exception:
            con.rollback()
            con.execute("PRAGMA foreign_keys = ON")
            con.close()
            raise
        con.execute("PRAGMA foreign_keys = ON")
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


def _backup_before_migration(con: sqlite3.Connection, path: Path, *, suffix: str,
                             validation_label: str) -> None:
    """Keep one byte-independent validated SQLite backup before a schema migration."""
    backup_path = path.with_name(path.name + suffix)
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
            raise sqlite3.DatabaseError(f"{validation_label} backup validation failed")
        os.replace(temp_path, backup_path)
        if not valid(backup_path):
            raise sqlite3.DatabaseError(
                f"published {validation_label} backup validation failed")
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _backup_before_engine_identity_v3(con: sqlite3.Connection, path: Path) -> None:
    """Keep one byte-independent SQLite backup of the pre-v3 evidence store."""
    _backup_before_migration(
        con, path, suffix=".pre-engine-identity-v3.bak", validation_label="pre-v3")


def _backup_before_target_cells_v4(con: sqlite3.Connection, path: Path) -> None:
    """Keep one byte-independent SQLite backup of the pre-v4 benchmark store."""
    _backup_before_migration(
        con, path, suffix=".pre-target-cells-v4.bak", validation_label="pre-v4")


def _backup_before_target_schema_v5(con: sqlite3.Connection, path: Path) -> None:
    """Keep one byte-independent SQLite backup of the pre-v5 target store."""
    _backup_before_migration(
        con, path, suffix=".pre-target-schema-v5.bak", validation_label="pre-v5")


def _backup_before_measurement_authority_v6(
        con: sqlite3.Connection, path: Path) -> None:
    """Keep one validated SQLite backup of the pre-v6 measurement store."""
    _backup_before_migration(
        con,
        path,
        suffix=".pre-measurement-authority-v6.bak",
        validation_label="pre-v6",
    )


def _json_object(raw) -> dict | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _benchmark_cell_values(*, engine_key: str | None, backend: str | None,
                           artifact_id: str | None, methodology_id: str | None,
                           target: dict | None, request_policy: dict | None) -> dict[str, str]:
    """Build the durable identity for one benchmark runtime cell."""
    from ara import targets

    mapped_runtime, mapped_backend, _canonical = targets.for_engine(engine_key)
    target = target if isinstance(target, dict) else {}

    def text_value(name: str, fallback: str) -> str:
        value = target.get(name)
        return value if isinstance(value, str) and value else fallback

    runtime = text_value("runtime", mapped_runtime)
    cell_backend = text_value(
        "backend", backend if isinstance(backend, str) and backend else mapped_backend)
    placement = text_value("placement", "unknown")
    config_key = text_value("config_key", "")
    if not config_key:
        digest = target.get("config_sha256")
        config_key = (f"cfg:v1:sha256:{digest}"
                      if isinstance(digest, str) and digest else "default")
    request_policy_key = (
        "policy:v1:" + _compact_json(request_policy)
        if isinstance(request_policy, dict) else "unknown")
    identity = {
        "runtime": runtime,
        "backend": cell_backend,
        "placement": placement,
        "artifact_id": text_value("artifact_id", artifact_id or "unknown"),
        "config_key": config_key,
        "request_policy_key": request_policy_key,
        "methodology_id": methodology_id or "unknown",
    }
    return {
        "evidence_key": "cell:v1:" + _compact_json(identity),
        "runtime": runtime,
        "backend": cell_backend,
        "placement": placement,
        "config_key": config_key,
        "request_policy_key": request_policy_key,
    }


def _benchmark_cells_v4_needed(con: sqlite3.Connection) -> bool:
    columns = {row["name"] for row in con.execute("PRAGMA table_info(benchmark_results)")}
    return "evidence_key" not in columns


def _migrate_benchmark_cells_v4(con: sqlite3.Connection) -> None:
    """Rebuild the benchmark PK so distinct runtime cells cannot overwrite one another."""
    rows = [dict(row) for row in con.execute("SELECT * FROM benchmark_results")]
    con.execute("DROP TABLE IF EXISTS benchmark_results_new")
    con.execute(_BENCHMARK_RESULTS_DDL.format(table="benchmark_results_new"))
    columns = [row["name"] for row in con.execute(
        "PRAGMA table_info(benchmark_results_new)")]
    for row in rows:
        target = _json_object(row.get("target_json"))
        request_policy = _json_object(row.get("request_policy_json"))
        cell = _benchmark_cell_values(
            engine_key=row.get("engine_key"), backend=row.get("backend"),
            artifact_id=row.get("artifact_id"), methodology_id=row.get("methodology_id"),
            target=target, request_policy=request_policy)
        values = {column: row.get(column) for column in columns}
        values.update(cell)
        con.execute(
            f"INSERT INTO benchmark_results_new ({','.join(columns)}) "  # noqa: S608
            f"VALUES ({','.join(':' + column for column in columns)})",
            values,
        )
    con.execute("DROP TABLE benchmark_results")
    con.execute("ALTER TABLE benchmark_results_new RENAME TO benchmark_results")


def _legacy_artifact_id(logical_model_id: str) -> str:
    import hashlib
    return "legacy-unverified:v1:" + hashlib.sha256(
        logical_model_id.encode("utf-8")).hexdigest()


def _artifact_classification(artifact_id: str | None) -> tuple[str, str]:
    """Return artifact source and evidence confidence without trusting a label."""
    if not isinstance(artifact_id, str) or not artifact_id:
        return "legacy", "legacy_unverified"
    if artifact_id.startswith("ollama-manifest-sha256:"):
        digest = artifact_id.removeprefix("ollama-manifest-sha256:")
        return ("ollama", "observed_digest") if (
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest.casefold())
        ) else ("ollama", "unknown")
    if artifact_id.startswith(("hf:", "hf-gguf:")):
        return "huggingface", "manifest_hash"
    if artifact_id.startswith("local-gguf:"):
        return "local", "strong"
    if artifact_id.startswith("legacy-unverified:v1:"):
        return "legacy", "legacy_unverified"
    return "unknown", "unknown"


def _characterization_identity(engine: str | None, logical_model_id: str,
                               artifact_id: str | None, config: dict | None) -> dict:
    from ara import targets

    runtime, backend, legacy_engine = targets.for_engine(engine)
    placement = config.get("placement") if isinstance(config, dict) else None
    if runtime == "ollama":
        backend = {
            "cpu": "cpu",
            "unified": "apple",
            "accelerator": "cuda",
            "partial_offload": "cuda",
        }.get(placement, backend)
    strong_artifact = isinstance(artifact_id, str) and bool(artifact_id)
    stored_artifact = artifact_id if strong_artifact else _legacy_artifact_id(logical_model_id)
    _source, confidence = _artifact_classification(stored_artifact)
    return {
        "runtime": runtime,
        "backend": backend,
        "legacy_engine": legacy_engine,
        "artifact_id": stored_artifact,
        "config_key": targets.characterization_config_key(engine, config),
        "artifact_confidence": confidence,
        "reusable": int(confidence in {
            "strong", "observed_digest", "manifest_hash"} and isinstance(config, dict)),
    }


def _target_schema_v5_needed(con: sqlite3.Connection) -> bool:
    char_columns = {
        row["name"] for row in con.execute("PRAGMA table_info(characterizations)")}
    calibration_columns = {
        row["name"] for row in con.execute("PRAGMA table_info(calibrations)")}
    return "runtime" not in char_columns or "runtime" not in calibration_columns


def _migrate_target_schema_v5(con: sqlite3.Connection) -> None:
    """Rebuild calibration and characterization keys around exact runtime targets."""
    from ara import targets

    calibration_rows = [dict(row) for row in con.execute("SELECT * FROM calibrations")]
    characterization_rows = [
        dict(row) for row in con.execute("SELECT * FROM characterizations")]
    con.execute("DROP TABLE IF EXISTS calibrations_new")
    con.execute("DROP TABLE IF EXISTS characterizations_new")
    con.execute(_CALIBRATIONS_V5_DDL.format(table="calibrations_new"))
    con.execute(_CHARACTERIZATIONS_V5_DDL.format(table="characterizations_new"))
    for row in calibration_rows:
        runtime, backend, legacy_engine = targets.for_engine(row.get("engine"))
        con.execute(
            "INSERT INTO calibrations_new "
            "(machine_key,runtime,backend,config_key,legacy_engine,fixed_overhead_gb,"
            "calibrated_at,wall_gb,safe_budget_gb,evidence_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row["machine_key"], runtime, backend, targets.calibration_config_key(),
             legacy_engine, row.get("fixed_overhead_gb"), row.get("calibrated_at"),
             row.get("wall_gb"), row.get("safe_budget_gb"), None),
        )
    for row in characterization_rows:
        config = _json_object(row.get("config_json"))
        identity = _characterization_identity(
            row.get("engine"), row["model_id"], row.get("artifact_id"), config)
        artifact_source, _confidence = _artifact_classification(identity["artifact_id"])
        _upsert_model_artifact(
            con, identity["artifact_id"], row["model_id"],
            artifact_source=artifact_source,
            artifact_confidence=identity["artifact_confidence"],
            observed_at=row.get("measured_at"),
        )
        con.execute(
            "INSERT INTO characterizations_new "
            "(machine_key,runtime,backend,artifact_id,config_key,logical_model_id,"
            "legacy_engine,safe_context,decode_context,config_json,points_json,evidence_json,"
            "artifact_confidence,reusable,measured_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["machine_key"], identity["runtime"], identity["backend"],
             identity["artifact_id"], identity["config_key"], row["model_id"],
             identity["legacy_engine"], row.get("safe_context"), row.get("decode_context"),
             row.get("config_json"), row.get("points_json"), None,
             identity["artifact_confidence"], identity["reusable"], row.get("measured_at")),
        )
    con.execute("DROP TABLE calibrations")
    con.execute("DROP TABLE characterizations")
    con.execute("ALTER TABLE calibrations_new RENAME TO calibrations")
    con.execute("ALTER TABLE characterizations_new RENAME TO characterizations")


def _measurement_authority_v6_needed(con: sqlite3.Connection) -> bool:
    calibration_columns = {
        row["name"] for row in con.execute("PRAGMA table_info(calibrations)")}
    characterization_columns = {
        row["name"] for row in con.execute("PRAGMA table_info(characterizations)")}
    return not {
        "environment_key", "authority_key", "memory_unit", "wall_bytes",
        "safe_budget_bytes", "authority_evidence_json",
    }.issubset(calibration_columns) or not {
        "environment_key", "authority_key", "memory_unit",
    }.issubset(characterization_columns)


def _v6_authority_values(row: dict) -> tuple[str, str, str, str | None, bool]:
    """Classify a pre-v6 row without inventing byte precision from stored floats."""
    authority_key = row.get("authority_key")
    environment_key = row.get("environment_key")
    memory_unit = row.get("memory_unit")
    if all(isinstance(value, str) and value for value in (
            authority_key, environment_key, memory_unit)):
        return (
            environment_key,
            authority_key,
            memory_unit,
            row.get("authority_evidence_json"),
            False,
        )

    from ara.engine_identity import canonical_engine

    is_mlx = row.get("runtime") == "mlx" or canonical_engine(
        row.get("legacy_engine")) == "mlx"
    if is_mlx:
        evidence = _compact_json({
            "schema": "legacy-unit-unknown:v1",
            "reason": "measurement predates exact byte authority",
        })
        return (
            LEGACY_UNIT_UNKNOWN_ENVIRONMENT_KEY,
            LEGACY_UNIT_UNKNOWN_AUTHORITY_KEY,
            "legacy-unit-unknown",
            evidence,
            True,
        )
    evidence = _compact_json({"schema": "unscoped-authority:v1", "scope": "unscoped"})
    return (
        UNSCOPED_ENVIRONMENT_KEY,
        UNSCOPED_AUTHORITY_KEY,
        "GiB",
        evidence,
        False,
    )


def _migrate_measurement_authority_v6(con: sqlite3.Connection) -> None:
    """Rekey measured rows by exact authority while retaining all historical evidence."""
    calibration_rows = [dict(row) for row in con.execute("SELECT * FROM calibrations")]
    characterization_rows = [
        dict(row) for row in con.execute("SELECT * FROM characterizations")]
    con.execute("DROP TABLE IF EXISTS calibrations_new")
    con.execute("DROP TABLE IF EXISTS characterizations_new")
    con.execute(_CALIBRATIONS_DDL.format(table="calibrations_new"))
    con.execute(_CHARACTERIZATIONS_DDL.format(table="characterizations_new"))

    for row in calibration_rows:
        environment_key, authority_key, memory_unit, authority_json, _legacy_mlx = (
            _v6_authority_values(row))
        con.execute(
            "INSERT INTO calibrations_new "
            "(machine_key,runtime,backend,config_key,legacy_engine,fixed_overhead_gb,"
            "calibrated_at,wall_gb,safe_budget_gb,evidence_json,environment_key,"
            "authority_key,memory_unit,wall_bytes,safe_budget_bytes,"
            "authority_evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["machine_key"], row["runtime"], row["backend"], row["config_key"],
                row.get("legacy_engine"), row.get("fixed_overhead_gb"),
                row.get("calibrated_at"), row.get("wall_gb"), row.get("safe_budget_gb"),
                row.get("evidence_json"), environment_key, authority_key, memory_unit,
                row.get("wall_bytes"), row.get("safe_budget_bytes"), authority_json,
            ),
        )

    for row in characterization_rows:
        environment_key, authority_key, memory_unit, _authority_json, legacy_mlx = (
            _v6_authority_values(row))
        con.execute(
            "INSERT INTO characterizations_new "
            "(machine_key,runtime,backend,artifact_id,config_key,logical_model_id,"
            "legacy_engine,safe_context,decode_context,config_json,points_json,evidence_json,"
            "artifact_confidence,reusable,measured_at,environment_key,authority_key,memory_unit) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["machine_key"], row["runtime"], row["backend"], row["artifact_id"],
                row["config_key"], row["logical_model_id"], row.get("legacy_engine"),
                row.get("safe_context"), row.get("decode_context"), row.get("config_json"),
                row.get("points_json"), row.get("evidence_json"),
                row["artifact_confidence"], 0 if legacy_mlx else row.get("reusable", 0),
                row.get("measured_at"), environment_key, authority_key, memory_unit,
            ),
        )

    con.execute("DROP TABLE calibrations")
    con.execute("DROP TABLE characterizations")
    con.execute("ALTER TABLE calibrations_new RENAME TO calibrations")
    con.execute("ALTER TABLE characterizations_new RENAME TO characterizations")


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
        columns = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}  # noqa: S608
        if "engine" in columns:
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
        columns = {
            row["name"] for row in con.execute(f"PRAGMA table_info({table})")}  # noqa: S608
        if table == "calibrations" and "runtime" in columns:
            pk_rest = ("runtime", "backend", "config_key")
            if "authority_key" in columns:
                pk_rest = (*pk_rest, "authority_key")
        elif table == "characterizations" and "runtime" in columns:
            pk_rest = ("runtime", "backend", "artifact_id", "config_key")
            if "authority_key" in columns:
                pk_rest = (*pk_rest, "authority_key")
        elif table == "benchmark_results" and "evidence_key" in columns:
            pk_rest = (*pk_rest, "evidence_key")
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
                   wall_gb: float | None = None, safe_budget_gb: float | None = None,
                   config_key: str | None = None, evidence: dict | None = None,
                   environment_key: str = UNSCOPED_ENVIRONMENT_KEY,
                   authority_key: str = UNSCOPED_AUTHORITY_KEY,
                   memory_unit: str = "GiB", wall_bytes: int | None = None,
                   safe_budget_bytes: int | None = None,
                   authority_evidence: dict | None = None) -> None:
    from ara import targets
    runtime, backend, legacy_engine = targets.for_engine(engine)
    config_key = config_key or targets.calibration_config_key()
    con.execute(
        "INSERT INTO calibrations "
        "(machine_key,runtime,backend,config_key,legacy_engine,fixed_overhead_gb,"
        "calibrated_at,wall_gb,safe_budget_gb,evidence_json,environment_key,authority_key,"
        "memory_unit,wall_bytes,safe_budget_bytes,authority_evidence_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(machine_key,runtime,backend,config_key,authority_key) DO UPDATE SET "
        "legacy_engine=excluded.legacy_engine, "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, calibrated_at=excluded.calibrated_at, "
        "wall_gb=excluded.wall_gb, safe_budget_gb=excluded.safe_budget_gb, "
        "evidence_json=excluded.evidence_json, environment_key=excluded.environment_key, "
        "memory_unit=excluded.memory_unit, wall_bytes=excluded.wall_bytes, "
        "safe_budget_bytes=excluded.safe_budget_bytes, "
        "authority_evidence_json=excluded.authority_evidence_json",
        (machine_key, runtime, backend, config_key, legacy_engine, fixed_overhead_gb,
         calibrated_at, wall_gb, safe_budget_gb, _compact_json(evidence), environment_key,
         authority_key, memory_unit, wall_bytes, safe_budget_bytes,
         _compact_json(authority_evidence)))
    con.commit()


def get_calibration(con: sqlite3.Connection, machine_key: str, engine: str,
                    *, config_key: str | None = None,
                    authority_key: str | None = None) -> dict | None:
    from ara import targets
    from ara.engine_identity import LEGACY_ENGINE_ALIASES
    runtime, backend, canonical = targets.for_engine(engine)
    columns = {row["name"] for row in con.execute("PRAGMA table_info(calibrations)")}
    if "runtime" not in columns:
        storage_keys = [canonical, *(legacy for legacy, replacement
                                     in LEGACY_ENGINE_ALIASES.items()
                                     if replacement == canonical)]
        placeholders = ",".join("?" for _ in storage_keys)
        row = con.execute(
            f"SELECT * FROM calibrations WHERE machine_key=? "  # noqa: S608
            f"AND engine IN ({placeholders}) ORDER BY calibrated_at DESC LIMIT 1",
            (machine_key, *storage_keys)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result.update(runtime=runtime, backend=backend,
                      config_key=targets.calibration_config_key(),
                      legacy_engine=canonical, evidence_json=None, evidence=None)
        result["engine"] = canonical
        return result
    clauses = ["machine_key=?", "runtime=?", "backend=?", "config_key=?"]
    values = [machine_key, runtime, backend, config_key or targets.calibration_config_key()]
    if authority_key is not None:
        clauses.append("authority_key=?")
        values.append(authority_key)
    row = con.execute(
        f"SELECT * FROM calibrations WHERE {' AND '.join(clauses)} "  # noqa: S608
        "ORDER BY calibrated_at DESC, authority_key DESC LIMIT 1",
        values).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["engine"] = result.get("legacy_engine") or canonical
    result["evidence"] = _json_object(result.get("evidence_json"))
    result["authority_evidence"] = _json_object(
        result.get("authority_evidence_json"))
    return result


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


def _upsert_model_artifact(
        con: sqlite3.Connection, artifact_id: str, logical_model_id: str, *,
        artifact_source: str, artifact_confidence: str,
        resolved_revision: str | None = None, manifest_hash: str | None = None,
        size_bytes: int | None = None, format: str | None = None,
        quantization: str | None = None, facts: dict | None = None,
        evidence: dict | None = None, observed_at: str | None = None,
        preserve_existing_evidence: bool = False) -> None:
    con.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id,logical_model_id,artifact_source,resolved_revision,manifest_hash,"
        "size_bytes,format,quantization,artifact_confidence,facts_json,evidence_json,observed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET "
        "logical_model_id=excluded.logical_model_id,artifact_source=excluded.artifact_source,"
        "resolved_revision=excluded.resolved_revision,manifest_hash=excluded.manifest_hash,"
        "size_bytes=excluded.size_bytes,format=excluded.format,quantization=excluded.quantization,"
        "artifact_confidence=excluded.artifact_confidence,facts_json=excluded.facts_json,"
        "evidence_json=CASE WHEN ? THEN COALESCE(excluded.evidence_json,"
        "model_artifacts.evidence_json) ELSE excluded.evidence_json END,"
        "observed_at=excluded.observed_at",
        (artifact_id, logical_model_id, artifact_source, resolved_revision, manifest_hash,
         size_bytes, format, quantization, artifact_confidence, _compact_json(facts),
         _compact_json(evidence), observed_at or _now(), preserve_existing_evidence),
    )


def upsert_model_artifact(
        con: sqlite3.Connection, artifact_id: str, logical_model_id: str, *,
        artifact_source: str, artifact_confidence: str,
        resolved_revision: str | None = None, manifest_hash: str | None = None,
        size_bytes: int | None = None, format: str | None = None,
        quantization: str | None = None, facts: dict | None = None,
        evidence: dict | None = None, observed_at: str | None = None) -> None:
    """Remember provenance facts for one immutable or observed model artifact."""
    _upsert_model_artifact(
        con, artifact_id, logical_model_id, artifact_source=artifact_source,
        artifact_confidence=artifact_confidence, resolved_revision=resolved_revision,
        manifest_hash=manifest_hash, size_bytes=size_bytes, format=format,
        quantization=quantization, facts=facts, evidence=evidence, observed_at=observed_at,
    )
    con.commit()


def get_model_artifact(con: sqlite3.Connection, artifact_id: str) -> dict | None:
    row = con.execute(
        "SELECT * FROM model_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["facts"] = _json_object(result.get("facts_json"))
    result["evidence"] = _json_object(result.get("evidence_json"))
    return result


# --- characterizations (fitted ceilings, per machine + engine + model) ---
def save_characterization(con: sqlite3.Connection, machine_key: str, engine: str,
                          model_id: str, *, safe_context: int | None,
                          points: list, measured_at: str | None = None,
                          decode_context: int | None = None,
                          config: dict | None = None,
                          artifact_id: str | None = None,
                          evidence: dict | None = None,
                          characterization_evidence: dict | None = None,
                          environment_key: str = UNSCOPED_ENVIRONMENT_KEY,
                          authority_key: str = UNSCOPED_AUTHORITY_KEY,
                          memory_unit: str = "GiB") -> None:
    stored_config = {} if config is None else config
    identity = _characterization_identity(
        engine, model_id, artifact_id, stored_config)
    artifact_source, _confidence = _artifact_classification(identity["artifact_id"])
    _upsert_model_artifact(
        con, identity["artifact_id"], model_id, artifact_source=artifact_source,
        artifact_confidence=identity["artifact_confidence"], evidence=evidence,
        preserve_existing_evidence=evidence is None,
    )
    con.execute(
        "INSERT INTO characterizations "
        "(machine_key,runtime,backend,artifact_id,config_key,logical_model_id,legacy_engine,"
        "safe_context,decode_context,config_json,points_json,evidence_json,artifact_confidence,"
        "reusable,measured_at,environment_key,authority_key,memory_unit) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(machine_key,runtime,backend,artifact_id,config_key,authority_key) "
        "DO UPDATE SET "
        "logical_model_id=excluded.logical_model_id,legacy_engine=excluded.legacy_engine,"
        "safe_context=excluded.safe_context, decode_context=excluded.decode_context, "
        "config_json=excluded.config_json,points_json=excluded.points_json,"
        "evidence_json=excluded.evidence_json,artifact_confidence=excluded.artifact_confidence,"
        "reusable=excluded.reusable,measured_at=excluded.measured_at,"
        "environment_key=excluded.environment_key,memory_unit=excluded.memory_unit",
        (machine_key, identity["runtime"], identity["backend"], identity["artifact_id"],
         identity["config_key"], model_id, identity["legacy_engine"], safe_context,
         decode_context, json.dumps(stored_config, sort_keys=True), json.dumps(points),
         _compact_json(characterization_evidence if characterization_evidence is not None
                       else evidence),
         identity["artifact_confidence"], identity["reusable"],
         measured_at or _now(), environment_key, authority_key, memory_unit))
    con.commit()


def _characterization_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["engine"] = d.get("legacy_engine")
    d["model_id"] = d["logical_model_id"]
    d["points"] = json.loads(d["points_json"]) if d["points_json"] else []
    d["config"] = json.loads(d["config_json"]) if d.get("config_json") is not None else None
    d["evidence"] = _json_object(d.get("evidence_json"))
    d["reusable"] = d.get("reusable") == 1
    return d


def _legacy_characterization_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    config = _json_object(d.get("config_json"))
    identity = _characterization_identity(
        d.get("engine"), d["model_id"], d.get("artifact_id"), config)
    d.update(identity)
    d["logical_model_id"] = d["model_id"]
    d["engine"] = identity["legacy_engine"]
    d["points"] = json.loads(d["points_json"]) if d.get("points_json") else []
    d["config"] = config
    d["evidence_json"] = None
    d["evidence"] = None
    d["reusable"] = identity["reusable"] == 1
    return d


def list_characterizations_for_display(
        con: sqlite3.Connection, machine_key: str, *, runtime: str | None = None,
        backend: str | None = None, logical_model_id: str | None = None) -> list[dict]:
    """Return all matching target history, including weak migrated display-only rows."""
    columns = {
        row["name"] for row in con.execute("PRAGMA table_info(characterizations)")}
    if "runtime" not in columns:
        rows = con.execute(
            "SELECT * FROM characterizations WHERE machine_key=? "
            "ORDER BY model_id,engine,measured_at",
            (machine_key,)).fetchall()
        parsed = [_legacy_characterization_row(row) for row in rows]
        return [row for row in parsed
                if (runtime is None or row["runtime"] == runtime)
                and (backend is None or row["backend"] == backend)
                and (logical_model_id is None
                     or row["logical_model_id"] == logical_model_id)]
    clauses = ["machine_key=?"]
    values = [machine_key]
    for column, value in (
        ("runtime", runtime), ("backend", backend),
            ("logical_model_id", logical_model_id)):
        if value is not None:
            clauses.append(f"{column}=?")
            values.append(value)
    rows = con.execute(
        f"SELECT * FROM characterizations WHERE {' AND '.join(clauses)} "  # noqa: S608
        "ORDER BY logical_model_id,runtime,backend,measured_at,artifact_id,config_key,"
        "authority_key",
        values).fetchall()
    return [_characterization_row(row) for row in rows]


def get_characterization_for_display(
        con: sqlite3.Connection, machine_key: str, *, runtime: str,
        backend: str, artifact_id: str, config_key: str) -> dict | None:
    """Return one exact target cell for history/display, regardless of reuse strength."""
    rows = list_characterizations_for_display(
        con, machine_key, runtime=runtime, backend=backend)
    return next((row for row in rows if row["artifact_id"] == artifact_id
                 and row["config_key"] == config_key), None)


def list_reusable_characterizations(
        con: sqlite3.Connection, machine_key: str, *, runtime: str | None = None,
        backend: str | None = None, logical_model_id: str | None = None,
        authority_key: str | None = None) -> list[dict]:
    """Return only explicitly reusable, artifact-bound target evidence."""
    return [row for row in list_characterizations_for_display(
        con, machine_key, runtime=runtime, backend=backend,
        logical_model_id=logical_model_id) if row["reusable"]
        and row["artifact_confidence"] in {"strong", "observed_digest", "manifest_hash"}
        and (authority_key is None or row.get("authority_key") == authority_key)]


def get_reusable_characterization(
        con: sqlite3.Connection, machine_key: str, *, runtime: str, backend: str,
        artifact_id: str, config_key: str, authority_key: str | None = None) -> dict | None:
    """Return the exact reusable target cell or ``None``; weak history never matches."""
    columns = {
        row["name"] for row in con.execute("PRAGMA table_info(characterizations)")}
    if "runtime" not in columns:
        rows = list_reusable_characterizations(
            con, machine_key, runtime=runtime, backend=backend,
            authority_key=authority_key)
        return next((row for row in rows if row["artifact_id"] == artifact_id
                     and row["config_key"] == config_key), None)
    authority_clause = " AND authority_key=?" if authority_key is not None else ""
    values = [machine_key, runtime, backend, artifact_id, config_key]
    if authority_key is not None:
        values.append(authority_key)
    row = con.execute(
        "SELECT * FROM characterizations WHERE machine_key=? AND runtime=? AND backend=? "
        f"AND artifact_id=? AND config_key=? AND reusable=1{authority_clause} "
        "ORDER BY measured_at DESC, authority_key DESC LIMIT 1",
        values).fetchone()
    if row is None:
        return None
    result = _characterization_row(row)
    return (result if result["artifact_confidence"] in {
        "strong", "observed_digest", "manifest_hash"} else None)


def get_reusable_characterization_for_engine(
        con: sqlite3.Connection, machine_key: str, engine: str,
        logical_model_id: str, *, config: dict,
        artifact_id: str | None = None,
        authority_key: str | None = None) -> dict | None:
    """Return the newest reusable native compatibility cell with an exact target config."""
    from ara import targets
    runtime, backend, _legacy_engine = targets.for_engine(engine)
    expected_key = targets.characterization_config_key(engine, config)
    rows = list_reusable_characterizations(
        con, machine_key, runtime=runtime, backend=backend,
        logical_model_id=logical_model_id, authority_key=authority_key)
    candidates = [row for row in rows
                  if row["config_key"] == expected_key
                  and (artifact_id is None or row["artifact_id"] == artifact_id)]
    return max(candidates, key=lambda row: (
        row.get("measured_at") or "", row["artifact_id"])) if candidates else None


def get_characterization(con: sqlite3.Connection, machine_key: str, engine: str,
                         model_id: str) -> dict | None:
    """Compatibility display read for the newest row under one legacy engine label."""
    from ara import targets
    runtime, backend, _canonical = targets.for_engine(engine)
    rows = list_characterizations_for_display(
        con, machine_key, runtime=runtime,
        backend=None if runtime == "ollama" else backend,
        logical_model_id=model_id)
    return max(rows, key=lambda row: (row.get("measured_at") or "",
                                      row["artifact_id"], row["config_key"])) if rows else None


def list_characterizations(con: sqlite3.Connection, machine_key: str,
                           engine: str | None = None) -> list[dict]:
    """Compatibility display list, optionally scoped through a legacy engine label."""
    if engine is None:
        return list_characterizations_for_display(con, machine_key)
    from ara import targets
    runtime, backend, _canonical = targets.for_engine(engine)
    return list_characterizations_for_display(
        con, machine_key, runtime=runtime,
        backend=None if runtime == "ollama" else backend)


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
                          methodology_id: str | None = None,
                          sample_size: int | None = None, tier: str = "measured",
                          refused_n: int | None = None, errored_n: int | None = None,
                          probe_context: int | None = None,
                          generation_cap: int | None = None,
                          repeat_count: int | None = None,
                          total_generations: int | None = None,
                          run_scores: list[float] | None = None,
                          artifact_id: str | None = None,
                          canonical_model_id: str | None = None,
                          target: dict | None = None,
                          request_policy: dict | None = None,
                          runtime_metrics: dict | None = None) -> None:
    from ara.engine_identity import canonical_engine
    canonical = canonical_engine(engine_key)
    cell = _benchmark_cell_values(
        engine_key=canonical, backend=backend, artifact_id=artifact_id,
        methodology_id=methodology_id, target=target, request_policy=request_policy)
    row = {
        "machine_key": machine_key,
        "model_id": model_id,
        "use_case": use_case,
        **cell,
        "engine_key": canonical,
        "base_model": base_model,
        "quant": quant,
        "benchmark_id": benchmark_id,
        "methodology_id": methodology_id,
        "tier": tier,
        "score": score,
        "max_score": max_score,
        "sample_size": sample_size,
        "refused_n": refused_n,
        "errored_n": errored_n,
        "probe_context": probe_context,
        "generation_cap": generation_cap,
        "repeat_count": repeat_count,
        "total_generations": total_generations,
        "run_scores_json": json.dumps(run_scores) if run_scores is not None else None,
        "artifact_id": artifact_id,
        "canonical_model_id": canonical_model_id,
        "target_json": _compact_json(target),
        "request_policy_json": _compact_json(request_policy),
        "runtime_metrics_json": _compact_json(runtime_metrics),
        "source": source,
        "measured_at": _now(),
    }
    columns = ", ".join(row)
    placeholders = ", ".join(f":{column}" for column in row)
    updates = ", ".join(
        f"{column}=excluded.{column}" for column in row
        if column not in {"machine_key", "model_id", "use_case", "evidence_key"})
    con.execute(
        f"INSERT INTO benchmark_results ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT(machine_key, model_id, use_case, evidence_key) DO UPDATE SET "
        f"{updates}",
        row)
    con.commit()


def _compact_json(value: dict | None) -> str | None:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"))
            if value is not None else None)


def _benchmark_row(row: sqlite3.Row) -> dict:
    result = dict(row)
    for name in ("target", "request_policy", "runtime_metrics"):
        raw = result.get(f"{name}_json")
        try:
            decoded = json.loads(raw) if raw is not None else None
        except (TypeError, ValueError):
            decoded = None
        result[name] = decoded if isinstance(decoded, dict) else None
    return result


def get_benchmark_result(con: sqlite3.Connection, machine_key: str, model_id: str,
                         use_case: str) -> dict | None:
    row = con.execute(
        "SELECT * FROM benchmark_results WHERE machine_key=? AND model_id=? AND use_case=? "
        "ORDER BY measured_at DESC, rowid DESC LIMIT 1",
        (machine_key, model_id, use_case)).fetchone()
    return _benchmark_row(row) if row else None


def list_benchmark_results(con: sqlite3.Connection, machine_key: str) -> list[dict]:
    """All benchmark runtime-cell results for a machine in deterministic evidence order."""
    rows = con.execute(
        "SELECT * FROM benchmark_results WHERE machine_key=? "
        "ORDER BY model_id, use_case, measured_at, evidence_key",
        (machine_key,)).fetchall()
    return [_benchmark_row(r) for r in rows]
