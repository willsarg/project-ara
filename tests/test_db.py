# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""db.py — the SQLite persistence foundation (calibrations, models, characterizations, profiles).

Profiles persistence: Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ara import db


def test_connect_creates_schema(store):
    tables = {r[0] for r in store.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"calibrations", "models", "characterizations"} <= tables


def test_connected_yields_working_connection_and_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "cm.db"))
    with db.connected() as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "models" in tables          # a live, schema-applied handle
    # closed on exit — using it now raises ProgrammingError
    import sqlite3
    import pytest
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_connected_readonly_never_creates_or_writes_store(tmp_path, monkeypatch):
    path = tmp_path / "readonly.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    assert hasattr(db, "connected_readonly")

    with pytest.raises(sqlite3.OperationalError):
        with db.connected_readonly():
            pass
    assert not path.exists()

    writable = db.connect()
    writable.execute("INSERT INTO models (model_id) VALUES ('org/model')")
    writable.commit()
    writable.close()
    with db.connected_readonly() as con:
        assert con.execute("SELECT model_id FROM models").fetchone()[0] == "org/model"
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM models")
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_connected_closes_on_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "cm.db"))
    import sqlite3
    import pytest
    captured = {}
    with pytest.raises(ValueError):
        with db.connected() as con:
            captured["con"] = con
            raise ValueError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        captured["con"].execute("SELECT 1")   # still closed despite the error


_LEGACY = "TestCPU|TestGPU|34359738368|Linux"   # byte-exact 32-GiB legacy key
_NEW = "ara1|TestCPU|TestGPU|32|Linux"           # its versioned GiB-rounded form


def _make_legacy_db(tmp_path, monkeypatch, name="legacy.db"):
    """A store holding rows under a legacy machine_key, with user_version forced below the
    rekey migration so a reconnect triggers the one-time auto-migration."""
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / name))
    con = db.connect()
    db.save_characterization(con, _LEGACY, "cpu", "m1", safe_context=4096, points=[])
    db.upsert_calibration(con, _LEGACY, "cpu", fixed_overhead_gb=1.0, calibrated_at="t")
    db.save_benchmark_result(con, _LEGACY, "m1", "coding", score=1.0, source="s")
    db.save_profile(con, _LEGACY, "{}")
    con.execute("PRAGMA user_version = 1")   # simulate a pre-rekey DB
    con.commit()
    con.close()


def test_connect_rekeys_legacy_keys_across_all_tables(tmp_path, monkeypatch):
    _make_legacy_db(tmp_path, monkeypatch)
    con = db.connect()                       # reconnect → one-time auto-migration
    assert db.get_characterization(con, _NEW, "cpu", "m1")["safe_context"] == 4096
    assert db.get_calibration(con, _NEW, "cpu") is not None
    assert db.get_benchmark_result(con, _NEW, "m1", "coding") is not None
    assert db.get_latest_profile(con, _NEW) is not None
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3
    assert db.get_characterization(con, _LEGACY, "cpu", "m1") is None   # nothing left under legacy


def test_rekey_merges_colliding_legacy_keys_keeping_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "collide.db"))
    a = "TestCPU|TestGPU|34359738368|Linux"       # 32 GiB exact
    b = "TestCPU|TestGPU|34359730000|Linux"       # a few KB less → same 32-GiB bucket
    con = db.connect()
    db.save_characterization(con, a, "cpu", "m1", safe_context=1000, points=[],
                             measured_at="2026-01-01T00:00:00+00:00")
    db.save_characterization(con, b, "cpu", "m1", safe_context=2000, points=[],
                             measured_at="2026-06-01T00:00:00+00:00")
    con.execute("PRAGMA user_version = 1")
    con.commit()
    con.close()
    con = db.connect()
    row = db.get_characterization(con, _NEW, "cpu", "m1")
    assert row is not None and row["safe_context"] == 2000     # newer measured_at wins
    n = con.execute("SELECT COUNT(*) FROM characterizations WHERE machine_key=?",
                    (_NEW,)).fetchone()[0]
    assert n == 1


def test_rekey_legacy_returns_rowcount_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "manual.db"))
    con = db.connect()                       # fresh DB: no legacy keys
    assert db._rekey_legacy(con) == 0        # idempotent no-op on a clean store
    db.save_characterization(con, _LEGACY, "cpu", "m1", safe_context=1, points=[])
    db.save_profile(con, _LEGACY, "{}")
    assert db._rekey_legacy(con) == 2        # 1 characterization + 1 profile row rekeyed
    assert db._rekey_legacy(con) == 0        # second call finds nothing


def test_db_path_uses_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "custom.db"))
    assert db._db_path() == tmp_path / "custom.db"


def test_db_path_defaults_to_platform_dir(monkeypatch):
    monkeypatch.delenv("ARA_DB_PATH", raising=False)
    p = db._db_path()
    assert p.name == "ara.db" and "ara" in str(p)


# --- per-engine calibration ---
def test_calibration_upsert_and_get(store):
    db.upsert_calibration(store, "machine-abc", "wmx", fixed_overhead_gb=1.5, calibrated_at="2026-06-19")
    row = db.get_calibration(store, "machine-abc", "wmx")
    assert row["fixed_overhead_gb"] == 1.5 and row["calibrated_at"] == "2026-06-19"


def test_calibration_get_missing_is_none(store):
    assert db.get_calibration(store, "nope", "wmx") is None


def test_connect_purges_legacy_decimal_wmx_calibrations(tmp_path, monkeypatch):
    """One-time data fix: wmx calibrations written before the GiB boundary conversion carry
    decimal-GB walls (~7.4% high). A float can't reveal its own units, so the rows are deleted
    (the next run re-calibrates honestly) — other engines' rows are untouched, and the purge
    runs once (user_version gate), so fresh GiB rows survive later connects.

    Slug: 2026-07-02-analytic-units-gib
    """
    import sqlite3

    path = tmp_path / "legacy.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    raw = sqlite3.connect(path)                      # a pre-migration store: user_version == 0
    raw.executescript(db.SCHEMA)
    raw.execute("INSERT INTO calibrations (machine_key, engine, fixed_overhead_gb, "
                "calibrated_at, wall_gb, safe_budget_gb) VALUES "
                "('m', 'wmx', 1.1, 't0', 25.8, 22.8)")     # decimal-era apple row
    raw.execute("INSERT INTO calibrations (machine_key, engine, fixed_overhead_gb, "
                "calibrated_at, wall_gb, safe_budget_gb) VALUES "
                "('m', 'wcx', 0.5, 't0', 7.6, 5.6)")       # wcx was always binary — keep
    raw.commit()
    raw.close()

    con = db.connect()
    assert db.get_calibration(con, "m", "wmx") is None           # purged
    assert db.get_calibration(con, "m", "wcx")["wall_gb"] == 7.6  # untouched
    # The migration chain now ends at v3; the single-field key 'm' is not a legacy machine key.
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3

    # A NEW (GiB-era) wmx calibration written after the purge must survive a reconnect.
    db.upsert_calibration(con, "m", "wmx", fixed_overhead_gb=1.0, calibrated_at="t1",
                          wall_gb=24.0, safe_budget_gb=21.0)
    con.close()
    con2 = db.connect()
    assert db.get_calibration(con2, "m", "wmx")["wall_gb"] == 24.0
    con2.close()


def test_calibration_upsert_replaces(store):
    db.upsert_calibration(store, "m", "wcx", fixed_overhead_gb=1.0, calibrated_at="t1")
    db.upsert_calibration(store, "m", "wcx", fixed_overhead_gb=2.0, calibrated_at="t2")
    assert db.get_calibration(store, "m", "wcx")["fixed_overhead_gb"] == 2.0


# --- measured wall persistence (Spec 2026-06-23-capability-pipeline) ---
def test_calibration_persists_measured_wall(store):
    # The measured wall + safe budget ride alongside the overhead so profile/recommend can
    # report what the engine actually measured, not just the heuristic.
    db.upsert_calibration(store, "m", "wcx", fixed_overhead_gb=0.6, calibrated_at="t",
                          wall_gb=24.0, safe_budget_gb=23.0)
    row = db.get_calibration(store, "m", "wcx")
    assert row["wall_gb"] == 24.0 and row["safe_budget_gb"] == 23.0


def test_calibration_wall_defaults_to_none(store):
    # Existing callers that don't pass a wall keep working — the columns stay NULL.
    db.upsert_calibration(store, "m", "wcx", fixed_overhead_gb=0.6, calibrated_at="t")
    row = db.get_calibration(store, "m", "wcx")
    assert row["wall_gb"] is None and row["safe_budget_gb"] is None


def test_migration_adds_wall_columns_to_old_schema(tmp_path, monkeypatch):
    """An existing DB without wall_gb/safe_budget_gb gets the columns after connect()."""
    import sqlite3 as _sqlite3
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("ARA_DB_PATH", str(db_path))

    # Build an old-schema calibrations table (no wall columns) and seed one row.
    OLD_SCHEMA = """
    CREATE TABLE IF NOT EXISTS calibrations (
        machine_key       TEXT NOT NULL,
        engine            TEXT NOT NULL,
        fixed_overhead_gb REAL,
        calibrated_at     TEXT,
        PRIMARY KEY (machine_key, engine)
    );
    """
    old_con = _sqlite3.connect(str(db_path))
    old_con.row_factory = _sqlite3.Row
    old_con.executescript(OLD_SCHEMA)
    old_con.execute(
        "INSERT INTO calibrations (machine_key, engine, fixed_overhead_gb, calibrated_at) "
        "VALUES (?, ?, ?, ?)", ("m", "wcx", 0.6, "2026-01-01T00:00:00+00:00"))
    old_con.commit()
    old_con.close()

    # connect() must run the migration, leaving the old row readable with NULL wall columns.
    con = db.connect()
    cols = {r["name"] for r in con.execute("PRAGMA table_info(calibrations)")}
    assert {"wall_gb", "safe_budget_gb"} <= cols
    row = db.get_calibration(con, "m", "wcx")
    assert row["fixed_overhead_gb"] == 0.6
    assert row["wall_gb"] is None and row["safe_budget_gb"] is None


# --- model catalog ---
def test_model_upsert_and_get(store):
    db.upsert_model(store, "org/m", modality="text", params=135_000_000, quant="fp16",
                    n_layers=30, hidden_size=576, kv_heads=3, head_dim=64,
                    weights_gb=0.27, max_context=8192)
    row = db.get_model(store, "org/m")
    assert row["modality"] == "text" and row["n_layers"] == 30 and row["kv_heads"] == 3


def test_model_get_missing_is_none(store):
    assert db.get_model(store, "nope") is None


def test_model_upsert_replaces_and_partial_fields(store):
    db.upsert_model(store, "m", modality="text", n_layers=10)
    db.upsert_model(store, "m", modality="text", n_layers=20)   # update
    row = db.get_model(store, "m")
    assert row["n_layers"] == 20 and row["quant"] is None       # unset field stays NULL


def test_list_models(store):
    db.upsert_model(store, "a", modality="text")
    db.upsert_model(store, "b", modality="vision")
    assert {m["model_id"] for m in db.list_models(store)} == {"a", "b"}


# --- characterizations (fitted ceilings) ---
def test_characterization_save_and_get(store):
    db.save_characterization(store, "m", "wcx", "org/model", safe_context=16000,
                             points=[(512, 1.4), (2048, 2.0)], measured_at="2026-06-19")
    row = db.get_characterization(store, "m", "wcx", "org/model")
    assert row["safe_context"] == 16000
    assert row["points"] == [[512, 1.4], [2048, 2.0]]   # JSON round-trip → lists
    assert row["measured_at"] == "2026-06-19"


def test_characterization_missing_is_none(store):
    assert db.get_characterization(store, "m", "wcx", "x") is None


def test_list_characterizations_for_machine_and_engine(store):
    db.save_characterization(store, "m", "wcx", "a", safe_context=1000, points=[[1, 2.0]])
    db.save_characterization(store, "m", "wcx", "b", safe_context=2000, points=[])
    db.save_characterization(store, "m", "wmx", "c", safe_context=3000, points=[])  # other engine
    rows = db.list_characterizations(store, "m", "wcx")
    assert {r["model_id"] for r in rows} == {"a", "b"}
    assert rows[0]["points"] == [[1, 2.0]]   # JSON parsed back


def test_list_characterizations_across_all_engines_when_engine_omitted(store):
    db.save_characterization(store, "m", "wcx", "a", safe_context=1000, points=[])
    db.save_characterization(store, "m", "wmx", "c", safe_context=3000, points=[])
    db.save_characterization(store, "other", "wcx", "z", safe_context=1, points=[])  # other machine
    rows = db.list_characterizations(store, "m")           # engine omitted → every engine
    assert [(r["model_id"], r["engine"]) for r in rows] == [("a", "cuda"), ("c", "mlx")]


# --- decode_context persistence ---
def test_save_characterization_with_decode_context_round_trips(store):
    db.save_characterization(store, "m", "wcx", "org/model", safe_context=16000,
                             points=[], decode_context=62000)
    row = db.get_characterization(store, "m", "wcx", "org/model")
    assert row["decode_context"] == 62000


def test_save_characterization_default_decode_context_is_none(store):
    db.save_characterization(store, "m", "wcx", "org/model", safe_context=16000, points=[])
    row = db.get_characterization(store, "m", "wcx", "org/model")
    assert row["decode_context"] is None


def test_characterization_artifact_id_round_trips(store):
    artifact = "ollama-manifest-sha256:" + "a" * 64
    db.save_characterization(
        store, "m", "ollama", "org/model", safe_context=16000, points=[],
        artifact_id=artifact,
    )
    row = db.get_characterization(store, "m", "ollama", "org/model")
    assert row["artifact_id"] == artifact


def test_legacy_characterization_migrates_with_unknown_artifact_id(tmp_path, monkeypatch):
    path = tmp_path / "legacy-artifact.db"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE characterizations (
        machine_key TEXT NOT NULL, engine TEXT NOT NULL, model_id TEXT NOT NULL,
        safe_context INTEGER, decode_context INTEGER, config_json TEXT,
        points_json TEXT, measured_at TEXT,
        PRIMARY KEY (machine_key, engine, model_id)
    );
    INSERT INTO characterizations VALUES
        ('m', 'ollama', 'org/model', 4096, NULL, '{}', '[]', '2026-01-01');
    PRAGMA user_version = 3;
    """)
    con.close()
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    migrated = db.connect()
    try:
        assert db.get_characterization(
            migrated, "m", "ollama", "org/model")["artifact_id"] is None
    finally:
        migrated.close()


def test_list_characterizations_includes_decode_context(store):
    db.save_characterization(store, "m", "wcx", "a", safe_context=8000, points=[], decode_context=50000)
    db.save_characterization(store, "m", "wcx", "b", safe_context=4000, points=[])
    rows = db.list_characterizations(store, "m", "wcx")
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["a"]["decode_context"] == 50000
    assert by_id["b"]["decode_context"] is None


def test_characterization_round_trips_measurement_config(store):
    db.save_characterization(
        store, "m", "mlx", "org/model", safe_context=8192, points=[],
        config={"kv_quant": "q4_0"},
    )
    row = db.get_characterization(store, "m", "mlx", "org/model")
    assert row["config"] == {"kv_quant": "q4_0"}
    assert json.loads(row["config_json"]) == {"kv_quant": "q4_0"}


def test_characterization_defaults_to_default_measurement_config(store):
    db.save_characterization(store, "m", "cpu", "org/model",
                             safe_context=8192, points=[])
    assert db.get_characterization(store, "m", "cpu", "org/model")["config"] == {}


def test_legacy_characterization_config_remains_unknown(tmp_path, monkeypatch):
    path = tmp_path / "legacy-config.db"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE characterizations (
        machine_key TEXT NOT NULL, engine TEXT NOT NULL, model_id TEXT NOT NULL,
        safe_context INTEGER, decode_context INTEGER, points_json TEXT, measured_at TEXT,
        PRIMARY KEY (machine_key, engine, model_id)
    );
    INSERT INTO characterizations VALUES ('m', 'mlx', 'org/model', 8192, NULL, '[]', 'then');
    """)
    con.close()
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    with db.connected() as migrated:
        row = db.get_characterization(migrated, "m", "mlx", "org/model")
        assert row["config"] is None


def test_migration_adds_decode_context_column_to_old_schema(tmp_path, monkeypatch):
    """An existing DB without decode_context gets the column after connect()."""
    import sqlite3 as _sqlite3
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("ARA_DB_PATH", str(db_path))

    # Build an old-schema DB (no decode_context column) and seed one row.
    OLD_SCHEMA = """
    CREATE TABLE IF NOT EXISTS characterizations (
        machine_key  TEXT NOT NULL,
        engine       TEXT NOT NULL,
        model_id     TEXT NOT NULL,
        safe_context INTEGER,
        points_json  TEXT,
        measured_at  TEXT,
        PRIMARY KEY (machine_key, engine, model_id)
    );
    """
    old_con = _sqlite3.connect(str(db_path))
    old_con.row_factory = _sqlite3.Row
    old_con.executescript(OLD_SCHEMA)
    old_con.execute(
        "INSERT INTO characterizations (machine_key, engine, model_id, safe_context, points_json, measured_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("m", "wcx", "org/model", 8000, "[]", "2026-01-01T00:00:00+00:00"))
    old_con.commit()
    old_con.close()

    # Now call connect() — it must run the migration
    con = db.connect()
    cols = {r["name"] for r in con.execute("PRAGMA table_info(characterizations)")}
    assert "decode_context" in cols

    # Old row must still be readable and decode_context should be None
    row = db.get_characterization(con, "m", "wcx", "org/model")
    assert row is not None
    assert row["safe_context"] == 8000
    assert row["decode_context"] is None


# --- system profiles (Slice 1; Spec 2026-06-23-capability-pipeline) ---
def test_connect_creates_profiles_table(store):
    tables = {r[0] for r in store.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "profiles" in tables


def test_profile_save_and_get_latest(store):
    db.save_profile(store, "machine-abc", '{"chip": "Apple M4 Pro"}')
    row = db.get_latest_profile(store, "machine-abc")
    assert row["profile_json"] == '{"chip": "Apple M4 Pro"}'
    assert row["captured_at"]   # default timestamp filled in when not given


def test_profile_get_latest_missing_is_none(store):
    assert db.get_latest_profile(store, "nope") is None


def test_profile_history_kept_latest_wins(store):
    db.save_profile(store, "m", '{"v": 1}', captured_at="2026-06-23T01:00:00+00:00")
    db.save_profile(store, "m", '{"v": 2}', captured_at="2026-06-23T02:00:00+00:00")
    assert db.get_latest_profile(store, "m")["profile_json"] == '{"v": 2}'
    rows = db.list_profiles(store, "m")
    assert [r["profile_json"] for r in rows] == ['{"v": 2}', '{"v": 1}']   # newest first, history kept


# --- benchmark results ---
def test_benchmark_result_save_and_get_round_trips_all_fields(store):
    db.save_benchmark_result(
        store, "machine-abc", "org/model", "coding",
        score=0.72, source="mmlu-v1",
        engine_key="wcx", backend="cuda", base_model="llama3", quant="q4_0",
        benchmark_id="run-001", max_score=1.0, sample_size=100, tier="measured")
    row = db.get_benchmark_result(store, "machine-abc", "org/model", "coding")
    assert row is not None
    assert row["machine_key"] == "machine-abc"
    assert row["model_id"] == "org/model"
    assert row["use_case"] == "coding"
    assert row["score"] == 0.72
    assert row["source"] == "mmlu-v1"
    assert row["engine_key"] == "cuda"
    assert row["backend"] == "cuda"
    assert row["base_model"] == "llama3"
    assert row["quant"] == "q4_0"
    assert row["benchmark_id"] == "run-001"
    assert row["max_score"] == 1.0
    assert row["sample_size"] == 100
    assert row["tier"] == "measured"
    assert row["measured_at"]   # auto-filled timestamp


def test_benchmark_result_get_missing_is_none(store):
    assert db.get_benchmark_result(store, "nope", "org/model", "coding") is None


def test_benchmark_result_upsert_updates_score_no_duplicate(store):
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.5, source="run-1")
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.8, source="run-2")
    row = db.get_benchmark_result(store, "m", "org/model", "coding")
    assert row["score"] == 0.8
    assert row["source"] == "run-2"
    # Only one row — no duplicate
    count = store.execute(
        "SELECT COUNT(*) FROM benchmark_results WHERE machine_key='m' AND model_id='org/model' AND use_case='coding'"
    ).fetchone()[0]
    assert count == 1


def test_list_benchmark_results_returns_machine_rows_only(store):
    db.save_benchmark_result(store, "m", "org/a", "coding", score=0.7, source="s1")
    db.save_benchmark_result(store, "m", "org/b", "math", score=0.6, source="s2")
    db.save_benchmark_result(store, "other-machine", "org/c", "coding", score=0.9, source="s3")
    rows = db.list_benchmark_results(store, "m")
    assert len(rows) == 2
    assert {r["model_id"] for r in rows} == {"org/a", "org/b"}
    # other-machine excluded
    assert all(r["machine_key"] == "m" for r in rows)


# --- benchmark result honesty columns: refused_n / errored_n (Spec 2026-07-02-benchmark-honesty-persistence) ---
def test_benchmark_result_persists_refused_and_errored_counts(store):
    """A partial-governance run stores its refusal/error counts so the score isn't misread as
    a clean full run — Rule #3. Spec 2026-07-02-benchmark-honesty-persistence."""
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.4, source="wmx probe",
                             refused_n=3, errored_n=2)
    row = db.get_benchmark_result(store, "m", "org/model", "coding")
    assert row["refused_n"] == 3
    assert row["errored_n"] == 2


def test_benchmark_result_refused_errored_default_none_is_legacy_unknown(store):
    """Omitting the counts stores NULL = legacy/unknown (distinct from 0 = measured clean run).
    Spec 2026-07-02-benchmark-honesty-persistence."""
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.4, source="s")
    row = db.get_benchmark_result(store, "m", "org/model", "coding")
    assert row["refused_n"] is None and row["errored_n"] is None


def test_benchmark_result_zero_counts_are_a_clean_run(store):
    """0 counts (a clean run) round-trip as 0, not NULL. Spec 2026-07-02-benchmark-honesty-persistence."""
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.9, source="s",
                             refused_n=0, errored_n=0)
    row = db.get_benchmark_result(store, "m", "org/model", "coding")
    assert row["refused_n"] == 0 and row["errored_n"] == 0


def test_benchmark_result_upsert_updates_refused_errored_counts(store):
    """Re-running overwrites the stored counts (ON CONFLICT UPDATE).
    Spec 2026-07-02-benchmark-honesty-persistence."""
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.4, source="run-1",
                             refused_n=3, errored_n=2)
    db.save_benchmark_result(store, "m", "org/model", "coding", score=0.8, source="run-2",
                             refused_n=0, errored_n=0)
    row = db.get_benchmark_result(store, "m", "org/model", "coding")
    assert row["refused_n"] == 0 and row["errored_n"] == 0 and row["score"] == 0.8


def test_migration_adds_refused_errored_columns_to_old_schema(tmp_path, monkeypatch):
    """An existing DB whose benchmark_results predates the honesty columns gets them after
    connect(), old rows readable with NULL counts. Spec 2026-07-02-benchmark-honesty-persistence."""
    import sqlite3 as _sqlite3
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("ARA_DB_PATH", str(db_path))

    # Build an old-schema benchmark_results table (no refused_n/errored_n) and seed one row.
    OLD_SCHEMA = """
    CREATE TABLE IF NOT EXISTS benchmark_results (
        machine_key  TEXT NOT NULL,
        model_id     TEXT NOT NULL,
        use_case     TEXT NOT NULL,
        tier         TEXT NOT NULL DEFAULT 'measured',
        score        REAL NOT NULL,
        source       TEXT NOT NULL,
        measured_at  TEXT NOT NULL,
        PRIMARY KEY (machine_key, model_id, use_case)
    );
    """
    old_con = _sqlite3.connect(str(db_path))
    old_con.row_factory = _sqlite3.Row
    old_con.executescript(OLD_SCHEMA)
    old_con.execute(
        "INSERT INTO benchmark_results (machine_key, model_id, use_case, score, source, measured_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("m", "org/model", "coding", 0.5, "legacy", "2026-01-01T00:00:00+00:00"))
    old_con.commit()
    old_con.close()

    con = db.connect()
    cols = {r["name"] for r in con.execute("PRAGMA table_info(benchmark_results)")}
    assert {"refused_n", "errored_n"} <= cols
    row = db.get_benchmark_result(con, "m", "org/model", "coding")
    assert row is not None and row["score"] == 0.5
    assert row["refused_n"] is None and row["errored_n"] is None


# --- canonical engine identity migration (user_version 2 -> 3) ---
def _v2_engine_identity_db(tmp_path, monkeypatch, name="identity.db"):
    path = tmp_path / name
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    con = sqlite3.connect(path)
    con.executescript(db.SCHEMA)
    con.execute("PRAGMA user_version = 2")
    return path, con


def test_v3_preserves_canonical_only_engine_rows_as_complete_rows(tmp_path, monkeypatch):
    _, con = _v2_engine_identity_db(tmp_path, monkeypatch, "canonical-only.db")
    calibrations = [
        ("mlx-machine", "mlx", 1.25, "2026-03-01", 24.0, None),
        ("cuda-machine", "cuda", None, None, 8.0, 7.0),
    ]
    con.executemany("INSERT INTO calibrations VALUES (?,?,?,?,?,?)", calibrations)
    characterizations = [
        ("mlx-machine", "mlx", "org/mlx-model", 4096, None,
         '[[512, 3.25], [4096, 7.5]]', "2026-03-02"),
        ("cuda-machine", "cuda", "org/cuda-model", None, 8192, None, None),
    ]
    con.executemany(
        "INSERT INTO characterizations (machine_key, engine, model_id, safe_context, "
        "decode_context, points_json, measured_at) VALUES (?,?,?,?,?,?,?)",
        characterizations)
    con.commit()
    con.close()

    migrated = db.connect()
    assert migrated.execute("PRAGMA user_version").fetchone()[0] == 3
    assert [tuple(row) for row in migrated.execute(
        "SELECT machine_key, engine, fixed_overhead_gb, calibrated_at, wall_gb, "
        "safe_budget_gb FROM calibrations ORDER BY machine_key"
    )] == sorted(calibrations)
    assert [tuple(row) for row in migrated.execute(
        "SELECT machine_key, engine, model_id, safe_context, decode_context, points_json, "
        "measured_at FROM characterizations ORDER BY machine_key"
    )] == sorted(characterizations)


def test_v3_migrates_engine_evidence_and_resolves_complete_row_collisions(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch)
    calibrations = [
        ("old", "wmx", 1.0, "2026-01-01", 10.0, 9.0),
        ("tie", "wmx", 1.0, "2026-02-01", 11.0, 10.0),
        ("tie", "mlx", 2.0, "2026-02-01", 22.0, 20.0),
        ("null-ts", "wcx", 1.0, None, 7.0, 6.0),
        ("null-ts", "cuda", 2.0, "2026-01-01", 8.0, 7.0),
    ]
    con.executemany("INSERT INTO calibrations VALUES (?,?,?,?,?,?)", calibrations)
    characterizations = [
        ("old", "wcx", "a", 100, 101, '["old"]', "2026-01-01"),
        ("collision", "wcx", "b", 100, 101, '["legacy"]', "2026-01-01"),
        ("collision", "cuda", "b", 200, 202, '["complete-winner"]', "2026-02-01"),
    ]
    con.executemany(
        "INSERT INTO characterizations (machine_key, engine, model_id, safe_context, "
        "decode_context, points_json, measured_at) VALUES (?,?,?,?,?,?,?)",
        characterizations)
    con.execute(
        "INSERT INTO benchmark_results (machine_key, model_id, use_case, engine_key, score, source, measured_at) "
        "VALUES ('m','model','coding','wcx',1.0,'wcx probe','2026-01-01')")
    profile = {
        "machine": {"engine": "wmx", "description": "wmx remains prose"},
        "projection": {"engine": "wcx", "nested": {"engine": "wmx"}},
        "engine": "wmx",
    }
    con.execute("INSERT INTO profiles VALUES ('m','2026-01-01',?)", (json.dumps(profile),))
    con.execute("INSERT INTO profiles VALUES ('m','2026-01-02',?)", (
        json.dumps({"machine": {"engine": "cpu"}, "projection": {"engine": "wcx"}}),))
    con.commit()
    con.close()

    migrated = db.connect()
    assert migrated.execute("PRAGMA user_version").fetchone()[0] == 3
    assert migrated.execute("SELECT engine FROM calibrations WHERE machine_key='old'").fetchone()[0] == "mlx"
    tie = dict(migrated.execute("SELECT * FROM calibrations WHERE machine_key='tie'").fetchone())
    assert tie["engine"] == "mlx" and tie["fixed_overhead_gb"] == 2.0 and tie["wall_gb"] == 22.0
    null_ts = dict(migrated.execute("SELECT * FROM calibrations WHERE machine_key='null-ts'").fetchone())
    assert null_ts["engine"] == "cuda" and null_ts["fixed_overhead_gb"] == 2.0
    char = dict(migrated.execute("SELECT * FROM characterizations WHERE machine_key='collision'").fetchone())
    assert char["engine"] == "cuda" and char["safe_context"] == 200
    assert char["decode_context"] == 202 and char["points_json"] == '["complete-winner"]'
    old_char = migrated.execute("SELECT engine FROM characterizations WHERE machine_key='old'").fetchone()
    assert old_char[0] == "cuda"
    bench = dict(migrated.execute("SELECT engine_key, source FROM benchmark_results").fetchone())
    assert bench == {"engine_key": "cuda", "source": "wcx probe"}
    migrated_profile = json.loads(migrated.execute("SELECT profile_json FROM profiles").fetchone()[0])
    assert migrated_profile["machine"]["engine"] == "mlx"
    assert migrated_profile["projection"]["engine"] == "cuda"
    assert migrated_profile["machine"]["description"] == "wmx remains prose"
    assert migrated_profile["projection"]["nested"]["engine"] == "wmx"
    assert migrated_profile["engine"] == "wmx"
    assert path.with_name(path.name + ".pre-engine-identity-v3.bak").exists()


def test_v3_malformed_profile_rolls_back_every_write(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "malformed.db")
    con.execute("INSERT INTO calibrations VALUES ('m','wmx',1.0,'t',2.0,1.0)")
    con.execute("INSERT INTO benchmark_results (machine_key,model_id,use_case,engine_key,score,source,measured_at) "
                "VALUES ('m','model','coding','wcx',1.0,'wcx probe','t')")
    con.execute("INSERT INTO profiles VALUES ('m','t','{not json')")
    con.commit()
    con.close()

    with pytest.raises(json.JSONDecodeError):
        db.connect()
    raw = sqlite3.connect(path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 2
    assert raw.execute("SELECT engine FROM calibrations").fetchone()[0] == "wmx"
    assert raw.execute("SELECT engine_key FROM benchmark_results").fetchone()[0] == "wcx"
    assert raw.execute("SELECT profile_json FROM profiles").fetchone()[0] == "{not json"


def test_v3_second_connect_is_noop_and_preserves_existing_backup(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "noop.db")
    con.execute("INSERT INTO calibrations VALUES ('m','wmx',1.0,'t',2.0,1.0)")
    con.commit()
    con.close()
    first = db.connect()
    first.close()
    backup = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    before = backup.read_bytes()
    second = db.connect()
    assert second.execute("PRAGMA user_version").fetchone()[0] == 3
    assert second.execute("SELECT engine FROM calibrations").fetchone()[0] == "mlx"
    second.close()
    assert backup.read_bytes() == before


def test_v3_backup_failure_leaves_no_final_then_retry_creates_valid_pre_v3_backup(
        tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "backup-retry.db")
    con.execute("INSERT INTO calibrations VALUES ('m','wmx',1.0,'t',2.0,1.0)")
    con.commit()
    backup = path.with_name(path.name + ".pre-engine-identity-v3.bak")

    class FailingBackup:
        def execute(self, sql):
            return con.execute(sql)

        def backup(self, target):
            raise RuntimeError("forced backup failure")

    with pytest.raises(RuntimeError, match="forced backup failure"):
        db._backup_before_engine_identity_v3(FailingBackup(), path)
    assert not backup.exists()
    assert list(tmp_path.glob("*.pre-engine-identity-v3.bak.*.tmp")) == []

    db._backup_before_engine_identity_v3(con, path)
    saved = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert saved.execute("PRAGMA user_version").fetchone()[0] == 2
    assert saved.execute("SELECT engine FROM calibrations").fetchone()[0] == "wmx"
    saved.close()
    con.close()


def test_v3_backup_replaces_invalid_existing_file(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "invalid-backup.db")
    con.commit()
    backup = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    backup.write_bytes(b"")

    db._backup_before_engine_identity_v3(con, path)

    saved = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    assert saved.execute("PRAGMA user_version").fetchone()[0] == 2
    saved.close()
    con.close()


def test_v3_backup_keeps_valid_concurrent_publisher(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "concurrent-backup.db")
    con.commit()
    backup = path.with_name(path.name + ".pre-engine-identity-v3.bak")

    real_replace = db.os.replace

    def concurrent_publish(source, destination):
        Path(destination).write_bytes(Path(source).read_bytes())
        real_replace(source, destination)

    monkeypatch.setattr(db.os, "replace", concurrent_publish)
    db._backup_before_engine_identity_v3(con, path)
    saved = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    assert saved.execute("PRAGMA user_version").fetchone()[0] == 2
    saved.close()
    assert list(tmp_path.glob("*.pre-engine-identity-v3.bak.*.tmp")) == []
    con.close()


def test_v3_backup_rejects_invalid_temporary_backup_and_cleans_it(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "invalid-temp-backup.db")
    con.commit()
    real_connect = db.sqlite3.connect

    def connect(database, *args, **kwargs):
        if isinstance(database, str) and ".tmp?mode=ro" in database:
            return real_connect(":memory:")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(db.sqlite3, "connect", connect)
    with pytest.raises(sqlite3.DatabaseError, match="pre-v3 backup validation failed"):
        db._backup_before_engine_identity_v3(con, path)
    assert not path.with_name(path.name + ".pre-engine-identity-v3.bak").exists()
    assert list(tmp_path.glob("*.pre-engine-identity-v3.bak.*.tmp")) == []
    con.close()


def test_v3_backup_rejects_invalid_published_final_and_cleans_temp(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "bad-concurrent-backup.db")
    con.commit()
    backup = path.with_name(path.name + ".pre-engine-identity-v3.bak")

    def publish_invalid(source, destination):
        Path(destination).write_bytes(b"")
        Path(source).unlink()

    monkeypatch.setattr(db.os, "replace", publish_invalid)
    with pytest.raises(sqlite3.DatabaseError, match="published pre-v3 backup validation failed"):
        db._backup_before_engine_identity_v3(con, path)
    assert list(tmp_path.glob("*.pre-engine-identity-v3.bak.*.tmp")) == []
    con.close()


def test_v3_backup_atomically_replaces_invalid_final_after_interleaving(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "interleaved-backup.db")
    con.commit()
    final = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    final.write_bytes(b"")
    real_replace = db.os.replace

    def interleaved_replace(source, destination):
        Path(destination).write_bytes(b"concurrent invalid file")
        real_replace(source, destination)

    monkeypatch.setattr(db.os, "replace", interleaved_replace)
    db._backup_before_engine_identity_v3(con, path)
    saved = sqlite3.connect(f"file:{final}?mode=ro", uri=True)
    assert saved.execute("PRAGMA user_version").fetchone()[0] == 2
    saved.close()
    con.close()


def test_v3_backup_does_not_depend_on_hard_links(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "no-hardlink.db")
    con.commit()
    monkeypatch.setattr(db.os, "link", lambda *args: (_ for _ in ()).throw(OSError("unsupported")))
    db._backup_before_engine_identity_v3(con, path)
    assert path.with_name(path.name + ".pre-engine-identity-v3.bak").exists()
    con.close()


def test_v3_backup_removes_only_stale_orphan_temps(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "orphan-backup.db")
    con.commit()
    final = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    stale = final.with_name(final.name + ".stale.tmp")
    fresh = final.with_name(final.name + ".fresh.tmp")
    stale.write_text("stale")
    fresh.write_text("fresh")
    old = 1_000_000_000
    db.os.utime(stale, (old, old))
    monkeypatch.setattr(db.time, "time", lambda: old + 172_800)

    db._backup_before_engine_identity_v3(con, path)

    assert not stale.exists()
    assert fresh.exists()
    con.close()


def test_v3_backup_tolerates_orphan_disappearing_during_cleanup(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "vanished-orphan.db")
    con.commit()
    final = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    orphan = final.with_name(final.name + ".vanished.tmp")
    orphan.write_text("orphan")
    real_stat = Path.stat

    def stat(candidate, *args, **kwargs):
        if candidate == orphan:
            candidate.unlink()
            raise FileNotFoundError
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    db._backup_before_engine_identity_v3(con, path)
    assert final.exists()
    con.close()


def test_v3_valid_backup_still_cleans_only_stale_orphan_temps(tmp_path, monkeypatch):
    path, con = _v2_engine_identity_db(tmp_path, monkeypatch, "valid-with-orphans.db")
    con.commit()
    db._backup_before_engine_identity_v3(con, path)
    final = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    stale = final.with_name(final.name + ".stale.tmp")
    fresh = final.with_name(final.name + ".fresh.tmp")
    stale.write_text("stale")
    fresh.write_text("fresh")
    old = 1_000_000_000
    db.os.utime(stale, (old, old))
    monkeypatch.setattr(db.time, "time", lambda: old + 172_800)

    db._backup_before_engine_identity_v3(con, path)

    assert not stale.exists()
    assert fresh.exists()
    con.close()
