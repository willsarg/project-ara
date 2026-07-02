# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""db.py — the SQLite persistence foundation (calibrations, models, characterizations, profiles).

Profiles persistence: Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

from ara import db


def test_connect_creates_schema(store):
    tables = {r[0] for r in store.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"calibrations", "models", "characterizations"} <= tables


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
    assert [(r["model_id"], r["engine"]) for r in rows] == [("a", "wcx"), ("c", "wmx")]


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


def test_list_characterizations_includes_decode_context(store):
    db.save_characterization(store, "m", "wcx", "a", safe_context=8000, points=[], decode_context=50000)
    db.save_characterization(store, "m", "wcx", "b", safe_context=4000, points=[])
    rows = db.list_characterizations(store, "m", "wcx")
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["a"]["decode_context"] == 50000
    assert by_id["b"]["decode_context"] is None


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
    assert row["engine_key"] == "wcx"
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
