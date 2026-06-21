"""db.py — the SQLite persistence foundation (machines, models, characterizations)."""
from __future__ import annotations

from ara import db


def test_connect_creates_schema(store):
    tables = {r[0] for r in store.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"machines", "models", "characterizations"} <= tables


def test_db_path_uses_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "custom.db"))
    assert db._db_path() == tmp_path / "custom.db"


def test_db_path_defaults_to_platform_dir(monkeypatch):
    monkeypatch.delenv("ARA_DB_PATH", raising=False)
    p = db._db_path()
    assert p.name == "ara.db" and "ara" in str(p)


# --- machine profiles (per machine + engine) ---
def test_machine_upsert_and_get(store):
    db.upsert_machine(store, "machine-abc", "wmx", fixed_overhead_gb=1.5, calibrated_at="2026-06-19")
    row = db.get_machine(store, "machine-abc", "wmx")
    assert row["fixed_overhead_gb"] == 1.5 and row["calibrated_at"] == "2026-06-19"


def test_machine_get_missing_is_none(store):
    assert db.get_machine(store, "nope", "wmx") is None


def test_machine_upsert_replaces(store):
    db.upsert_machine(store, "m", "wcx", fixed_overhead_gb=1.0, calibrated_at="t1")
    db.upsert_machine(store, "m", "wcx", fixed_overhead_gb=2.0, calibrated_at="t2")
    assert db.get_machine(store, "m", "wcx")["fixed_overhead_gb"] == 2.0


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
