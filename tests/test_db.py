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
