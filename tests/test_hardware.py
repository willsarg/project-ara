import datetime as dt
from ara import hardware as hw


def test_pwsh_json_wraps_single_object(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: '{"a":1}')
    assert hw._pwsh_json(["x"]) == [{"a": 1}]


def test_pwsh_json_passes_through_array(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: '[{"a":1},{"a":2}]')
    assert hw._pwsh_json(["x"]) == [{"a": 1}, {"a": 2}]


def test_pwsh_json_empty_on_failure(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: None)   # _run returns None on error
    assert hw._pwsh_json(["x"]) == []


def test_wmi_date_parses_dotnet_epoch():
    assert hw._wmi_date("/Date(1651104000000)/") == "2022-04-28"


def test_wmi_date_none_on_garbage():
    assert hw._wmi_date(None) is None and hw._wmi_date("nope") is None


def test_wmi_date_none_on_overflow():
    # A valid /Date(N)/ pattern but an astronomically large number triggers OverflowError.
    assert hw._wmi_date("/Date(999999999999999999999)/") is None


def test_clean_strips_and_nulls_placeholders():
    assert hw._clean("  G-Skill ") == "G-Skill"
    assert hw._clean("System Product Name") is None
    assert hw._clean("") is None


def test_gib_converts_bytes_to_gib():
    assert hw._gib(8 * hw.GB) == 8.0
    assert hw._gib(None) is None
    assert hw._gib("bad") is None


def test_gb_dec_converts_bytes_to_decimal_gb():
    assert hw._gb_dec(1_000_000_000) == 1.0
    assert hw._gb_dec(None) is None
    assert hw._gb_dec("bad") is None


def test_run_returns_none_on_failure():
    # A command that doesn't exist should return None (exception path).
    result = hw._run(["__no_such_cmd__xyz__"], timeout=1)
    assert result is None


def test_run_returns_none_on_nonzero_exit():
    # `false` exits 1; expect None.
    result = hw._run(["false"], timeout=3)
    assert result is None


def test_run_returns_stdout_on_success():
    result = hw._run(["echo", "hello"], timeout=3)
    assert result is not None and "hello" in result


def test_pwsh_json_empty_on_invalid_json(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: "not-json{{{")
    assert hw._pwsh_json(["x"]) == []
