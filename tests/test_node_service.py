# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node service layer — the systemd --user unit lifecycle, the Linux guard, and serve()."""
from __future__ import annotations

import sys
import types

import pytest

from ara.node import service


@pytest.fixture
def linux(monkeypatch):
    """Pin the platform to Linux so the systemd path runs."""
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")


@pytest.fixture
def calls(monkeypatch):
    """Capture every systemctl invocation; no real subprocess ever runs."""
    recorded = []
    monkeypatch.setattr(service, "_run", lambda cmd: recorded.append(cmd) or (0, "OUT", ""))
    return recorded


def test_run_is_the_subprocess_boundary(monkeypatch):
    class P:
        returncode = 0
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(service.subprocess, "run", lambda cmd, capture_output, text: P())
    assert service._run(["systemctl", "--user", "status", "x"]) == (0, "out", "err")


def test_require_linux_raises_off_linux(monkeypatch):
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Linux-only"):
        service._require_linux()


def test_require_linux_passes_on_linux(linux):
    assert service._require_linux() is None


@pytest.mark.parametrize("fn", [
    lambda: service.install("127.0.0.1", 9100),
    service.start, service.stop, service.status, service.uninstall,
])
def test_every_systemd_fn_is_guarded(monkeypatch, fn):
    monkeypatch.setattr(service.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError):
        fn()


def test_install_writes_unit_and_enables(tmp_path, monkeypatch, linux, calls):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path / "sd"))
    service.install("0.0.0.0", 9100)
    unit = tmp_path / "sd" / "ara-node.service"
    text = unit.read_text()
    assert f"ExecStart={service.sys.executable} -m ara.cli node run" in text   # push-only agent loop
    assert "Type=notify" in text                             # watchdog needs sd_notify handshake
    assert f"WatchdogSec={service.WATCHDOG_SEC}" in text
    assert "WantedBy=default.target" in text
    assert calls == [["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "--user", "enable", "--now", "ara-node.service"]]


def test_unit_dir_defaults_to_user_systemd(monkeypatch):
    monkeypatch.delenv("ARA_NODE_SYSTEMD_DIR", raising=False)
    monkeypatch.setattr(service.Path, "home", staticmethod(lambda: service.Path("/home/u")))
    assert service._unit_dir() == service.Path("/home/u/.config/systemd/user")


def test_start_stop_status(linux, calls):
    service.start()
    service.stop()
    assert service.status() == "OUT"           # returns systemctl stdout
    assert calls == [["systemctl", "--user", "start", "ara-node.service"],
                     ["systemctl", "--user", "stop", "ara-node.service"],
                     ["systemctl", "--user", "status", "ara-node.service"]]


def test_uninstall_disables_removes_and_reloads(tmp_path, monkeypatch, linux, calls):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / "ara-node.service"
    unit.write_text("stub")
    service.uninstall()
    assert not unit.exists()                   # file removed
    assert calls == [["systemctl", "--user", "disable", "--now", "ara-node.service"],
                     ["systemctl", "--user", "daemon-reload"]]


def test_uninstall_is_idempotent_when_absent(tmp_path, monkeypatch, linux, calls):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))   # no unit file present
    service.uninstall()                                          # must not raise
    assert calls[0] == ["systemctl", "--user", "disable", "--now", "ara-node.service"]


def test_serve_runs_uvicorn_with_built_app(monkeypatch):
    ran = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, host, port: ran.update(app=app, host=host, port=port)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr("ara.node.wiring.build_app", lambda version: f"APP:{version}")

    service.serve("127.0.0.1", 8088, version="3.3")
    assert ran == {"app": "APP:3.3", "host": "127.0.0.1", "port": 8088}
