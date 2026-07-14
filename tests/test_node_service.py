# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node service layer — the systemd --user unit lifecycle and the Linux guard."""
from __future__ import annotations

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
    service.install, service.start, service.stop, service.status, service.uninstall,
])
def test_every_systemd_fn_is_guarded(monkeypatch, fn):
    monkeypatch.setattr(service.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError):
        fn()


def test_install_writes_unit_and_enables(tmp_path, monkeypatch, linux, calls):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path / "sd"))
    service.install()
    unit = tmp_path / "sd" / "ara-node.service"
    text = unit.read_text()
    assert f'ExecStart="{service.sys.executable}" -m ara node run' in text
    assert "Type=notify" in text                             # watchdog needs sd_notify handshake
    assert f"WatchdogSec={service.WATCHDOG_SEC}" in text
    assert "WantedBy=default.target" in text
    assert calls == [["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "--user", "enable", "--now", "ara-node.service"]]


def test_unit_execstart_quotes_and_escapes_systemd_argv(monkeypatch):
    monkeypatch.setattr(service.sys, "executable", '/opt/ARA %build/py"thon\\bin')
    assert 'ExecStart="/opt/ARA %%build/py\\"thon\\\\bin" -m ara node run\n' \
        in service._unit_text()


@pytest.mark.parametrize("value", [
    "/opt/ara\npython", "/opt/ara\x00python", "/opt/ara\x1fpython", "/opt/ara\u0085python",
])
def test_unit_execstart_rejects_control_characters(monkeypatch, value):
    monkeypatch.setattr(service.sys, "executable", value)
    with pytest.raises(ValueError, match="control"):
        service._unit_text()


def test_install_stops_after_daemon_reload_failure(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        service, "_run",
        lambda cmd: calls.append(cmd) or (1, "reload stdout", "reload denied"),
    )
    with pytest.raises(RuntimeError, match="daemon-reload.*reload denied"):
        service.install()
    assert calls == [["systemctl", "--user", "daemon-reload"]]


def test_install_surfaces_enable_partial_state_failure(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    results = iter(((0, "", ""), (1, "Created symlink", "start job failed")))
    monkeypatch.setattr(service, "_run", lambda _cmd: next(results))
    with pytest.raises(RuntimeError, match="enable --now.*Created symlink.*start job failed"):
        service.install()


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


@pytest.mark.parametrize("fn,verb", [(service.start, "start"), (service.stop, "stop")])
def test_start_and_stop_surface_systemctl_failure(monkeypatch, linux, fn, verb):
    monkeypatch.setattr(service, "_run", lambda _cmd: (4, "daemon says no", "unit failed"))
    with pytest.raises(RuntimeError, match=rf"{verb}.*daemon says no.*unit failed"):
        fn()


@pytest.mark.parametrize("rc", [0, 3])
def test_status_accepts_active_and_inactive_state_blocks(monkeypatch, linux, rc):
    monkeypatch.setattr(service, "_run", lambda _cmd: (rc, "status block\n", ""))
    assert service.status() == "status block\n"


def test_status_rejects_unexpected_systemctl_failure(monkeypatch, linux):
    monkeypatch.setattr(service, "_run", lambda _cmd: (4, "", "unit not found"))
    with pytest.raises(RuntimeError, match="status.*unit not found"):
        service.status()


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
    assert calls == []


def test_uninstall_disable_failure_keeps_unit_and_skips_reload(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / service.UNIT_NAME
    unit.write_text("stub")
    calls = []
    monkeypatch.setattr(
        service, "_run",
        lambda cmd: calls.append(cmd) or (1, "still active", "disable denied"),
    )
    with pytest.raises(RuntimeError, match="disable --now.*disable denied"):
        service.uninstall()
    assert unit.exists()
    assert calls == [["systemctl", "--user", "disable", "--now", service.UNIT_NAME]]


def test_uninstall_reload_failure_is_surfaced_after_removal(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / service.UNIT_NAME
    unit.write_text("stub")
    results = iter(((0, "", ""), (1, "", "reload failed")))
    monkeypatch.setattr(service, "_run", lambda _cmd: next(results))
    with pytest.raises(RuntimeError, match="daemon-reload.*reload failed"):
        service.uninstall()
    assert not unit.exists()
