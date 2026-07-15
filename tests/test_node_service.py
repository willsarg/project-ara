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
    with pytest.raises(RuntimeError, match="Linux-only") as caught:
        service._require_linux()
    assert "ara node run" in str(caught.value)
    assert "ara node start" not in str(caught.value)


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
    assert f"ExecStart={service._systemd_quote(service.sys.executable)} -m ara node run" in text
    assert "Type=simple" in text                            # long jobs must not trip a loop watchdog
    assert "NotifyAccess=main" in text                      # STATUS works without a watchdog
    assert "WatchdogSec=" not in text
    assert f"RestartSec={service.RESTART_SEC}" in text      # daemon failures never rapid-loop
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


def test_status_info_returns_stable_machine_state(monkeypatch, linux):
    output = "LoadState=loaded\nActiveState=inactive\nSubState=dead\nStatusText=needs enrollment\n"
    monkeypatch.setattr(service, "_run", lambda _cmd: (0, output, ""))
    assert service.status_info() == {
        "installed": True, "active": False, "load_state": "loaded",
        "active_state": "inactive", "sub_state": "dead",
        "status_text": "needs enrollment",
    }


def test_status_info_reports_absent_unit(monkeypatch, linux):
    output = "LoadState=not-found\nActiveState=inactive\nSubState=dead\nStatusText=\n"
    monkeypatch.setattr(service, "_run", lambda _cmd: (0, output, ""))
    assert service.status_info()["installed"] is False


def test_status_info_ignores_unstructured_lines_and_defaults_unknown(monkeypatch, linux):
    monkeypatch.setattr(service, "_run", lambda _cmd: (0, "localized noise\n", ""))
    assert service.status_info() == {
        "installed": False, "active": False, "load_state": "unknown",
        "active_state": "unknown", "sub_state": "unknown", "status_text": None,
    }


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
    assert calls == [
        ["systemctl", "--user", "disable", "--now", service.UNIT_NAME],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_uninstall_missing_file_converges_loaded_systemd_unit(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(service, "_run", lambda cmd: calls.append(cmd) or (0, "disabled", ""))
    service.uninstall()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", service.UNIT_NAME],
        ["systemctl", "--user", "daemon-reload"],
    ]


@pytest.mark.parametrize("message", [
    "Unit ara-node.service does not exist.",
    "Failed to disable unit: Unit file ara-node.service does not exist.",
    "Unit ara-node.service not loaded.",
])
def test_uninstall_missing_file_accepts_explicit_absent_systemd_state(
        tmp_path, monkeypatch, linux, message):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    results = iter(((1, "", message), (0, "", "")))
    monkeypatch.setattr(service, "_run", lambda _cmd: next(results))
    service.uninstall()


def test_uninstall_missing_file_rejects_unclassified_disable_failure(
        tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    monkeypatch.setattr(service, "_run", lambda _cmd: (1, "", "permission denied"))
    with pytest.raises(RuntimeError, match="permission denied"):
        service.uninstall()


def test_uninstall_rejects_mixed_absence_and_fatal_bus_diagnostic(
        tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / service.UNIT_NAME
    unit.write_text("stub")
    calls = []
    diagnostic = (
        "Unit ara-node.service not loaded.\n"
        "Failed to connect to bus: Permission denied"
    )
    monkeypatch.setattr(
        service, "_run", lambda cmd: calls.append(cmd) or (1, "", diagnostic),
    )
    with pytest.raises(RuntimeError, match="Permission denied"):
        service.uninstall()
    assert unit.exists()
    assert calls == [["systemctl", "--user", "disable", "--now", service.UNIT_NAME]]


def test_uninstall_rejects_absence_text_with_unexpected_return_code(
        tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / service.UNIT_NAME
    unit.write_text("stub")
    calls = []
    monkeypatch.setattr(
        service, "_run", lambda cmd: calls.append(cmd) or (
            5, "", "Unit ara-node.service not loaded."),
    )
    with pytest.raises(RuntimeError, match="exited 5"):
        service.uninstall()
    assert unit.exists()
    assert calls == [["systemctl", "--user", "disable", "--now", service.UNIT_NAME]]


def test_uninstall_retry_after_reload_failure_converges(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("ARA_NODE_SYSTEMD_DIR", str(tmp_path))
    unit = tmp_path / service.UNIT_NAME
    unit.write_text("stub")
    first = iter(((0, "", ""), (1, "", "reload failed")))
    monkeypatch.setattr(service, "_run", lambda _cmd: next(first))
    with pytest.raises(RuntimeError, match="reload failed"):
        service.uninstall()
    assert not unit.exists()

    second = iter(((1, "", "Unit ara-node.service not loaded."), (0, "", "")))
    monkeypatch.setattr(service, "_run", lambda _cmd: next(second))
    service.uninstall()


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
