import datetime as dt
import sys

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


# ---------------------------------------------------------------------------
# Task 2: CPU detail
# ---------------------------------------------------------------------------

# --- macOS parser ---

def test_cpu_macos_apple_silicon_has_no_clock_or_l3():
    s = {
        "machdep.cpu.brand_string": "Apple M4 Pro",
        "hw.physicalcpu": "12",
        "hw.logicalcpu": "12",
        "hw.l1icachesize": "131072",
        "hw.l1dcachesize": "65536",
        "hw.l2cachesize": "4194304",
        # No hw.cpufrequency, hw.l3cachesize, machdep.cpu.features — Apple Silicon reality
    }
    c = hw._cpu_macos(s)
    assert c.brand == "Apple M4 Pro"
    assert c.vendor == "Apple"
    assert c.physical == 12 and c.logical == 12
    assert c.l1_kb == (131072 + 65536) // 1024   # 192
    assert c.l2_kb == 4194304 // 1024             # 4096
    assert c.l3_kb is None
    assert c.base_mhz is None and c.max_mhz is None
    assert c.features == []


def test_cpu_macos_x86_has_clock_and_features():
    """An Intel Mac sysctl dict — clock and features are present."""
    s = {
        "machdep.cpu.brand_string": "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz",
        "machdep.cpu.vendor": "GenuineIntel",
        "hw.physicalcpu": "8",
        "hw.logicalcpu": "16",
        "hw.cpufrequency": "2400000000",
        "hw.l1icachesize": "32768",
        "hw.l1dcachesize": "32768",
        "hw.l2cachesize": "262144",
        "hw.l3cachesize": "16777216",
        "machdep.cpu.features": "FPU VME DE PSE TSC",
    }
    c = hw._cpu_macos(s)
    assert c.brand == "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"
    assert c.vendor == "GenuineIntel"
    assert c.physical == 8 and c.logical == 16
    assert c.base_mhz == 2400 and c.max_mhz == 2400
    assert c.l2_kb == 256
    assert c.l3_kb == 16384
    assert "FPU" in c.features and "VME" in c.features


# --- Windows parser ---

def test_cpu_windows_from_wmi():
    proc = {
        "Name": "AMD Ryzen 9 5900X 12-Core Processor            ",
        "Manufacturer": "AuthenticAMD",
        "NumberOfCores": 12,
        "NumberOfLogicalProcessors": 24,
        "MaxClockSpeed": 3701,
        "L2CacheSize": 6144,
        "L3CacheSize": 65536,
    }
    c = hw._cpu_windows(proc, brand="AMD Ryzen 9 5900X")
    assert c.brand == "AMD Ryzen 9 5900X"
    assert c.vendor == "AuthenticAMD"
    assert c.physical == 12 and c.logical == 24
    assert c.max_mhz == 3701
    assert c.l2_kb == 6144 and c.l3_kb == 65536
    assert c.features == []


def test_cpu_windows_falls_back_to_wmi_name_if_no_registry_brand():
    """When brand arg is None, fall back to cleaning proc['Name']."""
    proc = {
        "Name": "AMD Ryzen 9 5900X 12-Core Processor            ",
        "Manufacturer": "AuthenticAMD",
        "NumberOfCores": 12,
        "NumberOfLogicalProcessors": 24,
        "MaxClockSpeed": 3701,
        "L2CacheSize": 6144,
        "L3CacheSize": 65536,
    }
    c = hw._cpu_windows(proc, brand=None)
    assert c.brand == "AMD Ryzen 9 5900X 12-Core Processor"


# --- Linux parser ---

_LINUX_CPUINFO = """\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
cpu MHz\t\t: 2200.000
cache size\t: 9216 KB
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu vme de pse tsc msr pae mce

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
cpu MHz\t\t: 2200.000
cache size\t: 9216 KB
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu vme de pse tsc msr pae mce
"""

_LINUX_CACHES = {
    "l1i": 32 * 1024,
    "l1d": 32 * 1024,
    "l2": 256 * 1024,
    "l3": 9 * 1024 * 1024,
}


def test_cpu_linux_parses_proc_cpuinfo():
    c = hw._cpu_linux(_LINUX_CPUINFO, _LINUX_CACHES, logical=2)
    assert c.brand == "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz"
    assert c.vendor == "GenuineIntel"
    assert "fpu" in c.features and "vme" in c.features
    assert c.l1_kb == 64       # (32 + 32) KiB
    assert c.l2_kb == 256
    assert c.l3_kb == 9216
    assert c.logical == 2


def test_cpu_linux_empty_cpuinfo():
    """Blank /proc/cpuinfo → all None / empty."""
    c = hw._cpu_linux("", {}, logical=None)
    assert c.brand is None and c.vendor is None and c.features == []


# --- _sysctl_many ---

def test_sysctl_many_parses_multi_key(monkeypatch):
    seen = {}
    output = "hw.physicalcpu: 12\nhw.logicalcpu: 12\n"

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["kw"] = k
        return output

    monkeypatch.setattr(hw, "_run", fake_run)
    result = hw._sysctl_many(["hw.physicalcpu", "hw.logicalcpu"])
    assert result == {"hw.physicalcpu": "12", "hw.logicalcpu": "12"}
    # Regression: must call `sysctl <keys>` WITHOUT `-n` (`-n` prints values only, no key labels)
    # AND with ignore_rc=True — sysctl exits non-zero when any requested key is unknown (e.g.
    # machdep.cpu.vendor on Apple Silicon) yet still prints the keys that exist. Both bugs made
    # real-macOS cpu_info() return all-None while the mocked tests passed.
    assert seen["cmd"] == ["sysctl", "hw.physicalcpu", "hw.logicalcpu"]
    assert "-n" not in seen["cmd"]
    assert seen["kw"].get("ignore_rc") is True


def test_run_ignore_rc_keeps_stdout_on_nonzero():
    # Real subprocess (cross-platform): prints to stdout, then exits 1.
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('hi'); sys.exit(1)"]
    assert hw._run(cmd) is None                       # default: gated on a 0 exit
    assert hw._run(cmd, ignore_rc=True) == "hi"       # ignore_rc keeps the partial stdout


def test_sysctl_many_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: None)
    assert hw._sysctl_many(["hw.physicalcpu"]) == {}


# --- _winreg_str (non-Windows: returns None) ---

def test_winreg_str_returns_none_on_non_windows(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    result = hw._winreg_str(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    assert result is None


# --- cpu_info() dispatcher ---

def test_cpu_info_dispatches_macos(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    sysctl_data = {
        "machdep.cpu.brand_string": "Apple M4 Pro",
        "hw.physicalcpu": "12",
        "hw.logicalcpu": "12",
        "hw.l1icachesize": "131072",
        "hw.l1dcachesize": "65536",
        "hw.l2cachesize": "4194304",
    }
    monkeypatch.setattr(hw, "_sysctl_many", lambda keys: sysctl_data)
    c = hw.cpu_info()
    assert c.brand == "Apple M4 Pro"
    assert c.vendor == "Apple"
    assert c.base_mhz is None and c.l3_kb is None


def test_cpu_info_dispatches_windows(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Windows")
    wmi_proc = [{
        "Name": "AMD Ryzen 9 5900X 12-Core Processor            ",
        "Manufacturer": "AuthenticAMD",
        "NumberOfCores": 12,
        "NumberOfLogicalProcessors": 24,
        "MaxClockSpeed": 3701,
        "L2CacheSize": 6144,
        "L3CacheSize": 65536,
    }]
    monkeypatch.setattr(hw, "_pwsh_json", lambda *a, **k: wmi_proc)
    monkeypatch.setattr(hw, "_winreg_str", lambda *a, **k: "AMD Ryzen 9 5900X")
    monkeypatch.setattr(_platform, "processor", lambda: "AMD64 Family 25")
    c = hw.cpu_info()
    assert c.brand == "AMD Ryzen 9 5900X"
    assert c.physical == 12 and c.logical == 24
    assert c.features == []


def test_cpu_info_dispatches_linux(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _LINUX_CPUINFO)
    monkeypatch.setattr(hw, "_linux_cpu_caches", lambda: _LINUX_CACHES)

    import psutil
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 2 if logical else 1)
    c = hw.cpu_info()
    assert c.brand == "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz"
    assert c.logical == 2


def test_cpu_info_unknown_platform_returns_empty(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "FreeBSD")
    c = hw.cpu_info()
    assert isinstance(c, hw.CpuInfo)
    assert c.brand is None


# --- _sysctl_many: line without ': ' separator (branch coverage) ---

def test_sysctl_many_skips_lines_without_colon_separator(monkeypatch):
    """Lines that don't have ': ' must be silently skipped."""
    output = "hw.physicalcpu: 12\nsome garbage line\nhw.logicalcpu: 12\n"
    monkeypatch.setattr(hw, "_run", lambda *a, **k: output)
    result = hw._sysctl_many(["hw.physicalcpu", "hw.logicalcpu"])
    assert result == {"hw.physicalcpu": "12", "hw.logicalcpu": "12"}




# --- _winreg_str on "Windows" with a mock winreg module ---

def test_winreg_str_reads_registry_value(monkeypatch):
    """On 'Windows', _winreg_str must open HKLM and return the stripped value."""
    import platform as _platform
    import sys
    import types

    monkeypatch.setattr(_platform, "system", lambda: "Windows")

    # Build a minimal fake winreg module
    fake_winreg = types.ModuleType("winreg")
    fake_winreg.HKEY_LOCAL_MACHINE = 0x80000002

    class _FakeKey:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    fake_winreg.OpenKey = lambda root, sub: _FakeKey()
    fake_winreg.QueryValueEx = lambda key, name: ("AMD Ryzen 9 5900X  ", None)

    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    result = hw._winreg_str(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    assert result == "AMD Ryzen 9 5900X"


def test_winreg_str_returns_none_on_exception(monkeypatch):
    """Exceptions from winreg (e.g., key not found) are silently caught → None."""
    import platform as _platform
    import sys
    import types

    monkeypatch.setattr(_platform, "system", lambda: "Windows")

    fake_winreg = types.ModuleType("winreg")
    fake_winreg.HKEY_LOCAL_MACHINE = 0x80000002
    fake_winreg.OpenKey = lambda root, sub: (_ for _ in ()).throw(OSError("not found"))

    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    result = hw._winreg_str(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    assert result is None


# --- _cpu_macos._int with bad value (ValueError branch) ---

def test_cpu_macos_handles_nonnumeric_sysctl_value():
    """A non-numeric sysctl value for a numeric field must produce None, not crash."""
    s = {
        "machdep.cpu.brand_string": "Apple M4 Pro",
        "hw.physicalcpu": "not-a-number",  # triggers ValueError path
        "hw.logicalcpu": "12",
        "hw.l1icachesize": "131072",
        "hw.l1dcachesize": "65536",
        "hw.l2cachesize": "4194304",
    }
    c = hw._cpu_macos(s)
    assert c.physical is None  # bad value → None
    assert c.logical == 12     # good value still parses


# --- _linux_cpu_caches ---

def test_linux_cpu_caches_returns_empty_when_no_sysfs(tmp_path, monkeypatch):
    """When /sys cache dir is absent, return {} gracefully."""
    import glob as _glob
    monkeypatch.setattr(_glob, "glob", lambda pattern: [])
    result = hw._linux_cpu_caches()
    assert result == {}


def test_linux_cpu_caches_parses_sysfs_dirs(tmp_path, monkeypatch):
    """Build a fake /sys/devices/system/cpu/cpu0/cache tree and verify parsing."""
    import glob as _glob

    cache_base = tmp_path / "cache"
    specs = [
        ("index0", "1", "Instruction", "32K"),
        ("index1", "1", "Data", "32K"),
        ("index2", "2", "Unified", "256K"),
        ("index3", "3", "Unified", "9M"),
    ]
    dirs = []
    for idx, level, kind, size in specs:
        d = cache_base / idx
        d.mkdir(parents=True)
        (d / "level").write_text(level)
        (d / "type").write_text(kind)
        (d / "size").write_text(size)
        dirs.append(str(d))

    monkeypatch.setattr(_glob, "glob", lambda pattern: dirs)
    result = hw._linux_cpu_caches()
    assert result["l1i"] == 32 * 1024
    assert result["l1d"] == 32 * 1024
    assert result["l2"] == 256 * 1024
    assert result["l3"] == 9 * 1024 * 1024


def test_linux_cpu_caches_skips_unreadable_index(tmp_path, monkeypatch):
    """A cache index directory missing a required file is silently skipped."""
    import glob as _glob

    # dir exists but has no files → open() raises FileNotFoundError
    bad_dir = tmp_path / "index0"
    bad_dir.mkdir()
    # intentionally no level/type/size files

    monkeypatch.setattr(_glob, "glob", lambda pattern: [str(bad_dir)])
    result = hw._linux_cpu_caches()
    assert result == {}


def test_linux_cpu_caches_skips_unknown_cache_level(tmp_path, monkeypatch):
    """A cache with an unrecognised level (e.g. L1 Unified) is skipped without error."""
    import glob as _glob

    d = tmp_path / "index0"
    d.mkdir()
    (d / "level").write_text("1")
    (d / "type").write_text("Unified")  # not 'Instruction' or 'Data' for L1 → no branch matches
    (d / "size").write_text("32K")

    monkeypatch.setattr(_glob, "glob", lambda pattern: [str(d)])
    result = hw._linux_cpu_caches()
    # No L1i or L1d should have been recorded
    assert "l1i" not in result and "l1d" not in result


def test_linux_cpu_caches_outer_exception_handled(monkeypatch):
    """If glob.glob itself raises, the outer except swallows it and returns {}."""
    import glob as _glob

    monkeypatch.setattr(_glob, "glob", lambda pattern: (_ for _ in ()).throw(OSError("no /sys")))
    result = hw._linux_cpu_caches()
    assert result == {}


# --- cpu_info() exception path ---

def test_cpu_info_returns_empty_on_internal_exception(monkeypatch):
    """If the dispatcher raises unexpectedly, return a blank CpuInfo rather than crashing."""
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    # Make _sysctl_many throw to trigger the outer except
    monkeypatch.setattr(hw, "_sysctl_many", lambda keys: (_ for _ in ()).throw(RuntimeError("boom")))
    c = hw.cpu_info()
    assert isinstance(c, hw.CpuInfo)
    assert c.brand is None
