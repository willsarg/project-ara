# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
import datetime as dt
import os
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
    # Exit 1 with no ignore_rc → None. Use the interpreter (cross-platform; `false` isn't on Windows).
    result = hw._run([sys.executable, "-c", "import sys; sys.exit(1)"], timeout=10)
    assert result is None


def test_run_returns_stdout_on_success():
    # Use the interpreter, not `echo` (a cmd builtin on Windows, not an executable).
    result = hw._run([sys.executable, "-c", "import sys; sys.stdout.write('hello')"], timeout=10)
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


# ---------------------------------------------------------------------------
# Task 3: Memory detail
# ---------------------------------------------------------------------------

# Fixtures from the plan (real winbox output)
_WIN_MEM_MODULES = [
    {"DeviceLocator": "DIMM_A1", "Capacity": 8589934592, "ConfiguredClockSpeed": 3400,
     "SMBIOSMemoryType": 26, "Manufacturer": "G-Skill", "PartNumber": "F4-3200C14-8GFX"},
    {"DeviceLocator": "DIMM_A2", "Capacity": 8589934592, "ConfiguredClockSpeed": 3400,
     "SMBIOSMemoryType": 26, "Manufacturer": "G-Skill", "PartNumber": "F4-3200C14-8GFX"},
    {"DeviceLocator": "DIMM_B1", "Capacity": 8589934592, "ConfiguredClockSpeed": 3400,
     "SMBIOSMemoryType": 26, "Manufacturer": "G-Skill", "PartNumber": "F4-3200C14-8GFX"},
    {"DeviceLocator": "DIMM_B2", "Capacity": 8589934592, "ConfiguredClockSpeed": 3400,
     "SMBIOSMemoryType": 26, "Manufacturer": "G-Skill", "PartNumber": "F4-3200C14-8GFX"},
]

_WIN_MEM_ARRAY = {"MemoryDevices": 4, "MaxCapacity": 134217728}

_WIN_TOTALS = (32.0, 28.0, 0.0)   # (total_gb, available_gb, swap_gb)

# macOS SPMemoryDataType text fixture (real Apple M4 Pro output shape)
_MACOS_SPMEMORY = """\
Hardware Overview:

      Memory: 24 GB
      Type: LPDDR5
      Manufacturer: Micron
"""

_MACOS_TOTALS = (24.0, 18.0, 0.0)

# Linux /proc/meminfo fixture (minimal)
_LINUX_MEMINFO = """\
MemTotal:       32686472 kB
MemFree:         1234567 kB
MemAvailable:   16000000 kB
SwapTotal:       2097152 kB
SwapFree:        2097152 kB
"""

# Linux dmidecode -t memory fixture — 1 populated DIMM + 1 empty slot (real dmidecode format:
# empty slots emit "Size: No Module Installed" and must NOT be counted or appended).
_LINUX_DMIDECODE = """\
# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.1.1 present.

Handle 0x1100, DMI type 17, 84 bytes
Memory Device
\tArray Handle: 0x1000
\tError Information Handle: 0x1101
\tTotal Width: 64 bits
\tData Width: 64 bits
\tSize: 8 GB
\tForm Factor: Row Of Chips
\tSet: None
\tLocator: Controller0-ChannelA-DIMM0
\tBank Locator: BANK 0
\tType: LPDDR4
\tType Detail: Unbuffered
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tSerial Number: --
\tAsset Tag: --
\tPart Number: K4EBE304EB-EGCG
\tRank: 2
\tConfigured Memory Speed: 3200 MT/s
\tMinimum Voltage: 1.0 V
\tMaximum Voltage: 1.2 V
\tConfigured Voltage: 1.1 V
\tMemory Technology: DRAM

Handle 0x1101, DMI type 17, 84 bytes
Memory Device
\tArray Handle: 0x1000
\tError Information Handle: 0x1102
\tTotal Width: Unknown
\tData Width: Unknown
\tSize: No Module Installed
\tForm Factor: Unknown
\tSet: None
\tLocator: Controller0-ChannelB-DIMM0
\tBank Locator: BANK 1
\tType: Unknown
\tType Detail: None
\tSpeed: Unknown
\tManufacturer: Unknown
\tSerial Number: --
\tAsset Tag: --
\tPart Number: Unknown
"""


# --- Windows parser ---

def test_mem_windows_four_modules_ddr4():
    """Real winbox fixture: 4 DDR4 DIMM_A1/A2/B1/B2, 8 GiB each, 3400 MT/s."""
    mi = hw._mem_windows(_WIN_MEM_MODULES, _WIN_MEM_ARRAY, _WIN_TOTALS)
    assert mi.total_gb == 32.0
    assert mi.available_gb == 28.0
    assert mi.swap_gb == 0.0
    assert mi.kind == "DDR4"
    assert mi.speed_mts == 3400
    assert mi.slots_used == 4
    assert mi.slots_total == 4
    assert len(mi.modules) == 4
    m0 = mi.modules[0]
    assert m0.slot == "DIMM_A1"
    assert m0.capacity_gb == 8.0
    assert m0.speed_mts == 3400
    assert m0.manufacturer == "G-Skill"
    assert m0.part_number == "F4-3200C14-8GFX"


def test_mem_windows_unknown_smbios_type_maps_to_none():
    """SMBIOSMemoryType 0 (unknown) must map to None kind, not crash."""
    modules = [{"DeviceLocator": "DIMM_A1", "Capacity": 8589934592,
                "ConfiguredClockSpeed": 3400, "SMBIOSMemoryType": 0,
                "Manufacturer": "Unknown", "PartNumber": ""}]
    array = {"MemoryDevices": 2}
    mi = hw._mem_windows(modules, array, (8.0, 6.0, 0.0))
    assert mi.kind is None
    assert mi.modules[0].part_number is None  # "" → _clean → None


def test_mem_windows_single_module_not_array():
    """A bare dict (single module) must be handled; _pwsh_json normalises to list — we pass list."""
    modules = [{"DeviceLocator": "DIMM_A1", "Capacity": 8589934592,
                "ConfiguredClockSpeed": 3200, "SMBIOSMemoryType": 34,
                "Manufacturer": "Samsung", "PartNumber": "M471A1G44AB0"}]
    array = {"MemoryDevices": 2}
    mi = hw._mem_windows(modules, array, (8.0, 6.0, 2.0))
    assert mi.kind == "DDR5"
    assert mi.slots_used == 1
    assert mi.slots_total == 2
    assert mi.swap_gb == 2.0


def test_mem_windows_empty_modules_list():
    """No modules (e.g. WMI failure) → MemoryInfo with totals but empty modules list."""
    mi = hw._mem_windows([], {}, (16.0, 8.0, 0.0))
    assert mi.modules == []
    assert mi.total_gb == 16.0
    assert mi.kind is None
    assert mi.slots_total is None


# --- macOS parser ---

def test_mem_macos_apple_silicon_no_modules():
    """Apple Silicon is soldered — kind and manufacturer parsed, but modules=[]."""
    mi = hw._mem_macos(_MACOS_SPMEMORY, _MACOS_TOTALS)
    assert mi.total_gb == 24.0
    assert mi.available_gb == 18.0
    assert mi.kind == "LPDDR5"
    assert mi.modules == []
    assert mi.slots_used is None
    assert mi.slots_total is None


def test_mem_macos_missing_type_line():
    """SPMemoryDataType with no 'Type:' line → kind=None (honest gap)."""
    text = "Hardware Overview:\n\n      Memory: 16 GB\n"
    mi = hw._mem_macos(text, (16.0, 10.0, 0.0))
    assert mi.kind is None
    assert mi.modules == []


# --- Linux parser ---

def test_mem_linux_non_root_no_modules():
    """Non-root Linux: dmidecode_text=None → modules=[], no crash."""
    mi = hw._mem_linux(_LINUX_MEMINFO, None, (31.2, 15.2, 2.0))
    assert mi.modules == []
    assert mi.total_gb == 31.2
    assert mi.slots_total is None


def test_mem_linux_root_parses_dmidecode():
    """Root Linux: 1 populated DIMM + 1 empty slot (No Module Installed).
    - Only the populated DIMM appears in modules (empty slot excluded).
    - capacity_gb is parsed from 'Size: 8 GB'.
    - slots_used == 1 (populated only), slots_total == 2 (all Memory Device blocks).
    """
    mi = hw._mem_linux(_LINUX_MEMINFO, _LINUX_DMIDECODE, (31.2, 15.2, 2.0))
    assert len(mi.modules) == 1                         # empty slot excluded
    m = mi.modules[0]
    assert m.slot == "Controller0-ChannelA-DIMM0"
    assert m.manufacturer == "Samsung"
    assert m.part_number == "K4EBE304EB-EGCG"
    assert m.speed_mts == 3200
    assert m.capacity_gb == 8.0                         # parsed from "Size: 8 GB"
    assert mi.kind == "LPDDR4"
    assert mi.slots_used == 1                           # only populated slots
    assert mi.slots_total == 2                          # all Memory Device blocks


def test_mem_linux_top_level_speed_from_modules():
    """Top-level MemoryInfo.speed_mts is aggregated (max) from the populated modules, the
    same as the Windows path — so `ara detect` shows a memory speed on Linux when dmidecode
    exposes it. (Linux previously left this None despite per-module speeds being known.)"""
    mi = hw._mem_linux(_LINUX_MEMINFO, _LINUX_DMIDECODE, (31.2, 15.2, 2.0))
    assert mi.speed_mts == 3200


def test_mem_linux_empty_meminfo():
    """Blank /proc/meminfo → totals from passed-in tuple, no crash."""
    mi = hw._mem_linux("", None, (0.0, 0.0, 0.0))
    assert mi.modules == []


def test_mem_linux_dmidecode_no_memory_device_blocks():
    """dmidecode text with no Memory Device blocks → slots_total=None (zero-block edge case)."""
    # Only header, no Memory Device entries at all
    text = "# dmidecode 3.3\nGetting SMBIOS data from sysfs.\n"
    mi = hw._mem_linux("", text, (0.0, 0.0, 0.0))
    assert mi.modules == []
    assert mi.slots_total is None


# Two-module dmidecode fixture — covers the "flush current when new block starts" path.
# Both blocks have "Size: 8 GB" so they are treated as populated and included.
_LINUX_DMIDECODE_TWO = """\
# dmidecode 3.3

Handle 0x1100, DMI type 17, 84 bytes
Memory Device
\tLocator: ChannelA-DIMM0
\tType: DDR4
\tSize: 8 GB
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tPart Number: M471A1G44AB0

Handle 0x1101, DMI type 17, 84 bytes
Memory Device
\tLocator: ChannelB-DIMM0
\tType: DDR4
\tSize: 8 GB
\tManufacturer: Samsung
\tPart Number: M471A1G44AB0
"""


def test_mem_linux_two_modules_dmidecode():
    """Two-module dmidecode: hits the 'flush current on new block' code path."""
    mi = hw._mem_linux(_LINUX_MEMINFO, _LINUX_DMIDECODE_TWO, (31.2, 15.2, 0.0))
    assert len(mi.modules) == 2
    assert mi.modules[0].slot == "ChannelA-DIMM0"
    assert mi.modules[1].slot == "ChannelB-DIMM0"
    assert mi.kind == "DDR4"


def test_parse_dmidecode_module_no_speed():
    """A module block with no Speed or Configured Memory Speed → speed_mts=None."""
    fields = {
        "Locator": "DIMM0",
        "Manufacturer": "Samsung",
        "Part Number": "M471A1G44AB0",
        # No 'Speed' or 'Configured Memory Speed'
    }
    m = hw._parse_dmidecode_module(fields)
    assert m.speed_mts is None
    assert m.slot == "DIMM0"


# Two-module fixture where both have Type set: second flush takes 'kind already set' branch.
_LINUX_DMIDECODE_TWO_TYPED = """\
Memory Device
\tLocator: ChannelA-DIMM0
\tType: DDR4
\tSize: 8 GB
\tSpeed: 3200 MT/s
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
\tLocator: ChannelB-DIMM0
\tType: DDR4
\tSize: 8 GB
\tSpeed: 3200 MT/s
\tManufacturer: Micron
\tPart Number: PART-B
"""


def test_mem_linux_two_typed_modules_kind_not_overwritten():
    """Second flush (kind already set) must not overwrite kind — branch 425->427 False path.
    Requires 3 modules so the second intermediate flush hits kind-already-set."""
    # Three modules: first intermediate flush sets kind, second intermediate flush skips it.
    three_modules = """\
Memory Device
\tLocator: DIMM0
\tType: DDR4
\tSize: 16 GB
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
\tLocator: DIMM1
\tType: DDR4
\tSize: 16 GB
\tManufacturer: Micron
\tPart Number: PART-B

Memory Device
\tLocator: DIMM2
\tType: DDR4
\tSize: 16 GB
\tManufacturer: Micron
\tPart Number: PART-C
"""
    mi = hw._mem_linux("", three_modules, (0.0, 0.0, 0.0))
    assert len(mi.modules) == 3
    assert mi.kind == "DDR4"


# A dmidecode text ending with a 'Memory Device' header followed by no fields (empty block at EOF).
# The first block has a "Size:" line so it is populated; the trailing block has no fields (no Size)
# so it is treated as empty and skipped.
_LINUX_DMIDECODE_TRAILING_HEADER = """\
Memory Device
\tLocator: ChannelA-DIMM0
\tType: DDR4
\tSize: 8 GB
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
"""


def test_mem_linux_dmidecode_trailing_empty_block():
    """A trailing 'Memory Device' header with no fields → empty trailing block is skipped.
    The first populated block (Size: 8 GB) is included; trailing empty block is not."""
    mi = hw._mem_linux("", _LINUX_DMIDECODE_TRAILING_HEADER, (0.0, 0.0, 0.0))
    # Only the first module (with fields and Size) is parsed; empty trailing block is skipped.
    assert len(mi.modules) == 1
    assert mi.modules[0].slot == "ChannelA-DIMM0"


# --- _SMBIOS_MEM map ---

def test_smbios_mem_has_required_codes():
    """The SMBIOS map must include DDR3/DDR4/DDR5/LPDDR4/LPDDR5 at minimum."""
    assert hw._SMBIOS_MEM[24] == "DDR3"
    assert hw._SMBIOS_MEM[26] == "DDR4"
    assert hw._SMBIOS_MEM[34] == "DDR5"
    assert hw._SMBIOS_MEM[30] == "LPDDR4"
    assert hw._SMBIOS_MEM[35] == "LPDDR5"


# --- memory_info() dispatcher ---

def test_memory_info_dispatches_macos(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _MACOS_SPMEMORY)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("vm", (), {"total": 24 * hw.GB, "available": 18 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    mi = hw.memory_info()
    assert mi.kind == "LPDDR5"
    assert mi.modules == []
    assert mi.total_gb == 24.0


def test_memory_info_dispatches_windows(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "Windows")

    def fake_pwsh_json(args):
        cmd = args[0] if args else ""
        if "Win32_PhysicalMemory" in cmd and "Array" not in cmd:
            return _WIN_MEM_MODULES
        if "Win32_PhysicalMemoryArray" in cmd:
            return [_WIN_MEM_ARRAY]
        return []

    monkeypatch.setattr(hw, "_pwsh_json", fake_pwsh_json)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("vm", (), {"total": 32 * hw.GB, "available": 28 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    mi = hw.memory_info()
    assert mi.kind == "DDR4"
    assert len(mi.modules) == 4
    assert mi.slots_total == 4


def test_memory_info_dispatches_linux_non_root(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _LINUX_MEMINFO)
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)  # non-root (os.geteuid is POSIX-only)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("vm", (), {"total": 31 * hw.GB, "available": 15 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 2 * hw.GB})())
    mi = hw.memory_info()
    assert mi.modules == []
    assert mi.slots_total is None


def test_memory_info_dispatches_linux_root(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "Linux")

    def fake_run(cmd, **k):
        if "dmidecode" in cmd:
            return _LINUX_DMIDECODE
        return _LINUX_MEMINFO

    monkeypatch.setattr(hw, "_run", fake_run)
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)  # root (os.geteuid is POSIX-only)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("vm", (), {"total": 31 * hw.GB, "available": 15 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 2 * hw.GB})())
    mi = hw.memory_info()
    assert len(mi.modules) == 1
    assert mi.kind == "LPDDR4"


def test_memory_info_unknown_platform_returns_empty(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("vm", (), {"total": 16 * hw.GB, "available": 8 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    mi = hw.memory_info()
    assert isinstance(mi, hw.MemoryInfo)
    assert mi.modules == []


def test_memory_info_exception_returns_empty(monkeypatch):
    import platform as _platform
    import psutil
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    mi = hw.memory_info()
    assert isinstance(mi, hw.MemoryInfo)
    assert mi.modules == []          # pin the "empty" claim, not just the type


# ---------------------------------------------------------------------------
# Task 4: Storage detail
# ---------------------------------------------------------------------------

# --- Windows fixture (Get-PhysicalDisk shape; generic drive models, no personal hardware) ---
# Drive FriendlyNames are placeholders — the parser only passes them through; classification keys
# entirely off MediaType/BusType. Those are friendly STRINGS ("SSD"/"NVMe"/...), NOT the uint16 enum
# codes the raw CIM carries — verified 2026-06-30 on a real PowerShell 5.1 box: `Get-PhysicalDisk |
# ConvertTo-Json` serializes the top-level fields the parser reads as strings (the ints surface only
# in the nested CimInstanceProperties dump we don't touch). So exact-string matching in
# _drives_windows is correct — unlike the Win32_PhysicalMemory path, whose SMBIOSMemoryType
# genuinely comes through as an int.
_WIN_PHYSICAL_DISKS = [
    {"FriendlyName": "Generic NVMe SSD 1TB", "MediaType": "SSD", "BusType": "NVMe",
     "Size": 1000204886016},
    {"FriendlyName": "Generic SATA HDD 2TB", "MediaType": "HDD", "BusType": "SATA",
     "Size": 2000398934016},
    {"FriendlyName": "Generic USB HDD 8TB", "MediaType": "Unspecified", "BusType": "USB",
     "Size": 8001456963584},
]

# Extended fixture to cover every branch in the media-type mapping
_WIN_PHYSICAL_DISKS_FULL = [
    {"FriendlyName": "NVMe Drive", "MediaType": "SSD", "BusType": "NVMe", "Size": 1000000000000},
    {"FriendlyName": "SATA SSD", "MediaType": "SSD", "BusType": "SATA", "Size": 500000000000},
    {"FriendlyName": "SATA HDD", "MediaType": "HDD", "BusType": "SATA", "Size": 2000000000000},
    {"FriendlyName": "USB Drive", "MediaType": "Unspecified", "BusType": "USB", "Size": 8000000000000},
    {"FriendlyName": "Unknown Device", "MediaType": "Unspecified", "BusType": "Unknown",
     "Size": 100000000000},
]

# macOS SPNVMeDataType fixture (real Apple M4 Pro / MacBook Pro output shape)
_MACOS_SPNVME = """\
NVMExpress:

    Apple SSD Controller:

      APPLE SSD AP0512Z:

          Capacity:          500.28 GB (500,277,792,768 bytes)
          TRIM Support:          Yes
          Model:                 APPLE SSD AP0512Z
          Revision:              0000000000000001
          Link Speed:            Unknown
          Link Width:            Unknown
          Detachable Drive:      No
          BSD Name:              disk0
          Partition Map Type:    GPT (GUID Partition Table)
          S.M.A.R.T. status:    Verified
"""

# Linux lsblk JSON fixture — mixed string and boolean rota values (real lsblk behaviour:
# util-linux < 2.33 emits strings "0"/"1"; >= 2.33 emits JSON booleans true/false).
_LINUX_LSBLK_JSON = """\
{
   "blockdevices": [
      {"name": "sda", "model": "Samsung SSD 870", "size": "500107862016", "rota": "0", "tran": "sata"},
      {"name": "sdb", "model": "WDC WD20EZRZ", "size": "2000398934016", "rota": true, "tran": "sata"},
      {"name": "sdc", "model": "SanDisk Cruzer", "size": "32016969728", "rota": false, "tran": "usb"},
      {"name": "nvme0n1", "model": "Samsung SSD 980 PRO", "size": "1000204886016", "rota": false, "tran": "nvme"}
   ]
}
"""

# String-form lsblk fixture ("1"/"0") — covers the legacy util-linux < 2.33 code path.
_LINUX_LSBLK_JSON_STR_ROTA = """\
{
   "blockdevices": [
      {"name": "sda", "model": "Old HDD", "size": "2000398934016", "rota": "1", "tran": "sata"},
      {"name": "sdb", "model": "Old SATA SSD", "size": "500107862016", "rota": "0", "tran": "sata"}
   ]
}
"""

# lsblk fixture where tran is null/missing (unknown fallback)
_LINUX_LSBLK_JSON_UNKNOWN = """\
{
   "blockdevices": [
      {"name": "sda", "model": "Mystery Drive", "size": "1000000000000", "rota": "0", "tran": null}
   ]
}
"""


# --- _drives_windows parser ---

def test_drives_windows_nvme_classification():
    """BusType=NVMe → 'nvme-ssd' regardless of MediaType."""
    drives = hw._drives_windows([{"FriendlyName": "Generic NVMe SSD 1TB",
                                   "MediaType": "SSD", "BusType": "NVMe",
                                   "Size": 1000204886016}])
    assert len(drives) == 1
    d = drives[0]
    assert d.model == "Generic NVMe SSD 1TB"
    assert d.media == "nvme-ssd"
    assert d.size_gb == round(1000204886016 / 1e9, 1)


def test_drives_windows_sata_ssd_classification():
    """MediaType=SSD + BusType=SATA → 'sata-ssd'."""
    drives = hw._drives_windows([{"FriendlyName": "SATA SSD",
                                   "MediaType": "SSD", "BusType": "SATA",
                                   "Size": 500000000000}])
    assert drives[0].media == "sata-ssd"


def test_drives_windows_hdd_classification():
    """MediaType=HDD → 'hdd'."""
    drives = hw._drives_windows([{"FriendlyName": "Generic SATA HDD 2TB",
                                   "MediaType": "HDD", "BusType": "SATA",
                                   "Size": 2000398934016}])
    assert drives[0].media == "hdd"


def test_drives_windows_usb_classification():
    """BusType=USB → 'usb'."""
    drives = hw._drives_windows([{"FriendlyName": "Generic USB HDD 8TB",
                                   "MediaType": "Unspecified", "BusType": "USB",
                                   "Size": 8001456963584}])
    assert drives[0].media == "usb"


def test_drives_windows_unknown_classification():
    """Unrecognised MediaType+BusType → 'unknown'."""
    drives = hw._drives_windows([{"FriendlyName": "Unknown Device",
                                   "MediaType": "Unspecified", "BusType": "Unknown",
                                   "Size": 100000000000}])
    assert drives[0].media == "unknown"


def test_drives_windows_full_winbox_fixture():
    """Full winbox Get-PhysicalDisk fixture: NVMe SSD + HDD + USB → 3 drives, correct media."""
    drives = hw._drives_windows(_WIN_PHYSICAL_DISKS)
    assert len(drives) == 3
    assert drives[0].media == "nvme-ssd"
    assert drives[1].media == "hdd"
    assert drives[2].media == "usb"
    assert drives[0].model == "Generic NVMe SSD 1TB"
    assert drives[1].model == "Generic SATA HDD 2TB"
    assert drives[2].model == "Generic USB HDD 8TB"


def test_drives_windows_empty_list():
    """Empty input → empty list (no crash)."""
    assert hw._drives_windows([]) == []


# --- _drives_macos parser ---

def test_drives_macos_apple_m4_pro_nvme():
    """Real macOS SPNVMeDataType output → 1 NVMe drive, correct model/size."""
    drives = hw._drives_macos(_MACOS_SPNVME)
    assert len(drives) == 1
    d = drives[0]
    assert d.model == "APPLE SSD AP0512Z"
    assert d.media == "nvme-ssd"
    assert d.size_gb == round(500277792768 / 1e9, 1)


def test_drives_macos_empty_text():
    """No NVMe output → empty list."""
    drives = hw._drives_macos("")
    assert drives == []


def test_drives_macos_missing_model_line():
    """Capacity without Model → drive skipped (model is None → not appended)."""
    text = "      Capacity:          500.28 GB (500000000000 bytes)\n"
    drives = hw._drives_macos(text)
    # No model found → nothing appended
    assert drives == []


def test_drives_macos_missing_bytes_in_capacity():
    """Capacity line with no '(N bytes)' → size_gb=None but drive still appended with model."""
    text = "      Model:                 APPLE SSD AP0512Z\n      Capacity:          500.28 GB\n"
    drives = hw._drives_macos(text)
    assert len(drives) == 1
    assert drives[0].size_gb is None


# --- _drives_linux parser ---

def test_drives_linux_mixed_types():
    """lsblk JSON fixture: sata-ssd (rota "0"), hdd (rota true boolean), usb (rota false),
    nvme-ssd (rota false) — covers both string and boolean rota shapes."""
    drives = hw._drives_linux(_LINUX_LSBLK_JSON)
    assert len(drives) == 4
    by_name = {d.model: d for d in drives}
    assert by_name["Samsung SSD 870"].media == "sata-ssd"   # rota "0" (string)
    assert by_name["WDC WD20EZRZ"].media == "hdd"           # rota true (boolean)
    assert by_name["SanDisk Cruzer"].media == "usb"         # rota false (boolean)
    assert by_name["Samsung SSD 980 PRO"].media == "nvme-ssd"  # rota false (boolean)


def test_drives_linux_string_rota_legacy():
    """String-form rota ("1"/"0") from util-linux < 2.33 must still classify correctly."""
    drives = hw._drives_linux(_LINUX_LSBLK_JSON_STR_ROTA)
    assert len(drives) == 2
    by_name = {d.model: d for d in drives}
    assert by_name["Old HDD"].media == "hdd"          # rota "1"
    assert by_name["Old SATA SSD"].media == "sata-ssd"  # rota "0"


def test_drives_linux_usb_rota_true_is_usb():
    """A USB-attached SSD whose bridge falsely reports rota=true must classify as 'usb',
    not 'hdd'. USB bridges routinely lie about rotation; transport is the reliable signal.
    (Real case: Crucial X6 external SSD on a ROG Ally, validated live 2026-06-21.)"""
    drives = hw._drives_linux(
        '{"blockdevices": [{"name": "sda", "model": "CT2000X6SSD9", '
        '"size": 2000398934016, "rota": true, "tran": "usb"}]}'
    )
    assert len(drives) == 1
    assert drives[0].media == "usb"


def test_drives_linux_unknown_tran():
    """ROTA=0 but tran=null → 'unknown'."""
    drives = hw._drives_linux(_LINUX_LSBLK_JSON_UNKNOWN)
    assert len(drives) == 1
    assert drives[0].media == "unknown"


def test_drives_linux_empty_json():
    """Empty blockdevices → empty list."""
    drives = hw._drives_linux('{"blockdevices": []}')
    assert drives == []


def test_drives_linux_bad_json():
    """Invalid JSON → empty list (no crash)."""
    drives = hw._drives_linux("not-json")
    assert drives == []


# --- storage_info() dispatcher ---

def test_storage_info_dispatches_macos(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _MACOS_SPNVME)
    monkeypatch.setattr(hw, "_disk_free_gb", lambda: 200.5)
    si = hw.storage_info()
    assert si.free_gb == 200.5
    assert len(si.drives) == 1
    assert si.drives[0].media == "nvme-ssd"


def test_storage_info_dispatches_windows(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_pwsh_json", lambda *a, **k: _WIN_PHYSICAL_DISKS)
    monkeypatch.setattr(hw, "_disk_free_gb", lambda: 400.0)
    si = hw.storage_info()
    assert si.free_gb == 400.0
    assert len(si.drives) == 3
    assert si.drives[0].media == "nvme-ssd"


def test_storage_info_dispatches_linux(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _LINUX_LSBLK_JSON)
    monkeypatch.setattr(hw, "_disk_free_gb", lambda: 150.0)
    si = hw.storage_info()
    assert si.free_gb == 150.0
    assert len(si.drives) == 4


def test_storage_info_unknown_platform_returns_empty(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(hw, "_disk_free_gb", lambda: None)
    si = hw.storage_info()
    assert isinstance(si, hw.StorageInfo)
    assert si.drives == []
    assert si.free_gb is None


def test_storage_info_exception_returns_empty(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(hw, "_disk_free_gb", lambda: None)
    si = hw.storage_info()
    assert isinstance(si, hw.StorageInfo)
    assert si.drives == [] and si.free_gb is None   # pin the "empty" claim, not just the type


# --- _drives_macos branch coverage ---

def test_drives_macos_empty_model_value_skipped():
    """'Model:   ' (blank after stripping) must be silently skipped (if val: False branch)."""
    text = "      Model:   \n      Capacity:   100 GB (100000000000 bytes)\n"
    drives = hw._drives_macos(text)
    # Empty model → if val: is False → nothing appended
    assert drives == []


def test_drives_macos_model_with_no_capacity():
    """A Model line with no Capacity line → size_gb=None (best_idx is None branch)."""
    text = "      Model:   APPLE SSD TEST\n"
    drives = hw._drives_macos(text)
    assert len(drives) == 1
    assert drives[0].size_gb is None
    assert drives[0].model == "APPLE SSD TEST"


def test_drives_macos_two_drives_nearest_capacity_pairing():
    """Two drives: each Model is paired with nearest Capacity.
    This covers the 'ci in used_cap_indices' skip + 'dist < best_dist' False path."""
    text = (
        "      Capacity:   100 GB (100000000000 bytes)\n"
        "      Model:   DRIVE A\n"
        "      Capacity:   200 GB (200000000000 bytes)\n"
        "      Model:   DRIVE B\n"
    )
    drives = hw._drives_macos(text)
    assert len(drives) == 2
    # DRIVE A is on line 1, capacity at line 0 → nearest cap is 100 GB
    # DRIVE B is on line 3, capacity at line 2 → nearest cap is 200 GB
    models = {d.model: d for d in drives}
    assert models["DRIVE A"].size_gb == round(100000000000 / 1e9, 1)
    assert models["DRIVE B"].size_gb == round(200000000000 / 1e9, 1)


# --- _disk_free_gb exception path ---

def test_disk_free_gb_returns_none_on_exception(monkeypatch):
    """_disk_free_gb must return None when disk_usage raises (exception branch)."""
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: (_ for _ in ()).throw(OSError("no disk")))
    result = hw._disk_free_gb()
    assert result is None


def test_disk_free_gb_uses_gib_not_decimal_gb(monkeypatch):
    """_disk_free_gb must divide by 1024**3 (GiB), not 1e9 (decimal GB).
    Pin: 107374182400 bytes (100 GiB exactly) must return 100.0, not 107.4."""
    import shutil
    _100_GiB = 100 * hw.GB   # 107374182400 bytes
    DiskUsage = type("DiskUsage", (), {"free": _100_GiB, "total": _100_GiB, "used": 0})
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DiskUsage())
    result = hw._disk_free_gb()
    assert result == 100.0   # GiB: 107374182400 / 1024**3 == 100.0


# ---------------------------------------------------------------------------
# Task 5: Board / firmware (board_info)
# ---------------------------------------------------------------------------

# --- Fixtures from the plan (REAL captured output) ---

# macOS SPHardwareDataType text fixture (real Apple M4 Pro / MacBook Pro)
_MACOS_SPHARDWARE = """\
Hardware Overview:

      Model Name:          MacBook Pro
      Model Identifier:    Mac16,8
      Model Number:        Z1CM000KVLL/A
      Chip:                Apple M4 Pro
      Total Number of Cores:  12 (8 performance and 4 efficiency)
      Memory:              24 GB
      System Firmware Version: 13822.81.10
      OS Loader Version:   13822.81.10
      Serial Number (system):  XXXXXXXXXX
      Hardware UUID:       XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Provisioning UDID:   XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Activation Lock Status: Disabled
"""

# Windows WMI fixture objects (real winbox)
_WIN_BASEBOARD = {"Manufacturer": "ASUSTeK COMPUTER INC.", "Product": "ROG STRIX X470-F GAMING"}
_WIN_BIOS = {"SMBIOSBIOSVersion": "6042", "ReleaseDate": "\\/Date(1651104000000)\\/"}
_WIN_SYSTEM = {"Manufacturer": "System manufacturer", "Model": "System Product Name"}

# Linux /sys/class/dmi/id file content map (real values on a typical bare-metal Linux)
_LINUX_DMI_FILES = {
    "board_vendor": "ASUSTeK COMPUTER INC.",
    "board_name": "ROG STRIX X470-F GAMING",
    "bios_version": "6042",
    "bios_date": "04/28/2022",
    "sys_vendor": "System manufacturer",   # placeholder → None
    "product_name": "System Product Name", # placeholder → None
}


# --- _board_macos parser ---

def test_board_macos_apple_mbp():
    """Real macOS SPHardwareDataType: system_vendor=Apple, system_model=MacBook Pro,
    bios_version from System Firmware Version, board_* = None."""
    b = hw._board_macos(_MACOS_SPHARDWARE)
    assert b.system_vendor == "Apple"
    assert b.system_model == "MacBook Pro"
    assert b.bios_version == "13822.81.10"
    assert b.bios_date is None          # macOS has no BIOS date
    assert b.board_vendor is None       # Macs have no separate motherboard
    assert b.board_model is None


def test_board_macos_missing_fields():
    """Empty SPHardwareDataType text → all None (no crash)."""
    b = hw._board_macos("")
    assert b.system_vendor is None
    assert b.system_model is None
    assert b.bios_version is None


def test_board_macos_no_model_name_leaves_vendor_none():
    """Any text that lacks 'Model Name:' → system_vendor is still None (not forced to Apple
    unless we actually found hardware info — honest gap)."""
    b = hw._board_macos("Some hardware text without expected keys\n")
    assert b.system_vendor is None


# --- _board_windows parser ---

def test_board_windows_asus_rog_winbox():
    """Real winbox fixture: board=ASUS ROG STRIX X470-F, bios=6042@2022-04-28,
    system=None/None (placeholders stripped by _clean)."""
    b = hw._board_windows(_WIN_BASEBOARD, _WIN_BIOS, _WIN_SYSTEM)
    assert b.board_vendor == "ASUSTeK COMPUTER INC."
    assert b.board_model == "ROG STRIX X470-F GAMING"
    assert b.bios_version == "6042"
    assert b.bios_date == "2022-04-28"
    assert b.system_vendor is None   # "System manufacturer" is a placeholder
    assert b.system_model is None    # "System Product Name" is a placeholder


def test_board_windows_real_system_fields():
    """Non-placeholder system fields are kept as-is."""
    system = {"Manufacturer": "Dell Inc.", "Model": "XPS 15 9520"}
    b = hw._board_windows(_WIN_BASEBOARD, _WIN_BIOS, system)
    assert b.system_vendor == "Dell Inc."
    assert b.system_model == "XPS 15 9520"


def test_board_windows_missing_bios_date():
    """BIOS with no ReleaseDate → bios_date=None (honest gap)."""
    bios = {"SMBIOSBIOSVersion": "6042"}
    b = hw._board_windows(_WIN_BASEBOARD, bios, _WIN_SYSTEM)
    assert b.bios_version == "6042"
    assert b.bios_date is None


def test_board_windows_empty_dicts():
    """All-empty WMI dicts → all None (no crash)."""
    b = hw._board_windows({}, {}, {})
    assert b.board_vendor is None
    assert b.board_model is None
    assert b.bios_version is None
    assert b.bios_date is None
    assert b.system_vendor is None
    assert b.system_model is None


# --- _board_linux parser ---

def test_board_linux_reads_dmi_files():
    """Real Linux DMI dict: board and bios populated; sys_vendor/product_name are
    placeholders so system_* = None."""
    b = hw._board_linux(_LINUX_DMI_FILES)
    assert b.board_vendor == "ASUSTeK COMPUTER INC."
    assert b.board_model == "ROG STRIX X470-F GAMING"
    assert b.bios_version == "6042"
    assert b.bios_date == "04/28/2022"
    assert b.system_vendor is None   # placeholder → _clean → None
    assert b.system_model is None    # placeholder → _clean → None


def test_board_linux_missing_files():
    """Missing DMI files (empty dict) → all None."""
    b = hw._board_linux({})
    assert b.board_vendor is None
    assert b.board_model is None
    assert b.bios_version is None


def test_board_linux_real_system_fields():
    """Non-placeholder sys_vendor/product_name → kept."""
    dmi = {
        "board_vendor": "ASUSTeK COMPUTER INC.",
        "board_name": "ROG STRIX X470-F GAMING",
        "bios_version": "6042",
        "bios_date": "04/28/2022",
        "sys_vendor": "ASUS",
        "product_name": "Custom Build",
    }
    b = hw._board_linux(dmi)
    assert b.system_vendor == "ASUS"
    assert b.system_model == "Custom Build"


# --- board_info() dispatcher ---

def test_board_info_dispatches_macos(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: _MACOS_SPHARDWARE)
    b = hw.board_info()
    assert b.system_vendor == "Apple"
    assert b.system_model == "MacBook Pro"
    assert b.bios_version == "13822.81.10"
    assert b.board_vendor is None


def test_board_info_dispatches_windows(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Windows")

    def fake_pwsh_json(args):
        cmd = args[0] if args else ""
        if "Win32_BaseBoard" in cmd:
            return [_WIN_BASEBOARD]
        if "Win32_BIOS" in cmd:
            return [_WIN_BIOS]
        if "Win32_ComputerSystem" in cmd:
            return [_WIN_SYSTEM]
        return []

    monkeypatch.setattr(hw, "_pwsh_json", fake_pwsh_json)
    b = hw.board_info()
    assert b.board_vendor == "ASUSTeK COMPUTER INC."
    assert b.board_model == "ROG STRIX X470-F GAMING"
    assert b.bios_version == "6042"
    assert b.bios_date == "2022-04-28"
    assert b.system_vendor is None
    assert b.system_model is None


def test_board_info_dispatches_linux(monkeypatch, tmp_path):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")

    # Create a fake /sys/class/dmi/id directory
    dmi_dir = tmp_path / "dmi_id"
    dmi_dir.mkdir()
    (dmi_dir / "board_vendor").write_text("ASUSTeK COMPUTER INC.\n")
    (dmi_dir / "board_name").write_text("ROG STRIX X470-F GAMING\n")
    (dmi_dir / "bios_version").write_text("6042\n")
    (dmi_dir / "bios_date").write_text("04/28/2022\n")
    (dmi_dir / "sys_vendor").write_text("System manufacturer\n")
    (dmi_dir / "product_name").write_text("System Product Name\n")

    monkeypatch.setattr(hw, "_DMI_ID_PATH", str(dmi_dir))
    b = hw.board_info()
    assert b.board_vendor == "ASUSTeK COMPUTER INC."
    assert b.board_model == "ROG STRIX X470-F GAMING"
    assert b.system_vendor is None
    assert b.system_model is None


def test_board_info_dispatches_linux_missing_files(monkeypatch, tmp_path):
    """Linux with some DMI files missing → those fields are None (no crash)."""
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")

    # Only board_vendor exists; rest missing
    dmi_dir = tmp_path / "dmi_id"
    dmi_dir.mkdir()
    (dmi_dir / "board_vendor").write_text("ASUS\n")

    monkeypatch.setattr(hw, "_DMI_ID_PATH", str(dmi_dir))
    b = hw.board_info()
    assert b.board_vendor == "ASUS"
    assert b.board_model is None
    assert b.bios_version is None


def test_board_info_unknown_platform_returns_empty(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "FreeBSD")
    b = hw.board_info()
    assert isinstance(b, hw.BoardInfo)
    assert b.board_vendor is None


def test_board_info_exception_returns_empty(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b = hw.board_info()
    assert isinstance(b, hw.BoardInfo)
    assert b.board_vendor is None   # pin the "empty" claim, not just the type


# ---------------------------------------------------------------------------
# probe() — bundles all four
# ---------------------------------------------------------------------------

def test_probe_returns_hardware_bundle(monkeypatch):
    """probe() must return a Hardware with all four substructures populated."""
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")

    # Patch the four dispatchers to return known values
    fake_cpu = hw.CpuInfo(brand="Apple M4 Pro", vendor="Apple")
    fake_mem = hw.MemoryInfo(total_gb=24.0, kind="LPDDR5")
    fake_storage = hw.StorageInfo(free_gb=200.0)
    fake_board = hw.BoardInfo(system_vendor="Apple", system_model="MacBook Pro",
                               bios_version="13822.81.10")

    monkeypatch.setattr(hw, "cpu_info", lambda: fake_cpu)
    monkeypatch.setattr(hw, "memory_info", lambda: fake_mem)
    monkeypatch.setattr(hw, "storage_info", lambda: fake_storage)
    monkeypatch.setattr(hw, "board_info", lambda: fake_board)

    result = hw.probe()
    assert isinstance(result, hw.Hardware)
    assert result.cpu is fake_cpu
    assert result.memory is fake_mem
    assert result.storage is fake_storage
    assert result.board is fake_board
    assert result.cpu.brand == "Apple M4 Pro"
    assert result.board.system_vendor == "Apple"


# ---------------------------------------------------------------------------
# Task 6: GpuInfo dataclass, vendor map, gpu_info() scaffold
# ---------------------------------------------------------------------------

def test_gpuinfo_defaults_and_vendor_map():
    from ara import hardware as hw
    g = hw.GpuInfo(vendor="amd")
    assert g.vendor == "amd"
    assert g.name is None and g.vram_gb is None and g.integrated is None
    assert g.driver_version is None and g.compute_runtime is None
    assert g.usable_backend is None
    # vendor map: PCI IDs and vendor strings normalise to canonical tokens
    assert hw._gpu_vendor("0x1002") == "amd"
    assert hw._gpu_vendor("0x10de") == "nvidia"
    assert hw._gpu_vendor("Advanced Micro Devices, Inc. [AMD/ATI]") == "amd"
    assert hw._gpu_vendor("NVIDIA Corporation") == "nvidia"
    assert hw._gpu_vendor("Intel Corporation") == "intel"
    assert hw._gpu_vendor("Apple") == "apple"
    assert hw._gpu_vendor("something else") == "unknown"


def test_hardware_has_gpus_and_probe_returns_list(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "gpu_info", lambda: [hw.GpuInfo(vendor="amd")])
    h = hw.probe()
    assert isinstance(h.gpus, list) and h.gpus[0].vendor == "amd"


def test_gpu_info_unknown_os_returns_empty(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw.platform, "system", lambda: "Plan9")
    assert hw.gpu_info() == []


def test_drm_gpu_amd_returns_none_integrated():
    """_drm_gpu always returns integrated=None for AMD; the APU heuristic lives in _gpus_linux."""
    from ara import hardware as hw
    g = hw._drm_gpu("0x1002", "0x15bf", 4294967296, "Phoenix1", cpu_vendor="AuthenticAMD")
    assert g.vendor == "amd"
    assert g.name == "Phoenix1"            # lspci name when available
    assert g.vram_gb == 4.3               # 4294967296 / 1e9 rounded (decimal GB)
    assert g.integrated is None           # resolved at list level in _gpus_linux, not here
    assert g.compute_runtime is None      # runtime filled in Task 3, not here


def test_drm_gpu_no_lspci_name_falls_back_generic():
    from ara import hardware as hw
    g = hw._drm_gpu("0x1002", "0x15bf", None, None, cpu_vendor="AuthenticAMD")
    assert g.name == "AMD Radeon Graphics"   # generic fallback
    assert g.vram_gb is None


def test_drm_gpu_nvidia_discrete_not_integrated():
    from ara import hardware as hw
    g = hw._drm_gpu("0x10de", "0x2484", 8589934592, "GA104 [GeForce RTX 3070]",
                    cpu_vendor="AuthenticAMD")
    assert g.vendor == "nvidia" and g.integrated is False


def test_gpus_linux_reads_sysfs(tmp_path, monkeypatch):
    from ara import hardware as hw
    # fake /sys/class/drm/card1/device
    dev = tmp_path / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    (dev / "mem_info_vram_total").write_text("4294967296\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {"0x15bf": "Phoenix1"})
    monkeypatch.setattr(hw, "cpu_info", lambda: hw.CpuInfo(vendor="AuthenticAMD"))
    # Stub the live Vulkan probe: the name priority is marketing-map → vulkaninfo → lspci, so on a
    # real Linux host with a Vulkan GPU the actual device name would win and mask the lspci fallback
    # this test exercises (caught on rog-ubuntu: real RADV name beat the mocked "Phoenix1").
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    gpus = hw._gpus_linux()
    assert len(gpus) == 1 and gpus[0].name == "Phoenix1" and gpus[0].vendor == "amd"


def test_gpus_linux_skips_cards_without_vendor(tmp_path, monkeypatch):
    from ara import hardware as hw
    (tmp_path / "card0" / "device").mkdir(parents=True)   # no vendor file
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {})
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])   # don't shell out to real vulkaninfo
    monkeypatch.setattr(hw, "cpu_info", lambda: hw.CpuInfo())
    assert hw._gpus_linux() == []


def test_gpus_linux_sole_amd_gpu_amd_cpu_is_integrated(tmp_path, monkeypatch):
    """Sole AMD GPU + AMD CPU → integrated=True (APU heuristic in _gpus_linux)."""
    from ara import hardware as hw
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    (dev / "mem_info_vram_total").write_text("4294967296\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {"0x15bf": "Phoenix1"})
    monkeypatch.setattr(hw, "cpu_info", lambda: hw.CpuInfo(vendor="AuthenticAMD"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 1
    assert gpus[0].vendor == "amd"
    assert gpus[0].integrated is True   # sole AMD GPU + AMD CPU → APU


def test_gpus_linux_two_gpus_amd_integrated_is_none(tmp_path, monkeypatch):
    """When there are two GPUs (e.g. AMD discrete + another), AMD integrated stays None.

    This prevents mislabelling a discrete AMD GPU as shared-VRAM in an AMD-CPU box.
    """
    from ara import hardware as hw
    for card, vendor, device in [("card0", "0x1002", "0x15bf"), ("card1", "0x10de", "0x2484")]:
        dev = tmp_path / card / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(f"{vendor}\n")
        (dev / "device").write_text(f"{device}\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {})
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])   # don't shell out to real vulkaninfo
    monkeypatch.setattr(hw, "cpu_info", lambda: hw.CpuInfo(vendor="AuthenticAMD"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 2
    amd_gpu = next(g for g in gpus if g.vendor == "amd")
    assert amd_gpu.integrated is None   # multiple GPUs → honest unknown, not mislabelled


def test_video_controller_gpu_nvidia():
    from ara import hardware as hw
    g = hw._video_controller_gpu({
        "Name": "NVIDIA GeForce RTX 2070", "AdapterCompatibility": "NVIDIA",
        "DriverVersion": "31.0.15.3623", "AdapterRAM": 4293918720})
    assert g.vendor == "nvidia" and g.name == "NVIDIA GeForce RTX 2070"
    assert g.driver_version == "31.0.15.3623"
    assert g.vram_gb is None          # 4293918720 ≈ uint32 cap ⇒ unknown, not 4.3


def test_video_controller_gpu_amd_small_adapterram_under_cap():
    # A sub-4GB AdapterRAM (here a 2GB integrated Radeon carveout) is BELOW the uint32 cap, so it's
    # trusted: vram_gb = bytes/1e9. (A real RX 6600 is 8GB and would report the uint32-pinned cap →
    # None, like the NVIDIA case above; don't pair a big-SKU name with a small reading.)
    from ara import hardware as hw
    g = hw._video_controller_gpu({
        "Name": "AMD Radeon(TM) Graphics", "AdapterCompatibility": "Advanced Micro Devices, Inc.",
        "DriverVersion": "31.0.21912.14", "AdapterRAM": 2147483648})
    assert g.vendor == "amd" and g.vram_gb == 2.1   # 2147483648/1e9


def test_video_controller_gpu_invalid_ram():
    from ara import hardware as hw
    g = hw._video_controller_gpu({
        "Name": "Intel HD Graphics", "AdapterCompatibility": "Intel",
        "DriverVersion": "27.20.100.8280", "AdapterRAM": None})
    assert g.vendor == "intel" and g.vram_gb is None


def test_gpus_windows_dispatch(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_pwsh_json", lambda args: [
        {"Name": "AMD Radeon RX 6600", "AdapterCompatibility": "AMD",
         "DriverVersion": "1.2", "AdapterRAM": 2147483648}])
    gpus = hw._gpus_windows()
    assert len(gpus) == 1 and gpus[0].vendor == "amd"


def test_gpus_macos_stub_returns_empty(monkeypatch):
    """_gpus_macos() on empty SPDisplaysDataType text returns []."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: "")
    assert hw._gpus_macos() == []


# ---------------------------------------------------------------------------
# Task 5: macOS GPU enumeration (_spdisplays_gpus / _gpus_macos)
# ---------------------------------------------------------------------------

_SPDISPLAYS_APPLE = """\
Graphics/Displays:
    Apple M4 Pro:
      Chipset Model: Apple M4 Pro
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 16
      Vendor: Apple (0x106b)
"""


def test_spdisplays_apple_unified():
    from ara import hardware as hw
    gpus = hw._spdisplays_gpus(_SPDISPLAYS_APPLE)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "apple" and g.name == "Apple M4 Pro"
    assert g.vram_gb is None and g.integrated is True


def test_spdisplays_empty_returns_empty():
    from ara import hardware as hw
    assert hw._spdisplays_gpus("") == []


_SPDISPLAYS_INTEL_DISCRETE = """\
Graphics/Displays:
    Intel Iris Pro:
      Chipset Model: Intel Iris Pro
      Type: GPU
      Bus: Built-In
      Vendor: Intel (0x8086)
      VRAM (Total): 1536 MB
"""

_SPDISPLAYS_INTEL_WITH_GB_VRAM = """\
Graphics/Displays:
    AMD Radeon Pro 5500M:
      Chipset Model: AMD Radeon Pro 5500M
      Type: GPU
      Bus: PCIe
      Vendor: AMD (0x1002)
      VRAM (Total): 8 GB
"""


def test_spdisplays_intel_no_gb_vram_line():
    """VRAM line present but no GB → vram_gb=None; Intel vendor; integrated=None (not apple)."""
    from ara import hardware as hw
    gpus = hw._spdisplays_gpus(_SPDISPLAYS_INTEL_DISCRETE)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "intel"
    assert g.vram_gb is None   # "1536 MB" — no GB match → None
    assert g.integrated is None  # not apple → None


def test_spdisplays_amd_vram_gb():
    """VRAM line with 'N GB' → vram_gb=float; AMD vendor; integrated=None."""
    from ara import hardware as hw
    gpus = hw._spdisplays_gpus(_SPDISPLAYS_INTEL_WITH_GB_VRAM)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "amd"
    assert g.vram_gb == 8.0
    assert g.integrated is None  # AMD on non-AMD CPU → None


def test_gpus_macos_calls_run_with_spdisplays(monkeypatch):
    """_gpus_macos() calls _run with SPDisplaysDataType and parses the result."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _SPDISPLAYS_APPLE)
    gpus = hw._gpus_macos()
    assert len(gpus) == 1
    assert gpus[0].vendor == "apple" and gpus[0].name == "Apple M4 Pro"


def test_gpus_macos_returns_empty_when_run_fails(monkeypatch):
    """When _run returns None (e.g. system_profiler absent), _gpus_macos returns []."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: None)
    assert hw._gpus_macos() == []


def test_gpu_info_dispatches_macos(monkeypatch):
    """gpu_info() on Darwin hits the macOS branch."""
    from ara import hardware as hw
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_gpus_macos", lambda: [hw.GpuInfo(vendor="apple", name="Apple M4 Pro",
                                                                 integrated=True)])
    result = hw.gpu_info()
    assert len(result) == 1 and result[0].vendor == "apple"


def test_with_runtime_unknown_vendor_passthrough(monkeypatch):
    """Unknown vendor returns None runtime and None backend (else branch)."""
    from ara import hardware as hw
    g = hw.GpuInfo(vendor="unknown")
    result = hw._with_runtime(g)
    assert result.compute_runtime is None and result.usable_backend is None


def test_gpu_info_dispatches_linux(monkeypatch):
    """gpu_info() on Linux hits the Linux branch (line 880)."""
    from ara import hardware as hw
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_gpus_linux", lambda: [hw.GpuInfo(vendor="amd")])
    result = hw.gpu_info()
    assert len(result) == 1 and result[0].vendor == "amd"


def test_gpu_info_dispatches_windows(monkeypatch):
    """gpu_info() on Windows hits the Windows branch (line 882)."""
    from ara import hardware as hw
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_gpus_windows", lambda: [hw.GpuInfo(vendor="nvidia")])
    result = hw.gpu_info()
    assert len(result) == 1 and result[0].vendor == "nvidia"


def test_gpu_info_exception_returns_empty(monkeypatch):
    """If _gpus_linux raises, gpu_info() returns [] (exception branch lines 887-888)."""
    from ara import hardware as hw
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_gpus_linux", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert hw.gpu_info() == []


# ---------------------------------------------------------------------------
# Task 7: _lspci_names, _drm_gpu intel/unknown branches, _read_text exception
# ---------------------------------------------------------------------------

def test_drm_gpu_intel_always_integrated():
    from ara import hardware as hw
    g = hw._drm_gpu("0x8086", "0x9bc4", 0, "Intel UHD Graphics", cpu_vendor="GenuineIntel")
    assert g.vendor == "intel" and g.integrated is True


def test_drm_gpu_unknown_vendor_integrated_none():
    from ara import hardware as hw
    g = hw._drm_gpu("0xffff", "0x0001", None, None, cpu_vendor=None)
    assert g.vendor == "unknown" and g.integrated is None and g.vram_gb is None


def test_lspci_names_returns_empty_when_lspci_absent(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda *a, **k: None)
    assert hw._lspci_names() == {}


def test_lspci_names_parses_split_ids_real_rog_ally(monkeypatch):
    """Real ROG Ally lspci -mm -nn line: split IDs, non-empty subsystem name.

    The subsystem name ('Phoenix1 [17f3]') must NOT be selected; only the device field
    immediately after the vendor field ('Phoenix1 [15bf]') is the correct source.
    """
    from ara import hardware as hw
    lspci_out = (
        '09:00.0 "VGA compatible controller [0300]" '
        '"Advanced Micro Devices, Inc. [AMD/ATI] [1002]" '
        '"Phoenix1 [15bf]" '
        '-r04 -p00 '
        '"ASUSTeK Computer Inc. [1043]" '
        '"Phoenix1 [17f3]"\n'
    )
    monkeypatch.setattr(hw, "_run", lambda *a, **k: lspci_out)
    names = hw._lspci_names()
    assert names.get("0x15bf") == "Phoenix1"
    # Confirm the subsystem name ('Phoenix1 [17f3]') was NOT selected
    assert names.get("0x17f3") is None


def test_lspci_names_parses_combined_ids_nvidia(monkeypatch):
    """Real discrete NVIDIA lspci -mm -nn line: combined IDs, inner bracket preserved in name.

    The inner '[GeForce RTX 3070]' bracket must NOT be stripped — only the final device-id
    bracket '[2484]' is removed. Expected name: 'GA104 [GeForce RTX 3070]'.
    """
    from ara import hardware as hw
    lspci_out = (
        '01:00.0 "VGA compatible controller [0300]" '
        '"NVIDIA Corporation [10de]" '
        '"GA104 [GeForce RTX 3070] [2484]" '
        '-ra1 '
        '"ASUSTeK Computer Inc. [1043]" '
        '"Device [8763]"\n'
    )
    monkeypatch.setattr(hw, "_run", lambda *a, **k: lspci_out)
    names = hw._lspci_names()
    assert names.get("0x2484") == "GA104 [GeForce RTX 3070]"


def test_lspci_names_skips_lines_without_match(monkeypatch):
    from ara import hardware as hw
    # A line that has no PCI vendor bracket we recognise → skipped cleanly
    lspci_out = '"0300" "Some other device" "Vendor [dead:beef]" "TheName" "" ""\n'
    monkeypatch.setattr(hw, "_run", lambda *a, **k: lspci_out)
    names = hw._lspci_names()
    assert names == {}


def test_lspci_names_skips_device_field_without_id_bracket(monkeypatch):
    """Device field immediately after vendor field has no [xxxx] bracket → line skipped."""
    from ara import hardware as hw
    # Vendor field has [1002] but device field has no id bracket → no match
    lspci_out = '"0300" "VGA [0300]" "AMD Corp [1002]" "NoIdBracketHere" "" ""\n'
    monkeypatch.setattr(hw, "_run", lambda *a, **k: lspci_out)
    names = hw._lspci_names()
    assert names == {}


def test_lspci_names_skips_empty_name_after_bracket_strip(monkeypatch):
    """Device field is just '[xxxx]' with no name text before it → name is empty, line skipped."""
    from ara import hardware as hw
    # Device field only has a bracket id, no name text before it
    lspci_out = '"0300" "VGA [0300]" "AMD Corp [1002]" "[15bf]" "" ""\n'
    monkeypatch.setattr(hw, "_run", lambda *a, **k: lspci_out)
    names = hw._lspci_names()
    assert names == {}


def test_read_text_returns_none_on_missing_file():
    from ara import hardware as hw
    result = hw._read_text("/nonexistent/path/that/cannot/exist")
    assert result is None


# ---------------------------------------------------------------------------
# Task 3: Compute-runtime detection (_vulkan_devices, _rocm_version, _with_runtime)
# ---------------------------------------------------------------------------

# Real vulkaninfo --summary fixture from a ROG Ally (verbatim, including space-padded fields).
_VULKANINFO_SUMMARY = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.318
\tdriverVersion      = 25.2.8
\tvendorID           = 0x1002
\tdeviceID           = 0x15bf
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Ryzen Z1 Extreme (RADV PHOENIX)
\tdriverID           = DRIVER_ID_MESA_RADV
\tdriverName         = radv
\tdriverInfo         = Mesa 25.2.8-0ubuntu0.24.04.2
\tconformanceVersion = 1.4.0.0
GPU1:
\tapiVersion         = 1.4.318
\tdriverVersion      = 25.2.8
\tvendorID           = 0x10005
\tdeviceID           = 0x0000
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 20.1.2, 256 bits)
\tdriverID           = DRIVER_ID_MESA_LLVMPIPE
\tdriverName         = llvmpipe
\tdriverInfo         = Mesa 25.2.8-0ubuntu0.24.04.2 (LLVM 20.1.2)
\tconformanceVersion = 1.3.1.1
"""

# Single-GPU fixture (no llvmpipe block) — the flush-on-apiVersion bug made this return [].
_VULKANINFO_SUMMARY_SINGLE = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.318
\tdriverVersion      = 25.2.8
\tvendorID           = 0x1002
\tdeviceID           = 0x15bf
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Ryzen Z1 Extreme (RADV PHOENIX)
\tdriverID           = DRIVER_ID_MESA_RADV
\tdriverName         = radv
\tdriverInfo         = Mesa 25.2.8-0ubuntu0.24.04.2
\tconformanceVersion = 1.4.0.0
"""


def test_vulkan_devices_parsed_filters_llvmpipe(monkeypatch):
    """Real multi-GPU fixture: llvmpipe filtered by deviceType _CPU; one AMD device returned."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY
                        if "--summary" in cmd else "cooperativeMatrix = true\n")
    devs = hw._vulkan_devices()
    assert len(devs) == 1                       # llvmpipe filtered out
    assert devs[0]["vendor"] == "amd"
    assert devs[0]["api"] == "1.4.318" and devs[0]["driver"] == "radv"
    assert devs[0]["coopmat"] is True


def test_vulkan_devices_coopmat_space_padded(monkeypatch):
    """Real vulkaninfo pads the value with spaces: 'cooperativeMatrix                   = true'.
    The old substring 'cooperativeMatrix = true' never matched → always False on real hardware."""
    from ara import hardware as hw
    padded_coop = "\tcooperativeMatrix                   = true\n"
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY
                        if "--summary" in cmd else padded_coop)
    devs = hw._vulkan_devices()
    assert len(devs) == 1
    assert devs[0]["coopmat"] is True


def test_vulkan_devices_coopmat_absent(monkeypatch):
    """No cooperativeMatrix line → coopmat is False."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY
                        if "--summary" in cmd else "some other output\n")
    devs = hw._vulkan_devices()
    assert len(devs) == 1
    assert devs[0]["coopmat"] is False


def test_vulkan_devices_single_gpu_no_llvmpipe(monkeypatch):
    """Single GPU with no trailing llvmpipe block — the flush-on-apiVersion bug dropped this.
    Must return exactly ONE device (the regression case)."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY_SINGLE
                        if "--summary" in cmd else "")
    devs = hw._vulkan_devices()
    assert len(devs) == 1
    assert devs[0]["vendor"] == "amd"
    assert devs[0]["api"] == "1.4.318"
    assert devs[0]["driver"] == "radv"


def test_with_runtime_amd_vulkan_usable(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_vulkan_devices",
        lambda: [{"vendor": "amd", "api": "1.4.318", "driver": "radv", "coopmat": True}])
    monkeypatch.setattr(hw, "_rocm_version", lambda: None)
    g = hw._with_runtime(hw.GpuInfo(vendor="amd", name="Phoenix1"))
    assert g.usable_backend == "vulkan"
    assert g.compute_runtime == "Vulkan 1.4.318 · radv · coopmat"


def test_with_runtime_amd_no_runtime_none(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    monkeypatch.setattr(hw, "_rocm_version", lambda: None)
    g = hw._with_runtime(hw.GpuInfo(vendor="amd"))
    assert g.usable_backend is None and g.compute_runtime is None


def test_with_runtime_amd_rocm_noted_not_usable(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    monkeypatch.setattr(hw, "_rocm_version", lambda: "6.0.2")
    g = hw._with_runtime(hw.GpuInfo(vendor="amd"))
    assert g.usable_backend is None and g.compute_runtime == "ROCm 6.0.2"


def test_with_runtime_apple_metal(monkeypatch):
    from ara import hardware as hw
    g = hw._with_runtime(hw.GpuInfo(vendor="apple", name="Apple M4 Pro GPU"))
    assert g.usable_backend == "mlx" and g.compute_runtime == "Metal"


def test_with_runtime_nvidia_cuda(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_cuda_version_smi", lambda: "12.4")
    g = hw._with_runtime(hw.GpuInfo(vendor="nvidia", name="RTX 4090"))
    assert g.usable_backend == "cuda" and g.compute_runtime == "CUDA 12.4"


def test_with_runtime_nvidia_no_cuda(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_cuda_version_smi", lambda: None)
    g = hw._with_runtime(hw.GpuInfo(vendor="nvidia", name="RTX 4090"))
    assert g.usable_backend is None and g.compute_runtime is None


def test_with_runtime_intel_vulkan_usable(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_vulkan_devices",
        lambda: [{"vendor": "intel", "api": "1.3.290", "driver": "anv", "coopmat": False}])
    monkeypatch.setattr(hw, "_rocm_version", lambda: None)
    g = hw._with_runtime(hw.GpuInfo(vendor="intel", name="Intel Arc A770"))
    assert g.usable_backend == "vulkan"
    assert g.compute_runtime == "Vulkan 1.3.290 · anv"


def test_with_runtime_unknown_vendor_none(monkeypatch):
    from ara import hardware as hw
    g = hw._with_runtime(hw.GpuInfo(vendor="unknown", name="Some GPU"))
    assert g.usable_backend is None and g.compute_runtime is None


def test_vulkan_devices_empty_when_vulkaninfo_absent(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: None)
    assert hw._vulkan_devices() == []


def test_rocm_version_returns_none_when_no_rocm(monkeypatch):
    from ara import hardware as hw
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(hw.os.path, "isdir", lambda _: False)
    assert hw._rocm_version() is None


def test_rocm_version_parses_version_file(monkeypatch):
    from ara import hardware as hw
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None)
    monkeypatch.setattr(hw, "_read_text", lambda path: "6.0.2-1234" if "version" in path else None)
    result = hw._rocm_version()
    assert result == "6.0.2"


def test_rocm_version_returns_unknown_when_no_version_file(monkeypatch):
    from ara import hardware as hw
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None)
    monkeypatch.setattr(hw, "_read_text", lambda path: None)
    result = hw._rocm_version()
    assert result == "unknown"


def test_cuda_version_smi_parses_version(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: "CUDA Version: 12.4\n")
    assert hw._cuda_version_smi() == "12.4"


def test_cuda_version_smi_returns_none_when_absent(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: None)
    assert hw._cuda_version_smi() is None


def test_vulkan_devices_no_coopmat(monkeypatch):
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY
                        if "--summary" in cmd else "")
    devs = hw._vulkan_devices()
    assert len(devs) == 1
    assert devs[0]["coopmat"] is False


def test_vulkan_devices_block_without_device_name_skipped(monkeypatch):
    """A GPU block that has apiVersion + driverName but no deviceName is silently skipped
    (covers the 'if not name: return' branch in _flush)."""
    from ara import hardware as hw
    # A summary with one block that has no deviceName line at all.
    summary_no_name = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.318
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdriverName         = radv
"""
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: summary_no_name
                        if "--summary" in cmd else "")
    devs = hw._vulkan_devices()
    assert devs == []


# ---------------------------------------------------------------------------
# GPU marketing-name policy
# ---------------------------------------------------------------------------

def test_marketing_gpu_name_z1_extreme_returns_780m():
    """Phoenix PCI id (1002:15bf) + Z1 Extreme CPU → AMD Radeon 780M (verified)."""
    from ara import hardware as hw
    assert hw._marketing_gpu_name("0x1002", "0x15bf", "AMD Ryzen Z1 Extreme") == "AMD Radeon 780M"


def test_marketing_gpu_name_z1_non_extreme_returns_none():
    """Phoenix PCI id but plain Z1 (not Extreme) → None (honest: 740M/760M not confirmed)."""
    from ara import hardware as hw
    assert hw._marketing_gpu_name("0x1002", "0x15bf", "AMD Ryzen Z1") is None


def test_marketing_gpu_name_nvidia_returns_none():
    """NVIDIA device → None (map is AMD/Linux-only for now)."""
    from ara import hardware as hw
    assert hw._marketing_gpu_name("0x10de", "0x2484", "Intel Core i9") is None


def test_marketing_gpu_name_no_cpu_brand_returns_none():
    """Phoenix PCI id but no CPU brand → None (can't disambiguate SKU)."""
    from ara import hardware as hw
    assert hw._marketing_gpu_name("0x1002", "0x15bf", None) is None


def test_marketing_gpu_name_normalises_0x_prefix():
    """Vendor/device ids without 0x prefix also work (normalisation)."""
    from ara import hardware as hw
    assert hw._marketing_gpu_name("1002", "15bf", "AMD Ryzen Z1 Extreme") == "AMD Radeon 780M"


def test_vulkan_devices_returns_name_key(monkeypatch):
    """_vulkan_devices() must include a 'name' key with the deviceName string."""
    from ara import hardware as hw
    monkeypatch.setattr(hw, "_run", lambda cmd, **k: _VULKANINFO_SUMMARY
                        if "--summary" in cmd else "")
    devs = hw._vulkan_devices()
    assert len(devs) == 1
    assert devs[0]["name"] == "AMD Ryzen Z1 Extreme (RADV PHOENIX)"


def test_vulkan_name_returns_first_match():
    """_vulkan_name returns the name of the first vulkan device whose vendor matches."""
    from ara import hardware as hw
    vk = [
        {"vendor": "amd", "name": "AMD Ryzen Z1 Extreme (RADV PHOENIX)", "api": "1.4", "driver": "radv", "coopmat": False},
        {"vendor": "intel", "name": "Intel Arc (ANV)", "api": "1.3", "driver": "anv", "coopmat": False},
    ]
    assert hw._vulkan_name(vk, "amd") == "AMD Ryzen Z1 Extreme (RADV PHOENIX)"
    assert hw._vulkan_name(vk, "intel") == "Intel Arc (ANV)"
    assert hw._vulkan_name(vk, "nvidia") is None


def test_vulkan_name_empty_list_returns_none():
    """Empty vulkan device list → None."""
    from ara import hardware as hw
    assert hw._vulkan_name([], "amd") is None


def test_gpus_linux_marketing_name_z1_extreme(tmp_path, monkeypatch):
    """ROG Ally sysfs fixture + cpu_info.brand='AMD Ryzen Z1 Extreme' → name='AMD Radeon 780M'."""
    from ara import hardware as hw
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    (dev / "mem_info_vram_total").write_text("4294967296\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {"0x15bf": "Phoenix1"})
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    monkeypatch.setattr(hw, "cpu_info",
        lambda: hw.CpuInfo(vendor="AuthenticAMD", brand="AMD Ryzen Z1 Extreme"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon 780M"


def test_gpus_linux_vulkan_name_tier(tmp_path, monkeypatch):
    """AMD GPU not in marketing map (cpu brand without z1 extreme) but vulkan device present
    → vulkan device name used."""
    from ara import hardware as hw
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    (dev / "mem_info_vram_total").write_text("4294967296\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {})
    monkeypatch.setattr(hw, "_vulkan_devices",
        lambda: [{"vendor": "amd", "name": "AMD Ryzen X (RADV GFX1234)",
                  "api": "1.4", "driver": "radv", "coopmat": False}])
    monkeypatch.setattr(hw, "cpu_info",
        lambda: hw.CpuInfo(vendor="AuthenticAMD", brand="AMD Ryzen 7 7840U"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Ryzen X (RADV GFX1234)"


def test_gpus_linux_lspci_name_tier(tmp_path, monkeypatch):
    """No marketing match, no vulkan, but lspci name present → lspci name used."""
    from ara import hardware as hw
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {"0x15bf": "Phoenix1"})
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    monkeypatch.setattr(hw, "cpu_info",
        lambda: hw.CpuInfo(vendor="AuthenticAMD", brand="AMD Ryzen 7 7840U"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 1
    assert gpus[0].name == "Phoenix1"


def test_gpus_linux_generic_name_tier(tmp_path, monkeypatch):
    """No marketing, no vulkan, no lspci → generic fallback name."""
    from ara import hardware as hw
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "device").write_text("0x15bf\n")
    monkeypatch.setattr(hw, "_DRM_GLOB", str(tmp_path / "card*"))
    monkeypatch.setattr(hw, "_lspci_names", lambda: {})
    monkeypatch.setattr(hw, "_vulkan_devices", lambda: [])
    monkeypatch.setattr(hw, "cpu_info",
        lambda: hw.CpuInfo(vendor="AuthenticAMD", brand="AMD Ryzen 7 7840U"))
    gpus = hw._gpus_linux()
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon Graphics"


# --------------------------------------------------------------------------- #
# cgroup-honest memory wall (Rule #1): containerized RAM ceiling
# --------------------------------------------------------------------------- #
import pytest


@pytest.fixture
def cgroup_files(monkeypatch):
    """Drive the single cgroup filesystem boundary from an in-memory {path: text} map."""
    files: dict[str, str] = {}
    monkeypatch.setattr(hw, "_read_cgroup_file", lambda path: files.get(path))
    return files


def test_read_cgroup_file_reads_existing_and_none_for_missing(tmp_path):
    p = tmp_path / "memory.max"
    p.write_text("123\n")
    assert hw._read_cgroup_file(str(p)) == "123\n"
    assert hw._read_cgroup_file(str(tmp_path / "nope")) is None       # OSError → None


def test_cgroup_v2_real_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V2] = "2147483648\n"
    assert hw.cgroup_memory_limit_bytes() == 2147483648


def test_cgroup_v2_max_is_no_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V2] = "max\n"
    assert hw.cgroup_memory_limit_bytes() is None


def test_cgroup_v2_garbage_is_no_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V2] = "not-a-number"
    assert hw.cgroup_memory_limit_bytes() is None


def test_cgroup_v1_real_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V1] = "1073741824"
    assert hw.cgroup_memory_limit_bytes() == 1073741824


def test_cgroup_v1_sentinel_is_no_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V1] = str(hw._V1_UNLIMITED)
    assert hw.cgroup_memory_limit_bytes() is None


def test_cgroup_v1_garbage_is_no_limit(cgroup_files):
    cgroup_files[hw._CGROUP_V1] = "nope"
    assert hw.cgroup_memory_limit_bytes() is None


def test_cgroup_missing_files_is_no_limit(cgroup_files):
    assert hw.cgroup_memory_limit_bytes() is None                     # empty map → None reads


def test_clamp_ram_to_cgroup_binds_below_physical(cgroup_files, monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    cgroup_files[hw._CGROUP_V2] = str(4 * hw.GB)
    assert hw.clamp_ram_to_cgroup(16 * hw.GB) == 4 * hw.GB


def test_clamp_ram_to_cgroup_no_limit_stays_physical(cgroup_files, monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    cgroup_files[hw._CGROUP_V2] = "max"
    assert hw.clamp_ram_to_cgroup(16 * hw.GB) == 16 * hw.GB


def test_clamp_ram_to_cgroup_limit_above_physical_stays_physical(cgroup_files, monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    cgroup_files[hw._CGROUP_V2] = str(64 * hw.GB)
    assert hw.clamp_ram_to_cgroup(16 * hw.GB) == 16 * hw.GB


def test_clamp_ram_to_cgroup_ignores_cgroup_off_linux(cgroup_files, monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    cgroup_files[hw._CGROUP_V2] = str(4 * hw.GB)                      # would bind IF read
    assert hw.clamp_ram_to_cgroup(16 * hw.GB) == 16 * hw.GB


def test_psutil_totals_clamps_total_to_cgroup(cgroup_files, monkeypatch):
    """The Rule #1 clamp lives at the source: _psutil_totals reports the cgroup wall, not host RAM."""
    import psutil
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: type("vm", (), {"total": 32 * hw.GB, "available": 20 * hw.GB})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    cgroup_files[hw._CGROUP_V2] = str(8 * hw.GB)                      # container capped at 8 GiB
    total_gb, available_gb, _swap = hw._psutil_totals()
    assert total_gb == 8.0                                            # clamped, not 32
    assert available_gb == 20.0                                       # available is left untouched
