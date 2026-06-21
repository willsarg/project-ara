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

# Fixtures from the plan (real willw11 output)
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

# Linux dmidecode -t memory fixture (minimal, 1 DIMM)
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
"""


# --- Windows parser ---

def test_mem_windows_four_modules_ddr4():
    """Real willw11 fixture: 4 DDR4 DIMM_A1/A2/B1/B2, 8 GiB each, 3400 MT/s."""
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
    """Root Linux with dmidecode output → 1 module parsed."""
    mi = hw._mem_linux(_LINUX_MEMINFO, _LINUX_DMIDECODE, (31.2, 15.2, 2.0))
    assert len(mi.modules) == 1
    m = mi.modules[0]
    assert m.slot == "Controller0-ChannelA-DIMM0"
    assert m.manufacturer == "Samsung"
    assert m.part_number == "K4EBE304EB-EGCG"
    assert m.speed_mts == 3200
    assert mi.kind == "LPDDR4"


def test_mem_linux_empty_meminfo():
    """Blank /proc/meminfo → totals from passed-in tuple, no crash."""
    mi = hw._mem_linux("", None, (0.0, 0.0, 0.0))
    assert mi.modules == []


# Two-module dmidecode fixture — covers the "flush current when new block starts" path.
_LINUX_DMIDECODE_TWO = """\
# dmidecode 3.3

Handle 0x1100, DMI type 17, 84 bytes
Memory Device
\tLocator: ChannelA-DIMM0
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tPart Number: M471A1G44AB0

Handle 0x1101, DMI type 17, 84 bytes
Memory Device
\tLocator: ChannelB-DIMM0
\tType: DDR4
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
\tSpeed: 3200 MT/s
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
\tLocator: ChannelB-DIMM0
\tType: DDR4
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
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
\tLocator: DIMM1
\tType: DDR4
\tManufacturer: Micron
\tPart Number: PART-B

Memory Device
\tLocator: DIMM2
\tType: DDR4
\tManufacturer: Micron
\tPart Number: PART-C
"""
    mi = hw._mem_linux("", three_modules, (0.0, 0.0, 0.0))
    assert len(mi.modules) == 3
    assert mi.kind == "DDR4"


# A dmidecode text that ends with a 'Memory Device' header but no fields (empty current at EOF).
_LINUX_DMIDECODE_TRAILING_HEADER = """\
Memory Device
\tLocator: ChannelA-DIMM0
\tType: DDR4
\tManufacturer: Micron
\tPart Number: PART-A

Memory Device
"""


def test_mem_linux_dmidecode_trailing_empty_block():
    """A trailing 'Memory Device' header with no fields → last empty current not appended."""
    mi = hw._mem_linux("", _LINUX_DMIDECODE_TRAILING_HEADER, (0.0, 0.0, 0.0))
    # Only the first module (with fields) is parsed; empty trailing block is skipped.
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


# ---------------------------------------------------------------------------
# Task 4: Storage detail
# ---------------------------------------------------------------------------

# --- Windows fixture (real willw11 Get-PhysicalDisk output) ---
_WIN_PHYSICAL_DISKS = [
    {"FriendlyName": "Samsung SSD 990 EVO 1TB", "MediaType": "SSD", "BusType": "NVMe",
     "Size": 1000204886016},
    {"FriendlyName": "ST2000DM008-2FR102", "MediaType": "HDD", "BusType": "SATA",
     "Size": 2000398934016},
    {"FriendlyName": "QNAP TR-004 DISK00", "MediaType": "Unspecified", "BusType": "USB",
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

# Linux lsblk JSON fixture
_LINUX_LSBLK_JSON = """\
{
   "blockdevices": [
      {"name": "sda", "model": "Samsung SSD 870", "size": "500107862016", "rota": "0", "tran": "sata"},
      {"name": "sdb", "model": "WDC WD20EZRZ", "size": "2000398934016", "rota": "1", "tran": "sata"},
      {"name": "sdc", "model": "SanDisk Cruzer", "size": "32016969728", "rota": "0", "tran": "usb"},
      {"name": "nvme0n1", "model": "Samsung SSD 980 PRO", "size": "1000204886016", "rota": "0", "tran": "nvme"}
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
    drives = hw._drives_windows([{"FriendlyName": "Samsung SSD 990 EVO 1TB",
                                   "MediaType": "SSD", "BusType": "NVMe",
                                   "Size": 1000204886016}])
    assert len(drives) == 1
    d = drives[0]
    assert d.model == "Samsung SSD 990 EVO 1TB"
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
    drives = hw._drives_windows([{"FriendlyName": "ST2000DM008-2FR102",
                                   "MediaType": "HDD", "BusType": "SATA",
                                   "Size": 2000398934016}])
    assert drives[0].media == "hdd"


def test_drives_windows_usb_classification():
    """BusType=USB → 'usb'."""
    drives = hw._drives_windows([{"FriendlyName": "QNAP TR-004 DISK00",
                                   "MediaType": "Unspecified", "BusType": "USB",
                                   "Size": 8001456963584}])
    assert drives[0].media == "usb"


def test_drives_windows_unknown_classification():
    """Unrecognised MediaType+BusType → 'unknown'."""
    drives = hw._drives_windows([{"FriendlyName": "Unknown Device",
                                   "MediaType": "Unspecified", "BusType": "Unknown",
                                   "Size": 100000000000}])
    assert drives[0].media == "unknown"


def test_drives_windows_full_willw11_fixture():
    """Full willw11 Get-PhysicalDisk fixture: NVMe SSD + HDD + USB → 3 drives, correct media."""
    drives = hw._drives_windows(_WIN_PHYSICAL_DISKS)
    assert len(drives) == 3
    assert drives[0].media == "nvme-ssd"
    assert drives[1].media == "hdd"
    assert drives[2].media == "usb"
    assert drives[0].model == "Samsung SSD 990 EVO 1TB"
    assert drives[1].model == "ST2000DM008-2FR102"
    assert drives[2].model == "QNAP TR-004 DISK00"


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
    """lsblk JSON fixture: sata-ssd, hdd, usb, nvme → correct media classification."""
    drives = hw._drives_linux(_LINUX_LSBLK_JSON)
    assert len(drives) == 4
    by_name = {d.model: d for d in drives}
    assert by_name["Samsung SSD 870"].media == "sata-ssd"
    assert by_name["WDC WD20EZRZ"].media == "hdd"
    assert by_name["SanDisk Cruzer"].media == "usb"
    assert by_name["Samsung SSD 980 PRO"].media == "nvme-ssd"


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

# Windows WMI fixture objects (real willw11)
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


def test_board_macos_sets_vendor_apple_always():
    """Any text that lacks 'Model Name:' → system_vendor is still None (not forced to Apple
    unless we actually found hardware info — honest gap)."""
    b = hw._board_macos("Some hardware text without expected keys\n")
    assert b.system_vendor is None


# --- _board_windows parser ---

def test_board_windows_asus_rog_willw11():
    """Real willw11 fixture: board=ASUS ROG STRIX X470-F, bios=6042@2022-04-28,
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
