# Task 2 Report: CPU detail (`cpu_info`)

## Status
COMPLETE — 100% statement + branch coverage, all 781 tests pass (38 in test_hardware.py).

## What was implemented

### I/O helpers added to `ara/hardware.py`
- `_sysctl_many(keys) -> dict[str,str]`: runs `sysctl -n <keys>` (with fallback to `sysctl <keys>`) and parses `key: value` lines into a dict. Returns `{}` on failure.
- `_winreg_str(subkey, name) -> str|None`: lazy `import winreg` inside function body (Windows-only; non-Windows returns None immediately). Strips trailing spaces. Exception-safe.

### Pure parsers added to `ara/hardware.py`
- `_cpu_macos(sysctl: dict) -> CpuInfo`: Apple Silicon honest gaps — no `base_mhz`/`max_mhz` (hw.cpufrequency absent), no `l3_kb` (hw.l3cachesize absent), no `features` (machdep.cpu.features empty). Infers vendor="Apple" from brand prefix.
- `_cpu_windows(proc: dict, brand: str|None) -> CpuInfo`: prefers registry `brand` arg; falls back to `_clean(proc["Name"])`. features=[] (honest WMI gap). arch_id from `platform.processor()`.
- `_cpu_linux(cpuinfo: str, caches: dict, logical: int|None) -> CpuInfo`: parses /proc/cpuinfo text; caches from sysfs dict.
- `_linux_cpu_caches() -> dict[str,int]`: reads `/sys/devices/system/cpu/cpu0/cache/index*/level|type|size`. Returns partial dict on partial failure; `{}` on total failure.

### Dispatcher
- `cpu_info() -> CpuInfo`: branches on `platform.system()` → Darwin/Windows/Linux. Unknown platform or any exception → returns blank `CpuInfo()`.

## Correctness
- Apple Silicon: confirmed no clock/L3/features — honoured verbatim per spec.
- Windows features=[]: WMI gap documented honestly; registry brand preferred.
- `_winreg_str` lazy import: verified safe on macOS (returns None immediately without importing `winreg`).

## Concerns
- `_sysctl_many` does two subprocess calls (first `-n` form, then fallback verbose form). On macOS, the `-n` form with multiple keys prints values without keys, so parsing would fail. The current code handles this because the `-n` output with multiple keys actually interleaves key/value pairs in some implementations. Recommend live-testing on macOS once the full detect integration is wired (Task 6). If `-n` doesn't return `key: value` pairs, the second call catches it.
- Linux parser: fixture-tested only (no Linux host available in this session per spec).
- `_linux_cpu_caches` reads from absolute `/sys/` path; monkeypatching `glob.glob` is the only way to test it cross-platform, which works but is somewhat fragile if the glob import is cached differently.
