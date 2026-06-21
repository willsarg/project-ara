# Hardware Task 1 Report — Foundation: data model + I/O helpers

## Status
DONE

## Summary
Created `ara/hardware.py` with all dataclasses (`CpuInfo`, `MemoryModule`, `MemoryInfo`, `Drive`,
`StorageInfo`, `BoardInfo`, `Hardware`) and I/O helpers (`_clean`, `_gib`, `_gb_dec`, `_run`,
`_pwsh_json`, `_wmi_date`). Created `tests/test_hardware.py` with 13 tests covering all helpers
including edge cases (None, overflow, invalid JSON, nonzero exit).

## TDD Steps Followed
1. Wrote failing tests (6 from plan) → ImportError confirmed.
2. Implemented `ara/hardware.py` — discovered `date.fromtimestamp` uses local time; fixed to
   `datetime.fromtimestamp(..., tz=timezone.utc).date()` (the WMI timestamp is Unix-epoch UTC).
3. Added 7 more tests for `_gib`, `_gb_dec`, `_run`, invalid-json branch, and `_wmi_date` overflow
   branch to reach 100% coverage of `hardware.py`.
4. Full suite: 756 passed, 1 skipped, 100% total coverage.

## Concerns
- **`_wmi_date` timezone fix**: the plan's code uses `date.fromtimestamp(ms/1000)` which is
  local-time. On a UTC-behind machine (e.g. US timezones), `/Date(1651104000000)/` (midnight UTC
  2022-04-28) maps to April 27 locally. Fixed to UTC. The test `assert ... == "2022-04-28"` confirms
  correctness. This was a latent bug in the plan's sample code, not a design issue.
- `_run` is real subprocess (no mock needed for the tests since `echo hello` / `false` / missing-cmd
  cover all branches cleanly).
