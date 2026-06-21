# Task 6 Implementation Report — Integrate into `detect.Machine`

## Status: COMPLETE

## What was done

### `ara/detect.py`
- Added `from ara import hardware as _hardware` import (no circular import — `hardware.py`
  does NOT import `detect`; verified with `python -c "import ara.detect, ara.hardware"`).
- `Machine` dataclass gains four new optional fields (with `default_factory` so existing
  callers that don't pass them still compile):
  - `cpu: _hardware.CpuInfo`
  - `memory: _hardware.MemoryInfo`
  - `storage: _hardware.StorageInfo`
  - `board: _hardware.BoardInfo`
- `profile()` calls `_hardware.probe()` once at the top; flat back-compat fields are now
  sourced from the returned `Hardware` struct (single source of truth):
  - `cpu_physical = hw.cpu.physical`
  - `cpu_logical = hw.cpu.logical`
  - `ram_total_gb = hw.memory.total_gb`
  - `ram_available_gb = hw.memory.available_gb`
  - `swap_gb = hw.memory.swap_gb`
  - `disk_free_gb = hw.storage.free_gb`
- `chip` field: `hw.cpu.brand or chip_name()` — on Windows this shows "AMD Ryzen 9 5900X"
  instead of the generic `platform.processor()` string.
- Removed the now-redundant direct calls to `_memory_gb()`, `_cpu_counts()`, `_disk_free_gb()`
  from `profile()` (those helpers are kept for use elsewhere; they remain tested).

### `tests/test_detect.py`
Added 8 new tests (TDD: written failing first, then implemented):
- `test_profile_embeds_hardware_structures` — verifies all four structures land on Machine
- `test_profile_chip_uses_cpu_brand_when_available` — chip = cpu.brand when present
- `test_profile_chip_falls_back_when_brand_none` — falls back to chip_name() sysctl
- `test_profile_flat_fields_sourced_from_hw_structures` — flat fields match hw struct values
- `test_profile_flat_fields_none_when_hw_fields_none` — None propagates correctly
- `test_import_no_circular` — import guard
- `test_chip_name_darwin_no_brand_falls_back` — Darwin + no sysctl → processor() fallback
- `test_memory_gb_returns_totals` — happy path for `_memory_gb()` (uncovered after refactor)

Also added `_make_hw()` helper and hardware imports to the test module header.

## Concerns
- None: no circular import, all 849 tests pass, 100% statement + branch coverage.
- `_memory_gb()`, `_cpu_counts()`, `_disk_free_gb()` remain in `detect.py` as public
  helpers (used by tests and potentially cli.py in the future). They are not dead code —
  but `profile()` no longer calls them directly. This is intentional per the plan.
