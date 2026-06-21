# Task 7 Report — Render in cli.py

## Status
COMPLETE — 879 tests pass, 100% statement + branch coverage, committed on main.

## Changes

### ara/cli.py
- `_det_system`: calls `_det_cpu_detail(c, m)` when `c.verbose`
- `_det_cpu_detail`: verbose CPU block — vendor, threads, base/max clock, L1/L2/L3 cache, features; each line skipped when value is None
- `_det_memory`: calls `_det_memory_detail(c, m)` when `c.verbose`
- `_det_memory_detail`: verbose memory block — kind, speed, slot summary (used/total), per-module rows; shows "(not reported on this system)" when modules=[] and no slot count at all
- `_det_storage`: calls `_det_storage_detail(c, m)` when `c.verbose`
- `_det_storage_detail`: per-drive rows — model · media · size; rows with all-None fields skipped
- `_det_board`: verbose-only BOARD section — board vendor/model, BIOS version/date, system vendor/model; section skipped entirely when all fields None or not verbose
- `_DETECT_RENDERERS`: added `("board", _det_board)` entry
- `_RECON_SECTIONS["detect"]`: added "board" key for --include/--exclude
- `render_detect` JSON: already includes cpu/memory/storage/board as nested dicts via `asdict(m)` (they're Machine dataclass fields); no structural change needed

### tests/test_cli.py
- Added import of hardware dataclasses
- Added Apple-Silicon and Windows/Ryzen fixture builders (_apple_cpu, _windows_cpu, _apple_memory, _windows_memory, _apple_storage, _windows_storage, _apple_board, _windows_board)
- 30 new tests covering: verbose CPU detail (all-None, base-only, features, Apple, Ryzen), verbose memory detail (4 modules, no modules/not-reported, partial slots, mixed-None module fields, all-None module with more following), verbose storage (drives, None fields, empty), BOARD section (Windows, Mac, all-None, non-verbose), JSON (nested cpu/memory/storage/board, existing keys intact, Apple memory)

## Concerns
- The "board" key added to _RECON_SECTIONS lets --include=board/--exclude=board work in non-verbose mode too, but the board renderer is a no-op then (c.verbose=False exits immediately). This is harmless but slightly inconsistent — could document that "board" only has effect with --verbose. Not a bug.
- MOCK NOTE: cli.py lines 790, 852, 1117-1118 (render_model_detail + render_characterize + render_profile JSON error paths) showed as uncovered when running test_cli.py alone — they are covered by test_cli_contract.py and other test files in the full suite.
