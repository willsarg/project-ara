#!/usr/bin/env bash
# scripts/mutation_probe.sh — layer 8 mutation-testing probe (mutmut) for ARA.
#
# Purpose: surface SURVIVING mutants — places where a small code change wouldn't fail any
# test, i.e. weak assertions hiding behind the green 100%-coverage gate. This is a periodic
# HEALTH CHECK, not a blocking gate: a survivor is a finding to triage, not a build failure.
#
# Why scoped, not a full sweep: mutmut (see [tool.mutmut] in pyproject.toml) reads its scope
# ONLY from pyproject.toml's [tool.mutmut] table — there is no CLI flag or env var to narrow
# it per-run. A full sweep of thousands of ARA statements reruns the fast unit suite once per
# mutant and takes HOURS, far past any CI budget. So this script computes a bounded file list
# (by default: ara/*.py touched in the last 7 days, i.e. since the previous weekly run),
# temporarily appends an `only_mutate` filter to pyproject.toml for the duration of the run,
# and always restores the original file afterward — even on failure or interrupt.
#
# A full, unscoped sweep is a manual/local long-run: `uv run mutmut run` with no args (after
# temporarily removing/widening `only_mutate`), left going for hours on a dev box.
#
# Usage:
#   scripts/mutation_probe.sh                          # scope = ara/*.py changed in the last 7 days
#   scripts/mutation_probe.sh ara/estimate.py ara/foo.py  # explicit scope (e.g. for local bring-up)
#
# Native engine package code (ara/_engine_packages/**) is never mutated regardless of scope —
# pyproject.toml's [tool.mutmut] carries a permanent `do_not_mutate` filter for it.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYPROJECT="pyproject.toml"
BACKUP="$(mktemp)"
cp "$PYPROJECT" "$BACKUP"
restore() {
    cp "$BACKUP" "$PYPROJECT"
    rm -f "$BACKUP"
}
trap restore EXIT

if [ "$#" -gt 0 ]; then
    FILES=("$@")
else
    # Default scope: ara/*.py touched in the last 7 days (the weekly schedule window),
    # excluding native engine package code. `git log --name-only` over that window, deduped,
    # filtered to
    # files that still exist (a file renamed/deleted since shouldn't be probed).
    mapfile -t FILES < <(
        git log --since="7 days ago" --name-only --pretty=format: -- 'ara/*.py' \
            | sort -u \
            | grep -E '^ara/' \
            | grep -v '^ara/_engine_packages/' \
            | while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done
    )
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "mutation_probe: no ara/*.py files in scope (nothing changed in the window) — nothing to probe."
    exit 0
fi

echo "mutation_probe: scoping mutmut to ${#FILES[@]} file(s):"
printf '  %s\n' "${FILES[@]}"

{
    printf '\n'
    printf '# --- appended by scripts/mutation_probe.sh for this run only; restored after ---\n'
    printf 'only_mutate = ['
    for f in "${FILES[@]}"; do
        printf '"%s", ' "$f"
    done
    printf ']\n'
} >>"$PYPROJECT"

# mutmut's own exit code reflects infra failures (e.g. the clean test run itself failing), not
# survivor counts — survivors are a finding, not a probe failure, but infra failures must fail the
# probe instead of leaving every mutant "not checked" behind a green workflow.
# Default to --max-children 1: mutmut's parallel workers share pytest tmp dirs and hit a cleanup
# race that leaves mutants "not checked" (verified). Serial is reliable; override for a faster
# local sweep with MUTMUT_MAX_CHILDREN=N if your box tolerates it.
uv run mutmut run --max-children "${MUTMUT_MAX_CHILDREN:-1}"

echo ""
echo "=== mutation_probe results ==="
uv run mutmut results
