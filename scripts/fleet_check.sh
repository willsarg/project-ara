#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# fleet_check.sh — ARA's live hardware fleet-check ritual (testing architecture layer 7).
#
# WHAT THIS IS
#   Layers 1-6 of ARA's test pyramid (unit / property / contract / etc.) run mocked, in CI,
#   on GitHub-hosted runners, on every PR. This layer is different on purpose: it is the
#   MANUAL PRE-RELEASE RITUAL that proves ARA's mocked engine seams match real hardware.
#   It is NOT wired into CI and never will be — the fleet boxes (a personal Ubuntu box and a
#   personal Windows box, reached over this operator's SSH config) are intentionally kept
#   OUTSIDE GitHub's trust boundary. No CI runner ever gets credentials to reach them, and
#   this script never runs unattended from a workflow. A human runs it, from a machine that
#   already has the fleet's SSH aliases trusted in ~/.ssh/known_hosts, before cutting a
#   release (or after touching an engine adapter / vendored engine source).
#
# WHAT IT DOES
#   For each target (a real, host-native machine — never a container, ARA's engines are
#   host-native by design):
#     1. Ship the current HEAD worktree over (git archive, not rsync — never node_modules,
#        never .venv, never any local cruft; only what `git` would commit).
#     2. `uv sync --frozen --group dev` — install the exact locked dev environment.
#     3. `uv run pytest` — the same mocked gate CI runs, but on real hardware/OS.
#     4. `uv run ara detect --json` — read-only live recon, proof the recon layer actually
#        talks to this real machine (not a fake/mock) and emits well-formed JSON.
#   Results are collected per target and printed as a summary table. Any FAIL or UNREACHABLE
#   target makes the whole run exit non-zero.
#
# HOW TO RUN IT
#   scripts/fleet_check.sh                      # all three: mac, rog-ubuntu, willw11
#   scripts/fleet_check.sh rog-ubuntu            # just one target
#   scripts/fleet_check.sh mac willw11           # a subset, any order
#
# SECURITY — HARD RULE, DO NOT VIOLATE
#   This script MUST NEVER disable SSH host-key verification. Never add
#   `-o StrictHostKeyChecking=no` or `-o UserKnownHostsFile=/dev/null` here, and reject any
#   change that does. It relies entirely on the operator's existing `known_hosts` (the
#   `rog-ubuntu` / `willw11` aliases are already trusted from prior manual `ssh` use). If a
#   host key is unknown, missing, or has changed, `ssh` MUST fail loudly and this script MUST
#   report that target as UNREACHABLE — never silently trust an unverified host. The only SSH
#   options used below (`BatchMode`, `ConnectTimeout`) affect liveness/prompting, not identity
#   verification.
#
# WINDOWS NOTE (willw11)
#   willw11's `ssh` lands directly in cmd.exe (Win32-OpenSSH default shell), NOT WSL and NOT
#   PowerShell. See the windows-shell-execution playbook for the full model; the two traps
#   that matter here:
#     - cmd.exe single quotes do NOT group tokens the way POSIX shells do (`'a b'` becomes
#       two args `'a` and `b'`); only double quotes group. We never rely on single-quote
#       grouping in a command sent to willw11.
#     - Don't POSIX-chain multiple semantically distinct steps into one `ssh` call assuming
#       shared shell semantics. Each meaningful step (sync / test / detect) is sent as its
#       OWN `ssh` invocation below, so a failure in one step is attributed precisely and we
#       never depend on cmd.exe interpreting a long chained command the way bash would.
#
# ROBUSTNESS
#   A target that can't be reached (bad DNS/alias, box asleep, network down, SSH refused) is
#   reported UNREACHABLE, distinct from FAIL (reachable, but pytest or `ara detect` failed).
#   One target's failure never aborts the others — we always attempt every requested target
#   and print one summary at the end. Exit code is non-zero if ANY target is not a clean PASS.

set -euo pipefail

# ---------------------------------------------------------------------------------------------
# Target registry
#
# Deliberately NOT an associative array: macOS ships bash 3.2 as /bin/bash (associative arrays
# need bash 4+), and this script must run unmodified on both the mac target's own shell and any
# Linux operator box. Target metadata is looked up via small case-statement helpers instead.
# ---------------------------------------------------------------------------------------------

ALL_TARGETS="mac rog-ubuntu willw11"

# Human-readable name for a target key.
target_display_name() {
    case "$1" in
        mac) echo "mac (local, MLX)" ;;
        rog-ubuntu) echo "rog-ubuntu (Ubuntu 24.04, Vulkan)" ;;
        willw11) echo "willw11 (Windows 11, CUDA/CPU)" ;;
        *) echo "$1" ;;
    esac
}

# SSH host alias for a target key. Assumed already present and trusted in the operator's SSH
# config / known_hosts (see security note above) — this script never configures SSH itself.
target_ssh_alias() {
    case "$1" in
        rog-ubuntu) echo "rog-ubuntu" ;;
        willw11) echo "willw11" ;;
        *) echo "" ;; # mac is local — no ssh alias
    esac
}

# Shell family for a target key: local | posix | windows-cmd.
target_kind() {
    case "$1" in
        mac) echo "local" ;;
        rog-ubuntu) echo "posix" ;;
        willw11) echo "windows-cmd" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

log() {
    printf '[fleet-check] %s\n' "$*" >&2
}

# SSH options shared by every remote call. NOTE: this list must never grow to include
# StrictHostKeyChecking=no or UserKnownHostsFile=/dev/null — see the security note above.
#   -o BatchMode=yes      : never prompt for a password/passphrase; fail fast instead of hanging.
#   -o ConnectTimeout=10  : bound how long an asleep/unreachable box can stall the whole run.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

# Best-effort JSON validity check for `ara detect --json` output. Prefers python3 (present on
# every target we care about — it's ARA's own runtime requirement) and falls back to a cheap
# structural sniff if python3 isn't on the caller's PATH for some reason.
is_valid_json() {
    local payload="$1"
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$payload" | python3 -c 'import json, sys; json.load(sys.stdin)' >/dev/null 2>&1
        return $?
    fi
    # Fallback heuristic: non-empty and starts/ends with the outer object braces.
    case "$payload" in
        '{'*'}') return 0 ;;
        *) return 1 ;;
    esac
}

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# ARA's version is derived from git tags by hatch-vcs, but we ship HEAD via `git archive`, which
# strips the .git dir — so the build can't see the tags and `uv sync` fails. Compute the version
# HERE (where .git exists) and hand it to every build via SETUPTOOLS_SCM_PRETEND_VERSION (hatch-vcs
# honors it). Strip a leading `v`; fall back to 0.0.0 if there's no tag. This is the version the
# fleet install reports — it's a check environment, so an approximate version is fine.
PRETEND_VERSION="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')"
PRETEND_VERSION="${PRETEND_VERSION:-0.0.0}"

# Results, as parallel indexed arrays (again: no associative arrays, bash-3.2-safe).
RESULT_TARGETS=()
RESULT_STATUS=()   # PASS | FAIL | UNREACHABLE
RESULT_DETAIL=()

record_result() {
    RESULT_TARGETS+=("$1")
    RESULT_STATUS+=("$2")
    RESULT_DETAIL+=("$3")
}

# ---------------------------------------------------------------------------------------------
# Local (mac) check: run the exact same three steps against a throwaway checkout of HEAD, so
# the mac result is directly comparable to the remote results (no "well it works in my
# already-hacked-on worktree" ambiguity).
# ---------------------------------------------------------------------------------------------

check_local() {
    local target="mac"
    local workdir
    workdir="$(mktemp -d "${TMPDIR:-/tmp}/ara-fleet-check.XXXXXX")"
    # ${workdir:-} guard: a RETURN trap referencing a `local` can fire in a scope where the local
    # is gone (bash trap+local under set -u), which would crash on an unbound var. Empty -> no-op.
    trap 'rm -rf "${workdir:-}"' RETURN
    export SETUPTOOLS_SCM_PRETEND_VERSION="$PRETEND_VERSION"  # subshells below inherit it

    log "$target: archiving HEAD into $workdir"
    if ! git -C "$REPO_ROOT" archive HEAD | tar -x -C "$workdir"; then
        record_result "$target" "FAIL" "git archive/extract failed"
        return
    fi

    log "$target: uv sync --frozen --group dev (version pinned via SETUPTOOLS_SCM_PRETEND_VERSION)"
    if ! (cd "$workdir" && uv sync --frozen --group dev) >&2; then
        record_result "$target" "FAIL" "uv sync failed"
        return
    fi

    log "$target: uv run pytest"
    if ! (cd "$workdir" && uv run pytest) >&2; then
        record_result "$target" "FAIL" "pytest failed"
        return
    fi

    log "$target: uv run ara detect --json"
    local detect_json
    if ! detect_json="$(cd "$workdir" && uv run ara detect --json)"; then
        record_result "$target" "FAIL" "ara detect --json exited non-zero"
        return
    fi
    if ! is_valid_json "$detect_json"; then
        record_result "$target" "FAIL" "ara detect --json did not return valid JSON"
        return
    fi

    record_result "$target" "PASS" "pytest green; ara detect --json valid"
}

# ---------------------------------------------------------------------------------------------
# Reachability probe, shared by both remote kinds. `echo ok` is a builtin in both a POSIX shell
# and cmd.exe, so it's a safe universal liveness check. We distinguish an SSH *connection*
# failure (exit 255 — box asleep, network down, host key rejected, auth failed) from every
# other kind of failure, which is what makes UNREACHABLE distinct from FAIL.
# ---------------------------------------------------------------------------------------------

is_reachable() {
    local alias="$1"
    local out rc
    set +e
    out="$(ssh "${SSH_OPTS[@]}" "$alias" "echo ok" 2>&1)"
    rc=$?
    set -e
    if [[ $rc -eq 255 ]]; then
        return 1
    fi
    [[ "$out" == *ok* ]]
}

# ---------------------------------------------------------------------------------------------
# POSIX remote check (rog-ubuntu): a real bash-family remote shell, so we can safely chain the
# three uv steps with POSIX `&&` semantics in a single ssh call per step (still one step per
# call, for the same precise-attribution reason as the Windows path — not because POSIX chaining
# is unsafe here).
# ---------------------------------------------------------------------------------------------

check_posix_remote() {
    local target="$1" alias="$2"

    if ! is_reachable "$alias"; then
        record_result "$target" "UNREACHABLE" "ssh connection failed (exit 255) or box unresponsive"
        return
    fi

    local remote_dir
    remote_dir="$(ssh "${SSH_OPTS[@]}" "$alias" 'mktemp -d "${TMPDIR:-/tmp}/ara-fleet-check.XXXXXX"')" || {
        record_result "$target" "FAIL" "could not create remote temp dir"
        return
    }
    # Best-effort remote cleanup no matter how this function returns.
    trap 'ssh "${SSH_OPTS[@]}" "'"$alias"'" "rm -rf '"$remote_dir"'" >/dev/null 2>&1 || true' RETURN

    log "$target: archiving HEAD -> $alias:$remote_dir"
    if ! git -C "$REPO_ROOT" archive HEAD | ssh "${SSH_OPTS[@]}" "$alias" "tar -x -C '$remote_dir'"; then
        record_result "$target" "FAIL" "git archive over ssh / remote tar extract failed"
        return
    fi

    # Run each remote uv step in an explicit LOGIN shell (`bash -lc`). uv installs to ~/.local/bin,
    # which is put on PATH by ~/.profile — sourced ONLY by a login shell. A plain `ssh host "cmd"`
    # runs a non-interactive, non-login shell that sources neither ~/.profile (login-only) nor
    # ~/.bashrc (Ubuntu's returns early when non-interactive), so `uv` isn't found (this exact bug
    # was caught by a live fleet run). A login shell gives the remote the same PATH the operator has
    # by hand. SETUPTOOLS_SCM_PRETEND_VERSION is pinned per call (git archive stripped .git; see the
    # global). $remote_dir/$PRETEND_VERSION expand LOCALLY (baked in literal), and are double-quoted
    # INSIDE the single-quoted login-shell arg for space-safety.
    log "$target: uv sync --frozen --group dev (version pinned)"
    if ! ssh "${SSH_OPTS[@]}" "$alias" "bash -lc 'cd \"$remote_dir\" && SETUPTOOLS_SCM_PRETEND_VERSION=\"$PRETEND_VERSION\" uv sync --frozen --group dev'" >&2; then
        record_result "$target" "FAIL" "uv sync failed"
        return
    fi

    log "$target: uv run pytest"
    if ! ssh "${SSH_OPTS[@]}" "$alias" "bash -lc 'cd \"$remote_dir\" && SETUPTOOLS_SCM_PRETEND_VERSION=\"$PRETEND_VERSION\" uv run pytest'" >&2; then
        record_result "$target" "FAIL" "pytest failed"
        return
    fi

    log "$target: uv run ara detect --json"
    local detect_json
    if ! detect_json="$(ssh "${SSH_OPTS[@]}" "$alias" "bash -lc 'cd \"$remote_dir\" && SETUPTOOLS_SCM_PRETEND_VERSION=\"$PRETEND_VERSION\" uv run ara detect --json'")"; then
        record_result "$target" "FAIL" "ara detect --json exited non-zero"
        return
    fi
    if ! is_valid_json "$detect_json"; then
        record_result "$target" "FAIL" "ara detect --json did not return valid JSON"
        return
    fi

    record_result "$target" "PASS" "pytest green; ara detect --json valid"
}

# ---------------------------------------------------------------------------------------------
# Windows-cmd remote check (willw11): see the WINDOWS NOTE header comment. Every meaningful
# step is its own ssh call with a single cmd.exe command line (`cd DIR && uv ...`), quoted with
# DOUBLE quotes (cmd's only grouping quote) around the whole command. We never assume `&&`
# chains three separate uv invocations together in one call, and we never rely on single
# quotes to group anything on the remote side.
# ---------------------------------------------------------------------------------------------

check_windows_remote() {
    local target="$1" alias="$2"

    if ! is_reachable "$alias"; then
        record_result "$target" "UNREACHABLE" "ssh connection failed (exit 255) or box unresponsive"
        return
    fi

    # No mktemp on cmd.exe: build a unique, space-free, relative directory name ourselves and
    # create it relative to whatever cwd Win32-OpenSSH lands us in (normally the user's home).
    # Relative + space-free sidesteps %TEMP% expansion/quoting edge cases entirely.
    local remote_dir="ara-fleet-check-$(date +%Y%m%d%H%M%S)-$$"

    log "$target: mkdir $remote_dir"
    if ! ssh "${SSH_OPTS[@]}" "$alias" "mkdir \"$remote_dir\"" >&2; then
        record_result "$target" "FAIL" "could not create remote temp dir"
        return
    fi
    # Best-effort remote cleanup no matter how this function returns (cmd's rmdir /s /q).
    trap 'ssh "${SSH_OPTS[@]}" "'"$alias"'" "rmdir /s /q \"'"$remote_dir"'\"" >/dev/null 2>&1 || true' RETURN

    log "$target: archiving HEAD -> $alias:$remote_dir"
    # bsdtar (shipped as tar.exe on Windows 10 1803+) reads the archive from stdin here.
    if ! git -C "$REPO_ROOT" archive HEAD | ssh "${SSH_OPTS[@]}" "$alias" "tar -x -C \"$remote_dir\""; then
        record_result "$target" "FAIL" "git archive over ssh / remote tar extract failed"
        return
    fi

    # SETUPTOOLS_SCM_PRETEND_VERSION on each remote uv call (git archive stripped .git). cmd.exe
    # form: `set "VAR=val"` (quoted, so no trailing-space corruption) chained with &&.
    local setver="set \"SETUPTOOLS_SCM_PRETEND_VERSION=$PRETEND_VERSION\""
    log "$target: uv sync --frozen --group dev (version pinned)"
    if ! ssh "${SSH_OPTS[@]}" "$alias" "cd \"$remote_dir\" && $setver && uv sync --frozen --group dev" >&2; then
        record_result "$target" "FAIL" "uv sync failed"
        return
    fi

    log "$target: uv run pytest"
    if ! ssh "${SSH_OPTS[@]}" "$alias" "cd \"$remote_dir\" && $setver && uv run pytest" >&2; then
        record_result "$target" "FAIL" "pytest failed"
        return
    fi

    log "$target: uv run ara detect --json"
    local detect_json
    if ! detect_json="$(ssh "${SSH_OPTS[@]}" "$alias" "cd \"$remote_dir\" && $setver && uv run ara detect --json")"; then
        record_result "$target" "FAIL" "ara detect --json exited non-zero"
        return
    fi
    if ! is_valid_json "$detect_json"; then
        record_result "$target" "FAIL" "ara detect --json did not return valid JSON"
        return
    fi

    record_result "$target" "PASS" "pytest green; ara detect --json valid"
}

# ---------------------------------------------------------------------------------------------
# Dispatch one target to the right checker. Each checker records exactly one result and never
# raises past this point — `set -e` is deliberately shielded per-target (see run_target) so one
# target's hard failure can't abort the others.
# ---------------------------------------------------------------------------------------------

run_target() {
    local target="$1"
    local kind
    kind="$(target_kind "$target")"

    case "$kind" in
        local)
            check_local
            ;;
        posix)
            check_posix_remote "$target" "$(target_ssh_alias "$target")"
            ;;
        windows-cmd)
            check_windows_remote "$target" "$(target_ssh_alias "$target")"
            ;;
        *)
            record_result "$target" "FAIL" "unknown target (not in: $ALL_TARGETS)"
            ;;
    esac
}

# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

main() {
    local targets=("$@")
    if [[ ${#targets[@]} -eq 0 ]]; then
        # shellcheck disable=SC2206 # intentional word-splitting of a known space-separated list
        targets=($ALL_TARGETS)
    fi

    log "targets: ${targets[*]}"
    log "repo root: $REPO_ROOT"

    for t in "${targets[@]}"; do
        log "=== $(target_display_name "$t") ==="
        # Shield the rest of the run from a single target's failure: this is a manual
        # diagnostic ritual, not a CI gate that should die on the first bad box.
        set +e
        run_target "$t"
        set -e
    done

    printf '\n'
    printf '%-14s %-30s %-13s %s\n' "TARGET" "NAME" "STATUS" "DETAIL"
    printf '%-14s %-30s %-13s %s\n' "------" "----" "------" "------"
    local overall_rc=0
    local i
    for ((i = 0; i < ${#RESULT_TARGETS[@]}; i++)); do
        printf '%-14s %-30s %-13s %s\n' \
            "${RESULT_TARGETS[$i]}" \
            "$(target_display_name "${RESULT_TARGETS[$i]}")" \
            "${RESULT_STATUS[$i]}" \
            "${RESULT_DETAIL[$i]}"
        if [[ "${RESULT_STATUS[$i]}" != "PASS" ]]; then
            overall_rc=1
        fi
    done
    printf '\n'

    if [[ $overall_rc -eq 0 ]]; then
        log "fleet check: ALL PASS"
    else
        log "fleet check: FAILURES PRESENT (see table above) — exiting non-zero"
    fi
    exit "$overall_rc"
}

main "$@"
