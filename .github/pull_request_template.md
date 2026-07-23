<!-- Thanks for contributing! Keep PRs focused on a single change. -->

## Summary

<!-- What does this change and why? -->

Related issue: <!-- e.g. Fixes #N -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Recon coverage (new tool / app / model store / interpreter)
- [ ] New backend
- [ ] Docs only
- [ ] Refactor / cleanup (no behavior change)

## ARA's rules (see [AGENTS.md](../AGENTS.md))

- [ ] **Recon stays read-only** — no new code path under `detect`/`status`/`python`/`apps`/`mlx`
      stresses the machine, loads a model, or mutates state.
- [ ] **`profile` stays engine-free and read-only** — it may reason over recon and stored display
      evidence, but never installs/imports/loads an engine, loads/downloads a model, or mutates state.
- [ ] **Measurement stays consent-gated** — `characterize` is the command that crosses into an
      engine/model and may download weights, only with explicit opt-in.
- [ ] **Advisory, not destructive** — nothing here runs or prescribes a state-mutating command.
- [ ] **Core stays engine-free** — no hardware-specific import outside a lazily-loaded backend.
- [ ] Reports the **user's** environment, not ARA's (venv stripped where relevant).

## Conventions

- [ ] `uv` only (no `--break-system-packages`); HF CLI is `hf`.
- [ ] Tests added/updated; `uv run pytest` green at **100%** statement + branch coverage.
- [ ] New curated-catalog entries note how they were verified.

## Evidence

<!-- Paste relevant command output (e.g. `ara apps` before/after), and note the machine. -->
