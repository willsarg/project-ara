# Security Policy

Project ARA is a small, community-maintained tool. Security reports are taken seriously and
handled on a best-effort basis.

## Supported versions

The project is pre-1.0 and moves on `main`. Only the **latest `main`** is supported — please
confirm an issue reproduces there before reporting.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Use GitHub's private vulnerability reporting:

1. Go to the repo's **[Security tab](https://github.com/willsarg/project-ara/security)**.
2. Click **"Report a vulnerability"**.
3. Describe the issue, the impact, and steps to reproduce.

This keeps the report private until a fix is available. Expect an initial response within
about a week. If accepted, we'll work on a fix and coordinate disclosure; if declined, we'll
explain why.

## Scope

ARA runs locally and has no network service, no authentication, and no remote attack surface
of its own. The areas most relevant to security are:

- **Reading untrusted metadata.** ARA reads `config.json`, `Info.plist`, package metadata,
  and Homebrew output from the local system. Parsing is best-effort and defensive, but a
  crafted file is the most plausible input-handling concern.
- **Subprocesses.** ARA shells out to read-only tools (`brew`, `nvidia-smi`,
  `system_profiler`, interpreters for `--version`/metadata). It does not pass untrusted input
  as shell strings, and recon never mutates state.
- **Model downloads.** `ara profile` can download a calibration model via `huggingface_hub`
  — only with explicit consent, into the standard HF cache.

If you find a way ARA could execute untrusted code, leak credentials (e.g. an HF token), or
mutate the system without consent, that's in scope — please report it privately.
