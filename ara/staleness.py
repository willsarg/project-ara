# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Is a stored characterization ceiling stale for the model's current revision?

ARA remembers each model's fitted ``safe_context`` with the timestamp it was measured. If the
model's cache artifacts are later updated (a re-download, a new quant of the same id), that stored
ceiling was measured against a *different* model — governance should say so rather than silently
trust it (Rule #3).

The staleness test itself (cache mtime vs the stored timestamp) is engine-agnostic and depends
only on the standard HF cache layout. ARA owns that pure, standard-library-only logic here so core
code never imports a nested engine package in-process.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_BLOB_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_STAGE_DISK_BUFFER_BYTES = 2_000_000_000


class StagedModel:
    """A verified operation-private model path with deterministic cleanup."""

    def __init__(self, path: Path, temporary: tempfile.TemporaryDirectory):
        self.path = str(path)
        self._temporary = temporary

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, *_exc) -> None:
        self._temporary.cleanup()


def _cache_dir(model_id: str) -> Path:
    default_hub = (Path(os.path.expanduser(os.environ["XDG_CACHE_HOME"]))
                   / "huggingface" / "hub"
                   if os.environ.get("XDG_CACHE_HOME") else _HUB)
    hub = (Path(os.path.expanduser(os.environ["HF_HUB_CACHE"]))
           if os.environ.get("HF_HUB_CACHE") else
           Path(os.path.expanduser(os.environ["HF_HOME"])) / "hub"
           if os.environ.get("HF_HOME") else default_hub)
    return hub / ("models--" + model_id.replace("/", "--"))


def _current_snapshot(repo_id: str) -> tuple[str, Path] | None:
    """Return the current ``main`` revision and snapshot directory for a cached HF repo."""
    root = _cache_dir(repo_id)
    try:
        revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not _REVISION_RE.fullmatch(revision):
        return None
    snapshot = _validated_snapshot(root, revision)
    if snapshot is None:
        return None
    return revision, snapshot


def _validated_snapshot(root: Path, revision: str) -> Path | None:
    """Return a real snapshot directory confined to this repository cache root."""
    snapshots = root / "snapshots"
    snapshot = snapshots / revision
    try:
        linklike = lambda path: path.is_symlink() or path.is_junction()
        if (linklike(root) or linklike(snapshots) or linklike(snapshot)
                or not snapshot.is_dir()):
            return None
        resolved_root = root.resolve(strict=True)
        if (resolved_root.name != root.name
                or resolved_root.parent != root.parent.resolve(strict=True)
                or snapshots.resolve(strict=True).parent != root.resolve(strict=True)
                or snapshot.resolve(strict=True).parent != snapshots.resolve(strict=True)):
            return None
    except OSError:
        return None
    return snapshot


def _content_digest(path: Path) -> tuple[int, str] | None:
    """Return a stable content digest, refusing a file that changes while it is read."""
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError:
        return None
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        return None
    return after.st_size, digest.hexdigest()


def _file_descriptor(snapshot: Path, path: Path) -> str | None:
    """Content-derived HF-cache identity, including Windows' direct-file fallback."""
    try:
        resolved = path.resolve(strict=True)
        relative = path.relative_to(snapshot).as_posix()
    except (OSError, ValueError):
        return None
    if "|" in relative:
        return None
    if path.is_symlink():
        # Normal HF snapshots link into this repository's content-addressed blob store. Any other
        # external link would let a lexical shard name load mutable bytes outside the authority.
        blob_dir = snapshot.parent.parent / "blobs"
        try:
            resolved_blob_dir = blob_dir.resolve(strict=True)
        except OSError:
            return None
        if resolved.parent != resolved_blob_dir or not _BLOB_RE.fullmatch(resolved.name):
            return None
        kind = f"blob:{resolved.name}"
    else:
        try:
            resolved.relative_to(snapshot.resolve(strict=True))
        except (OSError, ValueError):
            return None
        kind = "direct"
    content = _content_digest(resolved)
    if content is None:
        return None
    size, digest = content
    # huggingface_hub uses direct snapshot files when Windows cannot create symlinks.
    authority = f"{kind}:{size}:sha256:{digest}"
    return f"{relative}:{authority}"


def _selected_weights(snapshot: Path) -> list[Path] | None:
    """Exact bare-ref weight selection: all transformer shards or the smallest GGUF."""
    weights = [
        path for path in snapshot.rglob("*")
        if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES)
        and not (path.suffix.lower() == ".gguf"
                 and path.name.lower().startswith("mmproj"))
    ]
    ggufs = [path for path in weights if path.suffix.lower() == ".gguf"]
    transformer = [path for path in weights if path.suffix.lower() != ".gguf"]
    if ggufs and transformer:
        return None
    if ggufs:
        return [min(ggufs, key=lambda path: path.stat().st_size)]
    return transformer or None


def _transformer_manifest(snapshot: Path) -> list[Path] | None:
    """All files consumed from a transformer snapshot, with complete shard-index validation."""
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    indexes = [path for path in files
               if (path.name.endswith(".safetensors.index.json")
                   or path.name.endswith(".bin.index.json"))]
    if len(indexes) > 1:
        return None
    if indexes:
        try:
            index = json.loads(indexes[0].read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            names = set(weight_map.values()) if isinstance(weight_map, dict) else set()
            if not names:
                return None
            referenced = set()
            for name in names:
                logical = PurePosixPath(name) if isinstance(name, str) else None
                if (logical is None or logical.is_absolute() or "\\" in name
                        or any(part in ("", ".", "..") for part in name.split("/"))):
                    return None
                candidate = snapshot.joinpath(*logical.parts)
                if not candidate.is_file():
                    return None
                referenced.add(candidate)
            files = [path for path in files
                     if not path.name.lower().endswith(_WEIGHT_SUFFIXES)
                     or path in referenced]
        except (OSError, UnicodeError, ValueError, AttributeError):
            return None
    return files


def _authorized_snapshot(repo_id: str, artifact_id: str) -> Path | None:
    """Snapshot encoded by an already-verified HF artifact identity (never rereads refs/main)."""
    if not artifact_id.startswith(("hf:", "hf-gguf:")):
        return None
    authority = artifact_id.split(":", 2)[1]
    artifact_repo, separator, revision = authority.rpartition("@")
    if not separator or artifact_repo != repo_id or not _REVISION_RE.fullmatch(revision):
        return None
    root = _cache_dir(repo_id)
    snapshot = _validated_snapshot(root, revision)
    if snapshot is None:
        return None
    return snapshot


def _cache_updated_at(model_id: str) -> float | None:
    """Newest artifact mtime in any locally cached model snapshot."""
    root = _cache_dir(model_id)
    if not root.is_dir():
        return None
    latest: float | None = None
    for dirpath, _, filenames in os.walk(root / "snapshots"):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                mtime = max(path.lstat().st_mtime, path.stat().st_mtime)
            except OSError:
                continue
            latest = mtime if latest is None else max(latest, mtime)
    return latest


def fit_is_stale(model_id: str, measured_at: str | None) -> bool:
    """Whether cache artifacts are newer than a characterization run."""
    if not measured_at:
        return False
    try:
        measured = datetime.fromisoformat(measured_at)
    except (TypeError, ValueError):
        return False
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=timezone.utc)
    cache_mtime = _cache_updated_at(model_id)
    if cache_mtime is None:
        return False
    # DB timestamps use second precision; avoid false positives within that second.
    return cache_mtime > measured.timestamp() + 1.0


def artifact_identity(model: str, *, revision: str | None = None) -> str | None:
    """Identity of the exact local weights selected by *model*, without loading an engine."""
    if not isinstance(model, str):
        return None
    local = Path(model).expanduser()
    if model.lower().endswith(".gguf") and local.is_file():
        try:
            resolved = local.resolve(strict=True)
        except OSError:
            return None
        content = _content_digest(resolved)
        if content is None:
            return None
        size, digest = content
        return f"local-gguf:{resolved}:{size}:sha256:{digest}"

    repo, separator, filename = model.partition(":")
    repo_id = repo if separator and filename.lower().endswith(".gguf") else model
    if revision is not None:
        snapshot = (_validated_snapshot(_cache_dir(repo_id), revision)
                    if _REVISION_RE.fullmatch(revision) else None)
        current = (revision, snapshot) if snapshot is not None else None
    else:
        current = _current_snapshot(repo_id)
    if current is None:
        return None
    revision, snapshot = current
    if separator:
        selected = snapshot / filename
        descriptor = _file_descriptor(snapshot, selected) if selected.is_file() else None
        if descriptor is None:
            return None
        return f"hf-gguf:{repo_id}@{revision}:{descriptor}"
    try:
        selected = _selected_weights(snapshot)
        if selected is None:
            return None
        authority_files = (selected if selected[0].suffix.lower() == ".gguf"
                           else _transformer_manifest(snapshot))
        if authority_files is None:
            return None
        descriptors = []
        for path in sorted(authority_files):
            descriptor = _file_descriptor(snapshot, path)
            if descriptor is None:
                return None
            descriptors.append(descriptor)
    except (OSError, ValueError):
        return None
    return f"hf:{repo_id}@{revision}:" + "|".join(descriptors)


def pinned_model_ref(model: str, expected_artifact_id: str | None, *,
                     revision: str | None = None) -> str | None:
    """Resolve authorized evidence to an immutable local load reference.

    HF repository names are mutable. Governed commands must load the exact cached snapshot/file
    that characterization authorized, not ask the Hub for whatever ``main`` means later.
    """
    matches = (artifact_matches(model, expected_artifact_id, revision=revision)
               if revision is not None else artifact_matches(model, expected_artifact_id))
    if not matches:
        return None
    local = Path(model).expanduser()
    if model.lower().endswith(".gguf") and local.is_file():
        try:
            return str(local.resolve(strict=True))
        except OSError:
            return None
    repo, separator, filename = model.partition(":")
    repo_id = repo if separator and filename.lower().endswith(".gguf") else model
    snapshot = _authorized_snapshot(repo_id, expected_artifact_id)
    if snapshot is None:
        return None
    if separator:
        selected = snapshot / filename
        return str(selected) if selected.is_file() else None
    try:
        selected = _selected_weights(snapshot)
    except OSError:
        return None
    if selected and selected[0].suffix.lower() == ".gguf":
        return str(selected[0])
    return str(snapshot)


def _authority_entry(descriptor: str) -> tuple[str, tuple[int, str]] | None:
    """Decode one content descriptor from ARA's opaque stored artifact authority."""
    try:
        prefix, encoded_size, algorithm, digest = descriptor.rsplit(":", 3)
        size = int(encoded_size)
    except ValueError:
        return None
    if size < 0 or algorithm != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    if prefix.endswith(":direct"):
        relative = prefix[:-len(":direct")]
    else:
        relative, marker, blob = prefix.rpartition(":blob:")
        if not marker or not _BLOB_RE.fullmatch(blob):
            return None
    logical = PurePosixPath(relative)
    if (not relative or logical.is_absolute() or "\\" in relative or "|" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        return None
    return relative, (size, digest)


def _authority_manifest(expected_artifact_id: str) -> dict[str | None, tuple[int, str]] | None:
    """Extract the exact size/digest manifest that an artifact authority authorized."""
    if not isinstance(expected_artifact_id, str):
        return None
    if expected_artifact_id.startswith("local-gguf:"):
        try:
            _prefix, encoded_size, algorithm, digest = expected_artifact_id.rsplit(":", 3)
            size = int(encoded_size)
        except ValueError:
            return None
        if (size < 0 or algorithm != "sha256"
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            return None
        return {None: (size, digest)}
    if not expected_artifact_id.startswith(("hf:", "hf-gguf:")):
        return None
    try:
        descriptors = expected_artifact_id.split(":", 2)[2].split("|")
    except IndexError:
        return None
    entries = [_authority_entry(descriptor) for descriptor in descriptors]
    if not entries or any(entry is None for entry in entries):
        return None
    manifest = dict(entries)
    return manifest if len(manifest) == len(entries) else None


def _stage_file(source: Path, destination: Path, expected: tuple[int, str]) -> None:
    """Copy one source privately, then prove it is the exact authorized content."""
    resolved = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, destination)
    if _content_digest(destination) != expected:
        raise RuntimeError(f"{source} did not match its authorized digest while staging")


def stage_model_ref(model: str, expected_artifact_id: str, *,
                    revision: str | None = None) -> StagedModel:
    """Create a private load path whose files cannot be replaced through the source cache paths."""
    pinned = (pinned_model_ref(model, expected_artifact_id, revision=revision)
              if revision is not None else pinned_model_ref(model, expected_artifact_id))
    if pinned is None:
        raise RuntimeError(f"cannot pin the authorized artifact for {model}")
    manifest = _authority_manifest(expected_artifact_id)
    if manifest is None:
        raise RuntimeError(f"cannot decode the authorized artifact manifest for {model}")
    source = Path(pinned)
    if not source.exists():
        raise RuntimeError(f"authorized artifact disappeared before staging: {source}")
    stage_parent = source.parent if source.is_file() else source.parent.parent
    try:
        stage_parent = stage_parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve the staging volume for {model}") from exc
    required = sum(size for size, _digest in manifest.values()) + _STAGE_DISK_BUFFER_BYTES
    try:
        free = shutil.disk_usage(stage_parent).free
    except OSError as exc:
        raise RuntimeError(f"cannot verify free disk space for staging {model}") from exc
    if free < required:
        raise RuntimeError(
            f"not enough disk to stage {model}: needs the artifact size plus 2 GB headroom")
    temporary = tempfile.TemporaryDirectory(prefix=".ara-stage-", dir=stage_parent)
    root = Path(temporary.name)
    try:
        if source.is_file():
            if len(manifest) != 1:
                raise RuntimeError(f"authorized manifest does not describe one file for {model}")
            destination = root / source.name
            _stage_file(source, destination, next(iter(manifest.values())))
            staged_path = destination
        else:
            if None in manifest:
                raise RuntimeError(f"cannot verify transformer snapshot {source}")
            staged_path = root / "snapshot"
            for relative, expected in manifest.items():
                item = source.joinpath(*PurePosixPath(relative).parts)
                _stage_file(item, staged_path / relative, expected)
        matches = (artifact_matches(model, expected_artifact_id, revision=revision)
                   if revision is not None else artifact_matches(model, expected_artifact_id))
        if not matches:
            raise RuntimeError(f"artifact changed while staging: {model}")
        return StagedModel(staged_path, temporary)
    except BaseException:
        temporary.cleanup()
        raise


def artifact_matches(model: str, expected_artifact_id: str | None, *,
                     revision: str | None = None) -> bool:
    """Whether *model* still resolves to the exact artifact that authorized stored evidence."""
    current = (artifact_identity(model, revision=revision)
               if revision is not None else artifact_identity(model))
    if (current != expected_artifact_id and revision is None
            and isinstance(expected_artifact_id, str)):
        prefix = ("hf:", "hf-gguf:")
        if expected_artifact_id.startswith(prefix):
            repo, separator, filename = model.partition(":")
            repo_id = repo if separator and filename.lower().endswith(".gguf") else model
            authority = expected_artifact_id.split(":", 2)[1]
            _, separator, encoded_revision = authority.rpartition("@")
            if separator and _REVISION_RE.fullmatch(encoded_revision):
                current = artifact_identity(model, revision=encoded_revision)
    return (isinstance(expected_artifact_id, str) and bool(expected_artifact_id)
            and current == expected_artifact_id)


def artifact_size_gb(model: str) -> float | None:
    """Exact selected GGUF size for cataloging a quant variant; otherwise unknown."""
    if not isinstance(model, str):
        return None
    local = Path(model).expanduser()
    if model.lower().endswith(".gguf") and local.is_file():
        try:
            return round(local.stat().st_size / 1e9, 3)
        except OSError:
            return None
    repo, separator, filename = model.partition(":")
    if not separator or not filename.lower().endswith(".gguf"):
        return None
    root = _cache_dir(repo)
    try:
        revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
        return round((root / "snapshots" / revision / filename).stat().st_size / 1e9, 3)
    except (OSError, UnicodeError):
        return None


def ceiling_is_stale(model_id: str, measured_at: str | None) -> bool:
    """True when *model_id*'s HF cache is newer than *measured_at* — the stored ceiling predates
    the current cached files and should be re-characterized.

    Conservative by design: a missing timestamp or an uncached/unknown model returns ``False`` (we
    never nag without evidence), and this is advisory only — callers warn, they do not block, since
    the measured ceiling is still the best number on record until a fresh ``ara characterize`` runs.
    """
    return fit_is_stale(model_id, measured_at)
