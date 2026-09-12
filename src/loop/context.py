"""Assemble the planner prompt from the whitelist only (byte-capped, deny-first)."""

from __future__ import annotations

from pathlib import Path

import yaml

from loop.models import AssembledContext, FileSlice

ALWAYS_DENY_PREFIXES = ("logs/", "ledger/runs/", ".git/", "data/")
# Competition dumps / OOF blobs — never inline, even if a whitelist path matches.
ALWAYS_DENY_NAMES = frozenset(
    {
        "oof.csv",
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "submission.csv",
    }
)


def _norm(rel: str) -> str:
    """Normalize a relative path for prefix / basename deny checks."""
    return rel.replace("\\", "/").lstrip("./")


def is_denied(rel: str, deny: list[str]) -> bool:
    """True if `rel` is a dump (oof/csv), a deny prefix, or a listed deny path."""

    path = _norm(rel)
    if Path(path).name in ALWAYS_DENY_NAMES:
        return True
    prefixes = list(ALWAYS_DENY_PREFIXES) + [_norm(item) for item in deny]
    for prefix in prefixes:
        if not prefix:
            continue
        stem = prefix if prefix.endswith("/") else prefix + "/"
        if path == prefix.rstrip("/") or path.startswith(stem):
            return True
        if prefix.rstrip("/") in path.split("/"):
            # deny a path segment such as "logs" anywhere in the rel path
            if prefix.rstrip("/") in {"logs", ".git"}:
                return True
    return False


def _load_reads(path: Path) -> dict:
    """Parse planner_reads.yaml (`always` / `optional` / `deny` / byte caps)."""

    if not path.is_file():
        return {"always": [], "optional": [], "deny": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_candidates(rel: str, *, loop_root: Path, competition_root: Path | None) -> list[Path]:
    """Absolute paths to try for `rel` (competition root first, then loop root)."""

    rel_path = Path(_norm(rel))
    out: list[Path] = []
    if competition_root is not None:
        out.append((competition_root / rel_path).resolve())
    out.append((loop_root / rel_path).resolve())
    # de-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def assemble(
    *,
    loop_root: Path,
    reads_path: Path,
    competition_root: Path | None = None,
    max_bytes_per_file: int = 24000,
    max_total_bytes: int = 80000,
) -> AssembledContext:
    """Inline whitelist files into slices; skip denied/missing; honor byte caps."""

    spec = _load_reads(reads_path)
    deny = [str(x) for x in (spec.get("deny") or [])]
    always = [str(x) for x in (spec.get("always") or [])]
    optional = [str(x) for x in (spec.get("optional") or [])]
    max_bytes_per_file = int(spec.get("max_bytes_per_file") or max_bytes_per_file)
    max_total_bytes = int(spec.get("max_total_bytes") or max_total_bytes)

    ctx = AssembledContext()
    for rel in always + optional:
        if is_denied(rel, deny):
            ctx.skipped.append(f"{rel} (denied)")
            continue
        ctx.listing.append(rel)
        chosen: Path | None = None
        candidates = resolve_candidates(
            rel, loop_root=loop_root, competition_root=competition_root
        )
        for candidate in candidates:
            if candidate.is_file():
                chosen = candidate
                break
        if chosen is None:
            if rel in optional:
                ctx.skipped.append(f"{rel} (missing, optional)")
            else:
                ctx.slices.append(FileSlice(rel=rel, text="", missing=True))
                ctx.skipped.append(f"{rel} (missing)")
            continue
        if ctx.total_bytes >= max_total_bytes:
            ctx.skipped.append(f"{rel} (total budget)")
            continue
        data = chosen.read_bytes()
        truncated = False
        budget = min(max_bytes_per_file, max_total_bytes - ctx.total_bytes)
        if len(data) > budget:
            data = data[:budget]
            truncated = True
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text = text + "\n… [truncated by planner byte cap]\n"
        ctx.slices.append(FileSlice(rel=rel, text=text, truncated=truncated))
        ctx.total_bytes += len(data)
    return ctx
