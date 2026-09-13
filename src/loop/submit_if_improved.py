"""Submit to Kaggle and git-commit only when a run beats the best keep score.

Typical flow (default gate is CV — repo convention: CV is ground truth):

1. Read candidate CV from ``exps/<exp>/metrics.json`` (or ledger / runs).
2. Compare to the current best keep (keep ``metrics.json``, RESULTS ``ok`` rows,
   else the exp0010 floor 0.94552). Higher ROC AUC wins.
3. If the candidate does not beat best + ε: print ``kill`` and exit 1.
   Do not submit. Do not commit.
4. If it improves: predict if needed, ``python -m ev_s6e9 submit``, poll public
   LB (best-effort), append LB into RESULTS / EXPERIMENTS, then ``git add``
   only allowed paths and commit ``loop: KEEP expNNNN CV=... (+submit)``.

``--gate lb`` submits first, then keeps only if public LB improves. ``--dry-run``
prints the decision without submit/commit. ``--push`` is off by default.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ev_s6e9.schema import COMPETITION
from loop.ledger import append_result, parse_result_rows, update_result_lb, write_run_json
from loop.models import RunResult

FLOOR_CV = 0.94552
FLOOR_LB = 0.94561
FLOOR_EXP = "exp0010"
FLOOR_STRATEGY = "s010"
DEFAULT_EPS = 1e-5
DEFAULT_POLL_S = 240.0
DEFAULT_POLL_INTERVAL_S = 12.0

EXIT_OK = 0
EXIT_KILL = 1
EXIT_ERROR = 2

_EXP_ID = re.compile(r"exp(\d+)", re.I)
_SCORE = re.compile(r"^\d+\.\d+$")

DENIED_NAMES = {
    ".env",
    ".env.local",
    "kaggle.json",
    "access_token",
    "oof.csv",
    "submission.csv",
}
DENIED_SUFFIXES = (".csv", ".joblib", ".pkl", ".pickle", ".lgb", ".png", ".svg")
DENIED_DIR_PARTS = {"data", "outputs", "logs", ".venv", "venv", "__pycache__", ".git"}
ALLOWED_TOP = {
    "exps",
    "ledger",
    "src",
    "scripts",
    "tests",
    "config",
    "docs",
    "reports",
    "prompts",
}
ALLOWED_ROOT_FILES = {
    "EXPERIMENTS.md",
    "LEARNINGS.md",
    "STRATEGY.md",
    "STRATEGIES.md",
    "CURSOR.md",
    "README.md",
    "DEOTTE_BRIEF.md",
    "TOP20_PUBLIC_NOTEBOOKS.md",
}


@dataclass(frozen=True)
class ScoreRef:
    """One scored reference: numeric value plus where it came from."""

    value: float
    source: str
    kind: str = "cv"


@dataclass(frozen=True)
class GateVerdict:
    """Keep/kill decision for one candidate vs the current best keep."""

    keep: bool
    metric: str
    candidate: float
    best: ScoreRef
    eps: float

    @property
    def delta(self) -> float:
        """Candidate minus best (positive means the new run is ahead)."""

        return self.candidate - self.best.value

    def summary(self) -> str:
        """One-line keep/kill message for stdout."""

        verb = "keep" if self.keep else "kill"
        return (
            f"{verb}: {self.metric}={self.candidate:.5f} vs best "
            f"{self.best.source} {self.best.kind}={self.best.value:.5f} "
            f"+ eps={self.eps:g} (delta={self.delta:+.6f})"
        )


def beats(candidate: float, best: float, eps: float = DEFAULT_EPS) -> bool:
    """True when ``candidate`` is a higher ROC AUC than ``best`` by more than ``eps``."""

    return candidate > best + eps


def decide(candidate: float, best: ScoreRef, *, eps: float = DEFAULT_EPS) -> GateVerdict:
    """Build a keep/kill verdict (higher ROC AUC wins)."""

    return GateVerdict(
        keep=beats(candidate, best.value, eps),
        metric=best.kind,
        candidate=candidate,
        best=best,
        eps=eps,
    )


def exp_to_strategy_id(exp_id: str) -> str:
    """Map ``exp0019`` → ``s019`` (strategy ids are 3-digit)."""

    match = _EXP_ID.search(exp_id)
    if not match:
        raise ValueError(f"not an exp id: {exp_id}")
    return f"s{int(match.group(1)):03d}"


def resolve_exp(root: Path, name: str) -> Path:
    """Resolve ``exp0010`` / ``exps/exp0010`` to a folder that has ``config.json``."""

    p = Path(name)
    if p.is_dir() and (p / "config.json").exists():
        return p.resolve()
    cand = root / "exps" / name
    if cand.is_dir() and (cand / "config.json").exists():
        return cand.resolve()
    if not name.startswith("exp"):
        cand = root / "exps" / f"exp{name}"
        if cand.is_dir() and (cand / "config.json").exists():
            return cand.resolve()
    raise FileNotFoundError(f"experiment not found: {name} (expected exps/expNNNN/config.json)")


def read_cv(payload: dict) -> float | None:
    """Pull a CV float from metrics/config/run JSON (``cv_mean`` or a ``cv`` string)."""

    if payload.get("cv_mean") is not None:
        try:
            return float(payload["cv_mean"])
        except (TypeError, ValueError):
            return None
    raw = payload.get("cv")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().split("±")[0].split()[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def read_lb(payload: dict) -> float | None:
    """Pull an LB float from metrics/run JSON when present."""

    raw = payload.get("lb")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict:
    """Read a JSON object; missing/invalid files become ``{}``."""

    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def candidate_cv(root: Path, exp_dir: Path, strategy_id: str) -> float:
    """Candidate CV: exp metrics, then config, then ledger run JSON, then RESULTS."""

    for path in (exp_dir / "metrics.json", exp_dir / "config.json"):
        value = read_cv(_load_json(path))
        if value is not None:
            return value
    value = read_cv(_load_json(root / "ledger" / "runs" / f"{strategy_id}.json"))
    if value is not None:
        return value
    for row in parse_result_rows(root / "ledger" / "RESULTS.md"):
        if row.strategy_id == strategy_id and row.cv is not None:
            return row.cv
    raise FileNotFoundError(f"no CV for {exp_dir.name} / {strategy_id}")


def best_keep_score(
    root: Path, *, kind: str = "cv", exclude: str | None = None
) -> ScoreRef:
    """Best keep score: keep ``metrics.json``, else RESULTS ``ok`` rows, else the floor.

    ``exclude`` is an exp id or strategy id so a just-recorded candidate is not
    compared against itself.
    """

    floor = FLOOR_CV if kind == "cv" else FLOOR_LB
    skip = {s for s in (exclude, _other_id(exclude)) if s}
    best: ScoreRef | None = None
    exps = root / "exps"
    if exps.is_dir():
        for exp_dir in sorted(p for p in exps.iterdir() if p.is_dir()):
            if exp_dir.name in skip:
                continue
            metrics = _load_json(exp_dir / "metrics.json")
            cfg = _load_json(exp_dir / "config.json")
            status = str(metrics.get("status") or cfg.get("status") or "").lower()
            if status != "keep":
                continue
            value = read_cv(metrics) if kind == "cv" else read_lb(metrics)
            if value is None and kind == "cv":
                value = read_cv(cfg)
            if value is None:
                continue
            if best is None or value > best.value:
                best = ScoreRef(value, exp_dir.name, kind)
    for row in parse_result_rows(root / "ledger" / "RESULTS.md"):
        if row.strategy_id in skip or row.status != "ok":
            continue
        value = row.cv if kind == "cv" else row.lb
        if value is None:
            continue
        if best is None or value > best.value:
            best = ScoreRef(value, row.strategy_id, kind)
    return best or ScoreRef(floor, f"{FLOOR_EXP} floor", kind)


def _other_id(ident: str | None) -> str | None:
    """Map ``exp0019`` ↔ ``s019`` so exclude matches both ledgers and exp folders."""

    if not ident:
        return None
    if ident.startswith("exp"):
        try:
            return exp_to_strategy_id(ident)
        except ValueError:
            return None
    if ident.startswith("s") and ident[1:].isdigit():
        return f"exp{int(ident[1:]):04d}"
    return None


def is_allowed_git_path(rel: str) -> bool:
    """True if ``rel`` may be ``git add``-ed (never data CSVs, OOF, models, secrets)."""

    rel = rel.replace("\\", "/").lstrip("./")
    if not rel or rel.endswith("/"):
        return False
    name = Path(rel).name
    if name in DENIED_NAMES or name.startswith(".env"):
        return False
    if name.endswith(DENIED_SUFFIXES):
        return False
    parts = Path(rel).parts
    if any(part in DENIED_DIR_PARTS for part in parts):
        return False
    if len(parts) == 1:
        return name in ALLOWED_ROOT_FILES
    return parts[0] in ALLOWED_TOP


def filter_git_paths(paths: list[str]) -> list[str]:
    """Keep only paths the submit-if-improved commit is allowed to stage."""

    return [p for p in paths if is_allowed_git_path(p)]


def parse_public_score(row: dict) -> float | None:
    """Read a numeric public LB from a ``kaggle competitions submissions -v`` row."""

    for key in ("publicScore", "PublicScore", "score", "Score"):
        raw = str(row.get(key) or "").strip()
        if _SCORE.match(raw):
            return float(raw)
    return None


def parse_submissions_csv(text: str) -> list[dict]:
    """Parse verbose Kaggle submissions CSV into row dicts."""

    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


def pick_submission(rows: list[dict], *, needle: str | None = None) -> dict | None:
    """Prefer a row whose description contains ``needle``; else the first (latest) row."""

    if not rows:
        return None
    if needle:
        low = needle.lower()
        for row in rows:
            blob = " ".join(str(row.get(k) or "") for k in row).lower()
            if low in blob:
                return row
    return rows[0]


def poll_public_lb(
    *,
    competition: str = COMPETITION,
    needle: str | None = None,
    timeout_s: float = DEFAULT_POLL_S,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    run=subprocess.run,
    sleep=time.sleep,
    clock=time.monotonic,
    env: dict[str, str] | None = None,
) -> float | None:
    """Poll ``kaggle competitions submissions`` until a public score or timeout.

    Returns the score, or ``None`` if scoring is still pending / CLI failed.
    ``run`` / ``sleep`` / ``clock`` are injectable so tests never hit Kaggle.
    """

    deadline = clock() + timeout_s
    while clock() < deadline:
        proc = run(
            ["kaggle", "competitions", "submissions", "-c", competition, "-v"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode == 0:
            row = pick_submission(parse_submissions_csv(proc.stdout or ""), needle=needle)
            if row is not None:
                score = parse_public_score(row)
                if score is not None:
                    return score
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(interval_s, remaining))
    return None


def kaggle_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Copy ``base`` (or ``os.environ``) and attach ``~/.kaggle/access_token`` when present."""

    env = dict(base if base is not None else os.environ)
    token = Path.home() / ".kaggle" / "access_token"
    if token.exists():
        env["KAGGLE_API_TOKEN"] = token.read_text(encoding="utf-8").strip()
    return env


def py_bin(root: Path) -> str:
    """Prefer the repo ``.venv`` interpreter when present."""

    venv = root / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def ensure_predictions(
    root: Path,
    exp_dir: Path,
    *,
    skip: bool = False,
    force: bool = False,
    dry_run: bool = False,
    run=subprocess.run,
) -> Path:
    """Write ``outputs/submission.csv`` via ``python -m ev_s6e9 predict`` if needed."""

    sub = root / "outputs" / "submission.csv"
    if skip and not sub.exists():
        raise FileNotFoundError(f"{sub} missing; run predict or omit --skip-predict")
    # Guard against stale submissions: if the strategy's OOF (model output) is NEWER
    # than the existing submission.csv, the submission is stale and MUST be regenerated.
    # (Without this, a submit after training a new strategy reuses an old submission file.)
    oof = exp_dir / "oof.csv"
    if sub.exists() and not force:
        if oof.exists() and oof.stat().st_mtime > sub.stat().st_mtime:
            print(f"stale submission detected (oof {oof.name} newer than submission.csv) — regenerating", flush=True)
            force = True
        else:
            return sub
    cfg = _load_json(exp_dir / "config.json")
    args = list(cfg.get("predict_args") or [])
    if not args and cfg.get("strategy"):
        args = ["--strategy", str(cfg["strategy"])]
    cmd = [py_bin(root), "-m", "ev_s6e9", "predict", *args]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return sub
    env = os.environ.copy()
    env["EV_S6E9_ROOT"] = str(root)
    proc = run(cmd, cwd=str(root), env=env, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"predict failed (rc={proc.returncode})")
    if not sub.exists():
        raise FileNotFoundError(f"predict did not write {sub}")
    return sub


def submit_kaggle(
    root: Path,
    message: str,
    *,
    dry_run: bool = False,
    run=subprocess.run,
) -> None:
    """Call ``python -m ev_s6e9 submit`` with a message that includes exp id + CV."""

    cmd = [py_bin(root), "-m", "ev_s6e9", "submit", "-m", message]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    env = kaggle_env()
    env["EV_S6E9_ROOT"] = str(root)
    proc = run(cmd, cwd=str(root), env=env, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"submit failed (rc={proc.returncode})")


def porcelain_paths(root: Path, *, run=subprocess.run) -> list[str]:
    """Changed / untracked paths from ``git status --porcelain -u``."""

    proc = run(
        ["git", "status", "--porcelain", "-u"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    paths: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip().split(" -> ")[-1]
        if rel:
            paths.append(rel)
    return paths


def stage_allowed(root: Path, extra: list[str] | None = None, *, run=subprocess.run) -> list[str]:
    """``git add`` only allow-listed paths; never data CSVs, OOF, joblib, or secrets."""

    candidates = list(extra or [])
    candidates.extend(porcelain_paths(root, run=run))
    staged: list[str] = []
    seen: set[str] = set()
    for rel in candidates:
        rel = rel.replace("\\", "/").lstrip("./")
        if rel in seen or not is_allowed_git_path(rel):
            continue
        if not (root / rel).exists():
            continue
        seen.add(rel)
        proc = run(["git", "add", "--", rel], cwd=str(root), text=True, check=False)
        if proc.returncode == 0:
            staged.append(rel)
    return staged


def git_commit(
    root: Path,
    message: str,
    *,
    extra: list[str] | None = None,
    dry_run: bool = False,
    push: bool = False,
    run=subprocess.run,
) -> list[str]:
    """Stage allowed paths, commit ``message``, optionally ``git push``."""

    if dry_run:
        print(f"dry-run: would commit {message!r}", flush=True)
        return []
    staged = stage_allowed(root, extra, run=run)
    if not staged:
        print("nothing allowed to commit", flush=True)
        return []
    proc = run(["git", "commit", "-m", message], cwd=str(root), text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git commit failed (rc={proc.returncode})")
    if push:
        proc = run(["git", "push"], cwd=str(root), text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"git push failed (rc={proc.returncode})")
    return staged


def mark_keep(exp_dir: Path, *, lb: float | None = None) -> None:
    """Set ``status=keep`` on metrics/config (and record LB when we have it)."""

    for name in ("metrics.json", "config.json"):
        path = exp_dir / name
        data = _load_json(path)
        if not data and not path.exists():
            continue
        data["status"] = "keep"
        if lb is not None:
            data["lb"] = lb
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def record_lb(
    root: Path,
    *,
    exp_id: str,
    strategy_id: str,
    cv: float,
    lb: float | None,
    notes: str,
) -> None:
    """Write LB into RESULTS (update or append), EXPERIMENTS.md, and runs JSON."""

    results = root / "ledger" / "RESULTS.md"
    if lb is not None and update_result_lb(results, strategy_id, lb):
        pass
    elif not any(r.strategy_id == strategy_id for r in parse_result_rows(results)):
        append_result(
            results,
            RunResult(
                strategy_id=strategy_id,
                status="ok",
                cv=cv,
                lb=lb,
                notes=notes,
            ),
        )
    elif lb is None:
        pass
    lb_cell = f"{lb:.5f}" if lb is not None else "— (pending)"
    chunk = (
        f"### {date.today().isoformat()} — {exp_id} submit\n"
        f"- CV: {cv:.5f}\n"
        f"- LB: {lb_cell}\n"
        f"- Takeaway: {notes}\n"
    )
    experiments = root / "EXPERIMENTS.md"
    if experiments.exists():
        text = experiments.read_text(encoding="utf-8")
        prefix = "" if text.endswith("\n") else "\n"
        experiments.write_text(text + prefix + chunk + "\n", encoding="utf-8")
    else:
        experiments.write_text(chunk + "\n", encoding="utf-8")
    run_path = root / "ledger" / "runs" / f"{strategy_id}.json"
    existing = _load_json(run_path)
    write_run_json(
        run_path,
        RunResult(
            strategy_id=strategy_id,
            status=str(existing.get("status") or "ok"),
            cv=read_cv(existing) or cv,
            lb=lb if lb is not None else read_lb(existing),
            notes=str(existing.get("notes") or notes),
            phase=str(existing.get("phase") or ""),
        ),
    )


def submit_message(
    exp_id: str, strategy_id: str, cv: float, cfg: dict, override: str | None
) -> str:
    """Kaggle `-m` text: override, else config `submit_message` plus exp/CV, else ids."""

    if override:
        return override
    base = cfg.get("submit_message")
    if base:
        return f"{base} {exp_id} CV={cv:.5f}"
    return f"{exp_id} {strategy_id} CV={cv:.5f}"


def commit_paths_for(exp_id: str) -> list[str]:
    """Ledger + exp files this keep is expected to touch."""

    return [
        f"exps/{exp_id}/config.json",
        f"exps/{exp_id}/NOTES.md",
        f"exps/{exp_id}/metrics.json",
        f"exps/{exp_id}/cv.json",
        "ledger/RESULTS.md",
        "ledger/STRATEGIES.md",
        "EXPERIMENTS.md",
        "LEARNINGS.md",
        "STRATEGY.md",
        "reports/index.md",
    ]


def run(
    root: Path,
    exp: str,
    *,
    gate: str = "cv",
    eps: float = DEFAULT_EPS,
    dry_run: bool = False,
    push: bool = False,
    skip_predict: bool = False,
    force_predict: bool = False,
    poll_timeout: float = DEFAULT_POLL_S,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    message: str | None = None,
    run_cmd=subprocess.run,
    sleep=time.sleep,
    clock=time.monotonic,
) -> int:
    """Execute the gate. Return 0 on keep, 1 on kill, 2 on setup/submit/git errors."""

    try:
        exp_dir = resolve_exp(root, exp)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR
    exp_id = exp_dir.name
    try:
        strategy_id = exp_to_strategy_id(exp_id)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR

    try:
        cv = candidate_cv(root, exp_dir, strategy_id)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR

    if gate == "cv":
        verdict = decide(cv, best_keep_score(root, kind="cv", exclude=exp_id), eps=eps)
        print(verdict.summary(), flush=True)
        if not verdict.keep:
            return EXIT_KILL
    else:
        best_lb = best_keep_score(root, kind="lb", exclude=exp_id)
        print(
            f"gate=lb: will submit {exp_id} (CV={cv:.5f}) then compare public LB "
            f"to {best_lb.source} {best_lb.value:.5f} + eps={eps:g}",
            flush=True,
        )

    submit_msg = submit_message(
        exp_id, strategy_id, cv, _load_json(exp_dir / "config.json"), message
    )

    if dry_run and gate == "cv":
        print(f"dry-run: would submit -m {submit_msg!r}", flush=True)
        print(f"dry-run: would commit 'loop: KEEP {exp_id} CV={cv:.5f} (+submit)'", flush=True)
        if push:
            print("dry-run: would git push", flush=True)
        return EXIT_OK
    if dry_run and gate == "lb":
        print(f"dry-run: would submit -m {submit_msg!r} then gate on LB", flush=True)
        return EXIT_OK

    try:
        ensure_predictions(
            root,
            exp_dir,
            skip=skip_predict,
            force=force_predict,
            dry_run=False,
            run=run_cmd,
        )
        submit_kaggle(root, submit_msg, dry_run=False, run=run_cmd)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR

    lb = poll_public_lb(
        needle=exp_id,
        timeout_s=poll_timeout,
        interval_s=poll_interval,
        run=run_cmd,
        sleep=sleep,
        clock=clock,
        env=kaggle_env(),
    )
    if lb is None:
        print("LB pending (poll timed out); not blocking", flush=True)
    else:
        print(f"LB {lb:.5f}", flush=True)

    if gate == "lb":
        if lb is None:
            print("kill: LB pending — cannot confirm improve; not committing", flush=True)
            return EXIT_KILL
        verdict = decide(lb, best_keep_score(root, kind="lb", exclude=exp_id), eps=eps)
        print(verdict.summary(), flush=True)
        if not verdict.keep:
            return EXIT_KILL

    lb_note = f" LB={lb:.5f}" if lb is not None else " LB pending"
    notes = f"{exp_id} submitted CV={cv:.5f}{lb_note}"
    mark_keep(exp_dir, lb=lb)
    record_lb(root, exp_id=exp_id, strategy_id=strategy_id, cv=cv, lb=lb, notes=notes)
    commit_msg = f"loop: KEEP {exp_id} CV={cv:.5f}"
    if lb is not None:
        commit_msg += f" LB={lb:.5f}"
    commit_msg += " (+submit)"
    try:
        git_commit(
            root,
            commit_msg,
            extra=commit_paths_for(exp_id),
            dry_run=False,
            push=push,
            run=run_cmd,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR
    print(commit_msg, flush=True)
    return EXIT_OK


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach submit-if-improved flags to ``parser``."""

    parser.add_argument("exp", help="exp id or path (e.g. exp0019)")
    parser.add_argument("--root", default=None, help="Repo root (default: cwd / $LOOP_ROOT).")
    parser.add_argument(
        "--gate",
        choices=("cv", "lb"),
        default="cv",
        help="Decision metric. Default cv (submit only on CV personal best).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help=f"Minimum lift vs best keep (default {DEFAULT_EPS:g}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print keep/kill (and would-submit/commit) without calling Kaggle or git.",
    )
    parser.add_argument("--push", action="store_true", help="git push after a keep commit (off).")
    parser.add_argument("--skip-predict", action="store_true", help="Do not run predict.")
    parser.add_argument("--force-predict", action="store_true", help="Always run predict.")
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_S)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("-m", "--message", default=None, help="Override Kaggle submit message.")
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    """Dispatch a parsed argparse namespace to :func:`run`."""

    from loop.config import find_root

    root = find_root(getattr(args, "root", None))
    return run(
        root,
        args.exp,
        gate=args.gate,
        eps=args.eps,
        dry_run=args.dry_run,
        push=args.push,
        skip_predict=args.skip_predict,
        force_predict=args.force_predict,
        poll_timeout=args.poll_timeout,
        poll_interval=args.poll_interval,
        message=args.message,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m loop submit-if-improved`` and the scripts/ wrapper."""

    parser = argparse.ArgumentParser(
        prog="submit-if-improved",
        description="Submit to Kaggle and git-commit only when the run improves the score.",
    )
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))
