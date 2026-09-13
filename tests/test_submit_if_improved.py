"""CV/LB gate, allow-listed git paths, and dry-run submit-if-improved (no Kaggle)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.cli import main as loop_main
from loop.ledger import parse_result_rows, update_result_lb
from loop.submit_if_improved import (
    DEFAULT_EPS,
    EXIT_KILL,
    EXIT_OK,
    FLOOR_CV,
    FLOOR_EXP,
    ScoreRef,
    beats,
    best_keep_score,
    candidate_cv,
    decide,
    exp_to_strategy_id,
    filter_git_paths,
    is_allowed_git_path,
    parse_public_score,
    parse_submissions_csv,
    pick_submission,
    poll_public_lb,
    run,
    submit_message,
)


def _write_exp(root: Path, exp_id: str, *, cv: float, status: str, lb: float | None = None) -> Path:
    """Create a minimal `exps/expNNNN/{config,metrics}.json` tree."""

    exp = root / "exps" / exp_id
    exp.mkdir(parents=True, exist_ok=True)
    metrics = {"id": exp_id, "cv_mean": cv, "cv": f"{cv:.5f}", "status": status}
    if lb is not None:
        metrics["lb"] = lb
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    cfg = {
        "id": exp_id,
        "strategy": "deotte",
        "predict_args": ["--strategy", "deotte"],
        "submit_message": f"{exp_id} deotte",
        "status": status,
        "cv_mean": cv,
    }
    (exp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (exp / "NOTES.md").write_text(f"# {exp_id}\n", encoding="utf-8")
    return exp


def _write_results(root: Path, rows: str) -> Path:
    """Write a tiny RESULTS.md table."""

    path = root / "ledger" / "RESULTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Results\n\n| id | status | cv | lb | notes |\n|----|--------|----|----|-------|\n"
    path.write_text(header + rows, encoding="utf-8")
    return path


def _boom(*_a, **_k):
    """Fail if predict/submit/git/kaggle would have been invoked."""

    raise AssertionError("subprocess should not run on dry-run / unit paths")


def test_beats_and_decide_eps():
    assert beats(0.94553, 0.94552, eps=0) is True
    assert beats(0.94552, 0.94552, eps=0) is False
    assert beats(0.94553, 0.94552, eps=1e-5) is False
    assert beats(0.94554, 0.94552, eps=1e-5) is True
    verdict = decide(0.94, ScoreRef(FLOOR_CV, FLOOR_EXP, "cv"), eps=DEFAULT_EPS)
    assert verdict.keep is False
    assert "kill" in verdict.summary()


def test_exp_to_strategy_id():
    assert exp_to_strategy_id("exp0019") == "s019"
    assert exp_to_strategy_id("exp0010") == "s010"
    assert exp_to_strategy_id("exps/exp0001") == "s001"


def test_best_keep_from_keep_metrics_ignores_kill(tmp_path: Path):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep", lb=0.94561)
    _write_exp(tmp_path, "exp0012", cv=0.94559, status="kill")
    _write_results(tmp_path, "| s012 | fail | 0.94559 | — | noise |\n")
    best = best_keep_score(tmp_path, kind="cv")
    assert best.source == "exp0010"
    assert best.value == pytest.approx(0.94552)
    assert best_keep_score(tmp_path, kind="lb").value == pytest.approx(0.94561)


def test_best_keep_excludes_candidate_row(tmp_path: Path):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep")
    _write_results(
        tmp_path,
        "| s010 | ok | 0.94552 | 0.94561 | floor |\n| s019 | ok | 0.94610 | — | self |\n",
    )
    best = best_keep_score(tmp_path, kind="cv", exclude="exp0019")
    assert best.source == "exp0010"
    assert best.value == pytest.approx(0.94552)


def test_best_keep_falls_back_to_results_then_floor(tmp_path: Path):
    empty = best_keep_score(tmp_path, kind="cv")
    assert empty.value == pytest.approx(FLOOR_CV)
    assert "floor" in empty.source
    _write_results(tmp_path, "| s010 | ok | 0.94552 | 0.94561 | KEEP floor |\n")
    from_ledger = best_keep_score(tmp_path, kind="cv")
    assert from_ledger.value == pytest.approx(0.94552)
    assert from_ledger.source == "s010"


def test_candidate_cv_from_metrics_and_ledger(tmp_path: Path):
    exp = _write_exp(tmp_path, "exp0019", cv=0.94600, status="scored")
    assert candidate_cv(tmp_path, exp, "s019") == pytest.approx(0.94600)
    (exp / "metrics.json").unlink()
    (exp / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ledger" / "runs").mkdir(parents=True)
    (tmp_path / "ledger" / "runs" / "s019.json").write_text(
        json.dumps({"id": "s019", "cv": 0.9444}),
        encoding="utf-8",
    )
    assert candidate_cv(tmp_path, exp, "s019") == pytest.approx(0.9444)


def test_allowed_git_paths():
    assert is_allowed_git_path("exps/exp0019/metrics.json")
    assert is_allowed_git_path("exps/exp0019/config.json")
    assert is_allowed_git_path("exps/exp0019/NOTES.md")
    assert is_allowed_git_path("ledger/RESULTS.md")
    assert is_allowed_git_path("src/ev_s6e9/deotte.py")
    assert is_allowed_git_path("EXPERIMENTS.md")
    assert not is_allowed_git_path("exps/exp0019/oof.csv")
    assert not is_allowed_git_path("outputs/submission.csv")
    assert not is_allowed_git_path("data/raw/train.csv")
    assert not is_allowed_git_path("outputs/models/fold0.joblib")
    assert not is_allowed_git_path(".env")
    assert not is_allowed_git_path("kaggle.json")
    kept = filter_git_paths(
        [
            "exps/exp0019/metrics.json",
            "exps/exp0019/oof.csv",
            "data/raw/train.csv",
            ".env",
            "src/loop/cli.py",
        ]
    )
    assert kept == ["exps/exp0019/metrics.json", "src/loop/cli.py"]


def test_poll_lb_parses_csv_and_times_out_without_kaggle():
    rows = parse_submissions_csv(
        "fileName,date,description,status,publicScore\n"
        "sub.csv,2026-09-12,exp0019 s019 CV=0.94600,complete,0.94610\n"
    )
    assert parse_public_score(rows[0]) == pytest.approx(0.94610)
    assert pick_submission(rows, needle="exp0019")["description"].startswith("exp0019")

    calls = {"n": 0}

    def fake_run(*_a, **_k):
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stdout="fileName,publicScore\nsub.csv,\n")

    times = iter([0.0, 0.0, 10.0, 10.0, 999.0])
    slept: list[float] = []
    score = poll_public_lb(
        timeout_s=5,
        interval_s=10,
        run=fake_run,
        sleep=slept.append,
        clock=lambda: next(times),
    )
    assert score is None
    assert calls["n"] >= 1


def test_poll_lb_returns_score_on_second_tick():
    replies = [
        SimpleNamespace(returncode=0, stdout="fileName,publicScore\nsub.csv,\n"),
        SimpleNamespace(
            returncode=0,
            stdout="fileName,description,publicScore\nsub.csv,exp0019,0.94620\n",
        ),
    ]

    def fake_run(*_a, **_k):
        return replies.pop(0)

    times = iter([0.0, 0.0, 1.0, 1.0, 2.0])
    score = poll_public_lb(
        needle="exp0019",
        timeout_s=30,
        interval_s=1,
        run=fake_run,
        sleep=lambda _s: None,
        clock=lambda: next(times),
    )
    assert score == pytest.approx(0.94620)


def test_dry_run_kill_does_not_subprocess(tmp_path: Path, capsys):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep")
    _write_exp(tmp_path, "exp0019", cv=0.94552, status="scored")
    rc = run(tmp_path, "exp0019", dry_run=True, run_cmd=_boom)
    out = capsys.readouterr().out
    assert rc == EXIT_KILL
    assert "kill" in out
    assert "would submit" not in out


def test_dry_run_keep_prints_without_submit(tmp_path: Path, capsys):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep")
    _write_exp(tmp_path, "exp0019", cv=0.94610, status="scored")
    rc = run(tmp_path, "exp0019", dry_run=True, run_cmd=_boom)
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "keep" in out
    assert "would submit" in out
    assert "would commit" in out
    assert "loop: KEEP exp0019" in out


def test_dry_run_gate_lb_does_not_submit(tmp_path: Path, capsys):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep", lb=0.94561)
    _write_exp(tmp_path, "exp0019", cv=0.94610, status="scored")
    rc = run(tmp_path, "exp0019", gate="lb", dry_run=True, run_cmd=_boom)
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "gate=lb" in out
    assert "would submit" in out
    assert "then gate on LB" in out


def test_keep_mocked_submit_commit_and_lb(tmp_path: Path, capsys):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep")
    _write_exp(tmp_path, "exp0019", cv=0.94610, status="scored")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "submission.csv").write_text("id,Will_Buy_EV\n1,0.5\n")
    (tmp_path / "EXPERIMENTS.md").write_text("# Experiments\n\n", encoding="utf-8")
    _write_results(tmp_path, "| s019 | ok | 0.94610 | — | pending |\n")
    cmds: list[list[str]] = []

    def fake_run(argv, **_k):
        cmds.append(list(argv))
        if argv[:3] == ["kaggle", "competitions", "submissions"]:
            return SimpleNamespace(
                returncode=0,
                stdout="fileName,description,publicScore\nsub.csv,exp0019,0.94625\n",
            )
        if argv[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M exps/exp0019/metrics.json\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    rc = run(
        tmp_path,
        "exp0019",
        skip_predict=True,
        poll_timeout=1,
        poll_interval=0,
        run_cmd=fake_run,
        sleep=lambda _s: None,
        clock=lambda: 0.0,
    )
    assert rc == EXIT_OK
    assert any("submit" in c for c in cmds)
    assert any(c[:2] == ["git", "commit"] for c in cmds)
    assert not any(c[:2] == ["git", "push"] for c in cmds)
    rows = parse_result_rows(tmp_path / "ledger" / "RESULTS.md")
    assert rows[0].lb == pytest.approx(0.94625)
    text = (tmp_path / "EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "0.94625" in text
    metrics = json.loads((tmp_path / "exps" / "exp0019" / "metrics.json").read_text())
    assert metrics["status"] == "keep"
    assert "KEEP exp0019" in capsys.readouterr().out


def test_gate_lb_kills_without_commit(tmp_path: Path):
    _write_exp(tmp_path, "exp0010", cv=0.94552, status="keep", lb=0.94561)
    _write_exp(tmp_path, "exp0019", cv=0.94610, status="scored")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "submission.csv").write_text("id,Will_Buy_EV\n1,0.5\n")
    commits: list[list[str]] = []

    def fake_run(argv, **_k):
        if argv[:2] == ["git", "commit"]:
            commits.append(list(argv))
        if argv[:3] == ["kaggle", "competitions", "submissions"]:
            return SimpleNamespace(
                returncode=0,
                stdout="fileName,description,publicScore\nsub.csv,exp0019,0.94560\n",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    rc = run(
        tmp_path,
        "exp0019",
        gate="lb",
        skip_predict=True,
        poll_timeout=1,
        poll_interval=0,
        run_cmd=fake_run,
        sleep=lambda _s: None,
        clock=lambda: 0.0,
    )
    assert rc == EXIT_KILL
    assert commits == []


def test_update_result_lb(tmp_path: Path):
    path = _write_results(tmp_path, "| s019 | ok | 0.94610 | — | pending |\n")
    assert update_result_lb(path, "s019", 0.94625) is True
    row = parse_result_rows(path)[0]
    assert row.lb == pytest.approx(0.94625)
    assert row.cv == pytest.approx(0.94610)
    assert row.notes == "pending"


def test_submit_message_prefers_override():
    cfg = {"submit_message": "exp0019 deotte"}
    assert submit_message("exp0019", "s019", 0.946, cfg, "custom") == "custom"
    assert "CV=0.94600" in submit_message("exp0019", "s019", 0.946, cfg, None)
    assert submit_message("exp0019", "s019", 0.946, {}, None) == "exp0019 s019 CV=0.94600"


def test_cli_submit_if_improved_dry_run(tmp_path: Path, capsys):
    _write_exp(tmp_path, "exp0019", cv=0.94, status="scored")
    assert loop_main(["submit-if-improved", "exp0019", "--dry-run", "--root", str(tmp_path)]) == 1
    assert "kill" in capsys.readouterr().out


def test_stale_submission_regenerated_when_oof_newer(tmp_path: Path, capsys):
    """A stale submission.csv (older than the strategy's OOF) must be regenerated,
    not reused. Regression for the 0.88273-on-LB incident where a submit reused a
    submission file left over from an earlier strategy."""
    _write_exp(tmp_path, "exp0019", cv=0.94610, status="scored")
    (tmp_path / "outputs").mkdir(exist_ok=True)
    sub = tmp_path / "outputs" / "submission.csv"
    sub.write_text("id,Will_Buy_EV\n1,0.5\n")
    oof = tmp_path / "exps" / "exp0019" / "oof.csv"
    oof.write_text("id,Will_Buy_EV\n1,0.5\n")
    # make the OOF strictly newer than the stale submission
    sub.touch(); old = sub.stat().st_mtime
    oof.touch(); oof.stat()  # oof now has a later mtime than sub

    predict_cmds: list[list[str]] = []

    def fake_run(argv, **_k):
        if "ev_s6e9" in argv and "predict" in argv:
            predict_cmds.append(list(argv))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    from loop.submit_if_improved import ensure_predictions
    out = ensure_predictions(tmp_path, tmp_path / "exps" / "exp0019",
                             skip=False, force=False, dry_run=False, run=fake_run)
    assert predict_cmds, "predict should have been re-run because the OOF is newer than submission.csv"
    assert out == sub
