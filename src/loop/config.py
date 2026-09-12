"""Load `config/loop.yaml` into Settings (paths, adapters, stop conditions)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from loop.env import expand, load_dotenv


def find_root(cli_root: str | None = None) -> Path:
    """Resolve the loop repo: `--root`, `$LOOP_ROOT`, or cwd with `config/loop.yaml`."""

    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env_root = __import__("os").environ.get("LOOP_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "config" / "loop.yaml").is_file():
        return cwd.resolve()
    return cwd.resolve()


@dataclass
class Settings:
    """Typed view over loop.yaml plus the resolved repo `root`."""

    root: Path
    raw: dict

    @property
    def competition_name(self) -> str:
        """Competition slug shown in planner/executor prompts."""

        return str(self.raw.get("competition", {}).get("name") or "competition")

    @property
    def competition_root(self) -> Path | None:
        """Checkout the executor trains in (`.` in this instance)."""

        value = (self.raw.get("competition") or {}).get("root") or ""
        if not str(value).strip():
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return path if path.exists() else path

    @property
    def metric(self) -> str:
        """Primary CV metric name (roc_auc here)."""

        return str((self.raw.get("competition") or {}).get("metric") or "cv")

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger CV beats the previous best."""

        return bool((self.raw.get("competition") or {}).get("higher_is_better", True))

    @property
    def ledger_dir(self) -> Path:
        """Directory of STRATEGIES / RESULTS / CURRENT_STRATEGY."""

        return self.root / (self.raw.get("paths") or {}).get("ledger_dir", "ledger")

    @property
    def logs_dir(self) -> Path:
        """Long traces (`logs/<id>.log`); never planner-visible."""

        return self.root / (self.raw.get("paths") or {}).get("logs_dir", "logs")

    @property
    def runs_dir(self) -> Path:
        """Metrics-only JSON per run (`ledger/runs/<id>.json`)."""

        return self.root / (self.raw.get("paths") or {}).get("runs_dir", "ledger/runs")

    @property
    def planner_reads_path(self) -> Path:
        """Whitelist YAML that decides what the planner may see."""

        rel = (self.raw.get("paths") or {}).get("planner_reads", "config/planner_reads.yaml")
        return self.root / rel

    @property
    def planner_prompt(self) -> Path:
        """Markdown template filled before each plan call."""

        rel = (self.raw.get("planner") or {}).get("prompt", "prompts/planner.md")
        return self.root / rel

    @property
    def executor_prompt(self) -> Path:
        """Markdown template filled before each execute call."""

        rel = (self.raw.get("executor") or {}).get("prompt", "prompts/executor.md")
        return self.root / rel

    @property
    def max_iterations(self) -> int:
        """Default `loop run` count when `--iterations` is omitted."""

        return int((self.raw.get("loop") or {}).get("max_iterations") or 10)

    @property
    def target_cv(self) -> float | None:
        """Stop `run` early when an ok result meets this CV (or None)."""

        value = (self.raw.get("loop") or {}).get("target_cv")
        return float(value) if value is not None and value != "" else None

    @property
    def max_consecutive_failures(self) -> int:
        """Stop `run` after this many fails in a row."""

        return int((self.raw.get("loop") or {}).get("max_consecutive_failures") or 3)

    def section(self, *keys: str) -> dict:
        """Nested dict from `raw` (`planner`, `antigravity`, …); `{}` if missing."""

        cur: object = self.raw
        for key in keys:
            if not isinstance(cur, dict):
                return {}
            cur = cur.get(key) or {}
        return cur if isinstance(cur, dict) else {}


def load_settings(root: Path, config_path: Path | None = None) -> Settings:
    """Read `.env` + loop.yaml and expand `${VAR:-default}` placeholders."""

    load_dotenv(root / ".env")
    path = config_path or (root / "config" / "loop.yaml")
    raw = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(root=root, raw=expand(raw))
