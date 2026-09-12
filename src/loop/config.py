from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from loop.env import expand, load_dotenv


def find_root(cli_root: str | None = None) -> Path:
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
    root: Path
    raw: dict

    @property
    def competition_name(self) -> str:
        return str(self.raw.get("competition", {}).get("name") or "competition")

    @property
    def competition_root(self) -> Path | None:
        value = (self.raw.get("competition") or {}).get("root") or ""
        if not str(value).strip():
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return path if path.exists() else path

    @property
    def metric(self) -> str:
        return str((self.raw.get("competition") or {}).get("metric") or "cv")

    @property
    def higher_is_better(self) -> bool:
        return bool((self.raw.get("competition") or {}).get("higher_is_better", True))

    @property
    def ledger_dir(self) -> Path:
        return self.root / (self.raw.get("paths") or {}).get("ledger_dir", "ledger")

    @property
    def logs_dir(self) -> Path:
        return self.root / (self.raw.get("paths") or {}).get("logs_dir", "logs")

    @property
    def runs_dir(self) -> Path:
        return self.root / (self.raw.get("paths") or {}).get("runs_dir", "ledger/runs")

    @property
    def planner_reads_path(self) -> Path:
        rel = (self.raw.get("paths") or {}).get("planner_reads", "config/planner_reads.yaml")
        return self.root / rel

    @property
    def planner_prompt(self) -> Path:
        rel = (self.raw.get("planner") or {}).get("prompt", "prompts/planner.md")
        return self.root / rel

    @property
    def executor_prompt(self) -> Path:
        rel = (self.raw.get("executor") or {}).get("prompt", "prompts/executor.md")
        return self.root / rel

    @property
    def max_iterations(self) -> int:
        return int((self.raw.get("loop") or {}).get("max_iterations") or 10)

    @property
    def target_cv(self) -> float | None:
        value = (self.raw.get("loop") or {}).get("target_cv")
        return float(value) if value is not None and value != "" else None

    @property
    def max_consecutive_failures(self) -> int:
        return int((self.raw.get("loop") or {}).get("max_consecutive_failures") or 3)

    def section(self, *keys: str) -> dict:
        cur: object = self.raw
        for key in keys:
            if not isinstance(cur, dict):
                return {}
            cur = cur.get(key) or {}
        return cur if isinstance(cur, dict) else {}


def load_settings(root: Path, config_path: Path | None = None) -> Settings:
    load_dotenv(root / ".env")
    path = config_path or (root / "config" / "loop.yaml")
    raw = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(root=root, raw=expand(raw))
