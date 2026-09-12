from __future__ import annotations

from loop import context, ledger
from loop.config import Settings
from loop.factory import build_executor, build_planner, competition_root_label
from loop.log import log
from loop.models import Plan, RunResult
from loop.parse import parse_plan


class Loop:
    def __init__(self, settings: Settings, *, dry_run: bool = False):
        self.settings = settings
        self.dry_run = dry_run
        self.strategies = settings.ledger_dir / "STRATEGIES.md"
        self.results = settings.ledger_dir / "RESULTS.md"
        self.current = settings.ledger_dir / "CURRENT_STRATEGY.md"

    def run(self, iterations: int) -> int:
        failures = 0
        n = max(1, iterations)
        for i in range(1, n + 1):
            log(f"iteration {i}/{n} ({'dry-run' if self.dry_run else 'live'})")
            plan = self.plan_once()
            result = self.execute_once(plan)
            self.record(result)
            if result.status != "ok":
                failures += 1
                log(f"{result.strategy_id} failed ({failures} consecutive)")
                if failures >= self.settings.max_consecutive_failures:
                    log("stop: consecutive failures")
                    return 1
            else:
                failures = 0
                if self._hit_target(result):
                    log(f"stop: target CV {self.settings.target_cv} reached")
                    return 0
        return 0

    def plan_once(self) -> Plan:
        nxt = ledger.next_strategy_id(self.strategies)
        planner = build_planner(self.settings, default_id=nxt, dry_run=self.dry_run)
        prompt = self._planner_prompt(nxt)
        log(f"planning {nxt} via {planner.name}")
        plan = planner.plan(prompt)
        if not plan.strategy_id:
            plan.strategy_id = nxt
        ledger.write_current(self.current, plan)
        ledger.append_strategy(self.strategies, plan)
        log(f"wrote CURRENT_STRATEGY.md and appended {plan.strategy_id} to STRATEGIES.md")
        return plan

    def execute_once(self, plan: Plan | None = None) -> RunResult:
        if plan is None:
            plan = self._plan_from_current()
        executor = build_executor(self.settings, dry_run=self.dry_run)
        prompt = self._executor_prompt(plan)
        log(f"executing {plan.strategy_id} via {executor.name}")
        return executor.execute(prompt, plan)

    def record(self, result: RunResult) -> None:
        ledger.append_result(self.results, result)
        ledger.write_run_json(self.settings.runs_dir / f"{result.strategy_id}.json", result)
        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log(f"recorded {result.strategy_id} status={result.status} cv={result.cv}")

    def show_whitelist(self) -> AssembledView:
        ctx = self._assemble()
        return AssembledView(
            listing=ctx.listing,
            skipped=ctx.skipped,
            total_bytes=ctx.total_bytes,
            rendered=ctx.render(),
        )

    def _assemble(self):
        budget = self.settings.raw.get("budget") or {}
        return context.assemble(
            loop_root=self.settings.root,
            reads_path=self.settings.planner_reads_path,
            competition_root=self.settings.competition_root
            if self.settings.competition_root and self.settings.competition_root.exists()
            else None,
            max_bytes_per_file=int(budget.get("max_bytes_per_file") or 24000),
            max_total_bytes=int(budget.get("max_total_bytes") or 80000),
        )

    def _planner_prompt(self, next_id: str) -> str:
        ctx = self._assemble()
        template = self.settings.planner_prompt.read_text(encoding="utf-8")
        rows = ledger.parse_result_rows(self.results)
        best = ledger.best_cv(rows, higher_is_better=self.settings.higher_is_better)
        best_s = f"{best[1]:.6g} ({best[0]})" if best else "—"
        return template.format(
            competition_name=self.settings.competition_name,
            metric=self.settings.metric,
            higher_is_better=self.settings.higher_is_better,
            next_id=next_id,
            best_cv=best_s,
            phase_coverage=ledger.phase_coverage(self.strategies),
            whitelist_listing="\n".join(f"- {p}" for p in ctx.listing) or "- (empty)",
            assembled_context=ctx.render(),
        )

    def _executor_prompt(self, plan: Plan) -> str:
        template = self.settings.executor_prompt.read_text(encoding="utf-8")
        spec = plan.spec or ledger.current_spec(self.current)
        return template.format(
            loop_root=self.settings.root,
            competition_root=competition_root_label(self.settings),
            strategy_id=plan.strategy_id,
            metric=self.settings.metric,
            strategy_spec=spec,
        )

    def _plan_from_current(self) -> Plan:
        spec = ledger.current_spec(self.current)
        sid = ledger.next_strategy_id(self.strategies)
        parsed = parse_plan(spec, default_id=sid)
        if spec:
            parsed.spec = spec
        return parsed

    def _hit_target(self, result: RunResult) -> bool:
        target = self.settings.target_cv
        if target is None or result.cv is None or result.status != "ok":
            return False
        if self.settings.higher_is_better:
            return result.cv >= target
        return result.cv <= target


class AssembledView:
    def __init__(self, listing: list[str], skipped: list[str], total_bytes: int, rendered: str):
        self.listing = listing
        self.skipped = skipped
        self.total_bytes = total_bytes
        self.rendered = rendered
