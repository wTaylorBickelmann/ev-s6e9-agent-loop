# Logs moved out of the repo

Long run traces no longer live here. The orchestrator writes them under:

```text
$LOOP_LOGS_DIR   # default on this machine: ~/.cache/ev-s6e9-agent-loop/logs
```

That keeps Qwen Code / other agents from auto-ingesting multi-MB traces when the
cwd is the git checkout. See `.env.example`, `config/loop.yaml` (`paths.logs_dir`),
and `./scripts/restart_agent.sh` (agent pid/log files use the same cache root).
