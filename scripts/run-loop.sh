#!/usr/bin/env bash
# Kick off the plan → execute loop (Mac Studio or any machine with agy/qwen/Ollama).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
ITERATIONS="${1:-5}"
exec python -m loop run --iterations "$ITERATIONS"
