#!/usr/bin/env bash
# Bounce planner/executor model backends WITHOUT restarting `python -m loop`.
#
# The overnight loop owns the iteration state machine. This script only
# starts/stops the local brains it talks to:
#   deepseek  → llama-server --agent on :8080 (planner fallback)
#   qwen      → Ollama weights used by `qwen -p` (executor)
#   executor  → wedged `qwen` CLI process only (loop parent stays up)
#
# Usage:
#   scripts/restart_agent.sh status
#   scripts/restart_agent.sh deepseek  {up|down|restart|status}
#   scripts/restart_agent.sh qwen      {up|down|restart|status}
#   scripts/restart_agent.sh executor  bounce          # kill stuck qwen CLI; loop continues
#   scripts/restart_agent.sh swap-to-executor           # DeepSeek down, Qwen up
#   scripts/restart_agent.sh swap-to-planner [hot]     # Qwen down; pass 'hot' to also up DeepSeek
#   scripts/restart_agent.sh handoff executor|planner  # aliases for swap-to-*
#
# Env (from repo .env when present):
#   DEEPSEEK_PORT / PORT          default 8080
#   DEEPSEEK_MODEL                default deepseek-v4-flash-q3
#   OPENAI_MODEL                  default qwen3.8:27b-q4_K_M
#   OPENAI_BASE_URL               default http://127.0.0.1:11434/v1
#   AGENTS_PID_DIR                default <repo>/logs
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH:-/usr/bin:/bin}"

DEEPSEEK_PORT="${DEEPSEEK_PORT:-${PORT:-8080}}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash-q3}"
DEEPSEEK_HOST="${DEEPSEEK_HOST:-127.0.0.1}"
QWEN_MODEL="${OPENAI_MODEL:-qwen3.8:27b-q4_K_M}"
OLLAMA_HOST_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
# Derive Ollama base from OpenAI-compatible URL when set.
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  OLLAMA_HOST_URL="$(python3 - <<'PY' 2>/dev/null || echo "$OLLAMA_HOST_URL"
import os
from urllib.parse import urlparse
u = urlparse(os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
print(f"{u.scheme}://{u.hostname}:{u.port or (443 if u.scheme=='https' else 80)}")
PY
)"
fi

PID_DIR="${AGENTS_PID_DIR:-}"
if [[ -z "$PID_DIR" ]]; then
  if [[ -n "${LOOP_LOGS_DIR:-}" ]]; then
    PID_DIR="$(python3 -c 'import os,pathlib; print(pathlib.Path(os.path.expanduser(os.environ["LOOP_LOGS_DIR"])).resolve())')"
  else
    PID_DIR="$(python3 -c 'import pathlib; print(pathlib.Path.home()/".cache"/"ev-s6e9-agent-loop"/"logs")')"
  fi
fi
mkdir -p "$PID_DIR"
DEEPSEEK_PID_FILE="$PID_DIR/deepseek_harness.pid"
DEEPSEEK_LOG="$PID_DIR/deepseek_harness.log"

log() { printf '[agents] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

# --- process helpers -------------------------------------------------------

is_pid_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

loop_pids() {
  # Parent orchestrator only — never kill these from this script.
  pgrep -f 'python -m loop run' 2>/dev/null || true
}

deepseek_pids() {
  # llama-server serving the DeepSeek harness on our port / alias.
  local pids
  pids="$(lsof -nP -iTCP:"$DEEPSEEK_PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    pids="$(pgrep -f "llama-server.*${DEEPSEEK_MODEL}|llama-server.*deepseek-v4-flash|llama-server.*--port ${DEEPSEEK_PORT}" 2>/dev/null || true)"
  fi
  # shellcheck disable=SC2086
  echo $pids | tr ' ' '\n' | awk 'NF' | sort -u
}

qwen_cli_pids() {
  # Headless qwen-code executor CLIs (not the loop, not ollama).
  pgrep -f 'qwen-code/.*/cli-entry.js|qwen-code/.*/cli.js|/[q]wen -p |/[q]wen .* -p' 2>/dev/null || true
}

ollama_model_loaded() {
  need_cmd ollama
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$QWEN_MODEL"
}

curl_ok() {
  local url="$1"
  curl -fsS --max-time 3 "$url" >/dev/null 2>&1
}

# --- deepseek --------------------------------------------------------------

deepseek_status() {
  local pids http="down"
  pids="$(deepseek_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  if curl_ok "http://${DEEPSEEK_HOST}:${DEEPSEEK_PORT}/v1/models"; then
    http="up"
  fi
  log "deepseek: http=${http} port=${DEEPSEEK_PORT} model=${DEEPSEEK_MODEL} pids=${pids:-—}"
  if [[ "$http" == "up" ]]; then
    curl -fsS --max-time 3 "http://${DEEPSEEK_HOST}:${DEEPSEEK_PORT}/v1/models" 2>/dev/null \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); ids=[x.get("id") for x in d.get("data",[])]; print("[agents] deepseek models:", ", ".join(ids) or "—")' 2>/dev/null || true
  fi
}

deepseek_down() {
  local pids pid
  pids="$(deepseek_pids)"
  if [[ -z "$pids" ]]; then
    log "deepseek: already down"
    rm -f "$DEEPSEEK_PID_FILE"
    return 0
  fi
  log "deepseek: stopping pids $(echo "$pids" | tr '\n' ' ')"
  # Graceful then hard — do not touch loop/qwen.
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<<"$pids"
  sleep 2
  pids="$(deepseek_pids)"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    log "deepseek: kill -9 $pid"
    kill -KILL "$pid" 2>/dev/null || true
  done <<<"$pids"
  rm -f "$DEEPSEEK_PID_FILE"
  sleep 1
  if [[ -n "$(deepseek_pids)" ]]; then
    die "deepseek still listening on :${DEEPSEEK_PORT}"
  fi
  log "deepseek: down (freed harness RAM)"
}

deepseek_up() {
  need_cmd llama-server
  if curl_ok "http://${DEEPSEEK_HOST}:${DEEPSEEK_PORT}/v1/models"; then
    log "deepseek: already up on :${DEEPSEEK_PORT}"
    deepseek_status
    return 0
  fi
  # If something else owns the port, fail clearly.
  if [[ -n "$(deepseek_pids)" ]]; then
    die "port ${DEEPSEEK_PORT} busy but /v1/models not healthy; run: $0 deepseek down"
  fi
  log "deepseek: starting harness → ${DEEPSEEK_LOG}"
  # Background; keep loop independent. serve script execs llama-server.
  nohup bash "$ROOT/scripts/serve_deepseek_harness.sh" \
    >"$DEEPSEEK_LOG" 2>&1 &
  echo $! >"$DEEPSEEK_PID_FILE"
  # Wait for health (model load can take a while on Q3).
  local i
  for i in $(seq 1 90); do
    if curl_ok "http://${DEEPSEEK_HOST}:${DEEPSEEK_PORT}/v1/models"; then
      log "deepseek: up after ${i}s (pid file $(cat "$DEEPSEEK_PID_FILE"))"
      deepseek_status
      return 0
    fi
    sleep 2
  done
  die "deepseek failed to become healthy within 180s — see ${DEEPSEEK_LOG}"
}

deepseek_restart() {
  deepseek_down
  deepseek_up
}

# --- qwen / ollama ---------------------------------------------------------

qwen_status() {
  need_cmd ollama
  local loaded="no" serve="down"
  if curl_ok "${OLLAMA_HOST_URL}/api/tags"; then
    serve="up"
  fi
  if ollama_model_loaded; then
    loaded="yes"
  fi
  log "qwen: ollama=${serve} model=${QWEN_MODEL} loaded=${loaded}"
  log "qwen cli pids: $(qwen_cli_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo —)"
  ollama ps 2>/dev/null | sed 's/^/[agents] ollama ps: /' || true
}

qwen_down() {
  need_cmd ollama
  if ! curl_ok "${OLLAMA_HOST_URL}/api/tags"; then
    log "qwen: ollama serve not reachable at ${OLLAMA_HOST_URL} (nothing to unload)"
    return 0
  fi
  if ollama_model_loaded; then
    log "qwen: ollama stop ${QWEN_MODEL}"
    ollama stop "$QWEN_MODEL" || true
  else
    log "qwen: model ${QWEN_MODEL} not loaded"
  fi
  # Give runner a moment to drop RSS.
  sleep 1
  qwen_status
}

qwen_up() {
  need_cmd ollama
  if ! curl_ok "${OLLAMA_HOST_URL}/api/tags"; then
    die "ollama not reachable at ${OLLAMA_HOST_URL} — start \`ollama serve\` first"
  fi
  if ollama_model_loaded; then
    log "qwen: ${QWEN_MODEL} already loaded"
    qwen_status
    return 0
  fi
  log "qwen: warming ${QWEN_MODEL} (short generate)"
  # Pull into VRAM/RAM with a tiny completion; keeps runner alive.
  curl -fsS --max-time 600 "${OLLAMA_HOST_URL}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${QWEN_MODEL}\",\"prompt\":\"ping\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0}}" \
    >/dev/null
  log "qwen: warm complete"
  qwen_status
}

qwen_restart() {
  qwen_down
  qwen_up
}

# --- executor CLI bounce (not the loop) ------------------------------------

executor_bounce() {
  # Kills stuck `qwen -p` worker processes only. The orchestrator (`python -m
  # loop run`) stays alive; the current execute_once call will see a non-zero
  # exit and record a fail row, then the loop can continue to the next iter.
  #
  # This does NOT re-enter the same strategy automatically — use
  # `python -m loop execute-once` later if you need a manual retry.
  local pids pid loop_p
  pids="$(qwen_cli_pids)"
  if [[ -z "$pids" ]]; then
    log "executor: no qwen CLI workers to bounce"
    return 0
  fi
  loop_p="$(loop_pids | tr '\n' ' ')"
  log "executor: bouncing qwen CLI pids $(echo "$pids" | tr '\n' ' ') (loop pids stay: ${loop_p:-none})"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    # Safety: never kill the loop interpreter.
    if ps -p "$pid" -o command= 2>/dev/null | grep -q 'python -m loop'; then
      log "executor: skip loop pid $pid"
      continue
    fi
    kill -TERM "$pid" 2>/dev/null || true
  done <<<"$pids"
  sleep 2
  pids="$(qwen_cli_pids)"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if ps -p "$pid" -o command= 2>/dev/null | grep -q 'python -m loop'; then
      continue
    fi
    kill -KILL "$pid" 2>/dev/null || true
  done <<<"$pids"
  log "executor: bounce done — loop process left running"
  log "executor: note — in-flight execute_once will fail closed; iteration state machine continues"
}

# --- phase handoffs --------------------------------------------------------

swap_to_executor() {
  log "handoff: planner→executor (DeepSeek down, Qwen up)"
  deepseek_down
  qwen_up
}

swap_to_planner() {
  local hot="${1:-}"
  log "handoff: executor→planner (Qwen down${hot:+, DeepSeek up})"
  qwen_down
  if [[ "$hot" == "hot" || "$hot" == "--hot" || "$hot" == "up" ]]; then
    deepseek_up
  else
    log "planner handoff: leaving DeepSeek down (agy is primary; start with: $0 deepseek up)"
  fi
}

status_all() {
  log "repo: $ROOT"
  log "loop pids: $(loop_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo —)"
  deepseek_status
  qwen_status || true
}

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
  local target="${1:-}"
  local action="${2:-}"
  case "$target" in
    ""|-h|--help|help) usage; exit 0 ;;
    status) status_all ;;
    deepseek)
      case "$action" in
        up) deepseek_up ;;
        down|stop) deepseek_down ;;
        restart) deepseek_restart ;;
        status|"") deepseek_status ;;
        *) die "deepseek action must be up|down|restart|status" ;;
      esac
      ;;
    qwen|executor-model)
      case "$action" in
        up) qwen_up ;;
        down|stop) qwen_down ;;
        restart) qwen_restart ;;
        status|"") qwen_status ;;
        *) die "qwen action must be up|down|restart|status" ;;
      esac
      ;;
    executor)
      case "$action" in
        bounce|restart|kill) executor_bounce ;;
        status|"") qwen_status ;;
        *) die "executor action must be bounce|status" ;;
      esac
      ;;
    swap-to-executor|handoff)
      if [[ "$target" == "handoff" && "${action:-}" == "planner" ]]; then
        swap_to_planner "${3:-}"
      elif [[ "$target" == "handoff" && "${action:-}" == "executor" ]]; then
        swap_to_executor
      elif [[ "$target" == "swap-to-executor" ]]; then
        swap_to_executor
      else
        die "usage: $0 handoff executor|planner [hot]"
      fi
      ;;
    swap-to-planner)
      swap_to_planner "${action:-}"
      ;;
    *)
      die "unknown target '$target' (deepseek|qwen|executor|status|swap-to-executor|swap-to-planner)"
      ;;
  esac
}

main "$@"
