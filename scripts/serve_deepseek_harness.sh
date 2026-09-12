#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash Q3 via llama-server WITH the built-in agent tool harness.
# OpenAI-compatible base: http://127.0.0.1:8080/v1  model id: deepseek-v4-flash-q3
#
# Harness = llama-server --agent (CORS proxy + built-in tools: read/write/edit/grep/exec).
# Scoped to this repo when started from the checkout (or set WORKDIR).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/Models/deepseek-v4-flash-q3/UD-Q3_K_M}"
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -d "${MODEL_DIR}" ]]; then
    MODEL_PATH="$(ls "${MODEL_DIR}"/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00001-of-*.gguf 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found under ${MODEL_DIR}"
  echo "Expected Unsloth UD-Q3_K_M shards for DeepSeek-V4-Flash-0731."
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash-q3}"
WORKDIR="${WORKDIR:-$ROOT}"

EXTRA=( -ngl 99 )
if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  EXTRA=( -ngl 0 )
fi

# Harness on by default. Set HARNESS=0 for plain chat-only server.
HARNESS_ARGS=()
if [[ "${HARNESS:-1}" == "1" ]]; then
  # --agent enables all built-in tools + CORS proxy (local-only bind below).
  HARNESS_ARGS=( --agent )
fi

cd "${WORKDIR}"

echo "Serving: ${MODEL_PATH}"
echo "Workdir: ${WORKDIR}"
echo "OpenAI base: http://${HOST}:${PORT}/v1  model id: ${MODEL_NAME}"
echo "Harness: ${HARNESS:-1}  (llama-server --agent)"
echo "Ctrl-C to stop."

exec llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --alias "${MODEL_NAME}" \
  -c "${CTX}" \
  -fa on \
  --jinja \
  "${EXTRA[@]}" \
  "${HARNESS_ARGS[@]}" \
  "$@"
