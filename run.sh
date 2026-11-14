#!/usr/bin/env bash
# Holdover: start everything on this Mac. No internet is used.
#   ./run.sh         9B model (best answers)
#   ./run.sh 4B      smaller, faster model
set -euo pipefail
APP="$(cd "$(dirname "$0")" && pwd)"; ROOT="$APP"; [ -d "$APP/.venv" ] || ROOT="$(dirname "$APP")"
SIZE="$(echo "${1:-9B}" | tr 'B' 'b')"
PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/models/bonsai2-$SIZE-mlx"
LOGS="$APP/.logs"; mkdir -p "$LOGS"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # never reach for the network

pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

if ! curl -s localhost:8080/v1/models >/dev/null 2>&1; then
  echo "Starting Bonsai $SIZE..."
  "$PY" -m mlx_vlm.server --model "$MODEL" --port 8080 >"$LOGS/bonsai.log" 2>&1 & pids+=($!)
  until curl -s localhost:8080/v1/models >/dev/null 2>&1; do sleep 1; done
fi
echo "Starting the brain..."
"$PY" -m uvicorn server:app --app-dir "$APP/brain" --host 127.0.0.1 --port 8770 >"$LOGS/brain.log" 2>&1 & pids+=($!)
sleep 2
if [ -x "$APP/speech/.build/release/speechd" ]; then
  echo "Starting Argmax voice..."
  ( cd "$APP/speech" && ./.build/release/speechd ) >"$LOGS/speech.log" 2>&1 & pids+=($!)
else
  echo "Voice helper not built yet (cd speech && swift build -c release). Typing still works."
fi
open "http://127.0.0.1:8770/"
echo "Holdover is running at http://127.0.0.1:8770  (Ctrl+C to stop)"
wait
