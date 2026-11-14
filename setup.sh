#!/usr/bin/env bash
# One-time setup (needs internet once). After this, Holdover runs with WiFi off.
set -euo pipefail
cd "$(dirname "$0")"
SIZE="${1:-9B}"; L="$(echo "$SIZE" | tr 'B' 'b')"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -f "models/bonsai2-$L-mlx/model.safetensors" ] || .venv/bin/hf download "prism-ml/Ternary-Bonsai-2-$SIZE-mlx-2bit" --local-dir "models/bonsai2-$L-mlx"
# Argmax voice helper. Needs the Argmax registry login first (see speech/README.md).
( cd speech && swift build -c release && ./.build/release/speechd prerender lines.txt )
echo "Done. Start with ./run.sh"
