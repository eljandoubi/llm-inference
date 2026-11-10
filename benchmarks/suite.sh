#!/usr/bin/env bash
# Runs the scenarios behind the numbers in the README against two local servers (one
# with the prefix cache, one without) and writes everything to a text file.
#
#   benchmarks/suite.sh [model-dir] [out-file]
#
# The model is a random llava-style checkpoint (see randomhf/), it is created if the
# directory does not exist. PRESET=tiny only checks that the script runs: its
# 128-token window refuses the prompts.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-/tmp/vlmserve-bench-small}
OUT=${2:-benchmarks/results/latest.txt}
PRESET=${PRESET:-small}
PORT_ON=${PORT_ON:-8200}
PORT_OFF=${PORT_OFF:-8201}

if [ ! -d "$MODEL" ]; then
    uv run python -m randomhf "$MODEL" --preset "$PRESET"
fi
mkdir -p "$(dirname "$OUT")"

serve() {  # port, extra flags...
    local port=$1
    shift
    uv run vlmserve serve --model "$MODEL" --port "$port" --max-num-seqs 16 \
        --served-model-name bench --log-level warning "$@" &
    PIDS+=($!)
}

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
serve "$PORT_ON"
serve "$PORT_OFF" --no-prefix-caching
for port in "$PORT_ON" "$PORT_OFF"; do
    for _ in $(seq 1 120); do
        curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null && break
        sleep 1
    done
done

{
    echo "model: $MODEL ($PRESET), device: $(uv run vlmserve devices 2>/dev/null | sed -n 2p)"
    echo
} >"$OUT"

run() {  # title, port, load test flags...
    local title=$1 port=$2
    shift 2
    echo "### $title (port $port)" >>"$OUT"
    uv run python -m benchmarks.load_test --url "http://127.0.0.1:$port" "$@" >>"$OUT" 2>&1
    echo >>"$OUT"
}

TEXT=(--prompt-words 80 --max-tokens 64)
run "concurrency 1" "$PORT_ON" --requests 12 --concurrency 1 "${TEXT[@]}"
run "concurrency 4" "$PORT_ON" --requests 32 --concurrency 4 "${TEXT[@]}"
run "concurrency 16" "$PORT_ON" --requests 64 --concurrency 16 "${TEXT[@]}"

SHARED=(--requests 32 --concurrency 4 --shared-words 120 --prompt-words 10 --max-tokens 32)
run "shared 120-word system prompt, prefix caching ON" "$PORT_ON" "${SHARED[@]}"
run "shared 120-word system prompt, prefix caching OFF" "$PORT_OFF" "${SHARED[@]}"

IMAGES=(--requests 32 --concurrency 4 --images 1 --prompt-words 10 --max-tokens 32)
run "1 image, the same every time, caches ON" "$PORT_ON" "${IMAGES[@]}" --image-pool 1
run "1 image, a different one every time, caches ON" "$PORT_ON" "${IMAGES[@]}" --image-pool 32
run "1 image, the same every time, prefix cache OFF" "$PORT_OFF" "${IMAGES[@]}" --image-pool 1

echo "wrote $OUT"
