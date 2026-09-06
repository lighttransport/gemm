#!/bin/sh
# test_ds4f_serve.sh -- launch the runner + OpenAI frontend, send one chat request.
#
# Usage:  sh a64fx/llm/test_ds4f_serve.sh [stage_dir] [port]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE="${1:-/tmp/ds4f_single}"
PORT="${2:-8080}"
BASE="/tmp/ds4f_serve_test"
LIB="$HERE/../../libds4f_serve.so"
TOK="/mnt/disk1/models/ds4f-0731/tokenizer.json"

rm -f "$BASE".*
pkill -f ds4f_serve_runner.py 2>/dev/null || true
pkill -f ds4f_serve.py 2>/dev/null || true
sleep 1

echo "[test] starting the model runner (load takes ~2 min)..."
env DS4F_SERVE_BASE="$BASE" DS4F_STAGE_DIR="$STAGE" DS4F_SERVE_USE_HIP=1 \
    DS4F_MAXPOS=4096 DS4F_SERVE_LIB="$LIB" \
    python3 "$HERE/ds4f_serve_runner.py" --daemon

echo "[test] starting the OpenAI frontend on :$PORT..."
env PORT="$PORT" TOK="$TOK" DS4F_SERVE_BASE="$BASE" \
    python3 "$HERE/ds4f_serve.py" > /tmp/ds4f_frontend.log 2>&1 &
FRONT=$!

# wait for the runner to be ready
for i in $(seq 1 240); do
    if grep -q "serving on" "$BASE.reqseq" 2>/dev/null || grep -q "runner" /tmp/runner.log 2>/dev/null; then :; fi
    if grep -q "serving" /tmp/runner.log 2>/dev/null; then break; fi
    sleep 2
done
echo "[test] runner ready: $(grep 'serving' /tmp/runner.log | tail -1)"

sleep 2
echo "[test] sending a chat request..."
curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"ds4f","messages":[{"role":"user","content":"Hello there"}],"max_tokens":8}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('response:', json.dumps(d, ensure_ascii=False)[:400])"

kill $FRONT 2>/dev/null || true
pkill -f ds4f_serve_runner.py 2>/dev/null || true
echo "[test] done"
