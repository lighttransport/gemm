#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"

request() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  local out_file status
  out_file="$(mktemp)"
  if [[ -n "$data" ]]; then
    status="$(curl -sS -o "$out_file" -w "%{http_code}" -X "$method" "${BASE_URL}${path}" \
      -H 'content-type: application/json' --data "$data")"
  else
    status="$(curl -sS -o "$out_file" -w "%{http_code}" -X "$method" "${BASE_URL}${path}")"
  fi
  printf "HTTP %s " "$status"
  head -c 400 "$out_file"
  echo
  rm -f "$out_file"
}

echo "[smoke] GET /health"
request GET /health

echo "[smoke] GET /models"
request GET /models

echo "[smoke] POST /v1/infer (sam3 — expect 'ckpt path not set' unless --sam3-ckpt was passed)"
request POST /v1/infer '{"model":"sam3","task":"segmentation","backend":"cpu","inputs":{"text":"cat","image_base64":"aGVsbG8="}}'

echo "[smoke] POST /v1/infer (sam3.1 — expect 501 pending_runner)"
request POST /v1/infer '{"model":"sam3.1","task":"segmentation","backend":"cpu","inputs":{"text":"cat","image_base64":"aGVsbG8="}}'

echo "[smoke] POST /v1/infer (unsupported qwen backend)"
request POST /v1/infer '{"model":"qwen-image","task":"text-to-image","backend":"cuda","inputs":{"text":"cat"},"params":{"width":64,"height":64,"steps":1}}'

echo "[smoke] done"
