#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${1:-/tmp/diffusion-server-build/diffusion-server}"
JSONL_PATH="${2:-${SCRIPT_DIR}/stdio-examples.jsonl}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[smoke-stdio] binary not executable: ${BIN_PATH}" >&2
  exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
  echo "[smoke-stdio] examples file not found: ${JSONL_PATH}" >&2
  exit 1
fi

echo "[smoke-stdio] binary: ${BIN_PATH}"
echo "[smoke-stdio] input:  ${JSONL_PATH}"

OUT_FILE="$(mktemp)"
trap 'rm -f "${OUT_FILE}"' EXIT

"${BIN_PATH}" --stdio < "${JSONL_PATH}" | tee "${OUT_FILE}"

line_count="$(wc -l < "${JSONL_PATH}")"
out_count="$(wc -l < "${OUT_FILE}")"
if [[ "${out_count}" -lt "${line_count}" ]]; then
  echo "[smoke-stdio] expected >= ${line_count} output lines, got ${out_count}" >&2
  exit 1
fi

grep -q '"status":200' "${OUT_FILE}" || { echo "[smoke-stdio] missing status 200" >&2; exit 1; }
grep -q '"status":501' "${OUT_FILE}" || { echo "[smoke-stdio] missing status 501" >&2; exit 1; }
grep -q '"status":400' "${OUT_FILE}" || { echo "[smoke-stdio] missing status 400" >&2; exit 1; }
grep -q '"code":"not_implemented"' "${OUT_FILE}" || { echo "[smoke-stdio] missing not_implemented error" >&2; exit 1; }
grep -q '"code":"bad_request"' "${OUT_FILE}" || { echo "[smoke-stdio] missing bad_request error" >&2; exit 1; }

echo "[smoke-stdio] done"
