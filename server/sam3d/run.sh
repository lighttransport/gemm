#!/usr/bin/env bash
# Launch the sam-3d-body 3-pane demo server.
#
# Backends:
#   - ours    = cpu/sam3d_body/test_sam3d_body  (subprocess)
#   - pytorch = ref/sam3d-body/.venv            (in-process)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$ROOT/ref/sam3d-body/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[run.sh] missing venv: $VENV" >&2
  echo "[run.sh] see ref/sam3d-body/README.md to create it." >&2
  exit 1
fi
exec "$VENV/bin/python" "$HERE/app.py" \
    --c-runner        "$ROOT/cpu/sam3d_body/test_sam3d_body" \
    --safetensors-dir /mnt/disk01/models/sam3d-body/safetensors \
    --mhr-assets      /mnt/disk01/models/sam3d-body/safetensors \
    --rt-detr-model   /mnt/disk01/models/rt_detr_s/model.safetensors \
    --pytorch-ckpt    /mnt/disk01/models/sam3d-body/dinov3 \
    --web-root        "$ROOT/web" \
    --demo-html       "$ROOT/web/sam3d_demo.html" \
    --port 8765 "$@"
