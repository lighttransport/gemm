#!/bin/sh
# Stable CPU + RX 9070 XT DS4F server profile.
# Usage: run-ds4f-9070xt.sh STAGE_DIR [PORT] [THREADS] [SERVER_BIN]
set -eu

stage_dir=${1:?usage: $0 STAGE_DIR [PORT] [THREADS]}
port=${2:-8080}
threads=${3:-16}
server_bin=${4:-/tmp/ds4f-server-build/diffusion-server}

if [ ! -x "$server_bin" ]; then
    echo "missing server binary: $server_bin" >&2
    echo "build with cmake -S server -B /tmp/ds4f-server-build -DDIFFUSION_SERVER_ENABLE_DS4F_HETERO=ON -DDIFFUSION_SERVER_ENABLE_DS4F_HIP=ON" >&2
    exit 1
fi

exec "$server_bin" \
  --host 127.0.0.1 --port "$port" \
  --ds4f-model "$stage_dir" --ds4f-threads "$threads" \
  --ds4f-ep-size 8 --ds4f-ep-rank 0 --ds4f-max-pos 8192 \
  --ds4f-hip 1 --ds4f-hip-device 0 --ds4f-hip-async 1 \
  --ds4f-hip-prefill-attn 1 --ds4f-hip-expert-cache-mb -1 \
  --ds4f-hip-expert-cache-stats 1 \
  --ds4f-exact 1 --ds4f-mhc 1 --ds4f-tierb2 0 \
  --ds4f-tp-embed 0 --ds4f-tp-head 0
