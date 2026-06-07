#!/bin/bash
# int8-KV Lever B full validation (same job; SKIP_TOPO safe):
#   (1) coherence: real gen with CAL=64 so freeze fires mid-gen (~96 tokens under int8)
#   (2) memory: longctx16384 bf16 (fresh RSS baseline) vs int8 (RSS win + int8-at-scale)
# ctx capped at 16384 (the safe top; int8 RSS <= bf16 so no new OOM risk).
set +e
cd "$(dirname "$0")"
export SKIP_TOPO=1

echo "######## (1) coherence: int8 ON, CAL=64, real gen ########"
DS4F_INT8_KV=1 DS4F_INT8KV_CAL=64 GEN_OUT=gen_B64.txt MAX_NEW=320 \
  DS4F_GEN_SENTINEL=/tmp/int8kv_B64.txt DS4F_GEN_LOG=/tmp/int8kv_B64.log \
  ./run_ds4f_gen_11n.sh >/tmp/int8kv_B64.console 2>&1; echo "gen rc=$?"

echo "######## (2a) longctx16384 bf16 baseline ########"
DS4F_INT8_KV=0 DS4F_CTX_WARM=16384 DS4F_MAXGEN=16 \
  DS4F_LCTX_SENTINEL=/tmp/lctx_bf16.txt DS4F_LCTX_LOG=/tmp/lctx_bf16.log \
  ./run_ds4f_longctx_11n.sh >/dev/null 2>&1; echo "bf16 rc=$?"

echo "######## (2b) longctx16384 int8 (CAL=256) ########"
DS4F_INT8_KV=1 DS4F_INT8KV_CAL=256 DS4F_CTX_WARM=16384 DS4F_MAXGEN=16 \
  DS4F_LCTX_SENTINEL=/tmp/lctx_int8.txt DS4F_LCTX_LOG=/tmp/lctx_int8.log \
  ./run_ds4f_longctx_11n.sh >/dev/null 2>&1; echo "int8 rc=$?"

echo "######## summary ########"
echo "--- coherence gen B64 (CAL=64) divergence vs A (gen_A.txt) ---"
python3 - <<'PY'
import os
def rd(p): return open(p).read().split() if os.path.exists(p) and os.path.getsize(p) else []
a=rd("gen_A.txt"); b=rd("gen_B64.txt")
n=min(len(a),len(b)); div=next((i for i in range(n) if a[i]!=b[i]), n)
print(f"len A={len(a)} B64={len(b)}  identical-prefix-tokens={div}  (CAL=64,prompt24 -> freeze ~gen-token 40)")
PY
echo "--- RSS: bf16 vs int8 longctx16384 ---"
grep -hE 'LCTX_RC|nan=|argmax_distinct' /tmp/lctx_bf16.txt /tmp/lctx_int8.txt
echo "bf16 RSS:"; grep -m1 'RSS=' /tmp/lctx_bf16.txt
echo "int8 RSS:"; grep -m1 'RSS=' /tmp/lctx_int8.txt
echo "######## DONE ########"
