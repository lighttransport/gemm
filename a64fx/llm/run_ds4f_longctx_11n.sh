#!/bin/bash
# DS4F long-context Tier-B2 bench (demonstrates the O(topk) payoff on REAL weights).
#
# Wraps run_ds4f_11n.sh with the ctx-warm knob: ds4f_warm_kv + ds4f_warm_tb2 fill the
# synthetic KV + compressed (cmp_kv / idx_kv) caches to DS4F_CTX_WARM positions, then
# decode runs from there. Both warm fills are DETERMINISTIC (fixed per-layer splitmix64
# seeds) and RANK-INDEPENDENT (local caches only, no all-reduce) => lockstep preserved
# across all 11 EP ranks. This measures decode/attn cost at long context WITHOUT paying
# the O(ctx^2) cost of a real prefill of that length.
#
# Default config = combined bf16-pv dense + Tier-B2 sparse on real weights:
#     DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_TIERB2=1
# Reuses the /local blobs already staged by run_ds4f_stage_11n.sh (Step 1 of ds4f.md).
#
# Usage (inside a live 12-node alloc, from a64fx/llm; runs the 11 non-(0,0,0) EP nodes):
#     ./run_ds4f_longctx_11n.sh                          # ctx=16384 (safe top for f32 KV)
#     DS4F_CTX_WARM=8192 ./run_ds4f_longctx_11n.sh       # a lower curve point
#     setsid nohup ./run_ds4f_longctx_11n.sh >/tmp/lc.out 2>&1 &   # detached + sentinel
#
# HBM note: kv_cache = max_pos * kv_lora(512) * 4B (f32) * 43 layers. ctx=16384 => RSS
# ~28.9 GB (safe). ctx=32768 OOMs (~30.6 GB > usable HBM) for this f32-KV build; an f16
# KV cache would double the headroom (a TODO lever). DS4F_MAXPOS auto-sizes to
# CTX_WARM + MAXGEN + 256 headroom unless you set it explicitly.
set +e
cd "$(dirname "$0")" || exit 2

CTX=${DS4F_CTX_WARM:-16384}
GEN=${DS4F_MAXGEN:-16}
# auto-size max_pos unless the caller pinned it (runner asserts ctx_warm+maxgen<=max_pos)
MAXPOS=${DS4F_MAXPOS:-$((CTX + GEN + 256))}
SENT=${DS4F_LCTX_SENTINEL:-/tmp/ds4f_longctx_sentinel.txt}
LOG=${DS4F_LCTX_LOG:-/tmp/ds4f_longctx_run.log}
: > "$LOG"
{ echo "===== DS4F LONGCTX SENTINEL ====="; echo "LCTX_START $(date +%H:%M:%S) ctx=$CTX max_pos=$MAXPOS"; } > "$SENT"

# clear stale per-rank outputs so the perf/load counts reflect THIS run
rm -f ds4f_ep_perf_rank*.txt ds4f_ep_load_rank*.txt ds4f_ep_rank00.txt 2>/dev/null

t0=$(date +%s)
DS4F_REAL=${DS4F_REAL:-1} \
DS4F_FP8_BF16=${DS4F_FP8_BF16:-1} \
DS4F_TIERB2=${DS4F_TIERB2:-1} \
DS4F_CTX_WARM=$CTX DS4F_MAXPOS=$MAXPOS \
DS4F_PREFILL=${DS4F_PREFILL:-8} DS4F_MAXGEN=$GEN \
  ./run_ds4f_11n.sh > "$LOG" 2>&1
rc=$?
t1=$(date +%s)

perf=$(ls ds4f_ep_perf_rank*.txt 2>/dev/null | wc -l)
load=$(ls ds4f_ep_load_rank*.txt 2>/dev/null | wc -l)
topo=$(grep -m1 -E '^[0-9]' tofu_topo.txt 2>/dev/null)
crash=$(grep -ciE 'segmentation|sigsegv|abort|MISSING tensor|dtype .* !=|nbytes .* !=|exceed limit|bad_alloc|out of memory|Killed|file not found|No such file' "$LOG")
nan=$(grep -hoE 'NaNs?=[0-9]+' ds4f_ep_rank00.txt ds4f_ep_perf_rank00.txt 2>/dev/null | head -2 | tr '\n' ' ')
args=$(grep -hoE 'argmax[ =]+[0-9]+' ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE '[0-9]+$' | sort -u | tr '\n' ' ')
nuniq=$(echo $args | wc -w)

{
  echo "LCTX_RC rc=$rc wall=$((t1-t0))s perf=$perf/11 load=$load/11 crash_hits=$crash"
  echo "topo_row0=$topo  (claude node must be absent)"
  echo "nan=$nan"
  echo "argmax_distinct_across_ranks=$nuniq  values=[$args]  (1==lockstep)"
  echo "--- rank0 summary (per-phase decode @ ctx=$CTX) ---"
  head -36 ds4f_ep_rank00.txt 2>/dev/null
  echo "--- max RSS across load ranks ---"
  grep -hoE 'RSS[ =][0-9.]+ ?GB' ds4f_ep_load_rank*.txt 2>/dev/null | grep -oE '[0-9.]+' | sort -rn | head -3
  echo "--- first crash context (if any; rc=137 = OOM, lower DS4F_CTX_WARM) ---"
  grep -m6 -iE 'segmentation|sigsegv|abort|MISSING tensor|dtype .* !=|nbytes .* !=|exceed limit|bad_alloc|out of memory|Killed|file not found|No such file' "$LOG"
  echo "LCTX_END $(date +%H:%M:%S)"
} >> "$SENT"

cat "$SENT"
exit $rc
