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
# HBM note: kv_cache is now bf16 = max_pos * kv_lora(512) * 2B * 43 layers (halved). Even
# so, ctx<=~16384 is the safe top and ctx=32768 STILL OOM-kills under the gen config: the
# resident cost is dominated by the ~27.5 GB replicated dense weights (inflated +6 GB by
# DS4F_FP8_BF16=1), not KV, so halving a ~1.4 GB KV term doesn't move the ceiling. To go
# past 16K use DS4F_FP8_BF16=0 (FP8 dense, -6 GB, slower decode) and/or int8 KV (TODO).
# DS4F_MAXPOS auto-sizes to CTX_WARM + MAXGEN + 256 unless you set it explicitly.
#
# OOM-kill aftermath: a sig=9 OOM-kill leaves the remote plexec/PMIx service degraded, so
# every SUBSEQUENT launch dies pre-load in ~3s with "PLE 0080 plexec PMIx service error"
# (perf=0/11 load=0/11 rc=255). A short wait does not reliably clear it; the alloc usually
# must be recycled (new pjsub -> /local wiped -> re-stage). Don't push ctx past the known
# ceiling on a shared alloc you don't want to lose.
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
DS4F_Q8_DENSE=${DS4F_Q8_DENSE:-1} \
DS4F_HC_PAR=${DS4F_HC_PAR:-1} \
DS4F_HC_RMSPAR=${DS4F_HC_RMSPAR:-1} \
DS4F_CTX_WARM=$CTX DS4F_MAXPOS=$MAXPOS \
DS4F_PREFILL=${DS4F_PREFILL:-8} DS4F_MAXGEN=$GEN \
  ./run_ds4f_11n.sh > "$LOG" 2>&1
rc=$?
t1=$(date +%s)

perf=$(ls ds4f_ep_perf_rank*.txt 2>/dev/null | wc -l)
load=$(ls ds4f_ep_load_rank*.txt 2>/dev/null | wc -l)
topo=$(grep -m1 -E '^[0-9]' tofu_topo.txt 2>/dev/null)
crash=$(grep -ciE 'segmentation|sigsegv|abort|MISSING tensor|dtype .* !=|nbytes .* !=|exceed limit|bad_alloc|out of memory|Killed|file not found|No such file|terminated with the signal|sig=9|PMIx service error|PLE 0[0-9]+ plexec' "$LOG")
# pre-load death (sig=9 OOM aftermath / PMIx error) => 0 per-rank files despite rc!=0
[ "$rc" != "0" ] && [ "$load" = "0" ] && echo "  [warn] rc=$rc with load=0/11 => pre-load failure (OOM-kill aftermath? PMIx degraded? see log)" >&2
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
