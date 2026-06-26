#!/bin/bash
# GLM-5.2 FP8 KV-cache persistence (prompt caching) test on 384 non-contiguous A64FX nodes.
# (A) process a ~16k-token "system prompt" (tokenized gpt-5.5.md) and SAVE its Tier-A KV to storage;
# (B) LOAD that KV and process 1000 more tokens from there (cache hit -- no recompute);
# (C) full from-scratch reference over the same 17,384 tokens.
# Validates B's final argmax == C's (cache is lossless) and B is far faster (1k vs 17k tokens).

#PJM -g hp250467
#PJM -L "rscgrp=small,node=384,elapse=04:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=384"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-384}
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-0}; RUN_LAYERS=${GLM5_LAYERS:-0}
JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/kvcache_run_${JOB_TAG}"
KVDIR="$WORK/kv"; TOK=${GLM5_PROMPT_TOKENS:-$HOME/glm5_gpt55_17384.bin}
SYS=${GLM5_SYS_TOK:-16384}; NEW=${GLM5_NEW_TOK:-1000}; TOTAL=$((SYS+NEW))

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_fp8_prefill_$JOB_TAG}
export GLM5_NSHARDS=141
export GLM5_EP_SIZE=$NP
export GLM5_PREFILL_GROUPS=1   # single group: ep_size=NP must match the staged blobs (no auto-grouping)
export GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_TP_SHARED=1 GLM5_MAXPOS=${GLM5_MAXPOS:-32768}
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=${GLM5_PREFILL_ROLLING:-1}
export GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
th=${GLM5_THREAD_SWEEP:-24}; pc=${GLM5_PCHUNK_SWEEP:-256}
ovl=${GLM5_COMM_OVERLAP:-1}; arbf=${TP_AR_BF16:-1}   # set both 0 + th=1 for a deterministic (B==C) check
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS GLM5_COMM_OVERLAP=$ovl TP_AR_BF16=$arbf GLM5_PROMPT_TOKENS=$TOK"

echo "=== GLM5.2 FP8 KV-cache test: NP=$NP sys=$SYS new=$NEW total=$TOTAL tok=$TOK maxpos=$GLM5_MAXPOS job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" "$KVDIR" || exit 2; cd "$WORK" || exit 2

make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[kvcache] topo try $t failed; retry"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK"
echo "--- staging ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP ($(date))"

run() { # $1=label  rest=extra env
  local label=$1; shift
  echo "--- run $label ($(date)) ---"; rm -f glm5_ep_rank00.txt
  eval $RUNENV "$@" mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run $label"; exit 5; }
  cp glm5_ep_rank00.txt "rank00_${label}.txt" 2>/dev/null || true
  grep -hE "kv_save:|kv_load:|prefill_synth:|SENTINEL" "rank00_${label}.txt" 2>/dev/null
}

run A_save GLM5_PREFILL_SYNTH=$SYS   GLM5_KV_SAVE=$KVDIR
run B_load GLM5_PREFILL_SYNTH=$NEW   GLM5_KV_LOAD=$KVDIR
run C_full GLM5_PREFILL_SYNTH=$TOTAL

AB=$(grep -hoE "argmax=[0-9]+" rank00_B_load.txt 2>/dev/null|tail -1)
AC=$(grep -hoE "argmax=[0-9]+" rank00_C_full.txt 2>/dev/null|tail -1)
TB=$(grep -hoE "prefill_synth: [0-9]+ tok [0-9.]+ tok/s" rank00_B_load.txt 2>/dev/null|grep -oE "[0-9.]+ tok/s")
TC=$(grep -hoE "prefill_synth: [0-9]+ tok [0-9.]+ tok/s" rank00_C_full.txt 2>/dev/null|grep -oE "[0-9.]+ tok/s")
KVSZ=$(ls -la "$KVDIR/kv.bin" 2>/dev/null|awk '{printf "%.2f GB",$5/1e9}')
echo "=== KV-CACHE RESULT ==="
echo "saved KV: $KVSZ ($KVDIR/kv.bin)"
echo "B (load, $NEW new tok): $AB at $TB"
echo "C (full, $TOTAL tok):   $AC at $TC"
echo "VERDICT: $([ -n "$AB" ] && [ "$AB" = "$AC" ] && echo "MATCH (cache lossless)" || echo "MISMATCH")"
echo "=== kvcache test done $(date) ==="
