#!/bin/bash
# Regenerate the 3 GLM-5.2 int8 system-prompt KV caches, saving to ~/data (/vol0600, NOT HOME quota).
# Stages the int8 model once, then for each prompt: A_save (prefill SYS tok -> save Tier-A KV) + B_load verify.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=01:05:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-96}; JOB_TAG=${PJM_JOBID:-manual_$$}
OUTROOT=$HOME/data/glm5_kvcache
WORK="$OUTROOT/_logs_${JOB_TAG}"
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-int8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_int8_regen_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-233}
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1
export GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_TP_SHARED=1 GLM5_MAXPOS=${GLM5_MAXPOS:-32768}
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=${GLM5_PREFILL_ROLLING:-1}
export GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
th=48; pc=256
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=0 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1"
echo "=== GLM5.2 int8 KV-cache REGEN -> $OUTROOT  NP=$NP job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in $(seq 1 40); do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[regen] topo try $t failed; retry"; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK; staging ($(date))"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"
run() { local label=$1; shift; echo "--- run $label ($(date)) ---"; rm -f glm5_ep_rank00.txt
  eval $RUNENV "$@" mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run $label"; exit 5; }
  cp glm5_ep_rank00.txt "rank00_${label}.txt" 2>/dev/null || true
  grep -hE "kv_save:|kv_load:|prefill_synth:|SENTINEL" "rank00_${label}.txt" 2>/dev/null; }
# name | tokfile | sys | new
PROMPTS=(
  "gpt55|$HOME/glm5_gpt55_17384.bin|17384|0"
  "fable5|$HOME/glm5_fable5_1792.bin|1792|0"
  "opus48|$HOME/glm5_opus48_1792.bin|1792|0"
)
echo "=== KV-CACHE REGEN RESULTS ==="
for row in "${PROMPTS[@]}"; do
  IFS="|" read -r name tok sys new <<< "$row"
  KVDIR="$OUTROOT/$name/kv"; mkdir -p "$KVDIR"
  echo "### $name  tok=$tok sys=$sys ($(date))"
  run "${name}_A_save" GLM5_PREFILL_SYNTH=$sys GLM5_KV_SAVE=$KVDIR GLM5_PROMPT_TOKENS=$tok
  [ "$new" -gt 0 ] && run "${name}_B_load" GLM5_PREFILL_SYNTH=$new GLM5_KV_LOAD=$KVDIR GLM5_PROMPT_TOKENS=$tok
  KVSZ=$(stat -c%s "$KVDIR/kv.bin" 2>/dev/null||echo 0)
  AB=$(grep -hoE "argmax=[0-9]+" "rank00_${name}_B_load.txt" 2>/dev/null|tail -1)
  echo ">>> $name: kv.bin=$KVSZ bytes  B_load $AB  -> $KVDIR/kv.bin"
done
echo "=== regen done $(date) ==="
