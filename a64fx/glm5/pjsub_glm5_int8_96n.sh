#!/bin/bash
# GLM-5.2 INT8 (compressed-tensors w8a16) FULL-MODEL prefill perf on 96 A64FX nodes.
# Full 78 layers, TP on (head-sharded attention), big allreduce messages. Measures prefill tok/s.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/int8_run_${JOB_TAG}"
NL=${GLM5_NL:-78}                                   # full model
SYN=${GLM5_SYNTH:-2048}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-int8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_int8_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-233}
export GLM5_STAGE_LAYERS=$NL
NGRP=${GLM5_NGRP:-1}                                # data-parallel groups: G groups of NP/G ranks,
export GLM5_EP_SIZE=$((NP/NGRP)) GLM5_PREFILL_GROUPS=$NGRP GLM5_STATUS_DIR="$WORK"  # AR over NP/G ranks (shrinks the group)
export GLM5_TP=${GLM5_TP:-1} GLM5_MAXPOS=${GLM5_MAXPOS:-2304} GLM5_KV_BUDGET_GB=${GLM5_KV_BUDGET_GB:-16}
export GLM5_AR_HARD_CAP=${GLM5_AR_HARD_CAP:-2048} GLM5_AR_AUTO_CAP=${GLM5_AR_AUTO_CAP:-2048}
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=${GLM5_THREAD_SWEEP:-12}; pc=${GLM5_PCHUNK_SWEEP:-$SYN}
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_COMM_OVERLAP=1 TP_AR_BF16=1"

echo "=== GLM5.2 INT8 FULL: NP=$NP layers=$NL synth=$SYN pchunk=$pc TP=$GLM5_TP ar_cap=$GLM5_AR_HARD_CAP model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt   # persist through intermittent uTofu-init windows
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo window on try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging full int8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP  rank00: $(head -1 "$WORK"/glm5_stage_rank00.txt)"
GEN=${GLM5_GEN_NEW:-0}
echo "--- prefill_synth $SYN tok gen=$GEN th=$th ($(date)) ---"; rm -f glm5_ep_rank00.txt
eval $RUNENV GLM5_PREFILL_SYNTH=$SYN GLM5_GEN_NEW=$GEN mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run"; tail -20 glm5_ep_rank00.txt 2>/dev/null; exit 5; }
echo "=== INT8 FULL RESULT ==="
grep -hE "msa_on|CP OFF|prefill_synth:|gen: [0-9]|last argmax|NaNs|SENTINEL|arena|PROFILE prefill_synth (wall|attn|o_proj|qkv|shared|experts|dense_ffn|router)" glm5_ep_rank00.txt | head -28
echo "=== done $(date) ==="
