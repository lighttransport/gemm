#!/bin/bash
# GLM-5.2 code generation with precomputed system prompts on 384 non-contiguous A64FX nodes.
# (A) prefill the 3-doc system prompt (gpt-5.5 + fable5 + opus48, ~108k tokens) and SAVE its KV;
# (B) LOAD that KV, prefill the C++ query, and GENERATE code tokens; then detokenize to C++ text.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=384,elapse=03:30:00"
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
JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/codegen_run_${JOB_TAG}"
KVDIR="$WORK/kv"; TOK=${GLM5_PROMPT_TOKENS:-$HOME/glm5_codegen.bin}
SYS=${GLM5_SYS_TOK:-108474}; NEW=${GLM5_NEW_TOK:-58}; GEN=${GLM5_GEN_NEW:-256}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_bf16_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-282}
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_TP_SHARED=1 GLM5_MAXPOS=${GLM5_MAXPOS:-16384} GLM5_KV_BUDGET_GB=${GLM5_KV_BUDGET_GB:-16}
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=${GLM5_THREAD_SWEEP:-48}; pc=${GLM5_PCHUNK_SWEEP:-256}
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=0 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_PROMPT_TOKENS=$TOK"

echo "=== GLM5.2 code-gen: NP=$NP sys=$SYS query=$NEW gen=$GEN model=$GLM5_MODEL_DIR maxpos=$GLM5_MAXPOS job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" "$KVDIR" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK"
echo "--- staging ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP"

echo "--- A: precompute 3-doc system prompt KV ($(date)) ---"; rm -f glm5_ep_rank00.txt
eval $RUNENV GLM5_PREFILL_SYNTH=$SYS GLM5_KV_SAVE=$KVDIR mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: A precompute"; exit 5; }
cp glm5_ep_rank00.txt rank00_A_precompute.txt; grep -hE "kv_save:|prefill_synth:|SENTINEL" rank00_A_precompute.txt

echo "--- B: load KV + generate C++ ($(date)) ---"; rm -f glm5_ep_rank00.txt
eval $RUNENV GLM5_PREFILL_SYNTH=$NEW GLM5_KV_LOAD=$KVDIR GLM5_GEN_NEW=$GEN GLM5_GEN_OUT=$WORK/gen.ids mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: B generate"; exit 5; }
cp glm5_ep_rank00.txt rank00_B_generate.txt; grep -hE "kv_load:|gen:|GEN_IDS" rank00_B_generate.txt | head

echo "=== GENERATED C++ (detokenized) ==="
[ -s "$WORK/gen.ids" ] && python3 "$GLM5/glm5_tokenizer.py" decode-file "$WORK/gen.ids" | tee "$WORK/generated.cpp"
echo "=== codegen done $(date) ==="
