#!/bin/bash
# P2 — MTP draft-acceptance alpha @32n (the spec-decode go/no-go gate). Needs the MTP block (layer 78)
# staged, so GLM5_STAGE_LAYERS=79 (NOT 78) — cannot share P1's stage. GLM5_MTP=1 loads the block +
# measures alpha = P(MTP draft == real next token) in the gen loop; decode stays byte-identical (draft
# is a side computation). alpha>=~0.7 -> worth building the batched-K=2 causal verify (~1.7 tok/verify).
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=32,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=32; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/mtp_alpha_run_${JOB_TAG}_32n"; NL=78
export GLM5_MODEL_DIR=$HOME/models/glm52-int8 GLM5_STAGE_DIR=/local/glm5_int8_mtp_$JOB_TAG
export GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=79 GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=2048 GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
export LLM_THREADS=48 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
echo "=== P2 MTP alpha @${NP}n (STAGE_LAYERS=79) job=${PJM_JOBID:-?} ==="; date
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging 79 (78L + MTP block) ($(date)) ---"
mpiexec -np $NP "$LLM/build/glm5_stage" 2>stage.err || { echo "FATAL: stage"; tail -20 stage.err; exit 4; }
echo "staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"
python3 -c "import struct;d=open('$HOME/eval_prompts/coding.bin','rb').read();a=struct.unpack('<%dI'%(len(d)//4),d);open('$WORK/prompt.ids','w').write(' '.join(map(str,a[:256])))"

echo "--- MTP alpha run (GLM5_MTP=1, MAX_NEW=128) ($(date)) ---"; rm -f glm5_ep_rank00.txt
env GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_MTP=1 GLM5_PROMPT_IDS="$WORK/prompt.ids" GLM5_MAX_NEW=128 GLM5_GEN_OUT="$WORK/mtp.ids" \
    mpiexec -np $NP numactl --interleave=all "$LLM/build/glm5_ep_runner" | grep -hE 'gen:|NaN'
cp glm5_ep_stderr_rank00.txt mtp_stderr.txt 2>/dev/null   # SAVE before the control run truncates it (runner freopen "w")
echo "=== MTP load message (rank0 stderr) ==="; grep -hiE "MTP block loaded|MTP load|incomplete|layer .* missing" mtp_stderr.txt 2>/dev/null | head
grep -hE "MTP block loaded|MTP_ALPHA|NaN" glm5_ep_rank00.txt 2>/dev/null; mv glm5_ep_rank00.txt mtp_rank00.txt 2>/dev/null

echo "--- control: GLM5_MTP=0 (main token stream must equal the MTP=1 run: draft is side-only) ---"; rm -f glm5_ep_rank00.txt
env GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_MTP=0 GLM5_PROMPT_IDS="$WORK/prompt.ids" GLM5_MAX_NEW=128 GLM5_GEN_OUT="$WORK/ref.ids" \
    mpiexec -np $NP numactl --interleave=all "$LLM/build/glm5_ep_runner" | grep -hE 'gen:|NaN'
cmp -s "$WORK/mtp.ids" "$WORK/ref.ids" && echo "TOKEN-IDENTICAL (MTP draft is side-only) ✓" || echo "*** TOKEN DIFF MTP vs ref — draft leaked into the main stream (bug)"
echo "SENTINEL glm5_mtp_alpha_32n=done"; date
