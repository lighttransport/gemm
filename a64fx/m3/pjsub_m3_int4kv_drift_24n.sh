#!/bin/bash
# FU1 — quantify int4-KV DRIFT vs bf16-KV on longer/harder prompts (real weights, 24n fp8). The quality
# gate (job 49442681) showed int4-KV coherent but lossy (early EOS on 1 short prompt). This gens several
# prompts (MAX_NEW=128) with bf16-KV vs int4-KV and measures the MATCHING TOKEN-PREFIX length (how many
# greedy tokens agree before the int4 quantization forks the trajectory) + shows both texts for coherence.
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_int4kv_drift_24n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=2x3x4:torus,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=24"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-24}
RUN="$M3/int4drift_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1 TP_AR_BF16=1
export M3_MAXPOS=${M3_MAXPOS:-320} M3_MAX_NEW=${M3_MAX_NEW:-128}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 int4-KV DRIFT 24n fp8: NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[d] topo try $t"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }
echo "--- staging fp8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$RUN"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

prefix_match(){ awk 'NR==FNR{n=split($0,a," ");next}{m=split($0,b," ");c=0;for(i=1;i<=(n<m?n:m);i++){if(a[i]==b[i])c++;else break};print c" / "n}' "$1" "$2"; }

pi=0
while IFS= read -r prompt; do
  [ -z "$prompt" ] && continue
  pi=$((pi+1)); echo "================ PROMPT $pi: [$prompt] ================"
  python3 "$M3/m3_tokenizer.py" encode "$prompt" --bos > "$M3/pid_$pi.txt" || { echo "encode fail"; continue; }
  for kv in "bf16 0" "int4 1"; do set -- $kv; lbl=$1; flag=$2
    rm -f m3_ep_rank00.txt
    M3_REAL=1 M3_INT4_KV=$flag M3_MSTREAM=1 M3_PROMPT_IDS="$M3/pid_$pi.txt" M3_GEN_OUT="$M3/g_${pi}_${lbl}.txt" \
      mpiexec -np "$NP" "$LLM/build/m3_ep_runner" >/dev/null 2>&1 || echo "[p$pi $lbl] gen rc=$?"
    echo "[p$pi $lbl] text: $(python3 "$M3/m3_tokenizer.py" decode-file "$M3/g_${pi}_${lbl}.txt" 2>/dev/null)"
  done
  echo "[p$pi] matching greedy-token prefix (int4 vs bf16): $(prefix_match "$M3/g_${pi}_bf16.txt" "$M3/g_${pi}_int4.txt")"
done < "$M3/prompts_drift.txt"
echo "SENTINEL m3_int4kv_drift_24n=done"; echo "=== done $(date) ==="
