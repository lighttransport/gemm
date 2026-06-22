#!/bin/bash
# GLM-5.2 context-parallel (CP) prefill: A/B the batched attention flash-combine.
# Synthetic weights, few layers, static CP from token 0 (GLM5_CP_THRESHOLD<0) so the
# per-token vs batched combine is exercised on every chunk. Run INSIDE an interactive
# allocation (no pjsub). Compares GLM5_CP_COMBINE_BATCH=1 (batched) vs =0 (per-token):
# argmax must be IDENTICAL; comm% / calls / frags should drop sharply for batched.
#   ./run_glm5_cp_combine_ab.sh                 # NP=12, 8 layers, pchunk=64
#   NP=12 PCHUNK=128 LAYERS=8 ./run_glm5_cp_combine_ab.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
GLM5_DIR="$(cd "$(dirname "$0")" && pwd)"
LLM_DIR="$GLM5_DIR/../llm"
UTOFU_DIR="$GLM5_DIR/../utofu-tests"
cd "$GLM5_DIR"

NP=${NP:-12}
EXCLUDE=${EXCLUDE:-none}
VCOORD=${VCOORD:-vcoord_glm5cp.txt}

MPI_PLACE=()
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"; n=0
    for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1))
    done; done; done
    [ "$n" -ge "$NP" ] || { echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n < NP=$NP" >&2; exit 1; }
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
    MPI_PLACE=(-vcoordfile "$VCOORD")
fi

# ---- CP synth harness knobs ----
export LLM_THREADS=${LLM_THREADS:-24}
export OMP_NUM_THREADS=$LLM_THREADS
export GLM5_REAL=0
export GLM5_LAYERS=${LAYERS:-8}
export GLM5_EXPERTS=${EXPERTS:-16}
export GLM5_CP_THRESHOLD=-1          # static CP (legacy env-CP), no auto-tier
export GLM5_CP=1                     # force context-parallel from token 0
export GLM5_MSA=1                    # sparse attention (Tier-B style)
export GLM5_INT4_KV=${INT4_KV:-0}    # bf16 CP KV (exact) for a clean A/B
export GLM5_PREFILL_SYNTH=${SYNTH:-1024}
export GLM5_PREFILL_ONLY=1
export GLM5_PCHUNK=${PCHUNK:-64}
export GLM5_MAXPOS=${MAXPOS:-2048}
export GLM5_PREFILL_ROLLING=0
# Under CP the runner forces ar_tokens=1 (max_count=hidden=6144), which is SMALLER than the
# per-token combine SUM payload (nh+nh*hd=16448), so per-token already fragments and batching
# can't merge it. Raise the registered slot so batching actually collapses the fragment count.
# (hard cap is 64 -> max_count=6144*64=393216 > 16448.) Same value for both A and B, so the only
# variable is GLM5_CP_COMBINE_BATCH. Set AR_TOKENS=1 to see the small-slot regime.
export GLM5_AR_TOKENS=${AR_TOKENS:-64}

make -C "$LLM_DIR" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
BIN="$LLM_DIR/build/glm5_ep_runner"

# ---- topo on the placed nodes (retry the 1907 init flake) ----
if [ "${SKIP_TOPO:-0}" != "1" ]; then
    for try in 1 2 3 4 5; do
        rm -f tofu_topo.txt
        mpiexec -np "$NP" "${MPI_PLACE[@]}" "$UTOFU_DIR/tofu_topo_helper" \
            && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ] && break
        echo "[run] topo try $try failed (1907 flake?); retrying..." >&2; sleep 3
    done
fi
echo "--- topo ($(wc -l < tofu_topo.txt) rows), NP=$NP layers=$GLM5_LAYERS experts=$GLM5_EXPERTS pchunk=$GLM5_PCHUNK synth=$GLM5_PREFILL_SYNTH th=$LLM_THREADS ---"

for BATCH in 1 0; do
    export GLM5_CP_COMBINE_BATCH=$BATCH
    echo "=== GLM5_CP_COMBINE_BATCH=$BATCH ==="
    mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN" >/dev/null 2>&1 || true
    grep -hE "prefill_synth:|CP (ON|TIERED)|allreduce:|GLM5_P_ATTN|SENTINEL" glm5_ep_rank00.txt 2>/dev/null || cat glm5_ep_rank00.txt
done
