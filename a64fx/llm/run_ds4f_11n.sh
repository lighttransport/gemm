#!/bin/bash
# DeepSeek-V4-Flash synthetic EP harness on 11 A64FX nodes (Stage 2).
#
# SYNTHETIC weights — no GGUF/safetensors staging, no disk I/O. Each rank fills
# its ~1/11 expert shard + replicated dense directly in HBM, then runs synthetic
# prefill + decode with a per-layer tp_allreduce_sum MoE combine. Validates:
# per-node memory fit (~20-26 GB), cross-rank lockstep, and decode/prefill tok/s.
#
# Run INSIDE the existing allocation (NO pjsub):
#   ./run_ds4f_11n.sh
#   DS4F_FP8_BF16=1 ./run_ds4f_11n.sh     # predequant dense FP8->BF16 (faster, +6 GB)
#   DS4F_LAYERS=6 DS4F_MAXGEN=8 ./run_ds4f_11n.sh   # quick smoke
#   NP=12 EXCLUDE=none ./run_ds4f_11n.sh  # use the WHOLE alloc (DANGER: OOM-kills claude)
#
# By default we exclude the relative-(0,0,0) node (where a co-located login/claude
# session runs) so its ~32 GB cgroup is not filled by a rank's load peak -> NP=11.
set -e
# Prepend (do NOT replace) so both mpiexec AND the MPI/native compilers
# (mpiclang for tofu_topo_helper, fcc for the runner) stay on PATH.
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"
UTOFU_DIR="$LLM_DIR/../utofu-tests"

NP=${NP:-11}
EXCLUDE=${EXCLUDE:-0,0,0}          # relative coord to drop; "none" keeps the full alloc
VCOORD=${VCOORD:-vcoord_ds4f.txt}

# ---- generate the vcoordfile (all shape coords except $EXCLUDE) ----
MPI_PLACE=()
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"
    n=0
    for ((x=0; x<SX; x++)); do for ((y=0; y<SY; y++)); do for ((z=0; z<SZ; z++)); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"
        n=$((n+1))
    done; done; done
    if [ "$n" -lt "$NP" ]; then
        echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n nodes < NP=$NP" >&2; exit 1
    fi
    # keep exactly NP lines (head, since the alloc may be larger than 11+1)
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
    MPI_PLACE=(-vcoordfile "$VCOORD")
    echo "[run_ds4f] excluding relative ($EXCLUDE); placing $NP ranks via $VCOORD:"
    cat "$VCOORD" | sed 's/^/    /'
else
    echo "[run_ds4f] EXCLUDE=none: using the whole alloc (-np $NP, no vcoordfile)"
fi

# ---- forward harness knobs to the ranks (mpiexec forwards EXPORTED env only) ----
export LLM_THREADS=${LLM_THREADS:-48}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}
export DS4F_CMGS=${DS4F_CMGS:-4}
export DS4F_PREFILL=${DS4F_PREFILL:-8}
export DS4F_MAXGEN=${DS4F_MAXGEN:-16}
export DS4F_MAXPOS=${DS4F_MAXPOS:-4096}
export DS4F_LAYERS=${DS4F_LAYERS:-0}
export DS4F_FP8_BF16=${DS4F_FP8_BF16:-0}
# DS4F_DENSE_MXFP4=1 routes replicated dense through MXFP4 split (0.53 B/elem):
# leaner than FP8 AND faster, but compute-bound so slower than bf16-pv. Lean
# long-ctx default candidate. Overrides FP8/BF16.
export DS4F_DENSE_MXFP4=${DS4F_DENSE_MXFP4:-0}
# DS4F_SPARSE=1 enables the Stage-4 synthetic lightning-indexer attention: on
# sparse layers (compress_ratios[L]!=0) with nP>index_topk, a cheap compressed
# index selects topk positions and weighted-V runs over them only. Dense layers
# (0/1/last) and short ctx (nP<=topk) stay full-attention (byte-identical).
# HCA(R=128) layers go O(topk) flat; CSA(R=4) go O(nP/R). Default off.
export DS4F_SPARSE=${DS4F_SPARSE:-0}
# leave EMPTY when unset so the runner's auto-coupling (pv on iff predequant on)
# decides; set DS4F_BF16_PV=0 to force plain bf16, =1 to force pv.
export DS4F_BF16_PV=${DS4F_BF16_PV:-}
# DS4F_REAL=1 loads REAL DeepSeek-V4-Flash weights from each node's staged blob
# (rank<rr>.blob/.manifest in DS4F_STAGE_DIR) instead of the synthetic fill.
# REQUIRES run_ds4f_stage_11n.sh to have run first (same vcoordfile/node set).
# Dense is forced to FP8 on-demand; the synth dense knobs above are ignored.
export DS4F_REAL=${DS4F_REAL:-0}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
export DS4F_PROF=${DS4F_PROF:-1}
export TF_HW_BARRIER=${TF_HW_BARRIER:-1}
# TP_AR_BF16=1 halves the EP-combine reduce payload (16KB->8KB/all-reduce).
# Synthetic harness => bf16-rounded reduce is quality-irrelevant; all ranks stay
# bitwise-identical (lockstep preserved). Default off; flip to cut comm.
export TP_AR_BF16=${TP_AR_BF16:-0}

echo "=== DS4F EP harness on $NP node(s) ($([ "$DS4F_REAL" = 1 ] && echo "REAL weights <- $DS4F_STAGE_DIR" || echo synthetic)) ==="
echo "threads=$LLM_THREADS prefill=$DS4F_PREFILL maxgen=$DS4F_MAXGEN max_pos=$DS4F_MAXPOS layers=${DS4F_LAYERS:-43} dense=$([ "$DS4F_REAL" = 1 ] && echo "FP8(real)" || ([ "$DS4F_FP8_BF16" = 1 ] && echo BF16 || echo FP8))"

# ---- build (native fcc + OpenMP) ----
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
make -C "$LLM_DIR" ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null
BIN="$LLM_DIR/build/ds4f_ep_runner"

# ---- clean per-rank artifacts from any prior run ----
rm -f ds4f_ep_perf_rank*.txt ds4f_ep_load_rank*.txt ds4f_ep_stderr_rank*.txt ds4f_ep_rank00.txt

# ---- regenerate topo on EXACTLY the placed nodes (writes tofu_topo.txt, ranks 0..NP-1) ----
if [ "${SKIP_TOPO:-0}" != "1" ]; then
    mpiexec -np "$NP" "${MPI_PLACE[@]}" "$UTOFU_DIR/tofu_topo_helper"
else
    echo "[run_ds4f] SKIP_TOPO=1: using existing tofu_topo.txt"
fi
echo "--- tofu topo ($(wc -l < tofu_topo.txt) rows) ---"; cat tofu_topo.txt

# ---- launch the synthetic EP harness ----
echo "--- launching ds4f_ep_runner (NP=$NP) ---"
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN"

echo "=== per-rank load (alloc + first-touch + RSS) ==="; cat ds4f_ep_load_rank*.txt 2>/dev/null
echo "=== per-rank perf (compute / all-reduce comm / GB-s) ==="; cat ds4f_ep_perf_rank*.txt 2>/dev/null
echo "=== rank0 summary ==="; cat ds4f_ep_rank00.txt 2>/dev/null
