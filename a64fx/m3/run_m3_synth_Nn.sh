#!/bin/bash
# MiniMax-M3 (text) SYNTHETIC expert-parallel runner on N A64FX nodes.
#
# No weight staging — each rank fills its e%N expert shard + replicated dense in
# HBM, then runs synthetic prefill+decode with a per-MoE-layer tp_allreduce_sum
# combine. Validates: per-node memory fit, cross-rank lockstep (identical synthetic
# argmax on every rank), and prefill/decode tok/s split into compute vs all-reduce.
#
# Run INSIDE an interactive allocation (NO pjsub), mirroring run_ds4f_11n.sh:
#   ./run_m3_synth_Nn.sh                          # NP=11, default knobs
#   NP=7 M3_LAYERS=6 M3_EXPERTS=14 ./run_m3_synth_Nn.sh
#   NP=11 M3_LAYERS=8 M3_EXPERTS=22 M3_MAXPOS=512 ./run_m3_synth_Nn.sh
#
# The full model does NOT fit a small alloc; use M3_LAYERS / M3_EXPERTS to truncate
# (replicated dense + owned experts must fit ~28 GB/node). By default the
# relative-(0,0,0) node (co-located login/claude) is excluded -> NP ranks.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
M3_DIR="$(cd "$(dirname "$0")" && pwd)"
LLM_DIR="$M3_DIR/../llm"
UTOFU_DIR="$M3_DIR/../utofu-tests"
cd "$M3_DIR"

NP=${NP:-11}
EXCLUDE=${EXCLUDE:-0,0,0}
VCOORD=${VCOORD:-vcoord_m3.txt}

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
    if [ "$n" -lt "$NP" ]; then echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n < NP=$NP" >&2; exit 1; fi
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
    MPI_PLACE=(-vcoordfile "$VCOORD")
    echo "[run_m3] placing $NP ranks (excluding relative $EXCLUDE):"; sed 's/^/    /' "$VCOORD"
else
    echo "[run_m3] EXCLUDE=none: whole alloc (-np $NP)"
fi

# ---- harness knobs (exported so mpiexec forwards them) ----
export LLM_THREADS=${LLM_THREADS:-48}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}
export M3_PREFILL=${M3_PREFILL:-8}
export M3_DECODE=${M3_DECODE:-16}
export M3_MAXPOS=${M3_MAXPOS:-512}
export M3_LAYERS=${M3_LAYERS:-8}      # truncate (0 = full 60, won't fit small alloc)
export M3_EXPERTS=${M3_EXPERTS:-0}    # 0 = full 128 (uneven e%NP ok); set smaller to shrink

# ---- build (native) ----
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
make -C "$LLM_DIR" m3_ep_runner CC=fcc OPENMP=1 >/dev/null
BIN="$LLM_DIR/build/m3_ep_runner"
[ -x "$BIN" ] || { echo "build failed: $BIN missing" >&2; exit 1; }

# ---- regenerate topo on exactly the placed nodes ----
# The topo helper intermittently MPI_Aborts with CODE=1907 (uTofu init flake on
# some allocs/nodes); retry a few times before giving up (a fresh alloc is the
# ultimate fix). See memory ds4p-perf-calibration.
if [ "${SKIP_TOPO:-0}" != "1" ]; then
    topo_ok=0
    for try in 1 2 3 4 5; do
        rm -f tofu_topo.txt
        if mpiexec -np "$NP" "${MPI_PLACE[@]}" "$UTOFU_DIR/tofu_topo_helper" \
           && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
            topo_ok=1; break
        fi
        echo "[run_m3] topo helper try $try failed (CODE=1907 flake?); retrying..." >&2
        sleep 3
    done
    [ "$topo_ok" = 1 ] || { echo "FATAL: tofu_topo_helper failed $try times (1907 flake — try a fresh alloc)" >&2; exit 3; }
fi
echo "--- tofu topo ($(wc -l < tofu_topo.txt) rows) ---"; cat tofu_topo.txt

echo "--- launching m3_ep_runner (NP=$NP, layers=$M3_LAYERS experts=$M3_EXPERTS) ---"
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN"
echo "--- rank0 log ---"; cat m3_ep_rank00.txt 2>/dev/null || true
