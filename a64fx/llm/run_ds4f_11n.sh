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
# DS4F_PREFILL_BATCH=M_TILE>0 runs prefill as M-token GEMM tiles (needs EXACT+FP8_BF16).
# Sweet spot on 11 nodes is 64 (32L: batch64 70 tok/s > batch32 66 > batch128 59).
# M>=128 regresses (dense p_x/o_proj activations blow the 8MB/CMG L2). Default 0 (off).
# (Attention is now GEMM-ified per-head-block so its 32MB q/attn buffers no longer
#  bound L2 — that is why the old M>=128 cliff softened and 64 now wins.)
export DS4F_PREFILL_BATCH=${DS4F_PREFILL_BATCH:-0}
export DS4F_MAXGEN=${DS4F_MAXGEN:-16}
export DS4F_MAXPOS=${DS4F_MAXPOS:-4096}
# DS4F_CTX_WARM>0 fills synthetic KV+compressed caches to this ctx, then decodes
# from there (deterministic + rank-independent => lockstep preserved). Lets us
# measure long-ctx decode/attn cost without paying O(ctx^2) real prefill. Needs
# DS4F_MAXPOS > DS4F_CTX_WARM + DS4F_MAXGEN.
export DS4F_CTX_WARM=${DS4F_CTX_WARM:-0}
export DS4F_LAYERS=${DS4F_LAYERS:-0}
export DS4F_FP8_BF16=${DS4F_FP8_BF16:-0}
# DS4F_FP8_MAGIC=1 uses the gather-free "magic" FP8->f32 decode matvec (FTZ subnormals,
# exp==15 -> large finite) instead of the LUT-gather kernel; only active when dense is
# FP8 (DS4F_FP8_BF16=0). Default 0 (gather). Flip pending real-weight argmax-exact gate.
export DS4F_FP8_MAGIC=${DS4F_FP8_MAGIC:-0}
# DS4F_DENSE_MXFP4=1 routes replicated dense through MXFP4 split (0.53 B/elem):
# leaner than FP8 AND faster, but compute-bound so slower than bf16-pv. Lean
# long-ctx default candidate. Overrides FP8/BF16.
export DS4F_DENSE_MXFP4=${DS4F_DENSE_MXFP4:-0}
# DS4F_Q8_DENSE=1 repacks the 8 dominant dense tensors/layer (wq_a/wq_b/wkv/wo_a/wo_b/
# sh_w1/sh_w3/sh_w2) from bf16-pv to int8 W8A8 after load, so both decode matvec and prefill
# GEMM run svdot (~1.03 B/elem from HBM = half of bf16). Router gate + lm-head stay bf16-pv
# (argmax protection). REQUIRES DS4F_FP8_BF16=1 (the bf16-pv source); warn+no-op otherwise.
# The bf16 src is reclaimed via MADV_DONTNEED so net dense memory SHRINKS. VALIDATED: real-
# weight gen A/B is TOKEN-IDENTICAL to bf16 (per-64-block absmax scale preserves the greedy
# trajectory); decode 11.88->13.03 tok/s @ctx10240, RSS -5.5 GB. LOSSY (synthetic argmax
# shifts), so default OFF here to keep this base runner the clean bf16-pv reference; the
# perf/memory wrappers (run_ds4f_longctx_11n.sh, run_ds4f_gen_11n.sh) default it ON. (Lever 2l;
# the per-group o-proj fallback OPROJ_FUSE=0 is auto-overridden to fused for Q8 -- ds4f_row_slice
# can't slice the int8 layout.)
export DS4F_Q8_DENSE=${DS4F_Q8_DENSE:-0}
# DS4F_HC_PAR=1 parallelizes the mHC hc_pre/hc_post collapse/expand across the thread
# pool (was ~serial-scalar on tid0, the "other" phase). BIT-EXACT (disjoint per-stream/
# per-dim splits): real-weight 11n gen A/B TOKEN-IDENTICAL on vs off (64/64, NaNs=0,
# lockstep, 2026-06-11). Only active under DS4F_MHC=1; decode "other" 51.1->30.5 ms =>
# 9.46->11.80 tok/s (+24.7%). Default OFF here (clean reference); perf wrappers default ON.
export DS4F_HC_PAR=${DS4F_HC_PAR:-0}
# DS4F_HC_RMSPAR=1 (WS1b) folds the mHC RMS sum-of-squares INTO the mixes-matvec dispatch,
# killing the ~89us/call serial tid0 double-reduction (~35% of the post-HC_PAR "other").
# ss stays a double accumulation, so the parallel reassociation (~1e-13) is below the float
# rsq epsilon => BIT-IDENTICAL result: real-weight 11n gen A/B TOKEN-IDENTICAL on vs off
# (64/64, NaNs=0, lockstep, 2026-06-11). Only active under DS4F_MHC=1; decode "other"
# 30.5->24.1 ms => 11.80->12.77 tok/s (+8.2%). Default OFF here (clean ref); perf wrappers ON.
export DS4F_HC_RMSPAR=${DS4F_HC_RMSPAR:-0}
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
# DS4F_EXACT=1 swaps the dense forward stand-ins for the EXACT DeepSeek-V4-Flash
# math (RoPE/YaRN, per-head q-norm, MQA sliding-window+sink attn, grouped low-rank
# o-proj, sqrtsoftplus gate w/ selection bias, swiglu clamp). exact==0 is byte-
# identical to the stand-in path. Meaningful only with DS4F_REAL=1 (real weights).
export DS4F_EXACT=${DS4F_EXACT:-0}
# DS4F_MHC=1 enables exact manifold-constrained hyper-connections (4-stream).
export DS4F_MHC=${DS4F_MHC:-0}
# DS4F_TIERB2=1 enables the stateful compressor/indexer (CSA/HCA) decode path.
# Implies EXACT (q-norm/RoPE/window). With DS4F_REAL=1 the compressor/indexer
# tensors (staged as dense) are loaded by name and widened to f32 at load time.
export DS4F_TIERB2=${DS4F_TIERB2:-0}
# DS4F_IDX_INT8=1 swaps the Tier-B2 indexer index_score f32 svmla scan for a RESIDENT int8/SVE
# svdot scan: idx_kv quantized ONCE at write (per-position absmax scale) into idx_kv8, read int8
# directly (4 int8 MACs/lane). idx_kv is Hadamard-rotated+fp4'd at write so per-position int8 is
# safe. VALIDATED argmax-exact on real weights (96/96 ids). But NO decode win at <=16k ctx: the
# scan is only 1.2% of decode (TB2SCAN sub-timer), and M=1's per-query int8 quant (~85us) cancels
# the scan's 1.85x parallel win. Default 0 (f32). Projected payoff only at >=256k ctx (untestable,
# OOM). Kept as a gated building block. Conditional alloc => zero cost when off.
export DS4F_IDX_INT8=${DS4F_IDX_INT8:-0}
# DS4F_TOPK_NAIVE=1 forces the OLD O(k*T) linear-scan index_topk (reference). Default 0 =
# fast O(T*log k) heap+merge (selected set & order bit-identical; validated 432 trials +
# argmax-exact on real weights). The naive path was ~108ms/tok (37% of decode @ctx10240).
export DS4F_TOPK_NAIVE=${DS4F_TOPK_NAIVE:-0}
# DS4F_ATTN_SVE=0 forces the OLD scalar decode-attn inner loops (reference). Default 1 =
# SVE-widened dot/axpy over kv_lora=512 (PV bit-exact; QK dot reorders, argmax-safe).
export DS4F_ATTN_SVE=${DS4F_ATTN_SVE:-1}
# DS4F_OPROJ_FUSE=0 forces the OLD per-group o-proj (8 separate wo_a matvec dispatches,
# reference). Default 1 = fused block-diagonal wo_a in ONE pool dispatch (kills 7
# cross-CMG barriers/layer + load imbalance; BIT-EXACT, same kernel+accum order). o_proj
# was 28.8ms/tok = 25.6% of decode @ctx10240 (the dominant phase after attn/topk fixes).
export DS4F_OPROJ_FUSE=${DS4F_OPROJ_FUSE:-1}
# DS4F_QNR_PAR=0 forces the OLD serial per-head q-norm+RoPE (reference). Default 1 =
# split the n_heads heads across the pool (BIT-EXACT: heads independent, same per-head
# double-accum/scale/RoPE). Was 7.18ms/tok = 7.7% of decode @ctx10240 (scalar, serial on tid0).
export DS4F_QNR_PAR=${DS4F_QNR_PAR:-1}
# DS4F_TB2ROPE_PAR=0 forces the OLD serial per-head indexer RoPE+rotate+fp4 (reference).
# Default 1 = split the index_heads across the pool (BIT-EXACT, disjoint per-head slices).
# Was 4.68ms/tok = 5.0% of decode @ctx10240 (scalar, serial on tid0 inside ds4f_index_step).
export DS4F_TB2ROPE_PAR=${DS4F_TB2ROPE_PAR:-1}
# DS4F_INT8_KV=1 stores the window KV latent as int8 (per-channel STATIC scale calibrated
# on the first DS4F_INT8KV_CAL positions; S5 scheme), halving the KV footprint (the long-ctx
# memory dominator). LOSSY (~1% rel) -> argmax NOT bit-exact; coherence is the gate. Forces
# EXACT; incompatible with batched prefill. Default 0 (bf16 kv_cache). Off => zero cost.
export DS4F_INT8_KV=${DS4F_INT8_KV:-0}
export DS4F_INT8KV_CAL=${DS4F_INT8KV_CAL:-256}
# DS4F_INT8_CMP=1 stores the Tier-B2 compressed-latent cache cmp_kv as int8 (same S5
# static-per-channel scheme as INT8_KV, calibrated on the first DS4F_INT8CMP_CAL *slots*).
# cmp_kv is the dominant tierb2 physical (THP) footprint at high ctx; int8 reclaims ~3/4
# and lifts the ctx ceiling toward 256k. LOSSY -> coherence gate. Forces EXACT; tierb2 only.
# Default 0 (f32 cmp_kv). Off => zero cost.
export DS4F_INT8_CMP=${DS4F_INT8_CMP:-0}
export DS4F_INT8CMP_CAL=${DS4F_INT8CMP_CAL:-64}
export DS4F_PROF=${DS4F_PROF:-1}
export TF_HW_BARRIER=${TF_HW_BARRIER:-1}
# TP_AR_BF16=1 halves the EP-combine reduce payload (16KB->8KB/all-reduce).
# Synthetic harness => bf16-rounded reduce is quality-irrelevant; all ranks stay
# bitwise-identical (lockstep preserved). Default off; flip to cut comm.
export TP_AR_BF16=${TP_AR_BF16:-0}

echo "=== DS4F EP harness on $NP node(s) ($([ "$DS4F_REAL" = 1 ] && echo "REAL weights <- $DS4F_STAGE_DIR" || echo synthetic)$([ "$DS4F_EXACT" = 1 ] && echo " EXACT-math")$([ "$DS4F_TIERB2" = 1 ] && echo " TierB2")$([ "$DS4F_MHC" = 1 ] && echo " mHC")) ==="
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
if [ "${DS4F_QUIET:-0}" = "1" ]; then
    echo "--- tofu topo ($(wc -l < tofu_topo.txt) rows; DS4F_QUIET: see tofu_topo.txt) ---"
else
    echo "--- tofu topo ($(wc -l < tofu_topo.txt) rows) ---"; cat tofu_topo.txt
fi

# ---- launch the synthetic EP harness ----
echo "--- launching ds4f_ep_runner (NP=$NP) ---"
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN"

# Per-rank files are rank-identical by lockstep design (perf/argmax/RSS match across all
# ranks); DS4F_QUIET=1 prints only counts + the rank0 head, keeping interactive sessions
# lean. Full per-rank files remain on disk (ds4f_ep_{load,perf}_rank*.txt) for inspection.
if [ "${DS4F_QUIET:-0}" = "1" ]; then
    nload=$(ls ds4f_ep_load_rank*.txt 2>/dev/null | wc -l)
    nperf=$(ls ds4f_ep_perf_rank*.txt 2>/dev/null | wc -l)
    echo "=== per-rank compact (load=$nload/$NP perf=$nperf/$NP; ranks lockstep-identical; full files on disk) ==="
    echo "--- rank0 summary (first 20 lines; full in ds4f_ep_rank00.txt) ---"
    head -20 ds4f_ep_rank00.txt 2>/dev/null
else
    echo "=== per-rank load (alloc + first-touch + RSS) ==="; cat ds4f_ep_load_rank*.txt 2>/dev/null
    echo "=== per-rank perf (compute / all-reduce comm / GB-s) ==="; cat ds4f_ep_perf_rank*.txt 2>/dev/null
    echo "=== rank0 summary ==="; cat ds4f_ep_rank00.txt 2>/dev/null
fi
