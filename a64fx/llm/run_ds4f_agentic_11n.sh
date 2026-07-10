#!/bin/bash
# DS4F agentic-coding config for a ~12-node INTERACTIVE alloc (11 EP nodes; the 12th is reserved for
# the claude/login control process and excluded). Real-weight greedy generation tuned for coding
# tasks: the quality-preserving decode bundle (Tier-B2 indexer + mHC + bf16-pv dense + Q8) with the
# NUMA lever ON, sized to fit 11 EP nodes. Thin wrapper over run_ds4f_gen_11n.sh (which tokenizes the
# prompt, runs gen-mode, detokenizes) -> run_ds4f_11n.sh (topo + NUMA pinning + mpiexec).
#
# Context ceiling @11 EP nodes — use the EMPIRICAL figure. The pure weights+MLA-KV model says bf16
# fits ~64k, but the Tier-B2 indexer/compressed caches + warm-fill scratch push the real ceiling much
# lower: the ops log OOMs at ctx~32k, SAFE ~16k (sweet spot ~10k). An OOM kills the whole alloc
# (gotcha #6: SIGKILL degrades PMIx -> later launches die pre-load). DON'T probe past the ceiling.
#   KVBITS=16  bf16 KV (default) -> safe ~16k tokens (prompt+gen)
#   KVBITS=8   int8 KV (+ DS4F_INT8_CMP=1 for the indexer cache) -> extends, unproven
#   longer:    DS4F_CP=1 / run_ds4f_longctx_11n.sh -> 512k-1M (the validated long-ctx path)
#
# Prereq: weights staged to /local/ds4f on the SAME node set (run_ds4f_stage_11n.sh).
# Usage (inside the live 12-node alloc, from a64fx/llm):
#   PROMPT_FILE=task.txt ./run_ds4f_agentic_11n.sh                        # 64k-ceiling coding run
#   PROMPT_FILE=task.txt KVBITS=8 MAX_NEW=1024 ./run_ds4f_agentic_11n.sh  # 128k ceiling, longer gen
set -e
cd "$(dirname "$0")"
KVBITS=${KVBITS:-16}
export MAX_NEW=${MAX_NEW:-512}          # agentic completions run long; override as needed
export DS4F_NUMA=${DS4F_NUMA:-1}        # the ~1.40x bit-identical decode lever (in-runner interleave)
# quality-preserving decode bundle (== ds4f_ep_runner --preset decode); pinned so the config is explicit
# DS4F_TP_HEAD: vocab-shard the replicated bf16 lm_head. Memory (-0.96 GB RSS) AND decode-speed lever
# (11n A/B: 13.07->13.37 tok/s, +2.3%, after routing greedy decode through the cheap argmax-merge instead
# of a full-vocab all-reduce). BIT-EXACT (gen_ids 64/64 identical), Q8_DENSE-independent, no-op single-node.
# DS4F_TP_EMBED: vocab-shard the input embedding table. Pure MEMORY lever (-0.97 GB RSS, decode speed
# unchanged: only a 16 KB [hidden] reduce/token, not a full-vocab one). BIT-EXACT, no-op single-node.
export DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1
export DS4F_TP_HEAD=${DS4F_TP_HEAD:-1}
export DS4F_TP_EMBED=${DS4F_TP_EMBED:-1}
# Batched-verify prefill (comm-amortize + batched qproj/mHC): +74% prefill (13.3->23 tok/s), COHERENT
# not bit-identical (K-tile GEMM reassoc). Default ON for agentic coding (accepts a valid greedy path);
# set DS4F_PREFILL_GEMM=0 for bit-reproducible token-by-token prefill.
export DS4F_PREFILL_GEMM=${DS4F_PREFILL_GEMM:-1}
# 2026-07-10 session levers (real-weight 11n A/B'd; ds4f.md "2026-07-10 session"): SVE mHC
# (decode +12%), verify compute-shard + K=64 chunks (prefill 23.3->29.3 tok/s), wq_a+wkv fuse.
export DS4F_HC_SVE=${DS4F_HC_SVE:-1}
export DS4F_PF_TP=${DS4F_PF_TP:-1}
export DS4F_PREFILL_K=${DS4F_PREFILL_K:-64}
export DS4F_MV_FUSE=${DS4F_MV_FUSE:-1}
case "$KVBITS" in
  16) export DS4F_INT8_KV=0; CEIL="~16k safe" ;;
  8)  export DS4F_INT8_KV=1; CEIL="~16k+ (int8; add DS4F_INT8_CMP=1)" ;;
  *)  echo "KVBITS must be 16 (bf16 KV) or 8 (int8 KV)"; exit 2 ;;
esac
echo "[agentic] 11 EP nodes | KV=${KVBITS}-bit | ctx ceiling $CEIL | MAX_NEW=$MAX_NEW | NUMA=$DS4F_NUMA"
echo "[agentic] WARNING: OOM (past ~16k, or ctx~32k) kills the whole alloc (PMIx, gotcha #6)."
echo "[agentic] long ctx -> DS4F_CP=1 / run_ds4f_longctx_11n.sh (512k-1M). See a64fx/ds4f.md."
exec ./run_ds4f_gen_11n.sh
