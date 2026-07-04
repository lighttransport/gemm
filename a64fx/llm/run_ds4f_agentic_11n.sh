#!/bin/bash
# DS4F agentic-coding config for a ~12-node INTERACTIVE alloc (11 EP nodes; the 12th is reserved for
# the claude/login control process and excluded). Real-weight greedy generation tuned for coding
# tasks: the quality-preserving decode bundle (Tier-B2 indexer + mHC + bf16-pv dense + Q8) with the
# NUMA lever ON, sized to fit 11 EP nodes. Thin wrapper over run_ds4f_gen_11n.sh (which tokenizes the
# prompt, runs gen-mode, detokenizes) -> run_ds4f_11n.sh (topo + NUMA pinning + mpiexec).
#
# Memory ceiling @11 EP nodes (weights 22.7 GB/node, usable ~27; prompt+gen = context length):
#   KVBITS=16  bf16 KV (default) -> up to ~64k tokens
#   KVBITS=8   int8 KV           -> up to ~128k tokens
#   longer:    DS4F_CP=1 (context-parallel KV) -> 512k+   (see ds4f.md "Node configurations")
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
export DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1
case "$KVBITS" in
  16) export DS4F_INT8_KV=0; CEIL="~64k" ;;
  8)  export DS4F_INT8_KV=1; CEIL="~128k" ;;
  *)  echo "KVBITS must be 16 (bf16 KV) or 8 (int8 KV)"; exit 2 ;;
esac
echo "[agentic] 11 EP nodes | KV=${KVBITS}-bit (ctx ceiling $CEIL) | MAX_NEW=$MAX_NEW | NUMA=$DS4F_NUMA"
echo "[agentic] prompt+gen beyond $CEIL -> set KVBITS=8, or DS4F_CP=1 for 512k+ (see a64fx/ds4f.md)"
exec ./run_ds4f_gen_11n.sh
