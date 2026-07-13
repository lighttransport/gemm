#!/bin/bash
# Does decode tok/s depend on CONTEXT? The headline 22.45 was measured with the run script's
# DEFAULT synthetic prefill of 8 tokens. A real 70-token prompt reads 17.31. tb2prep (Tier-B2
# compressor/index prep) is the only big ctx-growing decode term -- so sweep prefill length and
# watch tb2prep. Same allocation, same flags: only ctx changes.
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0
export DS4F_CMP_LOCAL=1 DS4F_HC_SVE=1 DS4F_MV_FUSE=1 LLM_THREADS=47
free_nodes() {
    pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null; pkill -x plexec 2>/dev/null
    for i in $(seq 1 60); do
        [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec|plexec' || true)" = "0" ] && return 0
        sleep 1; done; }
: > ab_ctx.txt
for P in 8 64 256 1024; do
  free_nodes
  DS4F_PREFILL=$P DS4F_MAXGEN=16 DS4F_MAXPOS=$((P+64)) ./run_ds4fbase_12n.sh > ctx_$P.log 2>&1 || true
  { echo "### prefill(ctx)=$P"
    grep -hE "^decode:" ctx_$P.log 2>/dev/null | sed 's/^/    /'
    grep -hE "tb2prep" ctx_$P.log 2>/dev/null | tail -1 | sed 's/^/    /'
  } >> ab_ctx.txt
done
free_nodes
