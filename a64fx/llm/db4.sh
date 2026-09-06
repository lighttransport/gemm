#!/bin/bash
# [4/4] of the headline: batched decode throughput (DS4F_DB_BENCH sweep to M=16).
# Standalone + FULL LOG on purpose: bench_headline_12n.sh piped this run straight into grep, so when
# it produced no matching line there was nothing left to diagnose from. Never pipe a run's only
# output into a filter.
# Correctness of this path rests on VERIFY_GATE (DB_BENCH only TIMES steps; it never inspects output).
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0
export DS4F_CMP_LOCAL=1 DS4F_HC_SVE=1 DS4F_MV_FUSE=1 LLM_THREADS=47

pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null; pkill -x plexec 2>/dev/null
for i in $(seq 1 60); do
    [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec|plexec' || true)" = "0" ] && break
    sleep 1
done

DS4F_DB_BENCH=1 DS4F_DB_MAXM=${DS4F_DB_MAXM:-16} DS4F_DB_NTOK=32 DS4F_PREFILL=32 DS4F_MAXGEN=4 \
  ./run_ds4fbase_12n.sh > db4.log 2>&1
echo "EXIT=$?" >> db4.log
pkill -x ds4f_ep_runner 2>/dev/null; sleep 2; pkill -x mpiexec 2>/dev/null
