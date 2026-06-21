#!/bin/bash
# Decisive correctness control: does the PRISTINE committed binary (HEAD transformer.h,
# no attn_pp) emit token-0 "!!!!" too? If yes -> the degenerate output is pre-existing,
# not caused by Workstream C. If pristine emits sensible varied tokens -> my struct/alloc
# change regressed even the dormant path. Predictable prompt so we can eyeball sense.
# ONE output file + sentinel; per-rank stderr by hostname; watchdog+sweep+gap per run.
set -u
cd "$(dirname "$0")"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
OUT=pristine_check.out; : > "$OUT"
CAP=${CAP:-280}; MAXGEN=${MAXGEN:-16}
MODEL=/local/qwen36/27b/Qwen3.6-27B-BF16-00001-of-00002.gguf
export GGUF_LAZY_MMAP=1 LLM_THREADS=48 OMP_NUM_THREADS=48
export TP_PROMPT="The capital of France is" TP_MAXGEN=$MAXGEN TP_DUMP_TOKENS=1
export TP_IGNORE_EOS=1 QWEN27B_LOCAL_DIR=/local/qwen36/27b

say(){ echo "$@" | tee -a "$OUT"; }
sweep(){ mpiexec -np 11 -vcoordfile vcoord_no0.txt sh -c \
    'pkill -9 -f build/tp_runner >/dev/null 2>&1; true' >/dev/null 2>&1; }

run_one(){  # $1=binary $2=label  (env TF_ATTN_PP optional, exported by caller)
    rm -f tp_tokens_rank00.txt tp_load_rank*.txt rk_*.txt
    say "=== run $2  bin=$1  TF_ATTN_PP=${TF_ATTN_PP:-unset}  $(date +%H:%M:%S) ==="
    timeout "$CAP" mpiexec -np 11 -vcoordfile vcoord_no0.txt \
        sh -c 'exec "$@" > rk_$(hostname).txt 2>&1' sh "$1" "$MODEL"
    local rc=$?
    local lf; lf=$(ls tp_load_rank*.txt 2>/dev/null | wc -l)
    cp -f tp_tokens_rank00.txt "tokens_$2.txt" 2>/dev/null
    local toks; toks=$(tr '\n' ' ' < "tokens_$2.txt" 2>/dev/null)
    local nz; nz=$(grep -vc '^0$' "tokens_$2.txt" 2>/dev/null || echo 0)
    say "run $2: rc=$rc load=$lf/11 (124=hang)  nonzero_tokens=$nz"
    say "  token ids: $toks"
    # rank0 decoded text line if present
    local r0; r0=$(grep -l 'tensor-parallel decode' rk_*.txt 2>/dev/null | head -1)
    [ -n "$r0" ] && { say "  rank0 tail:"; tail -3 "$r0" | sed 's/^/    /' | tee -a "$OUT"; }
}

say "=== build PRISTINE (stash transformer.h) $(date +%H:%M:%S) ==="
STASHED=0
if git stash push -- ../../common/transformer.h >>"$OUT" 2>&1; then STASHED=1; fi
make tp_runner CC=fcc OPENMP=1 >>"$OUT" 2>&1 && cp -f build/tp_runner build/tp_runner_pristine && say "PRISTINE_BUILD_OK"
if [ "$STASHED" = 1 ]; then git stash pop >>"$OUT" 2>&1 && say "STASH_POP_OK"; fi
say "=== rebuild MINE (attn_pp) $(date +%H:%M:%S) ==="
make tp_runner CC=fcc OPENMP=1 >>"$OUT" 2>&1 && cp -f build/tp_runner build/tp_runner_mine && say "MINE_BUILD_OK"

unset TF_ATTN_PP
run_one build/tp_runner_pristine PRISTINE
sweep; sleep 8
export TF_ATTN_PP=0
run_one build/tp_runner_mine MINE_OFF
sweep; sleep 8
export TF_ATTN_PP=1
run_one build/tp_runner_mine MINE_ON

say "=== VERDICT ==="
np=$(grep -vc '^0$' tokens_PRISTINE.txt 2>/dev/null || echo 0)
if [ "${np:-0}" -gt 0 ]; then
    say "PRISTINE_PRODUCES_NONZERO=$np  -> degenerate output is NOT pre-existing; my change or config regressed"
else
    say "PRISTINE_ALL_ZERO -> degenerate '!!!!' is PRE-EXISTING (committed HEAD), not Workstream C"
fi
say "PRISTINE_CHECK_DONE $(date +%H:%M:%S)"
