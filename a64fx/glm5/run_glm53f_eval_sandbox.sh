#!/bin/bash
# Explicitly invoke with bash. The sandbox has no network, host home, or repo.
# Work directory must be dedicated evaluation output; it is the only writable bind.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 VENV EVALPLUS_CACHE WORK_DIR mbpp|humaneval" >&2
    exit 2
fi
venv=$(realpath "$1")
cache=$(realpath "$2")
work=$(realpath "$3")
dataset=$4
[[ $dataset == mbpp || $dataset == humaneval ]] || exit 2
[[ -f $work/samples.jsonl && -f $work/problems.jsonl && -f $venv/pyvenv.cfg ]] || exit 2
[[ ! -e $work/samples_eval_results.json ]] || { echo "Use a fresh work directory; existing results must not be reused." >&2; exit 2; }
[[ $work != / && $work != "$PWD" && $work != "$HOME" ]] || exit 2
# Apply the process cap inside the user namespace: an outer RLIMIT_NPROC
# counts unrelated jobs belonging to the host user and can prevent bwrap
# from creating its namespace before any generated code executes.
exec timeout --kill-after=10 1800 prlimit --as=17179869184 --cpu=1600 -- \
    bwrap --unshare-all --die-with-parent --new-session --clearenv \
    --setenv PATH /usr/bin:/bin --setenv LANG C.UTF-8 \
    --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind "$venv" /venv --ro-bind "$cache" /data \
    --bind "$work" /work --proc /proc --dev /dev --tmpfs /local \
    --setenv TMPDIR /local --setenv XDG_CACHE_HOME /work/cache \
    --setenv HUMANEVAL_OVERRIDE_PATH /work/problems.jsonl \
    --setenv MBPP_OVERRIDE_PATH /work/problems.jsonl \
    --setenv OPENBLAS_NUM_THREADS 1 --setenv OMP_NUM_THREADS 1 \
    --chdir /work /usr/bin/prlimit --nproc=32 -- /venv/bin/python -m evalplus.evaluate \
    --dataset "$dataset" --samples /work/samples.jsonl --parallel 4 \
    --min-time-limit 4 --gt-time-limit-factor 4 --test-details --i-just-wanna-run
