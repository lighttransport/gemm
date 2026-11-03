#!/bin/bash
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 SOURCE_CPP REPORT_DIR" >&2; exit 2; }
source_file=$(realpath "$1")
work=$(realpath "$2")
runner=$(realpath "$(dirname "$0")/test_glm53f_task_queue.py")
[[ -f $source_file && -d $work && $work != / && $work != "$PWD" && $work != "$HOME" ]] || exit 2
# The process limit is set after entering the private user namespace below.
exec timeout --kill-after=10 120 prlimit --as=4294967296 --cpu=90 -- \
    bwrap --unshare-all --die-with-parent --new-session --clearenv \
    --setenv PATH /usr/bin:/bin --setenv LANG C.UTF-8 \
    --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind "$source_file" /source.cpp --ro-bind "$runner" /test.py \
    --bind "$work" /work --proc /proc --dev /dev --tmpfs /local \
    --setenv TMPDIR /local --chdir /local \
    /usr/bin/prlimit --nproc=16 -- /usr/bin/python3 /test.py /source.cpp /work/report.json
