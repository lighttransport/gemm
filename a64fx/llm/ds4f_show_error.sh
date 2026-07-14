#!/bin/bash
# ds4f_show_error.sh — print WHY a ds4f run failed. Source it, or run it directly.
#
# WHY THIS EXISTS. mpiexec does not forward rank stdout/stderr, so the runner writes its diagnostics
# to per-rank files. The wrappers used to grep the *mpiexec stdout log* for crash context -- a file
# that never contains the error. The result: a run that died with a perfectly clear message
#
#     ds4f_load: cannot open /local/ds4f/rank00.manifest: No such file or directory
#     ds4f_load_real: no staged blob for rank 0 in /local/ds4f (run ds4f_stage first)
#
# presented to the operator as "died in 3s, no output, cause unknown". Hours were spent on that.
# The engine's failures were always legible; the plumbing hid them. This prints them.
#
#   ds4f_show_error [logdir]      # default: ./logs/latest

ds4f_show_error() {
    local ldir="${1:-${DS4F_LOG_DIR:-logs}/latest}"
    echo "---- ds4f failure diagnostics (${ldir}) ----"
    if [ ! -d "$ldir" ]; then
        echo "  no log dir at '$ldir'."
        echo "  If the run died in ~3s with NO output at all, mpiexec never launched the ranks --"
        echo "  that is a LAUNCHER/allocation problem, not a model problem. Check the job is alive:"
        echo "      mpiexec -np 1 /bin/sh -c 'hostname > \$HOME/livecheck.txt'"
        return 1
    fi
    local found=0
    for f in "$ldir"/rank*.err; do
        [ -s "$f" ] || continue
        # the interesting lines: our fatal path, the loader's messages, and any crash
        local hits
        hits=$(grep -iE 'FATAL|out of memory|cannot open|no staged blob|MISSING tensor|segmentation|sig(segv|bus|fpe)|backtrace|abort|Killed|will not fit|DS4F_EXACT' "$f")
        if [ -n "$hits" ]; then
            found=1
            echo "  == $(basename "$f") =="
            printf '%s\n' "$hits" | sed 's/^/    /' | head -20
        fi
    done
    if [ "$found" = 0 ]; then
        echo "  no fatal lines in any rank*.err. Tail of rank00.err:"
        tail -8 "$ldir/rank00.err" 2>/dev/null | sed 's/^/    /'
    fi
    echo "-------------------------------------------"
}

# allow direct execution as well as sourcing
if [ "${BASH_SOURCE[0]}" = "$0" ]; then ds4f_show_error "$@"; fi
