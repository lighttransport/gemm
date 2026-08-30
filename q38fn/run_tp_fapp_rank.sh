#!/bin/sh
set -eu

# Run under mpiexec.  Profiling every rank multiplies FAPP overhead and creates
# redundant reports; rank 0 is representative for TP compute and stage 0.
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
profile_rank=${Q38FN_FAPP_RANK:-0}
profile_dir=${Q38FN_FAPP_DIR:-/local/u14346/q38fn-tp-fapp}
profile_event=${Q38FN_FAPP_EVENT:-statistics}
profile_level=${Q38FN_FAPP_LEVEL:-0}

if [ "$rank" = "$profile_rank" ]; then
    mkdir -p "$profile_dir"
    exec fapp -C -d "$profile_dir" -Icpupa,nompi -L "$profile_level" \
        -Hevent="$profile_event" "$@"
fi
exec "$@"
