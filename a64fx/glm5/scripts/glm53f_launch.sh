# Shared MPI launch/preflight helpers. Source after glm53f_env.sh.
require_allocation() {
    [ "${PJM_MPI_PROC:-0}" = 12 ] && [ "${PJM_NODE:-0}" = 12 ] && [ -n "${PJM_JOBID:-}" ] || fail 'run inside a 12-node, 12-rank allocation'
}
mpi_run() {
    local label=$1
    shift
    last_log="$logdir/$label-$run_tag"
    "$mpiexec_bin" -n 12 -of-proc "$last_log" "$@"
}
require_ranks() {
    local rank files
    for rank in {0..11}; do
        files=("$last_log".*."$rank")
        [ "${#files[@]}" = 1 ] && grep -Eq "$1" "${files[0]}" || fail "rank $rank did not pass: $last_log"
    done
}

prepare_runtime() {
    # Fail before resident allocation if any rank is missing a staged manifest.
    stages=("$routed" '' "$shared" '' "$core" .core "$GLM53F_Q2_EMBED_STAGE" '' "$GLM53F_Q2_HEAD_STAGE" '')
    if [ "$native" = 1 ]; then
        for kind in dense sparse kda shexp; do key=GLM53F_Q2_${kind^^}_STAGE; stages+=("${!key}" ''); done
    fi
    mpi_run preflight bash -c '
        rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
        while [ "$#" -gt 0 ]; do
            printf -v manifest "%s/rank%02d%s.manifest" "$1" "$rank" "$2"
            [ -s "$manifest" ] || { echo "missing stage: $manifest" >&2; exit 2; }
            shift 2
        done
        echo GLM53F_STAGE_PREFLIGHT=PASS
    ' bash "${stages[@]}"
    require_ranks 'GLM53F_STAGE_PREFLIGHT=PASS'
    if [ -n "${GLM53F_TOPO_PATH:-}" ]; then
        topo_path=$(realpath -m "$GLM53F_TOPO_PATH")
    else
        topo_dir="$logdir/topology-$run_tag"
        mkdir -p "$topo_dir"
        (cd "$topo_dir" && "$mpiexec_bin" -n 12 "$GLM53F_BIN_DIR/tofu_topo_helper")
        topo_path="$topo_dir/tofu_topo.txt"
    fi
    [ "$(grep -vc '^#' "$topo_path")" = 12 ] || fail 'topology must contain 12 ranks'
    export GLM53F_UTOFU=1 TOFU_TOPO_PATH=$topo_path
}
