# Sourced by run_glm53f_12n.sh after environment/allocation validation.
stage_gguf() {
    local kind=$1 destination=$2
    mpi_run "$kind-stage" "$GLM53F_BIN_DIR/glm53f_q2_${kind}_stage" "$gguf" "$destination"
    require_ranks "SENTINEL glm53f_q2_${kind}_stage=(OK|REUSE)"
}
stage_compact() {
    local kind=$1 source=$2 destination=$3
    mpi_run "$kind-stage" bash -c '
        rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
        exec "$1" "$2" "$3" "$rank" "$4"
    ' bash "$GLM53F_BIN_DIR/glm53f_core_stage" "$source" "$destination" "$kind"
    require_ranks 'SENTINEL glm53f_rank_stage=(OK|REUSE)'
}
stage_weights() {
    [ -f "$gguf" ] || fail "missing GGUF: $gguf"
    # Patching is restricted to node-local copies; never mutate source images.
    for destination in "$core" "$shared"; do
        case "$(realpath -m "$destination")" in /local/?*) ;; *) fail "stage destination must be below /local: $destination";; esac
        [ "$(realpath -m "$destination")" != "$(realpath -m "$core_source")" ] || fail "core source and destination overlap"
        [ "$(realpath -m "$destination")" != "$(realpath -m "$shared_source")" ] || fail "shared source and destination overlap"
    done
    [ "$(realpath -m "$core")" != "$(realpath -m "$shared")" ] || fail "core and shared stages must be distinct"
    mpi_run routed-stage "$GLM53F_BIN_DIR/glm53f_q2_stage" "$gguf" "$routed"
    require_ranks 'SENTINEL glm53f_q2_stage=(OK|REUSE)'
    stage_gguf embed "$GLM53F_Q2_EMBED_STAGE"
    stage_gguf head "$GLM53F_Q2_HEAD_STAGE"
    # Bounded, resumable copies replace the former rm -rf + cp -a sequence.
    stage_compact core "$core_source" "$core"
    stage_compact model "$shared_source" "$shared"
    if [ "$native" = 1 ]; then
        for kind in dense sparse kda shexp; do
            key=GLM53F_Q2_${kind^^}_STAGE
            stage_gguf "$kind" "${!key}"
        done
        local rank_exec='rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}; exec "$1" "$2" "$3" "$rank" "${@:4}"'
        mpi_run core-routers bash -c "$rank_exec" bash "$GLM53F_BIN_DIR/glm53f_core_add_routers" "$model" "$core"
        require_ranks 'GLM53F_CORE_ROUTERS.*PASS'
        mpi_run core-patch bash -c "$rank_exec" bash "$GLM53F_BIN_DIR/glm53f_q2_core_patch" "$gguf" "$core" all
        require_ranks 'SENTINEL glm53f_q2_core_patch=OK'
        mpi_run shared-patch bash -c "$rank_exec" bash "$GLM53F_BIN_DIR/glm53f_q2_shared_patch" "$gguf" "$shared"
        require_ranks 'SENTINEL glm53f_q2_shared_patch=OK'
    fi
    echo "SENTINEL glm53f_stage_12n=OK quant=$quant native=$native logdir=$logdir"
}
