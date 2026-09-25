# Sourced by the native launcher. No staging, builds, or MPI calls here.
glm53f_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
glm53f_repo=$(cd "$glm53f_dir/../.." && pwd -P)
job=${PJM_JOBID:-manual}
quant=${GLM53F_QUANT:-q4}
case "$quant" in
    q4)
        gguf=${GLM53F_GGUF:-${GLM53F_Q4_MODEL:-$HOME/models/glm53f-gguf-all/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}}
        native=${GLM53F_NATIVE:-1} ;;
    q2)
        gguf=${GLM53F_GGUF:-${GLM53F_Q2_MODEL:-${GLM53F_Q2_ROOT:-$HOME/models/glm53f-gguf}/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf}}
        native=${GLM53F_NATIVE:-0} ;;
    *) echo "error: GLM53F_QUANT must be q4 or q2" >&2; return 2 ;;
esac
case "$native" in 0|1) ;; *) echo "error: GLM53F_NATIVE must be 0 or 1" >&2; return 2;; esac
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_STAGE_DIR:-${GLM53F_Q4_STAGE_DIR:-/local/glm53f-$quant-routed-$job}}
core_source=${GLM53F_CORE_SOURCE:-$model/a64fx_ep12_v2_core}
shared_source=${GLM53F_SHARED_SOURCE:-$model/a64fx_ep12_v1/shared}
core=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-$quant-core-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-$quant-shared-$job}
export GLM53F_Q2_EMBED_STAGE=${GLM53F_Q2_EMBED_STAGE:-/local/glm53f-$quant-embed-$job}
export GLM53F_Q2_HEAD_STAGE=${GLM53F_Q2_HEAD_STAGE:-/local/glm53f-$quant-head-$job}
if [ "$native" = 1 ]; then
    native_prefix=${GLM53F_NATIVE_PREFIX:-/local/glm53f-$quant-native-$job}
    core=${GLM53F_GGUF_CORE_STAGE:-${GLM53F_REPACK_DIR:-$native_prefix-core}}
    shared=${GLM53F_GGUF_SHARED_STAGE:-$native_prefix-shared}
    for kind in dense sparse kda shexp; do
        key=GLM53F_Q2_${kind^^}_STAGE
        export "$key=${!key:-$native_prefix-$kind}"
    done
else
    unset GLM53F_Q2_DENSE_STAGE GLM53F_Q2_SPARSE_STAGE GLM53F_Q2_KDA_STAGE GLM53F_Q2_SHEXP_STAGE
fi
export GLM53F_REPACK_DIR=$core GLM53F_REPACK_REQUIRE=$native
# MPI executes the binary on every node: binaries must be on shared storage.
export GLM53F_BIN_DIR=$(realpath -m "${GLM53F_BIN_DIR:-$glm53f_dir/build/glm53f}")
case "$GLM53F_BIN_DIR/" in
    /local/*) echo 'error: GLM53F_BIN_DIR must be on shared storage, not /local' >&2; return 2 ;;
esac
scratch_root=/local
[ -d /local ] || scratch_root=$glm53f_repo/tmp
export GLM53F_BUILD_DIR=$(realpath -m "${GLM53F_BUILD_DIR:-$scratch_root/glm53f-build-$job}")
logdir=${GLM53F_LOG_DIR:-${GLM53F_Q4_LOG_DIR:-${GLM53F_Q2_LOG_DIR:-$glm53f_repo/tmp/glm53f-$quant-$job}}}
logdir=$(realpath -m "$logdir")
run_tag=${GLM53F_RUN_TAG:-${GLM53F_Q2_RUN_TAG:-$$}}
case "$run_tag" in
    ''|*[!a-zA-Z0-9_.-]*) echo 'error: invalid GLM53F_RUN_TAG' >&2; return 2 ;;
esac
export OPAL_PREFIX=${GLM53F_MPI_HOME:-/opt/FJSVxtclanga/tcsds-1.2.43}
export MPI_HOME=$OPAL_PREFIX
export PATH="/opt/local/mpiexec:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=$GLM53F_BUILD_DIR
export GLM53F_MPICC=${GLM53F_MPICC:-mpifcc}
export GLM53F_FAST_MATH=${GLM53F_FAST_MATH:-1}
export GLM53F_NO_MATH_ERRNO=${GLM53F_NO_MATH_ERRNO:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores FLIB_BARRIER=HARD
mpiexec_bin=${GLM53F_MPIEXEC:-mpiexec}
