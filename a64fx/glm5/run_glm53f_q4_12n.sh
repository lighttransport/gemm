#!/bin/bash
# UD-Q4_K_XL on 12 nodes.  GLM53F_NATIVE=1 (default) runs every matrix from
# the GGUF's own blocks; GLM53F_NATIVE=0 keeps the hybrid (Q4 routed experts,
# safetensors FP8/BF16 core and shared expert).
set -euo pipefail
job=${PJM_JOBID:?Run inside a 12-node interactive allocation}
export GLM53F_Q2_MODEL=${GLM53F_Q4_MODEL:-$HOME/models/glm53f-gguf-all/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
export GLM53F_NATIVE=${GLM53F_NATIVE:-1}
export GLM53F_NATIVE_PREFIX=${GLM53F_NATIVE_PREFIX:-/local/glm53f-q4-native-$job}
export GLM53F_Q2_EMBED_STAGE=${GLM53F_Q2_EMBED_STAGE:-/local/glm53f-q4-embed-$job}
export GLM53F_Q2_HEAD_STAGE=${GLM53F_Q2_HEAD_STAGE:-/local/glm53f-q4-head-$job}
export GLM53F_STAGE_DIR=${GLM53F_STAGE_DIR:-/local/glm53f-q4-routed-$job}
export GLM53F_SHARED_STAGE_DIR=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-q4-shared-$job}
export GLM53F_REPACK_STAGE_DIR=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-q4-core-$job}
export GLM53F_Q2_LOG_DIR=${GLM53F_Q2_LOG_DIR:-../../tmp/glm53f-q4-$job}
exec bash "$(dirname "$0")/run_glm53f_q2_12n.sh"
