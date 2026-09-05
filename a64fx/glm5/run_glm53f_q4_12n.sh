#!/bin/bash
# Q4_K_XL routed experts; shared experts and core retain checkpoint precision.
set -euo pipefail
job=${PJM_JOBID:?Run inside a 12-node interactive allocation}
export GLM53F_Q2_MODEL=${GLM53F_Q4_MODEL:-$HOME/models/glm53f-q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
export GLM53F_STAGE_DIR=${GLM53F_STAGE_DIR:-/local/glm53f-q4-routed-$job}
export GLM53F_SHARED_STAGE_DIR=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-q4-shared-$job}
export GLM53F_REPACK_STAGE_DIR=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-q4-core-$job}
export GLM53F_Q2_LOG_DIR=${GLM53F_Q2_LOG_DIR:-../../tmp/glm53f-q4-$job}
exec bash "$(dirname "$0")/run_glm53f_q2_12n.sh"
