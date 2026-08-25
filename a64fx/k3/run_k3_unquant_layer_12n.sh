#!/bin/bash
# Optimized native K3 BF16+MXFP4 layer decode in the current 12-node job.
# The wrapped harness stages one layer only and never submits a batch job.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export K3_THREADS=${K3_THREADS:-47}
export K3_PROFILE=${K3_PROFILE:-1}
export K3_COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-0}
export K3_COMM_BF16=${K3_COMM_BF16:-1}
export K3_AR_GROUPS=${K3_AR_GROUPS:-3}
export K3_MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-row-aligned}
export K3_CMG_REPLICATE=${K3_CMG_REPLICATE:-1}
export K3_BF16_PV=${K3_BF16_PV:-1}
export K3_SITU_FAST=${K3_SITU_FAST:-1}
export K3_FAST_EXP=${K3_FAST_EXP:-1}
export K3_COMM_SPARSE_ROW=${K3_COMM_SPARSE_ROW:-1}
export K3_COMM_ASYNC_LATENT=${K3_COMM_ASYNC_LATENT:-1}
export K3_PREFILL_PIPELINE=${K3_PREFILL_PIPELINE:-auto}
export K3_PREFILL_PANEL=${K3_PREFILL_PANEL:-1024}
# The CMG-local expert-copy experiment was measured slower on the live 12-node
# path; keep the switch available, but do not enable the rejected default.
export K3_MOE_CMG_REPLICATE=${K3_MOE_CMG_REPLICATE:-0}
export K3_MOE_CMG_REPLICATE_MAX=${K3_MOE_CMG_REPLICATE_MAX:-16}
export K3_SHARED_FUSED_TEAM=${K3_SHARED_FUSED_TEAM:-1}
export K3_MLA_SPLIT_PROJ=${K3_MLA_SPLIT_PROJ:-1}
export K3_MLA_FAST_GATE=${K3_MLA_FAST_GATE:-1}
export K3_MLA_FLASH8=${K3_MLA_FLASH8:-1}
export K3_MLA_QK_MODE=${K3_MLA_QK_MODE:-auto}
export K3_MLA_FLASH_TRACE=${K3_MLA_FLASH_TRACE:-0}
export K3_SERIAL_VECTOR_OPS=${K3_SERIAL_VECTOR_OPS:-1}
export K3_COMM_DEFER_TCQ=${K3_COMM_DEFER_TCQ:-1}
export K3_COMM_DEFER_MRQ=${K3_COMM_DEFER_MRQ:-1}
export K3_MOE_LATE_SHARED_REDUCE=${K3_MOE_LATE_SHARED_REDUCE:-1}
# bit 0: attention output; bit 1: KDA final output; bit 2: MLA final output.
export K3_COMM_RABENSEIFNER=${K3_COMM_RABENSEIFNER:-7}
export K3_MOE_SCALE_ACTIVATION=${K3_MOE_SCALE_ACTIVATION:-1}

exec "$SCRIPT_DIR/run_k3_full_12n.sh" --expert-tp "$@"
