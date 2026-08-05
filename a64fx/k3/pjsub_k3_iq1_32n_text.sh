#!/bin/bash
# 32-node IQ1 GGUF generation smoke test: dequant/kernel check plus text.
#PJM -g hp250467
#PJM -L "node=32"
#PJM --mpi "proc=32"
#PJM -L "elapse=02:00:00"
#PJM -j
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
exec "$S/pjsub_k3_iq1_32n.sh"
