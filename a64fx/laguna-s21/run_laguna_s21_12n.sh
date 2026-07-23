#!/bin/bash
# Laguna S-2.1 INT4 launcher for an existing interactive 1xNP A64FX allocation.
#   MODE = self-test | stage | generate
#
# generate: discovers Tofu topology, (optionally) stages weights node-local, then
# runs the EP forward pass + greedy decode across NP nodes over uTofu.
#
# Env / flags:
#   LAGUNA_NP           ranks (default 12; = PJM_MPI_PROC)
#   LAGUNA_MODEL_DIR    source safetensors (default ~/models/laguna-s21-int4)
#   LAGUNA_STAGE_DIR    node-local dest    (default /local/$USER/laguna-s21-ep$NP)
#   --prompt "text"     natural-language prompt (tokenized here)
#   --ids FILE          pre-tokenized ids (overrides --prompt)
#   --max-new N         tokens to generate (default 48)
#   --layers L          truncate to first L layers (bring-up; default 48)
#   --no-stage          skip node-local staging (blobs already present)
set -euo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UTOFU="$REPO/a64fx/utofu-tests"

MODE="${1:-self-test}"; shift || true
NP="${LAGUNA_NP:-${PJM_MPI_PROC:-12}}"
MODEL="${LAGUNA_MODEL_DIR:-$HOME/models/laguna-s21-int4}"
STAGE="${LAGUNA_STAGE_DIR:-/local/$USER/laguna-s21-ep$NP}"
PROMPT="The capital of France is"
IDS=""; MAX_NEW=48; LAYERS=48; DO_STAGE=1
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt)  PROMPT="$2"; shift 2;;
    --ids)     IDS="$2"; shift 2;;
    --max-new) MAX_NEW="$2"; shift 2;;
    --layers)  LAYERS="$2"; shift 2;;
    --no-stage) DO_STAGE=0; shift;;
    *) PASS+=("$1"); shift;;
  esac
done

make -C "$HERE" all CC="${CC:-fcc}" OPENMP=1 >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null 2>&1 || true

case "$MODE" in
  self-test) exec "$HERE/build/laguna_s21_ep_runner" --self-test ;;
  stage)
    export LAGUNA_MODEL_DIR="$MODEL" LAGUNA_STAGE_DIR="$STAGE" LAGUNA_EP_SIZE="$NP"
    exec mpiexec -np "$NP" "$HERE/build/laguna_s21_stage" "${PASS[@]}" ;;
  generate) ;;  # handled below
  *) echo "usage: $0 {self-test|stage|generate} [flags]" >&2; exit 2;;
esac

# ---- generate ----
RUN_DIR="$HERE/gen_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

# Topology discovery (writes ./tofu_topo.txt used by the runner).
for try in 1 2 3 4 5; do
  rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && break
  [ "$try" = 5 ] && { echo "topology discovery failed" >&2; exit 3; }
done
[ "$(grep -vc '^#' tofu_topo.txt)" -eq "$NP" ] || { echo "topo has wrong node count" >&2; exit 3; }

# Stage node-local (per-rank blobs) unless skipped.
if [ "$DO_STAGE" = 1 ]; then
  echo "staging weights to $STAGE (this takes a few minutes) ..."
  LAGUNA_MODEL_DIR="$MODEL" LAGUNA_STAGE_DIR="$STAGE" LAGUNA_STATUS_DIR="$RUN_DIR" LAGUNA_EP_SIZE="$NP" \
    mpiexec -np "$NP" "$HERE/build/laguna_s21_stage" >stage.stdout 2>stage.stderr
  staged=$(find "$RUN_DIR" -maxdepth 1 -name 'laguna_stage_rank*.txt' | wc -l)
  [ "$staged" -eq "$NP" ] || { echo "staging incomplete: $staged/$NP" >&2; exit 4; }
fi

# Tokenize prompt (unless --ids given).
if [ -z "$IDS" ]; then
  IDS="$RUN_DIR/prompt.ids"
  # BOS (id 2) is required — without it the model degenerates to copying the last token.
  python3 "$HERE/tools/laguna_tok.py" encode "$PROMPT" --bos > "$IDS"
fi
echo "prompt ids: $(cat "$IDS")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-47}"   # leave one core free (a64fx-omp-leave-one-core)
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}" OMP_PLACES="${OMP_PLACES:-cores}"
export FLIB_BARRIER="${FLIB_BARRIER:-HARD}"
export XOS_MMM_L_PAGING_POLICY="${XOS_MMM_L_PAGING_POLICY:-demand:demand:demand}"

mpiexec -np "$NP" "$HERE/build/laguna_s21_ep_runner" --generate \
    --ids "$IDS" --max-new "$MAX_NEW" --layers "$LAYERS" \
    --stage-dir "$STAGE" --gen-out "$RUN_DIR/gen.ids" "${PASS[@]}"

echo "--- generated text ---"
python3 "$HERE/tools/laguna_tok.py" decode-file "$RUN_DIR/gen.ids" || true
