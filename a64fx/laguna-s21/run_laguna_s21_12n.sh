#!/bin/bash
# Laguna S-2.1 INT4 launcher for an existing interactive 1xNP A64FX allocation.
#   MODE = self-test | stage | generate | serve
#
# generate: discovers Tofu topology, (optionally) stages weights node-local, then
# runs the EP forward pass + decode across NP nodes over uTofu.
# serve:    the same, but leaves an HTTP endpoint up (see tools/laguna_cli.py).
#
# Configuration is by flag, not environment (PJM_MPI_PROC is the one exception:
# it is supplied by the batch system, not by the user):
#   --np N              ranks (default 12, or PJM_MPI_PROC)
#   --model-dir DIR     source safetensors (default depends on --bf16/--fp8)
#   --stage-dir DIR     node-local dest
#   --nshards N         safetensors shard count of the source checkpoint
#   --port N            serve mode: HTTP port
#   --maxpos N          serve mode: largest context to accept
#   --prompt-cache FILE serialize/reuse the fixed system-prompt KV prefix
#   --prompt "text"     natural-language prompt (tokenized here)
#   --ids FILE          pre-tokenized ids (overrides --prompt)
#   --max-new N         tokens to generate (default 48)
#   --layers L          truncate to first L layers (bring-up; default 48)
#   --no-stage          skip node-local staging (blobs already present)
#   --quality-cpp       4K no-think answer, compile/run, then one repair turn on failure
set -euo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UTOFU="$REPO/a64fx/utofu-tests"

MODE="${1:-self-test}"; shift || true
NP="${PJM_MPI_PROC:-12}"
PROMPT="The capital of France is"
IDS=""; MAX_NEW=48; MAX_NEW_SET=0; LAYERS=48; DO_STAGE=1; VARIANT=int4; CHAT=0; SYSMSG=""; NOTHINK=0; KV_FP16=0
QUALITY_CPP=0; PROMPT_CACHE=""
MODEL=""; STAGE=""; NSHARDS=""; PORT=""; MAXPOS=""
AR_GROUPS=""; EXPERT_GROUPS=1; COMM_ROBUST=2; COMM_POLL_SPINS=4
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt)  PROMPT="$2"; shift 2;;
    --chat)    PROMPT="$2"; CHAT=1; shift 2;;
    --system)  SYSMSG="$2"; shift 2;;
    --no-think) NOTHINK=1; shift;;
    --ids)     IDS="$2"; shift 2;;
    --max-new) MAX_NEW="$2"; MAX_NEW_SET=1; shift 2;;
    --layers)  LAYERS="$2"; shift 2;;
    --no-stage) DO_STAGE=0; shift;;
    --np)        NP="$2"; shift 2;;
    --model-dir) MODEL="$2"; shift 2;;
    --stage-dir) STAGE="$2"; shift 2;;
    --nshards)   NSHARDS="$2"; shift 2;;
    --port)      PORT="$2"; shift 2;;
    --maxpos)    MAXPOS="$2"; shift 2;;
    --prompt-cache)
      case "$2" in /*) PROMPT_CACHE="$2";; *) PROMPT_CACHE="$PWD/$2";; esac
      shift 2;;
    --ar-groups) AR_GROUPS="$2"; shift 2;;
    --expert-groups) EXPERT_GROUPS="$2"; shift 2;;
    --comm-robust) COMM_ROBUST="$2"; shift 2;;
    --comm-poll-spins) COMM_POLL_SPINS="$2"; shift 2;;
    --bf16)    VARIANT=bf16; shift;;
    --fp8)     VARIANT=fp8; shift;;
    --kv-fp16) KV_FP16=1; shift;;
    --quality-cpp) QUALITY_CPP=1; shift;;
    *) PASS+=("$1"); shift;;
  esac
done
if [ "$QUALITY_CPP" = 1 ]; then
  [ "$MODE" = generate ] || { echo "--quality-cpp requires generate mode" >&2; exit 2; }
  [ "$CHAT" = 1 ] || { echo "--quality-cpp requires --chat" >&2; exit 2; }
  NOTHINK=1
  [ "$MAX_NEW_SET" = 1 ] || MAX_NEW=4096
  if [ -z "$SYSMSG" ]; then
    SYSMSG="You are a meticulous senior software engineer. For code-generation requests, output the final answer only. Before finalizing, internally audit every required header, API contract, ownership and lifetime, synchronization predicate and matching notification, edge case, and assertion. Prefer safe value-returning interfaces. Ensure the complete program compiles and that its tests exercise the requested semantics. Correctness and internal consistency are more important than brevity."
  fi
fi
case "$COMM_ROBUST" in 0|1|2) ;; *) echo "--comm-robust must be 0, 1, or 2" >&2; exit 2;; esac
case "$COMM_POLL_SPINS" in 1|2|4|8|16|32|64|128|256|512|1024) ;;
  *) echo "--comm-poll-spins must be a power of two in [1,1024]" >&2; exit 2;; esac
if [ -n "$AR_GROUPS" ] && { [ "$AR_GROUPS" -lt 1 ] || [ $((NP % AR_GROUPS)) -ne 0 ]; }; then
  echo "--ar-groups must be a positive divisor of --np" >&2; exit 2
fi
if [ "$EXPERT_GROUPS" -lt 1 ] || [ $((NP % EXPERT_GROUPS)) -ne 0 ]; then
  echo "--expert-groups must be a positive divisor of --np" >&2; exit 2
fi

# int4 (production, default), pure-bf16 reference, or fp8 (bf16 linears + fp8
# experts). Each uses its own checkpoint, runner binary, and stage dir.
case "$VARIANT" in
  bf16) : "${MODEL:=$HOME/models/laguna-s21}";      : "${STAGE:=/local/$USER/laguna-s21-bf16-ep$NP}"; RUNNER="$HERE/build/laguna_s21_bf16_ep_runner"; : "${NSHARDS:=46}";;
  fp8)  : "${MODEL:=$HOME/models/laguna-s21-fp8}";  : "${STAGE:=/local/$USER/laguna-s21-fp8-ep$NP}";  RUNNER="$HERE/build/laguna_s21_fp8_ep_runner";  : "${NSHARDS:=24}";;
  *)    : "${MODEL:=$HOME/models/laguna-s21-int4}"; : "${STAGE:=/local/$USER/laguna-s21-ep$NP}";      RUNNER="$HERE/build/laguna_s21_ep_runner";      : "${NSHARDS:=15}";;
esac
if [ "$KV_FP16" = 1 ]; then
  [ "$VARIANT" = fp8 ] || { echo "--kv-fp16 currently requires --fp8" >&2; exit 2; }
  RUNNER="$HERE/build/laguna_s21_fp8_kvfp16_ep_runner"
fi

# Optional per-rank stdout capture. Fugaku's mpiexec does not deliver rank
# stdout to the launching process (a pipe or a redirect both come back empty);
# -of-proc PREFIX writes PREFIX.<step>.<rank> instead, and the prefix must be on
# the shared FS. Unset by default => behaviour unchanged; a64fx/llmgr sets it so
# a supervised runner's output lands in its log directory.
OFP=()
[ -n "${MPIEXEC_OF_PROC:-}" ] && OFP=(-of-proc "$MPIEXEC_OF_PROC")

targets=(all)
[ "$VARIANT" != int4 ] && targets+=("$VARIANT")
[ "$KV_FP16" = 1 ] && targets+=(fp8-kvfp16)
make -C "$HERE" "${targets[@]}" CC="${CC:-fcc}" OPENMP=1 >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null 2>&1 || true

case "$MODE" in
  self-test) exec "$RUNNER" --self-test ;;
  stage)
    exec mpiexec -np "$NP" "${OFP[@]+"${OFP[@]}"}" "$HERE/build/laguna_s21_stage" \
        --model-dir "$MODEL" --stage-dir "$STAGE" --ep-size "$NP" \
        --expert-groups "$EXPERT_GROUPS" --nshards "$NSHARDS" "${PASS[@]}" ;;
  generate|serve) ;;  # handled below
  *) echo "usage: $0 {self-test|stage|generate|serve} [flags]" >&2; exit 2;;
esac

# ---- generate / serve ----
RUN_DIR="$HERE/gen_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

# Topology discovery (writes ./tofu_topo.txt used by the runner).
for try in 1 2 3 4 5; do
  rm -f tofu_topo.txt
  mpiexec -np "$NP" "${OFP[@]+"${OFP[@]}"}" "$UTOFU/tofu_topo_helper" && break
  [ "$try" = 5 ] && { echo "topology discovery failed" >&2; exit 3; }
done
[ "$(grep -vc '^#' tofu_topo.txt)" -eq "$NP" ] || { echo "topo has wrong node count" >&2; exit 3; }

# Stage node-local (per-rank blobs) unless skipped.
if [ "$DO_STAGE" = 1 ]; then
  echo "staging weights to $STAGE (this takes a few minutes) ..."
  mpiexec -np "$NP" "${OFP[@]+"${OFP[@]}"}" "$HERE/build/laguna_s21_stage" \
      --model-dir "$MODEL" --stage-dir "$STAGE" --status-dir "$RUN_DIR" \
      --ep-size "$NP" --expert-groups "$EXPERT_GROUPS" --nshards "$NSHARDS" >stage.stdout 2>stage.stderr
  staged=$(find "$RUN_DIR" -maxdepth 1 -name 'laguna_stage_rank*.txt' | wc -l)
  [ "$staged" -eq "$NP" ] || { echo "staging incomplete: $staged/$NP" >&2; exit 4; }
fi

# Tokenize prompt (unless --ids given).  Serve mode takes its prompts over HTTP.
if [ "$MODE" = generate ] && [ -z "$IDS" ]; then
  IDS="$RUN_DIR/prompt.ids"
  if [ "$CHAT" = 1 ]; then
    # Instruct/chat: render the checkpoint's own chat_template.jinja.  Raw
    # continuation of an instruction-shaped prompt is out of distribution for this
    # model -- it answers properly only inside <system>/<user>/<assistant> markup.
    # The template emits BOS itself, so no --bos here.
    ARGS=(chat "$PROMPT" --show-prompt)
    [ -n "$SYSMSG" ] && ARGS+=(--system "$SYSMSG")
    [ "$NOTHINK" = 1 ] && ARGS+=(--no-think)
    LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
      python3 "$HERE/tools/laguna_tok.py" "${ARGS[@]}" > "$IDS" 2>"$RUN_DIR/prompt.txt"
    echo "--- chat prompt ---"; cat "$RUN_DIR/prompt.txt"
  else
    # BOS (id 2) is required — without it the model degenerates to copying the last token.
    LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
      python3 "$HERE/tools/laguna_tok.py" encode "$PROMPT" --bos > "$IDS"
  fi
fi
if [ "$MODE" = generate ]; then
  prompt_count=$(wc -w < "$IDS")
  if [ "$prompt_count" -le 256 ]; then
    echo "prompt ids: $(cat "$IDS")"
  else
    echo "prompt ids: $prompt_count tokens (full list omitted; $IDS)"
  fi
fi

# The template's system block is a literal token prefix of every following chat
# turn.  Build it independently so the runner can serialize exactly that KV state.
PCACHE_ARGS=()
if [ -n "$PROMPT_CACHE" ]; then
  if [ "$MODE" = generate ] && [ "$CHAT" != 1 ]; then
    echo "--prompt-cache requires --chat in generate mode" >&2; exit 2
  fi
  PCACHE_IDS="$RUN_DIR/system_prefix.ids"
  PCARGS=(system-prefix)
  [ -n "$SYSMSG" ] && PCARGS+=(--system "$SYSMSG")
  [ "$NOTHINK" = 1 ] && PCARGS+=(--no-think)
  LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
    python3 "$HERE/tools/laguna_tok.py" "${PCARGS[@]}" > "$PCACHE_IDS"
  PCACHE_ARGS=(--prompt-cache "$PROMPT_CACHE" --prompt-cache-ids "$PCACHE_IDS")
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-47}"   # leave one core free (a64fx-omp-leave-one-core)
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}" OMP_PLACES="${OMP_PLACES:-cores}"
# NB: do NOT set FLIB_BARRIER=HARD here -- it forces the OpenMP runtime to 48
# threads (oversubscribing all 48 cores), which ~4x-slows the matvec kernels.
export XOS_MMM_L_PAGING_POLICY="${XOS_MMM_L_PAGING_POLICY:-demand:demand:demand}"
export TP_AR_ROBUST="$COMM_ROBUST" TP_AR_POLL_SPINS="$COMM_POLL_SPINS"
[ -n "$AR_GROUPS" ] && export LAGUNA_AR_GROUPS="$AR_GROUPS"
RUNNER_PASS=(--expert-groups "$EXPERT_GROUPS" "${PASS[@]}")

if [ "$MODE" = serve ]; then
  [ -n "$PORT" ] || { echo "serve needs --port N" >&2; exit 2; }
  [ -n "$MAXPOS" ] || { echo "serve needs --maxpos N (largest context to accept)" >&2; exit 2; }
  echo "serving on port $PORT (maxpos $MAXPOS); client:"
  echo "  LAGUNA_TOKENIZER=$MODEL/tokenizer.json python3 $HERE/tools/laguna_cli.py --port $PORT chat 'hello'"
  exec mpiexec -np "$NP" "${OFP[@]+"${OFP[@]}"}" "$RUNNER" --serve \
      --port "$PORT" --maxpos "$MAXPOS" --layers "$LAYERS" \
      --stage-dir "$STAGE" "${PCACHE_ARGS[@]}" "${RUNNER_PASS[@]}"
fi

run_generate() {
  local prompt_ids="$1" gen_out="$2"
  mpiexec -np "$NP" "${OFP[@]+"${OFP[@]}"}" "$RUNNER" --generate \
      --ids "$prompt_ids" --max-new "$MAX_NEW" --layers "$LAYERS" \
      --stage-dir "$STAGE" --gen-out "$gen_out" "${PCACHE_ARGS[@]}" "${RUNNER_PASS[@]}"
}

run_generate "$IDS" "$RUN_DIR/gen.ids"

if [ "$QUALITY_CPP" = 1 ]; then
  echo "--- C++ quality check (executes generated code; timeout 20s) ---"
  if ! LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
      python3 "$HERE/tools/cpp_quality.py" "$RUN_DIR/gen.ids" --run \
        --prompt-ids "$IDS" --repair-out "$RUN_DIR/repair_prompt.ids"; then
    mv "$RUN_DIR/gen.ids" "$RUN_DIR/gen.initial.ids"
    echo "--- compiler/runtime feedback repair turn ---"
    run_generate "$RUN_DIR/repair_prompt.ids" "$RUN_DIR/gen.ids"
    LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
      python3 "$HERE/tools/cpp_quality.py" "$RUN_DIR/gen.ids" --run || {
        echo "repaired C++ answer still failed validation" >&2
        QUALITY_RC=5
      }
  fi
fi

echo "--- generated text ---"
LAGUNA_TOKENIZER="$MODEL/tokenizer.json" \
  python3 "$HERE/tools/laguna_tok.py" decode-file "$RUN_DIR/gen.ids" || true
exit "${QUALITY_RC:-0}"
