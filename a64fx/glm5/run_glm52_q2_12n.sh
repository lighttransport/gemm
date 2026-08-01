#!/bin/bash
# GLM-5.2 mixed-IQ full-model runner for an existing interactive 1x12 A64FX allocation.
#
# The runner (glm5_ep_runner) is arg-driven: every model/perf/generation knob is a --flag and the
# tuned 12n-Q2 production defaults are baked into the binary (glm5_bake_defaults), so no GLM5_*/TP_*
# config env is needed.  This launcher only ORCHESTRATES: topology discovery, node-local staging,
# build, prompt construction, and the perf gate.  It exports a small, documented set of
# library-runtime env vars that MUST exist before the process starts (they cannot be program args):
#   FLIB_BARRIER=HARD            Fujitsu OMP A64FX hardware barrier (13.3->15.0 tok/s; see GLM52 doc)
#   OMP_NUM_THREADS/PROC_BIND/PLACES   OpenMP runtime thread count + pinning
#   XOS_MMM_L_PAGING_POLICY     XOS heap paging policy (Fugaku presets demand:demand:prepage)
#
# Usage: run_glm52_q2_12n.sh MODE [extra runner flags...]
#   MODE = check | prefill | decode | generate | codegen | serve
# Orchestration knobs are flags too (parsed here, before MODE-independent runner flags):
#   --convert-dir DIR   shared source blobs   (default: a64fx-ep12-2w-v1)
#   --stage-dir DIR     node-local dest       (default: /local/$USER/glm52-2bit-ep12)
#   --np N              ranks                  (default: 12)
#   --tail-coord x,y,z  coordinate assigned to rank NP-1 (default: 1,1,1)
#   --last x,y,z        backward-compatible alias for --tail-coord
#   --tail-text FILE    append tokenized text at the exact end of a codegen prompt
#   --repeat N          run the model N times
#   --retries N         retry the model run on a transient uTofu barrier fan-in (default 3)
#   --load-timeout SEC  restart mpiexec if not all ranks load in time (default 300)
#   --no-stage          skip node-local staging (blobs already present)
#   --no-enforce        do not fail on a missed perf gate
#   --active-experts N  forwarded to the runner (3 => ~20 tok/s decode; default 8 exact)
# Any flag not consumed here is passed through verbatim to glm5_ep_runner.
if [ -z "${BASH_VERSION:-}" ] || shopt -oq posix; then exec /bin/bash "$0" "$@"; fi
set -euo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"

MODE="${1:-check}"; shift || true
case "$MODE" in check|prefill|decode|generate|codegen|serve) ;; *)
    echo "usage: $0 {check|prefill|decode|generate|codegen|serve} [flags]" >&2; exit 2;; esac

# --- orchestration defaults + flag parsing (unknown flags pass through to the runner) ---
NP=12
TAIL_COORD="${GLM52_TAIL_COORD:-1,1,1}"
TAIL_TEXT=""
REPEAT=1
DO_STAGE=1
ENFORCE=1
RETRIES=3                 # retry the model run this many times on a transient uTofu barrier fan-in
LOAD_TIMEOUT=300          # full loads normally take about two minutes; a missing child otherwise hangs forever
CONVERT_DIR="$HOME/models/glm52-2bit/a64fx-ep12-2w-v1"
STAGE_DIR="/local/$USER/glm52-2bit-ep12"
PREFILL_TARGET=34
DECODE_TARGET=15
RUNNER_FLAGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --np) NP="$2"; shift 2;;
        --last|--tail-coord) TAIL_COORD="$2"; shift 2;;
        --tail-text) TAIL_TEXT="$2"; shift 2;;
        --repeat) REPEAT="$2"; shift 2;;
        --retries) RETRIES="$2"; shift 2;;
        --load-timeout) LOAD_TIMEOUT="$2"; shift 2;;
        --no-stage) DO_STAGE=0; shift;;
        --no-enforce) ENFORCE=0; shift;;
        --convert-dir) CONVERT_DIR="$2"; shift 2;;
        --stage-dir) STAGE_DIR="$2"; shift 2;;
        --prefill-target) PREFILL_TARGET="$2"; shift 2;;
        --decode-target) DECODE_TARGET="$2"; shift 2;;
        *) RUNNER_FLAGS+=("$1"); shift;;
    esac
done

if [ -n "$TAIL_TEXT" ]; then
    case "$TAIL_TEXT" in /*) ;; *) TAIL_TEXT="$PWD/$TAIL_TEXT";; esac
fi

STAMP="${PJM_JOBID:-interactive}-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$HERE/logs/run-$STAMP-$MODE"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Library-runtime env boundary (cannot be program args; must precede process start).
export FLIB_BARRIER="${FLIB_BARRIER:-HARD}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}" OMP_PLACES="${OMP_PLACES:-cores}"
export XOS_MMM_L_PAGING_POLICY="${XOS_MMM_L_PAGING_POLICY:-demand:demand:demand}"

# Validate the requested source blobs.
for ((r=0;r<NP;r++)); do
    printf -v rr '%02d' "$r"
    test -s "$CONVERT_DIR/rank$rr.blob"
    grep -q '^# glm52-a64fx-ep12' "$CONVERT_DIR/rank$rr.manifest"
done

# Virtual-coordinate file: assign TAIL_COORD to rank NP-1.  Keeping the interactive
# coordinate (0,0,0) near rank 0 avoids the recurrent rank-11 child/load stall seen when
# (0,0,0) was the final entry.
VCOORD="$RUN_DIR/vcoord_glm52.txt"
SX="${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}"
SY="${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}"
SZ="${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}"
IFS=, read -r TX TY TZ EXTRA <<< "$TAIL_COORD"
if [ -n "${EXTRA:-}" ] || ! [[ "$TX" =~ ^[0-9]+$ && "$TY" =~ ^[0-9]+$ && "$TZ" =~ ^[0-9]+$ ]] ||
   (( TX >= SX || TY >= SY || TZ >= SZ || NP > SX*SY*SZ )); then
    echo "invalid --tail-coord $TAIL_COORD for shape ${SX}x${SY}x${SZ} and NP=$NP" >&2
    exit 2
fi
: > "$VCOORD"
for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do
    [ "$x,$y,$z" = "$TAIL_COORD" ] || echo "($x,$y,$z)" >> "$VCOORD"
done; done; done
echo "($TAIL_COORD)" >> "$VCOORD"
head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
[ "$(wc -l < "$VCOORD")" -eq "$NP" ] || { echo "invalid NP/shape/--tail-coord combination" >&2; exit 2; }

make -C "$UTOFU" tofu_topo_helper >/dev/null
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null
for try in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$UTOFU/tofu_topo_helper" && break
    [ "$try" = 5 ] && { echo "topology discovery failed" >&2; exit 3; }
done

if [ "$DO_STAGE" = 1 ]; then
    mpiexec -np "$NP" -vcoordfile "$VCOORD" \
        "$HERE/stage_glm52_q2_12n.sh" --source "$CONVERT_DIR" --dest "$STAGE_DIR" --status "$RUN_DIR" \
        >stage.stdout 2>stage.stderr
    staged=$(find "$RUN_DIR" -maxdepth 1 -name 'glm52_stage_rank*.txt' | wc -l)
    [ "$staged" -eq "$NP" ] || { echo "staging incomplete: $staged/$NP rank markers" >&2; exit 4; }
fi

# Build a natural-length prompt of `count` ids by repeating the sample block and tokenizing.
make_prompt() {
    local count="$1" out="$2" text="$RUN_DIR/prompt-repeat.txt" ids="$RUN_DIR/prompt-all.ids"
    : > "$text"
    while :; do
        printf '\n' >> "$text"
        sed -n '1,240p' "$HERE/prompt_fp8_1k.txt" >> "$text"
        python3 "$HERE/glm5_tokenizer.py" chat-file "$text" > "$ids"
        [ "$(wc -w < "$ids")" -ge "$count" ] && break
    done
    awk -v n="$count" '{for(i=1;i<=NF&&k<n;i++){printf "%s%s",$i,(++k<n?" ":"\n")}}' "$ids" > "$out"
}

# Common runner flags for every mode: node-local blobs, run-dir status, this run's log dir.
COMMON=(--stage-dir "$STAGE_DIR" --status-dir "$RUN_DIR")

THREADS=47   # leave one core free (a64fx-omp-leave-one-core); prefill overrides to 48
case "$MODE" in
    check)
        make_prompt 16 "$RUN_DIR/prompt.ids"
        MODE_FLAGS=(--layers 4 --max-new 1 --prompt-ids "$RUN_DIR/prompt.ids")
        ;;
    prefill)
        THREADS=48
        make_prompt 2048 "$RUN_DIR/prompt.ids"
        MODE_FLAGS=(--layers 78 --prefill-only --max-new 0 --prompt-ids "$RUN_DIR/prompt.ids")
        ;;
    decode)
        make_prompt 512 "$RUN_DIR/prompt.ids"
        MODE_FLAGS=(--layers 78 --max-new 128 \
                    --prompt-ids "$RUN_DIR/prompt.ids" --gen-out "$RUN_DIR/gen.ids")
        ;;
    generate)
        MODE_FLAGS=(--layers 78 --max-new 128 --gen-out "$RUN_DIR/gen.ids")
        # caller must pass --prompt-ids or --prompt-tokens via extra flags
        ;;
    serve)
        MODE_FLAGS=(--layers 78)
        ;;
    codegen)
        # Coding-agent usecase = the real long-context stability + coherence test.
        # Single phase: prefill the precomputed code prompt (real tokens), then greedy-generate
        # code and detokenize.  On 12 nodes a >~23k-token context exceeds the Tier-A bf16 KV
        # budget, so auto KV tiering engages CP-sharded Tier B + MSA mid-prefill.  Tier B is
        # int4 by default and BF16 under --stable-outputs.
        # prefill -- this run therefore exercises the long-context path end to end.  (KV save/load
        # reuse across the tier boundary is a separate item; a one-shot run needs neither.)
        CODEGEN_TOK="${CODEGEN_TOK:-$HOME/glm5_codegen.bin}"
        SYS_TOK="${SYS_TOK:-108474}"; GEN="${GEN:-256}"
        test -s "$CODEGEN_TOK" || { echo "codegen: missing prompt tokens $CODEGEN_TOK" >&2; exit 4; }
        if [ -n "$TAIL_TEXT" ]; then
            test -s "$TAIL_TEXT" || { echo "codegen: missing tail text $TAIL_TEXT" >&2; exit 4; }
            python3 "$HERE/make_long_prompt.py" --base-bin "$CODEGEN_TOK" \
                --tail-text "$TAIL_TEXT" --tokens "$SYS_TOK" \
                --output "$RUN_DIR/codegen-prompt.bin"
            CODEGEN_TOK="$RUN_DIR/codegen-prompt.bin"
        fi
        MAXPOS=$(( SYS_TOK + GEN + 128 ))
        export OMP_NUM_THREADS=47
        echo "--- codegen: prefill $SYS_TOK-token code prompt, generate $GEN, detokenize ---"
        codegen_ok=0
        for ((a=1;a<=RETRIES;a++)); do
            ATTEMPT_DIR="$RUN_DIR/attempt-$a"
            mkdir -p "$ATTEMPT_DIR"
            rm -f "$RUN_DIR"/glm5_ep_load_rank*.txt "$RUN_DIR"/glm5_ep_stderr_rank*.txt \
                  "$RUN_DIR/gen.ids" glm5_ep_rank00.txt
            # A separate process group lets the watchdog tear down mpiwrapp, org/mpiexec,
            # and plexec together. Killing only the wrapper leaves PLE coordinates reserved
            # briefly, causing every immediate retry to fail with PLE 0054.
            setsid mpiexec -np "$NP" -vcoordfile "$VCOORD" "$LLM/build/glm5_ep_runner" \
                --stage-dir "$STAGE_DIR" --status-dir "$RUN_DIR" \
                --layers 78 --threads 47 --ctx "$MAXPOS" --pchunk 512 \
                --prompt-tokens "$CODEGEN_TOK" --prefill-synth "$SYS_TOK" --prefill-only \
                --gen-new "$GEN" --gen-out "$RUN_DIR/gen.ids" \
                "${RUNNER_FLAGS[@]}" \
                > >(tee "$RUN_DIR/codegen-attempt-$a.stdout") \
                2> >(tee "$RUN_DIR/codegen-attempt-$a.stderr" >&2) &
            mpi_pid=$!
            load_start=$SECONDS
            while kill -0 "$mpi_pid" 2>/dev/null; do
                loaded=$(find "$RUN_DIR" -maxdepth 1 -name 'glm5_ep_load_rank*.txt' | wc -l)
                [ "$loaded" -ge "$NP" ] && break
                if (( SECONDS - load_start >= LOAD_TIMEOUT )); then
                    echo "codegen attempt $a: load timeout after ${LOAD_TIMEOUT}s ($loaded/$NP ranks); restarting mpiexec" >&2
                    kill -TERM -- "-$mpi_pid" 2>/dev/null || true
                    wait "$mpi_pid" 2>/dev/null || true
                    # PLE releases the remote child coordinates asynchronously; on a wedged
                    # rank this has taken over 30 seconds even after the local process group exits.
                    sleep 60
                    mpi_pid=
                    break
                fi
                sleep 2
            done
            if [ -n "${mpi_pid:-}" ]; then
                if wait "$mpi_pid" && grep -Eq '^SENTINEL .*=(done)$' glm5_ep_rank00.txt; then
                    codegen_ok=1
                    cp "$RUN_DIR/codegen-attempt-$a.stdout" "$RUN_DIR/codegen.stdout"
                    cp "$RUN_DIR/codegen-attempt-$a.stderr" "$RUN_DIR/codegen.stderr"
                    cp "$RUN_DIR"/glm5_ep_*.txt "$ATTEMPT_DIR"/ 2>/dev/null || true
                    break
                fi
            fi
            cp "$RUN_DIR"/glm5_ep_*.txt "$ATTEMPT_DIR"/ 2>/dev/null || true
            echo "codegen attempt $a did not complete; retrying ($a/$RETRIES)" >&2
        done
        [ "$codegen_ok" = 1 ] || { echo "codegen failed after $RETRIES attempt(s)" >&2; exit 5; }
        cp glm5_ep_rank00.txt "$RUN_DIR/rank00-codegen.txt" 2>/dev/null || true
        grep -Eq '^SENTINEL .*=(done)$' glm5_ep_rank00.txt || {
            echo "codegen: rank-0 sentinel missing" >&2; exit 5; }
        if [ -s "$RUN_DIR/gen.ids" ]; then
            echo "=== GENERATED (detokenized) ==="
            python3 "$HERE/glm5_tokenizer.py" decode-file "$RUN_DIR/gen.ids" | tee "$RUN_DIR/generated.txt"
        fi
        ln -sfn "$RUN_DIR" "$HERE/logs/latest-glm52-q2"
        echo "GLM52_RUN_DIR=$RUN_DIR"
        exit 0
        ;;
esac

export OMP_NUM_THREADS="$THREADS"

if [ "$MODE" = serve ]; then
    exec mpiexec -np "$NP" -vcoordfile "$VCOORD" "$LLM/build/glm5_ep_runner" \
        "${COMMON[@]}" "${MODE_FLAGS[@]}" --threads "$THREADS" "${RUNNER_FLAGS[@]}"
fi

for ((i=1;i<=REPEAT;i++)); do
    # A cold uTofu VCQ bring-up occasionally fails the bootstrap barrier ("barrier fan-in
    # (rc=-1)"), especially right after another job/run tore down the fabric.  It is transient and
    # clears on a fresh mpiexec, so retry the run (NOT a code bug).  Any other non-completion
    # (real error, NaN, OOM) is not retried -- its stderr won't match the transient signature.
    ok=0
    for ((a=1;a<=RETRIES;a++)); do
        : > glm5_ep_rank00.txt
        mpiexec -np "$NP" -vcoordfile "$VCOORD" "$LLM/build/glm5_ep_runner" \
            "${COMMON[@]}" "${MODE_FLAGS[@]}" --threads "$THREADS" "${RUNNER_FLAGS[@]}" \
            > >(tee "$RUN_DIR/run-$i.stdout") 2> >(tee "$RUN_DIR/run-$i.stderr" >&2) || true
        if grep -Eq '^SENTINEL .*=(done)$' glm5_ep_rank00.txt; then ok=1; break; fi
        if grep -qa 'barrier fan-in' glm5_ep_rank00.txt "$RUN_DIR"/glm5_ep_stderr_rank*.txt 2>/dev/null; then
            echo "run $i attempt $a: transient uTofu barrier fan-in; retrying ($a/$RETRIES)" >&2
            continue
        fi
        break   # non-transient failure: stop retrying, fall through to the error below
    done
    [ "$ok" = 1 ] || {
        echo "rank-0 completion sentinel missing after $RETRIES attempt(s); inspect $RUN_DIR/glm5_ep_stderr_rank*.txt" >&2
        exit 5
    }
    cp glm5_ep_rank00.txt "$RUN_DIR/rank00-$i.txt" 2>/dev/null || true
done

if [ "$ENFORCE" = 1 ] && { [ "$MODE" = prefill ] || [ "$MODE" = decode ]; }; then
    target="$PREFILL_TARGET"; [ "$MODE" = decode ] && target="$DECODE_TARGET"
    awk -v mode="$MODE" -v target="$target" '
        mode=="prefill" && $1=="gen_prefill_only:" { seen=1; rate=$4 }
        mode=="decode" && $1=="gen:" {
            for(i=1;i<=NF;i++) if($i=="decode"){ seen=1; rate=$(i+3) }
        }
        END {
            if(!seen){ print "missing " mode " metric" > "/dev/stderr"; exit 2 }
            printf "%s measured %.2f tok/s, target %.2f tok/s: %s\n",mode,rate,target,(rate>=target?"PASS":"FAIL")
            exit rate>=target?0:1
        }' "$RUN_DIR/rank00-$REPEAT.txt"
fi
ln -sfn "$RUN_DIR" "$HERE/logs/latest-glm52-q2"
echo "GLM52_RUN_DIR=$RUN_DIR"
