#!/bin/bash
# GLM-5.2 mixed-IQ full-model runner for an existing interactive 1x12 allocation.
if [ -z "${BASH_VERSION:-}" ] || shopt -oq posix; then exec /bin/bash "$0" "$@"; fi
set -euo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
MODE="${1:-check}"
case "$MODE" in check|prefill|decode|generate) ;; *) echo "usage: $0 {check|prefill|decode|generate}" >&2; exit 2;; esac

NP="${NP:-12}"
LAST="${LAST:-0,0,0}"
STAMP="${PJM_JOBID:-interactive}-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${GLM52_RUN_DIR:-$HERE/logs/run-$STAMP-$MODE}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

export GLM52_CONVERT_DIR="${GLM52_CONVERT_DIR:-$HOME/models/glm52-2bit/a64fx-ep12-v1}"
export GLM5_STAGE_DIR="${GLM5_STAGE_DIR:-/local/u14346/glm52-2bit-ep12}"
export GLM5_STATUS_DIR="$RUN_DIR"
export GLM5_REAL=1 GLM5_TP=1 GLM5_TP_ATTN=1 GLM5_TP_SHARED=1
export GLM5_TP_FFN=1 GLM5_TP_HEAD=1 GLM5_TP_EMBED=1
export GLM5_PREFILL_GROUPS=1
export GLM5_CP_THRESHOLD=-1 GLM5_CP=0 GLM5_INT4_KV=0 GLM5_MSA=0
export GLM5_MAXPOS="${GLM5_MAXPOS:-2304}" GLM5_PCHUNK="${GLM5_PCHUNK:-512}"
export GLM5_IQ_REF="${GLM5_IQ_REF:-1}"
export TP_AR_BF16="${TP_AR_BF16:-1}" TP_AR_ROBUST="${TP_AR_ROBUST:-1}"
export GLM5_BF16_GEMM_TOK="${GLM5_BF16_GEMM_TOK:-5}"
export GLM5_ATTN_QK="${GLM5_ATTN_QK:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}" OMP_PLACES="${OMP_PLACES:-cores}"
export TF_HW_BARRIER="${TF_HW_BARRIER:-1}"

for ((r=0;r<NP;r++)); do
    printf -v rr '%02d' "$r"
    test -s "$GLM52_CONVERT_DIR/rank$rr.blob"
    grep -q '^# glm52-a64fx-ep12-v1' "$GLM52_CONVERT_DIR/rank$rr.manifest"
done

VCOORD="$RUN_DIR/vcoord_glm52.txt"
SX="${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}"
SY="${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}"
SZ="${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}"
: > "$VCOORD"
for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do
    [ "$x,$y,$z" = "$LAST" ] || echo "($x,$y,$z)" >> "$VCOORD"
done; done; done
echo "($LAST)" >> "$VCOORD"
head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"

make -C "$UTOFU" tofu_topo_helper >/dev/null
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null
for try in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$UTOFU/tofu_topo_helper" && break
    [ "$try" = 5 ] && { echo "topology discovery failed" >&2; exit 3; }
done

if [ "${GLM52_STAGE:-1}" = 1 ]; then
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$HERE/stage_glm52_q2_12n.sh" \
        >stage.stdout 2>stage.stderr
fi

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

case "$MODE" in
    check)
        export GLM5_LAYERS="${GLM5_LAYERS:-4}" GLM5_MAX_NEW=1 LLM_THREADS="${LLM_THREADS:-47}"
        make_prompt 16 "$RUN_DIR/prompt.ids"
        export GLM5_PROMPT_IDS="$RUN_DIR/prompt.ids"
        ;;
    prefill)
        export GLM5_LAYERS=78 GLM5_PREFILL_ONLY=1 GLM5_MAX_NEW=0 LLM_THREADS="${LLM_THREADS:-48}"
        make_prompt 2048 "$RUN_DIR/prompt.ids"
        export GLM5_PROMPT_IDS="$RUN_DIR/prompt.ids"
        ;;
    decode)
        export GLM5_LAYERS=78 GLM5_MAX_NEW="${GLM5_MAX_NEW:-128}" LLM_THREADS="${LLM_THREADS:-47}"
        make_prompt 512 "$RUN_DIR/prompt.ids"
        export GLM5_PROMPT_IDS="$RUN_DIR/prompt.ids" GLM5_GEN_OUT="$RUN_DIR/gen.ids"
        ;;
    generate)
        export GLM5_LAYERS="${GLM5_LAYERS:-78}" GLM5_MAX_NEW="${GLM5_MAX_NEW:-128}" LLM_THREADS="${LLM_THREADS:-47}"
        test -s "${GLM5_PROMPT_IDS:?set GLM5_PROMPT_IDS for generate mode}"
        export GLM5_GEN_OUT="${GLM5_GEN_OUT:-$RUN_DIR/gen.ids}"
        ;;
esac
export OMP_NUM_THREADS="$LLM_THREADS"

repeat="${GLM52_REPEAT:-1}"
for ((i=1;i<=repeat;i++)); do
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$LLM/build/glm5_ep_runner" \
        > >(tee "$RUN_DIR/run-$i.stdout") 2> >(tee "$RUN_DIR/run-$i.stderr" >&2)
    grep -Eq '^SENTINEL .*=(done)$' glm5_ep_rank00.txt || {
        echo "rank-0 completion sentinel missing; inspect $RUN_DIR/glm5_ep_stderr_rank*.txt" >&2
        exit 5
    }
    cp glm5_ep_rank00.txt "$RUN_DIR/rank00-$i.txt" 2>/dev/null || true
done

if [ "${GLM52_ENFORCE_TARGETS:-1}" = 1 ] && { [ "$MODE" = prefill ] || [ "$MODE" = decode ]; }; then
    metric="$MODE"
    target="${GLM52_PREFILL_TARGET:-34}"
    [ "$MODE" = decode ] && target="${GLM52_DECODE_TARGET:-15}"
    awk -v mode="$metric" -v target="$target" '
        mode=="prefill" && $1=="gen_prefill_only:" { seen=1; rate=$4 }
        mode=="decode" && $1=="gen:" {
            for(i=1;i<=NF;i++) if($i=="decode"){ seen=1; rate=$(i+3) }
        }
        END {
            if(!seen){ print "missing " mode " metric" > "/dev/stderr"; exit 2 }
            printf "%s measured %.2f tok/s, target %.2f tok/s: %s\n",mode,rate,target,(rate>=target?"PASS":"FAIL")
            exit rate>=target?0:1
        }' "$RUN_DIR/rank00-$repeat.txt"
fi
ln -sfn "$RUN_DIR" "$HERE/logs/latest-glm52-q2"
echo "GLM52_RUN_DIR=$RUN_DIR"
