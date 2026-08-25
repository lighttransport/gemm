#!/bin/bash
# End-to-end full K3 C11 inference run.
# The two workload legs intentionally use separate contexts:
#   codegen: natural chat prompt -> fixed generated-ID budget (quality)
#   source:  8K prompt, no generation (prefill throughput)
#PJM -g hp250467
# Use the non-torus scalar placement accepted by the K3 96-node probes.
#PJM -L "rscgrp=small,node=96,elapse=08:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# No bare `#PJM -x NAME` directives: that form fails the gate check at 96 nodes
# (see pjsub_k3_full_96n_short_1h.sh for the canary evidence).  Pass overrides
# as `-x NAME=value` on the pjsub command line instead.
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
TOKENIZER_GGUF=${K3_TOKENIZER_GGUF:-/home/u14346/models/k3/iq1/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf}
NODES=96
THREADS=${K3_THREADS:-47}
BARRIER_ITERS=${K3_BARRIER_ITERS:-128}
COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}
SOURCE_COMM_DETERMINISTIC=${K3_SOURCE_COMM_DETERMINISTIC:-0}
COMM_BF16=${K3_COMM_BF16:-0}
COMM_ROBUST=${K3_COMM_ROBUST:-2}
COMM_POLL_SPINS=${K3_COMM_POLL_SPINS:-4}
COMM_A2A=${K3_COMM_A2A:-0}
COMM_A2A_MAX=${K3_COMM_A2A_MAX:-8192}
PREFETCH_MIB=${K3_PREFETCH_MIB:-0}
PROFILE=${K3_PROFILE:-0}
AR_GROUPS=${K3_AR_GROUPS:-16}
MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-row-aligned}
CODEGEN_PROMPT_MAX=${K3_CODEGEN_PROMPT_MAX:-1024}
CODEGEN_NEW_TOKENS=${K3_CODEGEN_NEW_TOKENS:-4096}
CODEGEN_QUALITY=${K3_CODEGEN_QUALITY:-1}
SOURCE_QUALITY=${K3_SOURCE_QUALITY:-0}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/full-96n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-$JOB_TAG"
if [ -n "${K3_FULL_STAGE_DIR:-}" ]; then STAGE_DIR=$K3_FULL_STAGE_DIR; fi
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
# Keep idle workers spinning between parallel regions.  perf on a KDA layer put
# __kmp_fork_barrier at 37% of runtime with __sched_yield at 5%, i.e. threads
# were sleeping and paying a wakeup per region.  Measured 2.146 -> 2.072
# ms/layer, and it also removes most of the run-to-run variance.
export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export K3_PYTHON="$K3/.venv-$(uname -m)/bin/python" K3_EXPERT_TP=1 K3_MOE_SHARD_LAYOUT="$MOE_SHARD_LAYOUT"
# At 96 nodes, each TP shard's dense BF16 projection is smaller than a 2 MiB
# page and is therefore commonly owned by one CMG. Replicate those projections
# across CMGs to avoid the cross-CMG read bottleneck. This costs about 4.14 GiB
# per rank over the full 93-layer image; retain an explicit escape hatch for
# memory-constrained allocations.
export K3_CMG_REPLICATE=${K3_CMG_REPLICATE:-1}
# The causal flash8 MLA path is numerically equivalent on the validated
# batched layer probe and is faster than the generic batched attention path.
export K3_MLA_FLASH8=${K3_MLA_FLASH8:-1} K3_MLA_QK_MODE=${K3_MLA_QK_MODE:-auto}
export K3_COMM_ASYNC_LATENT=${K3_COMM_ASYNC_LATENT:-1}
export K3_PREFILL_PIPELINE=${K3_PREFILL_PIPELINE:-on}

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/codegen" "$ROOT/source"
printf 'K3_FULL_CONFIG nodes=%s threads=%s barrier_iters=%s comm_deterministic=%s comm_bf16=%s comm_robust=%s comm_poll_spins=%s comm_a2a=%s comm_a2a_max=%s prefetch_mib=%s profile=%s\n' \
    "$NODES" "$THREADS" "$BARRIER_ITERS" "$COMM_DETERMINISTIC" "$COMM_BF16" \
    "$COMM_ROBUST" "$COMM_POLL_SPINS" "$COMM_A2A" "$COMM_A2A_MAX" "$PREFETCH_MIB" "$PROFILE" \
    | tee "$ROOT/config.txt"
printf 'stage_dir=%s ar_groups=%s moe_shard_layout=%s codegen_prompt_max=%s codegen_new_tokens=%s codegen_quality=%s source_quality=%s\n' \
    "$STAGE_DIR" "$AR_GROUPS" "$MOE_SHARD_LAYOUT" "$CODEGEN_PROMPT_MAX" \
    "$CODEGEN_NEW_TOKENS" "$CODEGEN_QUALITY" "$SOURCE_QUALITY" | tee -a "$ROOT/config.txt"
printf 'source_comm_deterministic=%s prefill_pipeline=%s mla_flash8=%s\n' \
    "$SOURCE_COMM_DETERMINISTIC" "${K3_PREFILL_PIPELINE:-on}" "${K3_MLA_FLASH8:-1}" | tee -a "$ROOT/config.txt"
printf 'cmg_replicate=%s\n' "$K3_CMG_REPLICATE" | tee -a "$ROOT/config.txt"
"$K3/k3_setup_python.sh"

make -C "$K3" full-runner k3_prompt_ids k3_decode_ids >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null

mpiexec -np "$NODES" "$UTOFU/tofu_topo_helper" >"$ROOT/topology.log"
mv tofu_topo.txt "$ROOT/tofu_topo.txt"
[[ $(grep -vc '^#' "$ROOT/tofu_topo.txt") -eq "$NODES" ]]

echo "K3_FULL_BARRIER_BEGIN nodes=$NODES iterations=$BARRIER_ITERS"
BARRIER_PREFIX="$ROOT/barrier.rank"
mpiexec -np "$NODES" -of-proc "$BARRIER_PREFIX" "$K3/k3_full_runner" \
    --mode barrier --nodes "$NODES" --topo "$ROOT/tofu_topo.txt" \
    --barrier-iters "$BARRIER_ITERS"
grep -h 'K3FULL_BARRIER' "$BARRIER_PREFIX".* >"$ROOT/barrier.log"
echo "K3_FULL_BARRIER_END status=PASS nodes=$NODES iterations=$BARRIER_ITERS"

cat >"$ROOT/codegen_prompt.txt" <<'EOF'
You are an expert C++20 systems programmer. Implement a production-quality
bounded multi-producer multi-consumer queue with lock-free progress, precise
memory-ordering comments, cache-line padding, ABA protection, shutdown, and a
complete stress-test suite. Return compilable C++20 code first, then explain
the linearization points and failure cases.
EOF
cat >"$ROOT/source_prompt.txt" <<'EOF'
Analyze this large C++ service as if preparing a correctness and performance
review. Track ownership, lifetime, atomics, lock ordering, exception safety,
allocator behavior, cache locality, and test gaps. Produce a concrete patch
plan with code-level findings. The repeated translation units below represent
independent service files and must be treated as source, not prose.

namespace service {
struct WorkItem { std::shared_ptr<void> payload; std::atomic<unsigned> state{0}; };
class Reactor {
 public:
  void submit(WorkItem item);
  void stop() noexcept;
 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<WorkItem> queue_;
  std::vector<std::thread> workers_;
  bool stopping_{false};
};
void Reactor::submit(WorkItem item) {
  std::lock_guard<std::mutex> lock(mu_);
  if (stopping_) throw std::runtime_error("stopped");
  queue_.push_back(std::move(item));
  cv_.notify_one();
}
}
EOF

# Tokenization happens before the 1.56-TB checkpoint staging.  The semantic
# code-generation probe keeps the natural user-message length: repeating one
# instruction until it fills 1K tokens strongly biases the completion and is
# not a meaningful quality test.  The independent 8K source leg remains a
# fixed-size throughput benchmark.
"$K3/k3_prompt_ids" "$TOKENIZER_GGUF" "$ROOT/codegen_prompt.txt" \
    "$ROOT/codegen.ids" "$CODEGEN_PROMPT_MAX" --chat --thinking --natural-length
CODEGEN_TOKENS=$(wc -w < "$ROOT/codegen.ids")
CODEGEN_MAX_SEQ=$((CODEGEN_TOKENS + CODEGEN_NEW_TOKENS))
"$K3/k3_prompt_ids" "$TOKENIZER_GGUF" "$ROOT/source_prompt.txt" \
    "$ROOT/source.ids" 8192 --chat --thinking --repeat-to

# One rank per node writes its own /local image unless a prepared mixed image
# is supplied through K3_FULL_STAGE_DIR.
if [ -n "${K3_FULL_STAGE_DIR:-}" ]; then
    [[ -s "$STAGE_DIR/rank000.manifest" && -s "$STAGE_DIR/rank095.manifest" ]] || {
        echo "K3_FULL_STAGE_DIR is missing prepared rank manifests: $STAGE_DIR" >&2; exit 4;
    }
else
    mpiexec -np "$NODES" -of-proc "$ROOT/stage.rank" sh -c \
        "exec '$K3/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' '$NODES' \"\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no MPI rank}}}\""
fi

# The output ID streams and rank sidecars are durable on the result filesystem.
CODEGEN_PROFILE_ARGS=()
SOURCE_PROFILE_ARGS=()
if [ "$PROFILE" -ne 0 ]; then
    CODEGEN_PROFILE_ARGS=(--profile "$ROOT/codegen/profile.txt")
    SOURCE_PROFILE_ARGS=(--profile "$ROOT/source/profile.txt")
fi
# A16 activations avoid applying a second int8 quantizer on top of IQ weights
# during the semantic quality probe.  It is deliberately configurable because
# the fast Q8 path remains the appropriate decode-throughput measurement.
export K3_QUANT_QUALITY="$CODEGEN_QUALITY"
mpiexec -np "$NODES" -of-proc "$ROOT/codegen/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/codegen.ids" \
    --output "$ROOT/codegen/output.txt" --prefill-tokens "$CODEGEN_TOKENS" \
    --new-tokens "$CODEGEN_NEW_TOKENS" --prefill-chunk "$CODEGEN_TOKENS" \
    --prefill-path batched --max-seq "$CODEGEN_MAX_SEQ" --threads "$THREADS" \
    --comm-deterministic "$COMM_DETERMINISTIC" --comm-bf16 "$COMM_BF16" \
    --comm-robust "$COMM_ROBUST" --comm-poll-spins "$COMM_POLL_SPINS" \
    --comm-a2a "$COMM_A2A" --comm-a2a-max "$COMM_A2A_MAX" \
    --prefetch-mib "$PREFETCH_MIB" \
    --ar-groups "$AR_GROUPS" "${CODEGEN_PROFILE_ARGS[@]}"
"$K3/k3_python.sh" "$K3/validate_k3_full_output.py" "$ROOT/codegen/output.txt" \
    --nodes "$NODES" | tee "$ROOT/codegen/validation.txt"
GENERATED_IDS=$(sed -n 's/^generated_ids://p' "$ROOT/codegen/output.txt")
# The runner generates a fixed token budget for reproducible timing. Present
# only the semantic response: anything after K3's end-of-message token belongs
# to a subsequent turn and makes a correct completion look corrupted.
"$K3/k3_decode_ids" "$TOKENIZER_GGUF" --stop-after 163586 $GENERATED_IDS \
    | tee "$ROOT/codegen/generated.cpp.txt" "$ROOT/codegen/decode.txt"
"$K3/k3_python.sh" "$K3/validate_k3_codegen.py" \
    "$ROOT/codegen/generated.cpp.txt" | tee "$ROOT/codegen/quality.txt"

export K3_QUANT_QUALITY="$SOURCE_QUALITY"
mpiexec -np "$NODES" -of-proc "$ROOT/source/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/source.ids" \
    --output "$ROOT/source/output.txt" --prefill-tokens 8192 \
    --new-tokens 0 --prefill-only --prefill-chunk 1024 --prefill-path batched --max-seq 8192 \
    --threads "$THREADS" --comm-deterministic "$SOURCE_COMM_DETERMINISTIC" \
    --comm-bf16 "$COMM_BF16" --comm-robust "$COMM_ROBUST" \
    --comm-poll-spins "$COMM_POLL_SPINS" --comm-a2a "$COMM_A2A" \
    --comm-a2a-max "$COMM_A2A_MAX" --ar-groups "$AR_GROUPS" \
    --prefetch-mib "$PREFETCH_MIB" \
    "${SOURCE_PROFILE_ARGS[@]}"
"$K3/k3_python.sh" "$K3/validate_k3_full_output.py" "$ROOT/source/output.txt" \
    --nodes "$NODES" | tee "$ROOT/source/validation.txt"

printf 'K3_FULL_96 status=PASS codegen=%s source_prefill=%s stage_dir=%s\n' \
    "$ROOT/codegen/output.txt" "$ROOT/source/output.txt" "$STAGE_DIR"
