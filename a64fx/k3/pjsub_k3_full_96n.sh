#!/bin/bash
# End-to-end full K3 C11 inference run.
# The two workload legs intentionally use separate contexts:
#   codegen: 1K prompt -> 4K generated IDs (decode throughput)
#   source:  8K prompt, no generation (prefill throughput)
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=08:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
# Propagate launch-time overrides from the login1 environment into the job.
#PJM -x K3_BARRIER_ITERS
#PJM -x K3_THREADS
#PJM -x K3_COMM_DETERMINISTIC
#PJM -x K3_COMM_BF16
#PJM -x K3_COMM_ROBUST
#PJM -x K3_COMM_POLL_SPINS
#PJM -x K3_COMM_A2A
#PJM -x K3_COMM_A2A_MAX
#PJM -x K3_PREFETCH_MIB
#PJM -x K3_PROFILE
#PJM -x K3_MODEL_DIR
#PJM -x K3_FULL_STAGE_DIR
#PJM -x K3_AR_GROUPS
#PJM -x K3_MOE_SHARD_LAYOUT
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
NODES=96
THREADS=${K3_THREADS:-47}
BARRIER_ITERS=${K3_BARRIER_ITERS:-128}
COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}
COMM_BF16=${K3_COMM_BF16:-0}
COMM_ROBUST=${K3_COMM_ROBUST:-2}
COMM_POLL_SPINS=${K3_COMM_POLL_SPINS:-4}
COMM_A2A=${K3_COMM_A2A:-0}
COMM_A2A_MAX=${K3_COMM_A2A_MAX:-8192}
PREFETCH_MIB=${K3_PREFETCH_MIB:-16}
PROFILE=${K3_PROFILE:-0}
AR_GROUPS=${K3_AR_GROUPS:-16}
MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-replicated}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/full-96n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-$JOB_TAG"
if [ -n "${K3_FULL_STAGE_DIR:-}" ]; then STAGE_DIR=$K3_FULL_STAGE_DIR; fi
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export K3_PYTHON="$K3/.venv-$(uname -m)/bin/python" K3_EXPERT_TP=1 K3_MOE_SHARD_LAYOUT="$MOE_SHARD_LAYOUT"

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/codegen" "$ROOT/source"
printf 'K3_FULL_CONFIG nodes=%s threads=%s barrier_iters=%s comm_deterministic=%s comm_bf16=%s comm_robust=%s comm_poll_spins=%s comm_a2a=%s comm_a2a_max=%s prefetch_mib=%s profile=%s\n' \
    "$NODES" "$THREADS" "$BARRIER_ITERS" "$COMM_DETERMINISTIC" "$COMM_BF16" \
    "$COMM_ROBUST" "$COMM_POLL_SPINS" "$COMM_A2A" "$COMM_A2A_MAX" "$PREFETCH_MIB" "$PROFILE" \
    | tee "$ROOT/config.txt"
printf 'stage_dir=%s ar_groups=%s moe_shard_layout=%s\n' "$STAGE_DIR" "$AR_GROUPS" "$MOE_SHARD_LAYOUT" | tee -a "$ROOT/config.txt"
"$K3/k3_setup_python.sh"

make -C "$K3" full-runner >/dev/null
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

# Tokenization happens before the 1.56-TB checkpoint staging, so missing Python
# tokenizer dependencies fail cheaply.  The C runner consumes only integer IDs.
"$K3/k3_python.sh" "$K3/make_k3_prompt_ids.py" --vocab "$MODEL_DIR/tiktoken.model" \
    --text "$ROOT/codegen_prompt.txt" --output "$ROOT/codegen.ids" \
    --tokens 1024 --bos --repeat-to
"$K3/k3_python.sh" "$K3/make_k3_prompt_ids.py" --vocab "$MODEL_DIR/tiktoken.model" \
    --text "$ROOT/source_prompt.txt" --output "$ROOT/source.ids" \
    --tokens 8192 --bos --repeat-to

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
mpiexec -np "$NODES" -of-proc "$ROOT/codegen/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/codegen.ids" \
    --output "$ROOT/codegen/output.txt" --prefill-tokens 1024 \
    --new-tokens 4096 --prefill-chunk 256 --max-seq 5120 --threads "$THREADS" \
    --comm-deterministic "$COMM_DETERMINISTIC" --comm-bf16 "$COMM_BF16" \
    --comm-robust "$COMM_ROBUST" --comm-poll-spins "$COMM_POLL_SPINS" \
    --comm-a2a "$COMM_A2A" --comm-a2a-max "$COMM_A2A_MAX" \
    --prefetch-mib "$PREFETCH_MIB" \
    --ar-groups "$AR_GROUPS" "${CODEGEN_PROFILE_ARGS[@]}"
"$K3/k3_python.sh" "$K3/validate_k3_full_output.py" "$ROOT/codegen/output.txt" \
    --nodes "$NODES" | tee "$ROOT/codegen/validation.txt"
"$K3/k3_python.sh" "$K3/decode_k3_output.py" "$ROOT/codegen/output.txt" \
    --vocab "$MODEL_DIR/tiktoken.model" \
    --text-output "$ROOT/codegen/generated.cpp.txt" | tee "$ROOT/codegen/decode.txt"

mpiexec -np "$NODES" -of-proc "$ROOT/source/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/source.ids" \
    --output "$ROOT/source/output.txt" --prefill-tokens 8192 \
    --new-tokens 0 --prefill-only --prefill-chunk 256 --max-seq 8192 \
    --threads "$THREADS" --comm-deterministic "$COMM_DETERMINISTIC" \
    --comm-bf16 "$COMM_BF16" --comm-robust "$COMM_ROBUST" \
    --comm-poll-spins "$COMM_POLL_SPINS" --comm-a2a "$COMM_A2A" \
    --comm-a2a-max "$COMM_A2A_MAX" --ar-groups "$AR_GROUPS" \
    --prefetch-mib "$PREFETCH_MIB" \
    "${SOURCE_PROFILE_ARGS[@]}"
"$K3/k3_python.sh" "$K3/validate_k3_full_output.py" "$ROOT/source/output.txt" \
    --nodes "$NODES" | tee "$ROOT/source/validation.txt"

printf 'K3_FULL_96 status=PASS codegen=%s source_prefill=%s stage_dir=%s\n' \
    "$ROOT/codegen/output.txt" "$ROOT/source/output.txt" "$STAGE_DIR"
