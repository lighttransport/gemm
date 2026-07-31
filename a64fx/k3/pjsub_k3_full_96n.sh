#!/bin/bash
# End-to-end full K3 C11 inference run.
# The two workload legs intentionally use separate contexts:
#   codegen: 1K prompt -> 4K generated IDs (decode throughput)
#   source:  8K prompt, no generation (prefill throughput)
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=08:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
NODES=96
THREADS=${K3_THREADS:-47}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/full-96n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-$JOB_TAG"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/codegen" "$ROOT/source"

make -C "$K3" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null

mpiexec -np "$NODES" "$UTOFU/tofu_topo_helper" >"$ROOT/topology.log"
mv tofu_topo.txt "$ROOT/tofu_topo.txt"
[[ $(grep -vc '^#' "$ROOT/tofu_topo.txt") -eq "$NODES" ]]

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
python3 "$K3/make_k3_prompt_ids.py" --vocab "$MODEL_DIR/tiktoken.model" \
    --text "$ROOT/codegen_prompt.txt" --output "$ROOT/codegen.ids" \
    --tokens 1024 --bos --repeat-to
python3 "$K3/make_k3_prompt_ids.py" --vocab "$MODEL_DIR/tiktoken.model" \
    --text "$ROOT/source_prompt.txt" --output "$ROOT/source.ids" \
    --tokens 8192 --bos --repeat-to

# One rank per node writes its own /local image.  PMIX_RANK is the launcher
# rank used by the Fujitsu MPI in the K3 allocations.
mpiexec -np "$NODES" -of-proc "$ROOT/stage.rank" sh -c \
    "exec '$K3/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' '$NODES' \"\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no MPI rank}}}\""

# The output ID streams and rank sidecars are durable on the result filesystem.
mpiexec -np "$NODES" -of-proc "$ROOT/codegen/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/codegen.ids" \
    --output "$ROOT/codegen/output.txt" --prefill-tokens 1024 \
    --new-tokens 4096 --prefill-chunk 256 --max-seq 5120 --threads "$THREADS"
python3 "$K3/validate_k3_full_output.py" "$ROOT/codegen/output.txt" \
    --nodes "$NODES" | tee "$ROOT/codegen/validation.txt"
python3 "$K3/decode_k3_output.py" "$ROOT/codegen/output.txt" \
    --vocab "$MODEL_DIR/tiktoken.model" \
    --text-output "$ROOT/codegen/generated.cpp.txt" | tee "$ROOT/codegen/decode.txt"

mpiexec -np "$NODES" -of-proc "$ROOT/source/rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/source.ids" \
    --output "$ROOT/source/output.txt" --prefill-tokens 8192 \
    --new-tokens 0 --prefill-only --prefill-chunk 256 --max-seq 8192 --threads "$THREADS"
python3 "$K3/validate_k3_full_output.py" "$ROOT/source/output.txt" \
    --nodes "$NODES" | tee "$ROOT/source/validation.txt"

printf 'K3_FULL_96 status=PASS codegen=%s source_prefill=%s stage_dir=%s\n' \
    "$ROOT/codegen/output.txt" "$ROOT/source/output.txt" "$STAGE_DIR"
