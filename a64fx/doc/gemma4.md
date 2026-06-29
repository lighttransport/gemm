# Gemma4 31B Q4 on A64FX

This note describes the current single-node Gemma4 31B Q4_0 runner path on
A64FX. It assumes the QAT GGUF model is available at:

```bash
$HOME/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf
```

## Build

```bash
make -C a64fx/llm CC=fcc OPENMP=1
```

Verify that the binary is using the hugepage malloc path:

```bash
ldd a64fx/llm/build/llm_runner | grep mpg
readelf -h a64fx/llm/build/llm_runner | grep Type:
```

Expected:

```text
libmpg.so.1 => /opt/FJSVxos/mmm/lib64/libmpg.so.1
Type:                              EXEC (Executable file)
```

## Run

Use the wrapper:

```bash
cd a64fx/llm
./run_gemma4_31b_q4.sh
```

The wrapper defaults to the A64FX hugepage and NUMA placement settings:

```bash
XOS_MMM_L_HPAGE_TYPE=hugetlbfs
XOS_MMM_L_PAGING_POLICY=demand:demand:demand
XOS_MMM_L_ARENA_FREE=2
NO_MMAP=1
NUMA_DISTRIBUTE=1
NUMA_N_CMGS=4
NUMA_CMG_BUDGET_GB=7
NUMA_ALIGNMENT=2097152
OMP_NUM_THREADS=48
numactl -C12-59 -m4-7
```

The 4 CMGs are nodes 4-7. The runner uses 48 cores, 12 per CMG, and a 7 GB
per-CMG budget. This leaves about 1 GB per CMG for OS/runtime overhead.

## Loader Requirements

Do not use file-backed `mmap` for this model on A64FX. File-backed pages do not
get the useful hugepage behavior here and cause severe TLB pressure. Use:

```bash
NO_MMAP=1
```

With `NUMA_DISTRIBUTE=1`, large tensors are loaded later by
`transformer_numa_setup()` using parallel `pread`. This first-touches each row
partition from the worker that will consume it, spreading weight pages across
CMGs.

Small tensors are different: some are copied during `transformer_load()` before
`transformer_numa_setup()` runs. The loader therefore eagerly loads 1D and small
tensors even in NUMA-distributed mode. This is required for tensors such as
`rope_freqs.weight`; leaving it deferred produces NaNs in Gemma4 RoPE.

Default eager threshold:

```bash
NUMA_EAGER_TENSOR_BYTES=16777216
```

Use `TF_NUMA_VERIFY=1` for sampled tensor-byte verification, or
`TF_NUMA_VERIFY=2` for full tensor-byte verification. These are diagnostic only
and should not be used for normal timing.

## Memory Layout

Current measured placement for the 31B Q4_0 model:

```text
numa: phase 1 done (16.4GB weights loaded)
numa: per-CMG usage: CMG0=4208.4MB CMG1=4203.3MB CMG2=4203.3MB CMG3=4203.3MB (budget=7.0GB)
```

Model data is allocated on a 2 MB boundary for hugepage compatibility. Large
batch scratch buffers are allocated with 256-byte alignment for SVE-friendly
access.

## Current Measurements

Command shape:

```bash
env XOS_MMM_L_HPAGE_TYPE=hugetlbfs \
    XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    XOS_MMM_L_ARENA_FREE=2 \
    NO_MMAP=1 NUMA_DISTRIBUTE=1 NUMA_CMG_BUDGET_GB=7 \
    OMP_NUM_THREADS=48 TF_KV_DTYPE=f16 TF_PREFILL_GEMM=1 \
    numactl -C12-59 -m4-7 \
    ./build/llm_runner "$HOME/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf" \
        --prompt "The capital of France is" \
        --max-seq 65536 --max-gen 1 --llm-threads 48
```

64k context smoke:

```text
TOTAL:       3454.6 ms (4.05 tok/s)   # profiled 14-token batch prefill body
prefill total: 15.18 s (14 tokens, 0.9 tok/s)
logits: nan=0 inf=0
```

8-token decode smoke, `TF_NO_EOS_STOP=1`:

```text
TOTAL:       3324.3 ms (4.21 tok/s)   # profiled 14-token batch prefill body
prefill total: 15.03 s (14 tokens, 0.9 tok/s)
gen: 8 tokens in 5.20 s (1.54 tok/s)
logits: nan=0 inf=0 for all dumped decode steps
```

The generated text for the short prompt included `Paris.`.

## Current Bottleneck

Decode is still far below the expected 40+ tok/s target. The NUMA loader and
hugepage path are now working, but steady-state decode is dominated by the
output/lm_head matvec:

```text
lm_head: 1740.16 ms, 19.7300 GFLOPs, 11.3 GFLOP/s, 7 calls
```

The next optimization target is therefore not the loader. Focus on the
single-token decode matvec path, especially the weight-tied output projection
over the 262144-token vocabulary.

## Known Pitfalls

- `NUMA_DISTRIBUTE=1` without eager-loading small tensors can corrupt copied
  runtime metadata. The symptom found for Gemma4 was all-NaN hidden state from
  layer 5 onward because `rope_freqs.weight` had not been loaded before
  proportional RoPE setup.
- Pinning the main thread permanently to one core causes later ad-hoc pthread
  kernels to inherit a single-core affinity mask. The thread pool now pins the
  main thread only while it executes worker 0's task, then restores the broader
  `numactl` CPU mask.
- `TF_NUMA_VERIFY=2` rereads the whole model and is intentionally slow. Use it
  only when validating loader changes.

---

# Gemma4 12B BF16 multinode (A64FX / Fugaku, uTofu)

A second, independent path: the **12B BF16** model (`gemma-4-12b-it-BF16.gguf`,
~24 GB) run across **2–12 nodes** with tensor parallelism (TP) and speculative
decode (MTP draft + batched-TP verify). All code is in `a64fx/gemma4-mn/`. MPI
only *places* ranks (one rank/node, no `MPI_Init`); all comm is uTofu one-sided
Put via `a64fx/utofu-tests/tp_allreduce.h`.

```bash
TARGET=$HOME/models/gemma4/12b/gemma-4-12b-it-BF16.gguf       # ~24 GB, all BF16
MTP=$HOME/models/gemma4/12b/mtp-gemma-4-12b-it.gguf           # 4-block draft head, Q8_0
```

## Two parallelism modes

- **Pipeline-parallel (PP)** — splits the 48 layers across N stages. **Memory
  win only, no single-stream speedup** (stages are serial for one token): decode
  is ~flat 2.5 tok/s at every N, per-node footprint 12.9→3.6 GB. Use only to fit
  a bigger model; for speed use TP. Files: `gemma4_stage.c` (`pp` mode),
  `gemma4_pp_load.h`, `gemma4_pp_runner.c`, `sweep_pp.sh`.
- **Tensor-parallel (TP)** — Megatron row/col shard of every layer. **This is the
  speed path.** Files: `gemma4_stage.c` (`tp` mode), `gemma4_tp_load.h`,
  `gemma4_tp_runner.c`, `gemma4_mtp.h`, `sweep_tp.sh`.

## Build

Native A64FX compilers; the runners need only `-ltofucom` (no MPI lib):

```bash
cd a64fx/gemma4-mn
fcc -Nclang -O2 -D_GNU_SOURCE -I../../common gemma4_stage.c -o gemma4_stage
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
    -I../../common gemma4_tp_runner.c -lm -lpthread -lhwb -ltofucom -o gemma4_tp_runner
```

## Run (TP + spec decode)

`sweep_tp.sh <N> <maxgen>` does the whole flow (vcoordfile from the 2×3×2 grid
minus the login node, topo helper, stage, run). The pieces, for N nodes:

```bash
VC=vcoord_g4tp_n${N}.txt                 # one "(x,y,z)" per line, excl login (0,0,0)
mpiexec -np N -vcoordfile $VC ../utofu-tests/tofu_topo_helper      # writes tofu_topo.txt
# 1) stage the per-rank tensor slice to node-local /local (memory-safe, <=1 GB chunks)
mpiexec -np N -vcoordfile $VC sh -c \
  "mkdir -p /local/gemma4_tp; exec ./gemma4_stage $TARGET /local/gemma4_tp \$PMIX_RANK N tp"
# 2) WARM-UP a fresh alloc with one throwaway run (cold misreads ~10x slow), then:
# 3) speculative decode: MTP draft + batched-TP verify
mpiexec -np N -vcoordfile $VC sh -c \
  "cd $PWD; LLM_THREADS=48 GEMMA4_TP_MTP=$MTP exec ./gemma4_tp_runner \
   $TARGET /local/gemma4_tp prompt_long.txt 128"
```

Result line: `TPSPEC N=.. tok_s=.. alpha=.. tok_fwd=.. first=..` (also written to
`gemma4_tp_result.txt`; mpiexec does not forward rank stdout). Drop `GEMMA4_TP_MTP`
for plain M=1 TP decode.

## How TP works

The model is **all BF16** (2 B/elem, no block quant), so weights slice
element-wise into **dense per-rank shards** and the standard SVE matvec / batched
GEMM works directly — no custom sliced kernel.

| tensor | shard | comm |
|--------|-------|------|
| `attn_q`, `ffn_gate`, `ffn_up` | ROW (output dim) | none (disjoint) |
| `attn_output`, `ffn_down` | COL (contraction) | all-reduce-SUM |
| `attn_k/v`, `token_embd`, norms | REPLICATE | — |
| lm_head (tied `token_embd`) | VOCAB slice | all-reduce-argmax |

Forward wiring is in `transformer.h` (`tf_forward_blocks_range` for M=1 decode,
`tf_gemma4_prefill_batch` for the batched verify), gated on `m->tp_size > 1` and
**byte-identical when off**. Heads split `[r*16/N, (r+1)*16/N)`; `kv_h` uses the
global head index since KV is replicated. Correctness model: disjoint-output
shards are bit-exact, contraction shards coherent via all-reduce (argmax-exact).

## Speculative decode (MTP)

`gemma4_mtp.h` is the gemma4-assistant draft head (4 blocks @1024-dim) that reads
the **target's KV** (replicated on every rank) and **replicated MTP weights**, so
every rank drafts *identically with no comm*. The verify is the batched TP forward
(argmax-exact); accept/rollback is deterministic on the replicated logits, so the
token stream stays lockstep without extra comm. The MTP gguf is **Q8_0** —
pre-dequantized to BF16 once at load. Draft and verify both vocab-shard the
lm_head and block-shard the FFN; `GEMMA4_MTP_NOSHARD=1` turns the draft shards off.

## Results (warm, 9-token prompt, K=4, argmax-exact throughout)

```text
M=1 TP decode :  N=2 2.56   N=4 5.95   N=6 5.33  tok/s   (TP scales, peak ~N=4)
spec decode   :  N=6 14.59 -> 30.0 -> 36.44 tok/s        (best = 36.44 @N=6)
```

The spec-decode arc at N=6 (each step argmax-exact, alpha 0.78 / tok-per-fwd 4.1):

| step | tok/s | x |
|------|-------|---|
| replicated draft (Q8_0), full lm_head | 14.59 | — |
| + vocab-shard the verify lm_head | 30.0 | 2.06 |
| + block-shard the MTP FFN | **36.44** | 1.22 |

**Best: ~36 tok/s, ~14.5× single-node M=1 (~2.5).** Round is balanced verify 57%
(weight-bandwidth-bound) / draft 42%. Measured (`TP_SKIP_AR=1` + the spec-loop
`t_draft`/`t_verify` timers): verify comm is only ~20% of the verify, so the real
gains came from removing the **redundant full-vocab lm_head** (`token_embd` 2 GB
replicated, recomputed over all 262144 vocab/rank) and the **replicated draft**,
not from the per-layer all-reduce. Remaining levers are all ~1.1×.

## Operational gotchas (multinode)

- **WARM a fresh allocation** with one throwaway run before measuring — the first
  run on cold nodes (anon-load not resident) misreads ~10× slow (e.g. 0.24 vs
  5.95 tok/s). Memory-stage in ≤1 GB fadvise'd chunks; never a plain 24 GB `cp`
  (kswapd thrash hangs the node).
- **Re-stage after every job restart** — a new allocation gets different physical
  nodes, so `/local` shards vanish and `tofu_topo.txt` goes stale (`my coords not
  in topo` / `pp load failed`). Re-run the topo helper + stager for the new coords.
- **Stale-shard guard**: `gemma4_tp_load` aborts if the manifest `nranks` ≠ runtime
  N (a prior sweep at a different N clobbers the same physical nodes' `/local`).
  Always re-stage for the exact N.
- **Stragglers**: a timed-out/killed runner strands a compute-node process →
  `PLE 0054 ... exceed the limit on virtual coordinate` blocks the next run;
  login-side `pkill` cannot kill it. Wait for it to die, leave a gap between
  back-to-back runs, and give cold M=1 baselines ≥200 s. `pkill -9 mpiexec plexec`
  clears login-side launchers.
- The MTP draft + verify use OpenMP; the transformer pthread pool busy-spins when
  idle and would starve them — the spec path calls `tf_pool_shutdown` for its
  duration (the verify is all-OMP GEMM, needs no pool).

