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

