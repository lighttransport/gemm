# Qwen3.8 DSpark A64FX work guide

This is the reproduction and continuation guide for the standalone DSpark
drafter in `a64fx/dspark`. Work on an A64FX compute node and compile and run
the binaries directly on that node.

## Checkpoints and local staging

The authoritative shared-filesystem checkpoints are:

```text
/home/u14346/models/qwen38/radixark/Qwen3.8-27B-DSpark
/home/u14346/models/qwen38/radixark/Qwen3.8-27B-NVFP4
```

They occupy about 3.5 GiB and 21 GiB respectively. **Always stage both trees
into `/local` before validation, loading, or benchmarking.** Do not point a
full validator or benchmark at the shared paths. `/local` is erased when the
allocation ends, so repeat staging for every fresh allocation.

A single `cp` of the roughly 25 GiB tree can build a large dirty page cache on
a 32 GiB A64FX node. Use bounded 1 GiB writes with an fsync after each chunk:

```sh
cd /vol0006/mdt0/data/hp250467/work/gemm/glm53f

SRC_ROOT=/home/u14346/models/qwen38/radixark
STAGE_ROOT=/local/u14346/qwen38-radixark
mkdir -p "$STAGE_ROOT"

stage_one() {
    src=$1
    dst=$2
    size=$(stat -c%s "$src")
    if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" = "$size" ] && \
       [ ! "$src" -nt "$dst" ]; then
        echo "reuse $dst ($size bytes)"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    truncate -s 0 "$dst"
    chunks=$(( (size + 1073741823) / 1073741824 ))
    i=0
    while [ "$i" -lt "$chunks" ]; do
        dd if="$src" of="$dst" bs=1M count=1024 \
           skip=$((i * 1024)) seek=$((i * 1024)) \
           conv=notrunc,fsync status=none
        i=$((i + 1))
        echo "  $dst: chunk $i/$chunks"
        sleep 1
    done
    truncate -s "$size" "$dst"
    touch -r "$src" "$dst"
}

for model in Qwen3.8-27B-DSpark Qwen3.8-27B-NVFP4; do
    find "$SRC_ROOT/$model" -type f -print0 |
    while IFS= read -r -d '' src; do
        rel=${src#"$SRC_ROOT/"}
        stage_one "$src" "$STAGE_ROOT/$rel"
    done
done

DRAFT=$STAGE_ROOT/Qwen3.8-27B-DSpark
TARGET=$STAGE_ROOT/Qwen3.8-27B-NVFP4
du -sh "$DRAFT" "$TARGET"
test "$(stat -c%s "$DRAFT/model.safetensors")" = \
     "$(stat -c%s "$SRC_ROOT/Qwen3.8-27B-DSpark/model.safetensors")"
for shard in "$TARGET"/model-*.safetensors; do
    source_shard=$SRC_ROOT/Qwen3.8-27B-NVFP4/$(basename "$shard")
    test "$(stat -c%s "$shard")" = "$(stat -c%s "$source_shard")"
done
```

The size checks catch incomplete staging. For archival or release validation,
also compare checksums, but do that outside timed runs because it rereads all
checkpoint bytes.

## Native build and baseline correctness

Keep compiler temporaries inside the repository; `/tmp` is not available on
the compute node. The Makefile already sets `TMPDIR` appropriately.

```sh
cd /vol0006/mdt0/data/hp250467/work/gemm/glm53f
make -C a64fx/dspark clean all CC=fcc \
  CFLAGS='-Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast'

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/test_dspark
```

Expected result is `dspark tests: PASS`. The production-length BF16 test
should remain close to the recorded scalar/SVE agreement:

```text
long-K GEMM rel_l2=1.45107e-06 scaled_max=2.29086e-06
```

Do not continue to real-weight performance work if the unit test or long-K
comparison fails.

## Reproduce real-weight validation

All commands below deliberately use only `/local` checkpoint paths:

```sh
STAGE_ROOT=/local/u14346/qwen38-radixark
DRAFT=$STAGE_ROOT/Qwen3.8-27B-DSpark
TARGET=$STAGE_ROOT/Qwen3.8-27B-NVFP4
test -s "$DRAFT/model.safetensors"
test -s "$TARGET/model.safetensors.index.json"

./a64fx/dspark/validate_dspark --headers "$DRAFT" "$TARGET"

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/validate_dspark --full "$DRAFT" "$TARGET"
```

The header check must pass before the resident load. The full check should
report `headers PASS`, `load PASS backend=sve`, and `proposal PASS`. Resident
weights use about 6.7 GiB; the default 8192-token state adds about 160 MiB.
Record `MemAvailable` before and after the run:

```sh
awk '/MemAvailable/{print}' /proc/meminfo
```

For the independent checkpoint/PyTorch oracle, first create the fixture on a
machine with PyTorch, Transformers 5.8.1, and safetensors. Copy only the small
JSON fixture back to this checkout, then validate the locally staged weights:

```sh
STAGE_ROOT=/local/u14346/qwen38-radixark
python3 a64fx/dspark/export_golden.py \
  "$STAGE_ROOT/Qwen3.8-27B-DSpark" \
  "$STAGE_ROOT/Qwen3.8-27B-NVFP4" \
  a64fx/dspark/tmp/qwen38_golden.json

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/validate_dspark \
  --golden a64fx/dspark/tmp/qwen38_golden.json --full \
  "$STAGE_ROOT/Qwen3.8-27B-DSpark" \
  "$STAGE_ROOT/Qwen3.8-27B-NVFP4"
```

Token IDs must match exactly. Confidence absolute error and selected-logit
relative error must each remain within the validator's `2e-3` tolerance.

## Reproduce the performance baseline

Run the benchmark only after the full real-weight validator passes:

```sh
STAGE_ROOT=/local/u14346/qwen38-radixark
DRAFT=$STAGE_ROOT/Qwen3.8-27B-DSpark
TARGET=$STAGE_ROOT/Qwen3.8-27B-NVFP4

OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/dspark/bench_dspark "$DRAFT" "$TARGET" |
  tee a64fx/dspark/tmp/bench-a64fx.txt
```

The benchmark measures 1, 12, and 48 threads and compares every SVE result
with the scalar backend. The current reference for one accepted context row
is 1746.9/204.2/69.4 ms at 1/12/48 threads. All seven token IDs must match;
maximum confidence difference must be at most `2e-4`, and selected-logit
relative difference at most `5e-4`. Treat timings as comparable only when the
node frequency and OpenMP placement are unchanged.

For each performance change, save:

- commit or exact source diff;
- compiler and complete flags;
- node frequency and thread placement;
- all benchmark rows, not only the best result;
- token match and both numerical error maxima;
- at least three repetitions when claiming a speedup.

## Current state and continuation tasks

The standalone implementation and first SVE optimization are in commits
`fc741faa` and `691bd7ff`. On the current A64FX node, a clean native rebuild
passes all scalar/SVE unit tests. A synthetic 5120-by-5120, width-seven BF16
GEMM measured 177.8 GFLOP/s with 12 threads and 643.1 GFLOP/s with 48 threads;
this is a kernel sanity check, not a substitute for the real-weight proposal
benchmark.

Native real-weight validation on job `51819460` staged both checkpoint trees
under `/local/u14346/qwen38-radixark`. The original 48-way concurrent `pread`
loader took 1595.2 seconds and accumulated 1384.5 seconds of system time even
though a direct read of the local 3.5 GiB draft took about two seconds. LLIO
was serializing the concurrent reads into anonymous pages. The loader now
performs one sequential read into a temporary anonymous buffer, followed by a
parallel first-touch copy into the resident HBM arena. Results:

- load: 1595.2 seconds to 6.7 seconds (238x faster);
- full proposal: PASS with exactly unchanged token IDs, confidence, and
  selected logits;
- transient peak RSS: 8,572,160 KiB, zero swaps;
- post-fix benchmark: 1756.0/205.5/75.2 ms at 1/12/48 threads;
- scalar/SVE: all seven tokens match, maximum confidence difference
  `1.24037e-4`, maximum selected-logit relative difference `4.05867e-4`.

The drafter therefore fits safely on one A64FX node. Do not introduce tensor
or data parallelism for the standalone validator; reserve the other allocated
nodes for independent experiments or later target-model integration.

`DSPARK_PROFILE=1` enables proposal phase timing. Four fresh-load profiles on
the same job produced three stable totals of 65.6--67.8 ms and one 98.2 ms
scheduling outlier. In the stable runs, FFN up/gate plus down projections cost
27.4--28.1 ms, the NVFP4 LM head cost 16.1--16.8 ms, and sequential
Markov/confidence correction cost 9.3--10.5 ms. Those phases account for about
80% of proposal time. Two-token attention normally cost less than 1 ms; one
32 ms attention sample was the source of the scheduling outlier. Use multiple
profiles and medians when evaluating changes.

The first profile-guided change combines each FFN gate/up projection pair in
one OpenMP workshare, with the two matrices processed concurrently while
retaining the same per-row SVE accumulation order. Four fresh-load runs gave
stable totals of 64.3--66.6 ms and gate/up times of 17.4--17.9 ms, versus
65.6--67.8 ms total and 18.8--19.1 ms for gate/up before the change. Real
checkpoint proposals remained exactly unchanged. A dedicated paired-GEMM
scalar/SVE test covers both outputs.

The second profile-guided change moves a complete attention score row and
BF16 value reduction into SVE kernels, avoiding millions of tiny dot calls and
scalar BF16 conversions at long context. Synthetic zero-KV context probes
measured 1024-token attention at 4.1 ms (previously 15.8 ms) and 8192-token
attention at 58.1 ms (previously 158.3 ms). The 8192-token total proposal fell
from 223.0 to 122.0 ms. The full real-weight scalar/SVE gate still matches all
seven tokens; maximum confidence difference is `1.31279e-4` and maximum
selected-logit relative difference is `3.58070e-4`. A direct scalar/SVE test
covers split persistent/current K/V inputs.

Continue in this order:

1. Generate the independent golden fixture and run the golden validator; the
   native full validator and scalar/SVE cross-check already pass.
2. Reproduce the 1/12/48-thread real-weight baseline at least three times and
   characterize the observed 48-thread run-to-run variation.
3. Optimize only the measured dominant phase. Start with the width-seven FFN
   BF16 GEMMs, then the NVFP4 output panel and Markov correction. Preserve
   FP32 accumulation and the existing numerical thresholds. Re-profile at a
   representative long context before changing attention.
4. Re-run unit, long-K, full-checkpoint, golden, and scalar/SVE benchmark
   gates after every kernel change. Revert changes that move cost elsewhere or
   improve only a synthetic microbenchmark.
5. Integrate the drafter with target verification only after standalone
   correctness and performance are stable. The target remains authoritative:
   append only accepted target features, and use truncate/reset transactionally
   after rejected speculative rows.

Do not commit generated binaries, staged weights, golden fixtures, or timing
logs. Never use `/tmp`, and never benchmark from the shared checkpoint path.
