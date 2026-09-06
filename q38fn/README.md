# Qwen3.8-Flash-Next A64FX bring-up

The Fugaku checkpoint is `~/models/q38fn/bf16`. Its text backbone is a
Qwen4-Exp model with 48 layers, hidden size 2560, 24 attention heads, 2 KV
heads, 512 routed experts with top-10 activation, and four persistent
hyper-connection streams. Three of every four layers use Gated DeltaNet; the
fourth uses full attention. PLE is enabled at zero-based layer 1 (the second
decoder block).

## 51B n-gram embedding

The PLE table is 51,200,245,760 BF16 parameters (95.37 GiB), stored as 128
packed row splits. Each split is `[2,500,012, 160]`, or 800,003,840 bytes.
The packed table has 320,001,536 rows, with 16 hash heads: eight bigram and
eight trigram heads. The lookup returns 16 × 160 BF16 values (5,120 bytes per
token) before PLE projection/convolution.

For a current token `t0` and preceding tokens `t1`, `t2`:

```text
bigrams = (t0 * m0) XOR (t1 * m1)
trigrams = bigrams XOR (t2 * m2)
row[h] = head_offset[h] + unsigned_remainder(hash, head_vocab_size[h])
```

The multiplications intentionally wrap int64. EOS resets missing/earlier
context. `common/q38fn_arch.h` contains the checkpoint-provided constants.
The hash scheme follows the Qwen/Hugging Face implementation and the
corresponding NVIDIA reference documentation.

## Runner and probes

### Tensor-parallel BF16 runner (TP12)

`q38fn_tp_runner` is the lockstep, single-stream path intended to replace the
pipeline runner for decode.  Every decoder layer executes on all 12 nodes:
projection output rows or columns are sharded, each routed expert keeps only
its local intermediate slice, the vocabulary endpoints are row-sharded, and
the 51B n-gram table is owner-sharded.  The per-rank blobs are checksummed and
published atomically, so a partial stage is never mistaken for a runnable
model.

Stage directly from Fugaku's shared model filesystem into each compute node's
own `/local` filesystem (do not rsync model weights through bash-over-http):

```bash
make -C q38fn q38fn_tp_stage_mpi MPICC=mpifcc
mpiexec -n 12 q38fn/q38fn_tp_stage_mpi \
  "$HOME/models/q38fn/bf16" /local/u14346/q38fn-tp
```

For a bounded layout/runtime check, set `Q38FN_TP_STAGE_LAYERS=1` and use
`--layers 1 --probe-token 42`.  A complete decode requires the default
48-layer stage.

Build the native A64FX/uTofu runner after unloading a conflicting OSS LLVM MPI
module if one is active:

```bash
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
export PATH=/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH
CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp' \
  MPICC=mpifcc make -C q38fn q38fn_tp_runner_utofu
mkdir -p q38fn/runs/tp12
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
Q38FN_TP_TRACE_DIR="$PWD/q38fn/runs/tp12" \
  mpiexec -n 12 q38fn/q38fn_tp_runner_utofu \
    "$HOME/models/q38fn/bf16" --local-base /local/u14346/q38fn-tp \
    --prompt 'Write a complete C11 Fibonacci program. Return only code.' \
    --max-gen 128 --max-seq 512
```

Before every multi-node runner launch, regenerate the uTofu topology from the
current PJM allocation.  `tofu_topo_helper` calls `utofu_query_my_coords()` on
each MPI rank and writes the rank-ordered six-dimensional coordinates.  Do not
reuse a topology file from another allocation:

```bash
make -C a64fx/utofu-tests tofu_topo_helper \
  MPICC=mpifcc
rm -f a64fx/utofu-tests/tofu_topo.txt
mpiexec -n 12 a64fx/utofu-tests/tofu_topo_helper
cat a64fx/utofu-tests/tofu_topo.txt

TOFU_TOPO_PATH="$PWD/a64fx/utofu-tests/tofu_topo.txt" \
  mpiexec -n 12 q38fn/q38fn_tp_runner_utofu \
    "$HOME/models/q38fn/bf16" --local-base /local/$USER/q38fn-tp
```

Run the helper and the application as separate, serialized `mpiexec` launches;
never start two uTofu/MPI applications concurrently in one interactive job.
For the current 12-node interactive allocation, `PJM_MPI_SHAPE_X=12` and the
full allocation includes virtual coordinate `(0,0,0)`.  Consequently, use the
plain 12-rank launch shown above: a vcoord file that excludes `(0,0,0)` has
only 11 usable entries.  Use `-vcoordfile` only for a strict subset, and build
it from the actual `PJM_MPI_SHAPE_*` values.  The helper's `tofu_topo.txt` is
the placement/discovery contract consumed by the uTofu transport; keep it in
the runner's working directory or set `TOFU_TOPO_PATH` explicitly.

For a placement-independent uTofu smoke test after topology discovery:

```bash
mpiexec -n 12 a64fx/utofu-tests/tofu_topo_helper
mpiexec -n 12 a64fx/utofu-tests/tp_ar_diag_bench
```

The second command is deliberately issued only after the first exits.  On
Fugaku, `CODE=1907` means retry the helper once in the same allocation and
then retry the application; do not leave overlapping MPI launchers running.
When issuing these commands through the local bash-over-HTTP wrapper, use
explicit remote paths such as `/home/u14346/models/q38fn/bf16` carefully: the
local shell can expand an unquoted `$HOME` before the command reaches Fugaku.

The helper must be run as the allocation's MPI-launched command, not from a
shell already running as a bash-over-HTTP/PJM rank; nested `mpiexec` launches
are rejected by Fugaku's `plexec`.  If uTofu initialization returns transient
`CODE=1907`, remove the topology file and retry the helper, then retry the
runner in the same allocation.  For TP4 or TP6, use the matching `-n` value
and runner target; the topology file must contain exactly that many fresh
rank entries.  `TOFU_TOPO_PATH` is optional when the runner starts in the
directory containing `tofu_topo.txt`, but setting it explicitly avoids stale
files when staging and running from different directories.

MPI is used for launch, uTofu VCQ-ID bootstrap, and prompt broadcasts only.
All generated-token tensor-parallel sums and the sharded-vocabulary argmax use
one-sided uTofu collectives.  Blob loading uses parallel 64 MiB `pread`s: with
48 bound workers this first-touches resident weights across all four CMGs
instead of pinning the model to CMG0.  Decode matvecs process eight BF16 rows
per SVE kernel call, including the selected routed experts and LM-head shard.
Per-rank traces are written when `Q38FN_TP_TRACE_DIR` is set.

Scalar prefill keeps the recurrent PLE and decoder layers strictly
autoregressive, but batches the independent token-embedding reductions and
PLE key/value projections in eight-token windows. The projected PLE values are
then consumed in token order, preserving the recurrent state while reducing
prompt-side uTofu reductions, n-gram lookups, and repeated projection setup. This does not use
the experimental layer-major path. Set `Q38FN_TP_DISABLE_EMBED_BATCH=1` for
an A/B comparison or legacy behavior.
BF16 window projections use one SVE/OpenMP traversal over the resident rows;
the scalar fallback remains available on non-SVE hosts.

TP layout v5 omits unused `model.visual.*` tensors and physically
packs only the DeltaNet Q/K/V rows executed by each rank. Decoder and expert
tensors are anonymous/resident. Owned n-gram tensors are staged directly as
signed Q5 blocks (32 weights plus an FP16 scale in 22 bytes), loaded into
anonymous HBM, and dequantized only for the 16 selected rows. This removes
decode-time file faults. The loader also converts legacy BF16 n-gram entries
in bounded chunks, dropping source pages as it advances.

### Exact speculative decode

Use `--spec-width 4` or `--spec-width 8` to enable exact greedy speculative
decode. The first token in every block is the target model's carried greedy
token; later candidates come from repeated-history matching and the staged
native `mtp.*` layer. Native MTP is currently isolated behind
`Q38FN_TP_ENABLE_MTP=1`: it changes target logits when interleaved with target
prefill and has not passed the exactness gate. The safe default therefore uses
history proposals only. The target verifies every candidate, commits only the
longest matching prefix, and restores DeltaNet, PLE, attention, MTP, and
hyper-connection state to that prefix. Consequently its committed token IDs
must match `--spec-width 1` exactly.

On TP12, the safe one-token fast path reproduced 16/16 baseline token IDs at
16.49 tok/s (scalar baseline 16.98 tok/s). Transactional replay remains active
only for blocks wider than one. An MPI-only MTP collective diagnostic produced
the same divergence as uTofu, ruling out collective-channel ordering; inspect
MTP tensor shapes and write bounds before enabling it.
State-integrity probes found no changes to target DeltaNet, attention KV, PLE,
saved/live hyper activations, or the target LM-head scratch when MTP uses its
dedicated model view. The remaining interference is hidden process-global or
kernel runtime state, so in-process MTP remains unsupported by default.

### Experimental prefill windows

`Q38FN_TP_PREFILL_WIDTH=8 Q38FN_TP_LAYER_MAJOR=1` processes prompt tokens in
eight-token layer-major windows and batches embedding/n-gram communication.
Prefill does not allocate or copy speculative rollback journals. On the
83-token repeat-8 prompt this reduced prefill from 4.126 s to 3.745 s (9.2%)
and reduced collective calls from 17,836 to 17,528 for the complete prefill +
eight-token decode run. It remains opt-in: an earlier journaled run matched
8/8 generated IDs, but the journal-free timing changed token IDs after token
two, exposing the runner's unresolved numerical instability. Do not use this
path for quality claims until a deterministic scalar baseline and per-layer
checksum gate pass repeatedly.

`Q38FN_TP_MPI_SUM=1` forces all tensor-parallel sums through MPI while keeping
the same runner binary. Use it as the determinism oracle: compare two scalar
per-layer checksum probes and then repeat with MTP prefill enabled. If MPI is
stable while uTofu changes, the defect is in uTofu completion/slot reuse rather
than model state or prefill scheduling.

`--spec-adaptive` starts by probing the requested maximum width and width four,
then chooses the better observed committed-token rate while periodically
narrowing after poor acceptance. The runtime reports proposal, verification,
commit time, source counts, and an acceptance histogram in `Q38FN_SPEC`.

```bash
mpiexec -n 12 q38fn/q38fn_tp_runner_utofu \
  "$HOME/models/q38fn/bf16" --local-base /local/$USER/q38fn-tp \
  --prompt 'Write a portable C11 merge sort. Return only code.' \
  --max-gen 128 --max-seq 512 --spec-width 8 --spec-adaptive
```

The final hyper-connection mixer and LM head evaluate all candidate activations
as Q8 block passes and use packed reductions plus one batched distributed
argmax. PLE performs one packed reduction per block; its n-gram requests are
sorted by resident shard/row, deduplicated, and prefetched.
Sharded token embeddings are also gathered with one packed reduction per
candidate block instead of one collective per token.
Target verification is token-major by default: each candidate completes all
layers before the next candidate starts, preserving the model's cross-layer
recurrent state exactly. `Q38FN_TP_LAYER_MAJOR=1` enables the experimental
layer-major scheduler for kernel development, but it currently fails the
greedy-token exactness gate and must not be used for quality or throughput
claims. Final-mixer and LM-head evaluation is also scalar by default;
`Q38FN_TP_BATCHED_HEAD=1` enables its experimental packed implementation.
Width-aware embedding, PLE, and n-gram kernels remain available on the exact
path.

The full-attention window path is experimental and disabled by default because
the reordered layer path does not yet preserve target token IDs. Set
`Q38FN_TP_ATTN_WINDOW=1` only for numerical A/B work. The default retains the
layer-major scheduler while executing each full-attention layer through the
original exact path. Compare committed token IDs—not decoded text alone—against
`--spec-width 1` before accepting a new kernel result.

The stager retains the checkpoint's `mtp.*` tensors for exact target-verified
speculative decoding. MTP attention and MoE tensors use the same TP ownership
rules as one full-attention decoder layer; small fusion and normalization
tensors are replicated. Token embeddings and the LM head remain shared with
the target model rather than duplicated.

### Four-node low-bit path (TP4)

Build-time `Q38FN_TP_RANKS=4` support covers tensor ownership, DeltaNet's 12
local value heads, six local full-attention heads, n-gram ownership, runtime
state sizing, and uTofu peer tables. The dedicated targets are:

```bash
make -C q38fn q38fn_tp4_stage_mpi q38fn_tp4_runner_utofu \
  MPICC='mpifcc -Nclang' \
  CFLAGS='-O3 -Wall -Wextra -Wpedantic -march=armv8.2-a+sve -fopenmp'
```

`q38fn_lowbit_plan` computes storage from the checkpoint metadata and the
actual TP ownership rules. For four ranks it reports the same totals on every
rank:

```text
BF16 main       58.781 GiB   BF16 n-gram       23.842 GiB
Q5 main         20.306 GiB   Q5 n-gram          8.196 GiB
Q5 main+ngram   28.502 GiB   Q5+MXFP4 n-gram   26.639 GiB
```

All-Q5 is selected because it fits 32 GiB HBM with about 3.5 GiB remaining
and preserves n-gram cosine better than MXFP4. On real A64FX routed-expert
weights, the reusable Q5 SVE kernel measured 180--192 GB/s BF16-equivalent,
relative L2 0.0465, and cosine 0.99892. On a real 160-wide n-gram shard, Q5
measured relative L2 0.0455 and cosine 0.99896; MXFP4 measured 0.1166 and
0.99318 respectively. `make -C q38fn test` builds and runs both TP12 and TP4
layout/runtime tests plus scalar Q5 and compressed-blob tests.

The TP4 stager quantizes every matrix whose local inner dimension is a whole
32-value Q5 block. The 80-column hyperconnection mixer slice remains BF16;
the storage calculator includes that fallback. A native four-rank bounded
stage (`Q38FN_TP_STAGE_LAYERS=1`) produced a 669,693,440-byte rank-0 blob:
19 Q5 tensors occupied 664,634,080 bytes. The one-layer uTofu runner completed
on all four ranks with identical checksum `2d6a48d5875ac370`, finite output,
and 15.14 ms layer time. This gate exercises Q5 embedding, hyperconnection,
DeltaNet, routed/shared MoE, five uTofu collectives, and four-rank state/layout
dimensions. A second two-layer stage proved 32 owned n-gram tensors were
resident Q5 (`8,800,042,240` bytes) rather than file-mapped BF16.

For the bounded gate:

```bash
Q38FN_TP_STAGE_LAYERS=1 OMP_NUM_THREADS=48 \
  OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -n 4 q38fn/q38fn_tp4_stage_mpi \
    "$HOME/models/q38fn/bf16" /local/$USER/q38fn-tp4-q5
Q38FN_TP_PROFILE=1 OMP_NUM_THREADS=48 \
  OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -n 4 q38fn/q38fn_tp4_runner_utofu \
    "$HOME/models/q38fn/bf16" --local-base /local/$USER/q38fn-tp4-q5 \
    --layers 1 --probe-token 42
```

The corrected full 48-layer runner is coherent. For token 42, serial and TP12
layer-0 comparisons have Delta output cosine `0.999999999999` (max absolute
error `3.28e-6`) and final layer output cosine `0.999999999999` (RMSE
`2.22e-8`). The first material defect was an OpenMP four-head recurrent loop
that updated only local head 0 on the Fugaku MPI runtime; the four stateful
head updates are serial pending integration into a persistent worker pool.
Hard gates now abort on non-finite layer/final values, invalid token IDs, or
non-finite logits.

A 29-token Fibonacci coding prompt generated a coherent 32-token prefix.
Fusing the top-10 routed-expert gate/up and down phases into one OpenMP region
per phase improved the same robust short run from 5.67 to **6.42 tok/s**
(`4.544 s` prefill, `4.984 s` decode), while the layer-0 oracle remained at
MoE cosine 1.0 and maximum absolute error `2.68e-7`. The generated prefix was
`The user wants a complete C11 program that prints the first ten Fibonacci
numbers.` The prior robust uTofu baseline measured 5.67 tok/s (`4.886 s`
prefill, `5.647 s` for 32 decode tokens), with `3.041 s` in 11,956 collectives
over the complete run. The SVE eight-row path
is oracle-valid but measured 5.56 tok/s, so `Q38FN_TP_EXACT_MV=1` remains the
faster baseline. Direct all-to-all uTofu reduction (`TP_AR_A2A=1`) reached
6.53 tok/s and stayed rank-coherent, but rank-order reassociation changed the
generated branch after token 17; it is therefore not the exact default.
`TP_AR_ROBUST=2` reached 6.26 tok/s in a short run but also changed logits
materially and is not a validated/default mode.

A robust 202-output-token run measured **7.16 tok/s**, stopped at
`<|im_end|>`, and produced a complete Fibonacci C11 program. The extracted
code compiled with `fcc -Nclang -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror`
and printed the exact sequence `0 1 1 2 3 5 8 13 21 34`. Repeated-prompt tests
at 143 and 257 input tokens each generated coherent 32-token continuations.

An evaluated v3 experiment replicated each decoder layer's 320-wide
hyperconnection input mixer to remove 96 reductions per token. It passed the
layer oracle but increased cold layer time to 38.5 ms because every rank did
12 times the mixer work; strided emulation still reached only 4.96 tok/s.
Layout v4 restores physically packed rank-local mixer slices. The local
`q38fn_tp_hc_repack_mpi` conversion takes seconds and preserves the serial
oracle (layer cosine 1.0, RMSE `1.15e-8`).

Combining v4 with direct A2A reduction and a single contiguous uTofu Put per
peer reached **6.45 tok/s** with LLVM 21 kernels. Fusing routed and shared MoE
work into three OpenMP regions raised this to 6.50 tok/s while retaining the
layer oracle. CMG-striped first touch at 64 KiB granularity for small and
routed-expert tensors produced the current short-run best: **7.37 tok/s**
(`3.772 s` prefill, `4.339 s` decode for 32 tokens, `1.965 s` aggregate
communication, `48.94 s` load). Striping every large tensor measured 7.32
tok/s, so non-expert tensors of at least 16 MiB retain 4 MiB load chunks.

The one-Put path reduced
transport time to 1.35 s when built wholly with Fujitsu's compiler, but its
slower generated matvecs offset that gain. A mixed-compiler executable did not
retain the Fujitsu transport result because runtime selection follows the
linking MPI wrapper. Active OpenMP waiting also regressed a 16-token run to
5.84 tok/s. Overlapping local TCQ completion with inbound A2A traffic preserved
tokens but regressed to 6.23 tok/s and was reverted. These variants are
measurements, not production defaults. The
30 tok/s target is not yet met.

For rank-0 FAPP collection of the actual PP3 x TP4/uTofu decode path, build
the instrumented runner and place the rank-selective wrapper inside `mpiexec`:

```bash
make -C q38fn q38fn_tp4_runner_utofu_fapp \
  MPICC='mpifcc -Nclang' \
  CFLAGS='-O3 -march=armv8.2-a+sve -fopenmp'
Q38FN_FAPP_DIR=/local/$USER/q38fn-pp3-fapp \
Q38FN_TP_FAPP=1 Q38FN_TP_PIPELINE=1 Q38FN_TP_SKIP_NGRAM=1 \
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -n 12 sh q38fn/run_tp_fapp_rank.sh \
    q38fn/q38fn_tp4_runner_utofu_fapp ~/models/q38fn/bf16 \
    --local-base /local/$USER/q38fn-pp3-tp4 \
    --prompt hello --max-gen 8 --max-seq 64
fapp -A -d /local/$USER/q38fn-pp3-fapp -ttext
```

The wrapper profiles rank 0 only by default, avoiding twelve redundant FAPP
collectors. Set `Q38FN_FAPP_RANK` to inspect a different pipeline stage.

For TP12 DeltaNet analysis, level 1 is required because the detailed regions
are nested under `decode`.  `-Icpupa` exposes the PA2 counters in CSV output:

```bash
Q38FN_TP_FAPP=1 Q38FN_TP_FAPP_COMPONENTS=1 \
Q38FN_TP_FAPP_DETAIL=1 Q38FN_FAPP_LEVEL=1 Q38FN_FAPP_EVENT=pa2 \
Q38FN_FAPP_DIR=/local/$USER/q38fn-detail-pa2 \
  mpiexec -n 12 sh q38fn/run_tp_fapp_rank.sh \
    q38fn/q38fn_tp_runner_utofu_fapp ~/models/q38fn/bf16 \
    --local-base /local/$USER/q38fn-tp12-q8 --prompt hello --max-gen 8
fapp -A -Icpupa -d /local/$USER/q38fn-detail-pa2 -tcsv
```

The TP12 PA2 split measured recurrent, projection, and output at 18.59, 16.92,
and 7.81 billion cycles respectively.  Recurrent load stalls were 43.6% of
cycles.  Its 32 `(head,column-tile)` tasks are now assigned to eight cores in
each CMG using the same permutation as first-touch loading; HC-up similarly
returns each 64 KiB block to its first-touch CMG.  Both schedules are default,
with `Q38FN_TP_NO_DELTA_CMG_STRIPE=1` and
`Q38FN_TP_NO_HC_CMG_STRIPE=1` available for A/B tests.  With
`FLIB_FASTOMP=TRUE`, the matched 64-token TP12 run improved from 14.93 to
16.32 tok/s while preserving all 64 token IDs exactly.

A later resident BF16-base/Q8-overlay profile reached **18.83 tok/s** for a
16-token TP12 decode.  The corrected eight-row HC-up schedule reduced HC-up
from 6.24 to 5.61 ms/token (10.1%) and raised end-to-end throughput from 18.57
to 18.66 tok/s.  Rank-0 PA2 counters on that path show the remaining limit is
distributed load/commit latency rather than one kernel: HC spends 60.5% of
cycles with no commit and 35.1% in load completion wait; MoE is 53.2% and
27.3%; Delta projection is 52.7% and 28.8%; Delta recurrent is about 59.6%
and 39.3%.  A serial Q8 router and a 12-thread generic Q8 limit both regress,
so the adaptive Q8 thread count remains the default.

When a BF16 base is combined with a Q8 overlay, set
`Q38FN_TP_SKIP_Q8_OVERLAYED=1`.  The loader then reserves base virtual
addresses but does not fault shadowed BF16 tensors into HBM; without it, the
base, HC sidecar, and Q8 overlay exceed practical per-node HBM headroom.  The
TP12 stager target now defines `Q38FN_TP_STAGE_DEFAULT_Q5=1` so fresh staging
uses the intended compact base by default.

`Q38FN_TP_PROFILE=1` records layer times and finite-value statistics.
`Q38FN_TP_TRACE_DIR=/local/...` writes per-rank tokens and summaries without
depending on Fugaku's MPI/PLE stdout forwarding. `Q38FN_DUMP_DIR=/local/...`
writes component vectors for `q38fn/compare_f32`; for a one-layer probe set
`Q38FN_TP_LOAD_LAYERS=1` to avoid loading unrelated resident weights.

Stage new v4 blobs directly with `q38fn_tp_stage_mpi`. Interrupted staging
resumes at the last manifest-recorded tensor boundary: uncommitted blob tails
are truncated and completed tensors are not reread. Existing v1 blobs can be
upgraded in place at local-SSD speed without rereading shared safetensors:

```bash
mpiexec -n 12 q38fn/q38fn_tp_upgrade_mpi /local/u14346/q38fn-tp
```

The converter renames the recoverable v1 blob to v2, packs QKV into the start
of each existing entry reservation, and atomically publishes a checksummed v2
manifest. It needs no second 29 GB blob, which matters when PP staging already
occupies `/local`.

The temporary replicated-HC v3 format can be repacked locally to v4 without
rereading shared safetensors:

```bash
make -C q38fn q38fn_tp_hc_repack_mpi MPICC=mpifcc
mpiexec -n 12 q38fn/q38fn_tp_hc_repack_mpi /local/u14346/q38fn-tp
```

### Twelve-node resident runner

The production path must not stream checkpoint weights from shared storage.
First run one staging rank per node; each rank copies its four decoder layers,
its share of the 128 PLE table tensors, and its endpoint weights to a distinct
node-local directory:

```bash
mpiexec -n 12 q38fn/q38fn_stage_mpi ~/models/q38fn/bf16 /local/u14346/q38fn
```

The stager is idempotent by file size: rerunning it keeps complete staged
files and copies newly required or incomplete files. After `fsync`, it issues
`POSIX_FADV_DONTNEED` for destination files (including files retained by an
idempotent rerun). This is required before anonymous HBM upload: recently
written 34 GB payloads otherwise retain enough page cache to OOM-kill a rank.
Rebuild the stager after changing its tensor-selection logic so an old
executable cannot leave a rank with a stale shard manifest.

The resulting payload roots are `/local/u14346/q38fn/rank-00` through
`rank-11` on their respective nodes. Then run the decoder:

```bash
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -n 12 q38fn/q38fn_pp_runner ~/models/q38fn/bf16 \
  --local-base /local/u14346/q38fn \
  --prompt 'Write a C11 function that returns the nth Fibonacci number. Return only code.' \
  --max-gen 256 --max-seq 512
```

At startup, every rank materializes all BF16 tensors for its four layers into
anonymous HBM2 under an interleaved CMG memory policy. Expert slices point
into the encompassing resident expert tensors. The PLE table remains
distributed by logical n-gram shard: endpoint ranks 0 and 11 own nine tables
each, while ranks 1 through 10 own eleven each. Each BF16 n-gram row is
quantized once to signed INT8 with a per-row FP32 scale, reducing an 800 MB
shard to 410 MB while preserving random HBM-resident lookup. Rank 0 caches
only the 5 KiB input-embedding rows actually touched instead of retaining the
full 1.27 GB vocabulary table; rank 11 retains the final mixer and LM head.
A per-token collective assembles only the 16 selected rows. The stager copies every physical
safetensors file needed by a rank's logical table assignment; shared storage
is used for metadata only after staging. Payload reads larger than Linux's
single-read limit are split into 64 MiB `pread` operations.

The 12-node A64FX validation cached 29.223--29.539 GB per rank in HBM2 and
loaded it from `/local` in 37.0--38.7 seconds. A 29-token Fibonacci coding
prompt generated a coherent, warning-clean C11 program and stopped at
`<|im_end|>` after 175 output tokens. Measured prefill was 62.9 seconds and
decode was initially 0.467 token/s. Direct resident-BF16 MoE and LM-head
matvecs, batched top-10 expert phases, fused shared-expert projections, and
FP32/SVE-friendly reductions improved the same 29-token/32-output benchmark
to 11.0 seconds prefill and 2.963 token/s decode (6.34x). Reusing persistent
runtime scratch buffers and reading hyperconnection, DeltaNet convolution,
PLE, attention-normalization, and final-mixer weights directly from resident
BF16 storage further reduced the same run to 7.71 seconds prefill and 4.225
token/s decode (9.05x over the original baseline, 42.6% over 2.963 token/s).
The generated token prefix remained unchanged. Explicit one- and
four-accumulator SVE-intrinsic variants reached 2.882 and 2.960 token/s, so the
simpler compiler-vectorized reduction remains selected. Thread-topology tests
also retained `OMP_NUM_THREADS=48 OMP_PROC_BIND=close`: 24-thread and 36-thread
spread placement reached 4.181 and 4.211 token/s respectively. A longer
29-token-prompt/128-output run measured 8.057 seconds prefill and 4.131
token/s decode (30.987 seconds), 2.2% below the short 32-output sample as the
attention context grew.

On job 51078755, a freshly staged BF16 n-gram preload exceeded the practical
HBM limit and SIGKILLed ranks at 27.9--29.5 GB resident. Destination-cache
eviction plus row-scaled INT8 n-gram residency reduced rank 0 to 24.495 GB and
left 5.9 GB `MemAvailable`; all 12 ranks then completed. The one-time
BF16-to-INT8 conversion made preload 110.7 seconds. Eight-row SVE kernels
reduced a four-layer stage from about 61 to 23 ms/token, and the same kernel
reduced the LM-head projection from 54.5 to 14.4 ms/token. Their combined
short run produced the same coherent 32-token prefix at **3.590 tok/s**.
A 63-input-token/256-output-token run remained coherent and measured
**3.059 tok/s** (`18.268 s` prefill, `83.682 s` decode). Rank timing proves
that activation sends (roughly 0.04 ms/token) and the n-gram collective
(roughly 0.11 ms/token) are negligible; the serial sum of 12 layer stages is
the pipeline bottleneck.
Extending the identical prompt to 512 output tokens completed without NaNs,
rank divergence, timeout, or OOM at **2.706 tok/s** (`18.192 s` prefill,
`189.190 s` decode). It remained grammatical and correctly reasoned about
the `uint64_t` requirement for `F(93)`, but exhausted the token limit while
still thinking rather than emitting the requested source. Therefore this run
is a long-context stability check, not the code-generation quality gate; the
greedy-exact TP code/compile/run validation above remains that gate.

### Decode roofline

The active BF16 path reads approximately 13.3 GB/token: about 12.0 GB from
the 48 decoder layers, 1.27 GB from the LM head, and the remaining endpoint
and PLE projections. Using the measured 922 GB/s/node sequential-HBM result,
a serial pipeline has a bandwidth-only ceiling of about **69.3 tok/s**.
Consequently 30 tok/s requires 399 GB/s, or **43.3%** of that calibrated
roof. The best validated TP result, 7.37 tok/s, corresponds to about 98 GB/s
and **10.6%** of the roof; the final PP short run, 3.590 tok/s, corresponds to
48 GB/s and **5.2%**. These are effective active-weight rates, not a claim
that all accesses are long sequential streams. The TP profile additionally
spent 61.4 ms/token in collectives, so eliminating communication alone caps
that measured implementation near 13.5 tok/s. Reaching 30 requires both a
substantially faster small-row-shard matvec schedule and fewer dependent
reductions; pipeline-only tuning cannot supply it for batch-one autoregressive
decode.

Compile the bounded checkpoint validator:

```bash
cc -std=c11 -Wall -Wextra -Wpedantic -I../common \
  q38fn_runner.c -o q38fn_runner -lm
./q38fn_runner ~/models/q38fn/bf16 --layers 3
```

The validator reads only safetensors headers. It does not load the 123 GB of
logical weights represented by the first three MoE layers.

`--probe-token ID` performs a real numerical checkpoint operation: it reads
one BF16 embedding row, converts it to float, and reports a stable checksum.
Adding `--probe-hc` executes the layer-0 attention hyperconnection prelude
(grouped RMSNorm, low-rank input mixing, and block injection) directly from
the safetensors weights. This remains a component probe, not model decode, so
it intentionally does not report tokens/second.

`--probe-delta` extends that path through one stateful layer-0 Gated DeltaNet
decode step. Its persistent FP32 state contains a four-sample depthwise causal
convolution history and 48 matrices of shape `[128, 128]`. The probe executes
the QKV/Z/A/B projections, recurrent delta update, sigmoid-gated RMSNorm, and
output projection, then injects the result into all four residual streams.

The runner also supports prompt preflight: it renders the Q38FN user/
assistant-thinking template and tokenizes it from `tokenizer.json` without
loading model weights:

```bash
make -C q38fn q38fn_runner
q38fn/q38fn_runner ~/models/q38fn/bf16 --layers 3 \
  --prompt 'Explain why the sky appears blue in two concise sentences.' \
  --target 'Blue light is scattered more strongly than red light.'
```

`--target` validates a teacher-forced assistant continuation, including atomic
`</think>` and `<|im_end|>` tokens; it is not reported as model-generated text.
A longer hardware-side check produced 75 input tokens and 121 coherent target
tokens with no invalid IDs. Numerical logit generation remains a separate
distributed graph stage.

For verbatim code-output validation, pass `--target-file`. The A64FX check
tokenized a 44-token programming prompt and a 152-token C11 continuation,
then compiled the exact source with `-Werror` and verified this output:

```text
0 1 1 2 3 5 8 13 21 34
```

The n-gram probe separates metadata location from payload storage:

```bash
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -I../common \
  ngram_probe.c -o ngram_probe
./ngram_probe ~/models/q38fn/bf16 /local/q38fn_probe 1000 0
```

The last argument fixes the packed split, allowing a single shard to be staged
to `/local` safely. Without it, the probe performs true hashed lookups across
all required shard files. Each logical lookup performs 16 random 320-byte
reads. Compare `payload_GB_s` and `lookups_s` between shared storage and
`/local`; the model needs asynchronous prefetch and batching if filesystem
latency approaches the compute time.

The measured sequential `/local` bandwidth was 1.119 GB/s for a 512 MiB
read, while synchronous hashed-style reads reached only 69.1 lookups/s
(about 0.35 MB/s useful payload). The gap is random-read latency, not SSD
bandwidth.

For the batched experiment, compile `ngram_async_probe.c` and pass
`iterations partition workers duplicate_period`:

```bash
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -pthread -I../common \
  ngram_async_probe.c -o ngram_async_probe
./ngram_async_probe ~/models/q38fn/bf16 /local/q38fn_probe 1000 0 4 0
```

On the single A64FX node, 1 worker achieved 59.1 logical lookups/s, 4
workers achieved 86.9/s, and 8 workers achieved 83.2/s. A synthetic 4-way
duplicate workload reached 187.1 logical lookups/s with 800 physical reads
instead of 3,200. These are 16-row batches from one staged split; real
lookups span splits, so production should use a persistent four-worker queue,
row-ID deduplication, and contiguous read coalescing.

The full 12-node design should row-shard the packed table across nodes and
prefetch/deduplicate rows per token batch. The 51B table should remain off
HBM; only the selected 5,120-byte result and a small prefetch queue belong in
the A64FX memory budget.

`ngram_hbm_probe` now exercises a four-token in-flight lookup window by
default: it
deduplicates row IDs across the window, prefetches every unique 320-byte row,
then consumes the rows. The row copy is kept no-inline so the compiler cannot
turn the benchmark into a two-lane sample; on SVE targets it uses full-row
halfword vector loads/stores. This is a probe for memory-level parallelism,
not a claim that the full runner can retain four tokens without applying its
own KV/activation memory budget.

### Resident-HBM probe

For an allocation-local resident test, compile with Fugaku's native compiler
and read the source shard directly from shared storage (do not transfer the
full model through the HTTP bridge):

```bash
make -C q38fn hbm_probe CC='fcc -Nclang'
Q38FN_LOOKUP_WINDOW=4 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  q38fn/ngram_hbm_probe ~/models/q38fn/bf16 ~/models/q38fn/bf16 100000 0 0 128
```

Set `Q38FN_LOOKUP_WINDOW=1..16` to sweep the number of independent token
lookups exposed to HBM. The default is 4; record the best window on the same
allocation because the useful random-row rate is latency- and contention-
limited, unlike the sequential stream ceiling.

`Q38FN_PREFETCH_LINES=0..5` controls how many 64-byte lines of each 320-byte
row receive a low-locality software prefetch before the batch copy. The probe
default is 3 (the first, middle, and final row lines); this knob is diagnostic
until repeated runs on the same allocation establish a stable win.
Set `Q38FN_HBM_MIN_EFF_PCT=95` to make the probe return status 3 when the
contiguous stream falls below the 95% efficiency gate; unset it for an
informational run.

The probe loads the 800,003,840-byte n-gram tensor directly from the staged
3.2 GB safetensor shard into pre-touched anonymous resident A64FX memory; it
does not allocate a tensor-sized intermediate staging buffer or perform a
second full memcpy. It then copies the resident data into four explicitly
first-touched CMG buffers, pins 12 worker cores to each CMG, and performs direct
in-memory row reads. A current native run measured `load_s=3.325` seconds,
`stream_GB_s=922.32` (99.98%), and `logical_payload_GB_s=107.23`, with the
95% gate passing. Earlier buffered-load results in the 4-node allocation were:

```text
load_s=11.890..12.443 stream_passes=32 stream_GB_s=910.47..911.05
stream_eff_pct=98.70..98.76 (relative to the 922.47 GB/s calibration)
```

On a normal-mode allocation, LLVM clang measured 741.8 GB/s (80.4%), while
native `fcc -Nclang` measured 911.17 GB/s (98.78%). Use fcc for the A64FX
performance gate. Random-row lookup is latency/contention limited and is not
directly comparable to the contiguous stream roofline.

With the current interleave-8 copy schedule, a longer native sweep measured
107.17 and 104.45 GB/s at lookup windows 4 and 8, respectively, while the
contiguous stream reached 921--922 GB/s (99.87--100.00%). The HBM-probe
default is therefore window 4; window 8 remains available for integrated
workloads whose transport needs more outstanding requests.

The default three prefetch hints are spread over row cache lines 0, 2, and 4
(rather than the old contiguous 0, 1, 2 footprint). The row-copy helper also
declares its destination, row-ID list, and resident image non-aliasing, which
lets the native compiler keep the independent load chains exposed. Two
consecutive native window-8 runs reached 95.51 and 95.73 GB/s useful
random-row payload. The probe now issues prefetches in bounded 16-row groups;
this reached 102.14 GB/s on repeat, versus 97.14 GB/s for the all-at-once
schedule, 82.30 GB/s for batch 8, and 100.49 GB/s for batch 32. The no-prefetch
control produced 40.96 GB/s with the same checksum. Keep batch 16 and three
lines as the defaults; use `Q38FN_PREFETCH_BATCH` and
`Q38FN_PREFETCH_LINES` only for controlled experiments.

The HBM probe accepts the stager's raw `/local/shard-XXX.bin` files directly
and uses byte zero as their payload origin; when those files are absent it
falls back to a copied safetensors shard and applies its header/tensor offset.
On the active A64FX node, a staged 800 MB shard loaded in 0.484 s and reached
921.04 GB/s (99.84%) on the 16-pass stream, with 106.79 GB/s logical random
payload at the default lookup settings. The corresponding resident owner
probe loaded at 1.593 GB/s from `/local` and copied 32-row responses at 9.942
GB/s.

After deleting the queued MPI job and reusing the active interactive
allocation, a fresh single-rank run measured 921.84 GB/s (99.93%) on a
32-pass HBM stream and 99.86 GB/s logical random payload. The resident owner
response path reached 4.834 GB/s for 8-row spans and 9.832 GB/s for 32-row
spans after loading the staged shard at 1.657 GB/s.

The stream benchmark binds its 48 OpenMP workers once before warmup and timing;
affinity syscalls are excluded from the HBM measurement. Before cache-line lane
alignment, this measured 910.47--911.05 GB/s (98.70--98.76%) with an unchanged
checksum.
CMG lane boundaries are rounded down to 64-byte cache-line boundaries (the
final lane absorbs the remainder), which makes the alignment contract valid for
all 48 stream workers. Two subsequent 32-pass runs measured 918.16--922.85
GB/s (99.53--100.04%).
The longer 100,000-iteration gate also measured 919.06 GB/s (99.63%) and
returned success with `Q38FN_HBM_MIN_EFF_PCT=95`.
`Q38FN_STREAM_NT=1` selects SVE non-temporal loads for comparison; a native
32-pass run reached 908.26 GB/s (98.46%), so normal loads remain the default.
The result line reports `stream_nt` to make this choice explicit.
Requested bounded batch sizes are rounded up to a whole number of copy groups,
so non-multiples such as 10 still cover every row exactly once.

On A64FX, the lookup row copier now specializes the 160-halfword row to five
full 512-bit SVE vectors, removing per-vector predicate construction while
keeping the generic variable-width fallback. Two 100,000-iteration native
runs reached 115.76--117.22 GB/s logical payload with identical checksums; the
contiguous stream remained 921.64--921.76 GB/s (99.91--99.92%). The per-thread
lookup destination is explicitly 64-byte aligned so the fast path's store
alignment assumption is valid.

The lookup probe now uses a fixed 512-entry open-addressing table for
cross-window deduplication instead of an O(N²) scan. With window 8 and unique
rows, the optimized run reached 95.73 GB/s; with duplicate period 4 it reached
827.41 GB/s logical payload and 25.86 GB/s unique payload, preserving checksum
behavior.

For the complete four-node resident experiment, submit
`a64fx/q38fn/pjsub_q38fn_ngram_4n.sh` as a PJM MPI job:

```bash
pjsub --no-check-directory \
  -x Q38FN_PIPE_ITERS=1000 a64fx/q38fn/pjsub_q38fn_ngram_4n.sh
```

The script builds the MPI-linked stager and uTofu probe, creates a fresh
rank-ordered topology, stages all 128 shards into `/local`, and runs the
resident pipeline before the allocation ends. `/local` is node-private, so
staging and measurement intentionally happen in the same job. Submit this as
a PJM-launched job; a command entered through the interactive HTTP server is
already inside `plexec` and cannot launch another `mpiexec`.
The table and row scratch slab are now worker-lifetime state with generation
tags, avoiding per-window table clearing and stack setup; a repeat unique-row
run sustained 95.51 GB/s with the same checksum.

The updated probe was also run across four ranks in the 4-node allocation with
per-rank result files. The four hosts reached 904.46--908.19 GB/s/node, or
98.05--98.45% of the same calibration:

```text
a31-4208c  906.30 GB/s  98.25%
a31-4200c  907.78 GB/s  98.41%
a31-4009c  908.19 GB/s  98.45%
a31-4001c  904.46 GB/s  98.05%
```

### TP6 decode benchmark

Run `a64fx/q38fn/pjsub_q38fn_tp6_decode.sh` as the top-level PJM MPI command.
It stages the decoder weights to each node's `/local`, generates a fresh
uTofu topology, and runs the six-rank decode with grouped indexed-Q5 enabled.
The HTTP development shell is already a `plexec` rank and must not be used to
launch this MPI command. `Q38FN_TP_NGRAM_Q5=1` enables the optional BF16
n-gram-to-Q5 conversion; it is disabled by default because the converted
22-shard n-gram buffers consume several additional GiB per rank. The default
keeps n-gram rows file-backed while decoder weights use the resident loader.

```bash
pjsub --no-check-directory \
  -x Q38FN_MAX_GEN=32 a64fx/q38fn/pjsub_q38fn_tp6_decode.sh
```

The full-row lookup path issues prefetches for every unique row before copying
the batch. It measures 2.408 GB/s useful payload for random rows (470,310
logical batches/s). A duplicate-period-4 run measures 19.137 GB/s logical
payload and 4.784 GB/s unique payload at 3.738M batches/s. Here `lookup_s`
counts complete 16-head batches; random lookup is latency-limited and is not
expected to reach the sequential HBM roofline.

The uTofu owner assigns request slots to two bounded service threads per peer
by default (`Q38FN_UTOFU_SERVICE_THREADS=2`). This allows independent
resident-HBM reads to overlap while keeping VCQ puts serialized and preserving
the owner credit limit (`Q38FN_UTOFU_OWNER_CREDITS`, default 2). Set both to 1
for a conservative baseline, or increase them together only after measuring
MRQ pressure on the target allocation. The PJM launcher unloads the conflicting
LLVM module and builds MPI targets with TCSDS `mpifcc -Nclang`, which is needed
for C11 atomics on this Fugaku environment.

The prefetch-enabled probe was also validated across four ranks. Per-host
random useful payload was 2.399--2.447 GB/s/node, while the stream measured
904.53--908.25 GB/s/node (98.06--98.46% of the calibration):

```text
a31-4009c  908.12 GB/s  98.44%  2.399 GB/s random payload
a31-4200c  908.25 GB/s  98.46%  2.444 GB/s random payload
a31-4001c  904.53 GB/s  98.06%  2.447 GB/s random payload
a31-4208c  905.49 GB/s  98.16%  2.444 GB/s random payload
```

The earlier 93.25 GB/s result was caused by heap allocation and hard-coded
affinity. The mmap/runtime-affinity path reaches 98.47--98.50% on one node and
98.05--98.45% across four nodes. The HBM lookup itself is node-local and does
not require communication for this measurement.

The response length is rounded to uTofu's 256-byte transfer unit so the tail of
each 320-byte BF16 row is transferred correctly.  All Put and TCQ-poll
operations on a VCQ are serialized by one VCQ lock because the uTofu API must
not be entered concurrently from service threads.

The portable pipeline's result materializer uses the same A64FX fast path as
the resident owner copier: when source and destination are 64-byte aligned and
the runtime SVE width is 512 bits, each 160-halfword row is copied as five full
vectors. This removes per-vector predicate construction from both the
scratch-to-result reorder path and the final caller-output path; variable SVE
widths and unaligned buffers retain the predicated fallback.

For a four-rank resident-HBM correctness gate, use one pipeline worker per
rank first:

```bash
Q38FN_HBM_PREFETCH_ROWS=4 \
Q38FN_UTOFU_REQUEST_RETRY_MS=1000 Q38FN_UTOFU_RESPONSE_RETRY_MS=1000 \
mpiexec -np 4 -of-proc "$PWD/pipe4" \
  q38fn/ngram_utofu_pipeline_probe ~/models/q38fn/bf16 tofu_topo.txt \
  4 3 1 4 resident
```

To make repeated probes reuse allocation-local shard staging, add
`Q38FN_NGRAM_STAGE_BASE=/local/$USER/q38fn-ngram-stage`; the loader creates one
directory per rank and validates each cached shard before using it.  `/local`
is allocation-local and is erased when the session ends, so this is a
within-allocation cache only.

For transport/HBM diagnostics that do not need the complete table, set
`Q38FN_NGRAM_SHARD_LIMIT=4`.  Each rank then loads only the low-numbered
rank-owned shards; this is intended for owner validation and focused routing
tests, not for a full-model lookup benchmark.

The adapter defaults to two in-flight credits per remote owner because the
allocation's MRQ depth is smaller than an unrestricted multi-worker burst.
Set `Q38FN_UTOFU_OWNER_CREDITS=1` for the conservative single-credit mode.
Credit two is the recommended production setting; credit three and four remain
controlled transport experiments because the allocation's MRQ depth is finite.
Synthetic traffic is validated through 32 workers/rank, while real nonresident
traffic is validated through four workers/rank.  Higher settings remain stress
tests and must not be interpreted as HBM results after a timeout or MRQ
overflow.

With 32 workers/rank and 100 sixteen-row requests, the corrected slot-bounded
sweep measured 0.288--0.307 GB/s/rank at credit 1, 0.360--0.416 GB/s/rank at
credit 2, and 0.300--0.453 GB/s/rank at credit 4; all 12 runs completed with
`rc=0`. A fresh no-environment real four-node run with eight workers/rank and
ten iterations also completed on every rank at 694--1,176 logical rows/s,
with identical checksums.

The deeper eight-token window also completed correctly with the default
two-credit transport: 136--198 logical rows/s/rank across the four ranks and
the expected window checksum. Its lower rate reflects 1,280 fragmented
sixteen-row transfers rather than a local-HBM bandwidth limit.

For the uTofu transport sweep, use the synthetic probe's final worker-count
argument to exercise concurrent same-owner requests:

```bash
mpiexec -np 4 q38fn/ngram_utofu_probe tofu_topo.txt 4 16 100 32
```

Compare worker counts 1, 2, 4, 8, and 16. The adapter has four bounded slots
per peer and applies a per-owner credit of two by default. Use
`Q38FN_UTOFU_OWNER_CREDITS=1` or `=3..4` for controlled experiments; each
value is bounded to the four allocated slots.

The corrected slot-bounded stress sweep was run at 32 workers/rank and 100
iterations: credit one reached 0.288--0.307 GB/s/rank, credit two
0.360--0.416 GB/s/rank, and credit four 0.300--0.453 GB/s/rank; all ranks
completed with `rc=0`. The real eight-token pipeline also passes with 32-row
capacity, identical checksum `12337086998269232710`, and 724--1,116 logical
rows/s/rank in the latest run.

The probe stops uTofu service threads before unmapping HBM, which is required
on error and normal-exit paths alike.

For the integrated resident-HBM sweep, use
`a64fx/q38fn/sweep_ngram_hbm.sh` after building `utofu_pipeline_probe`. It
compares `Q38FN_HBM_PREFETCH_ROWS=0,2,4,8` with service-thread pinning enabled
and disabled. Pinning interleaves service CPUs across the four CMGs by default;
set `Q38FN_UTOFU_CMG_COUNT` for a different allocation, or
`Q38FN_UTOFU_PIN_SERVICE=0` to disable it. Response acknowledgments default to
one Put; set `Q38FN_UTOFU_ACK_PUTS=2..8` only for reliability experiments.
