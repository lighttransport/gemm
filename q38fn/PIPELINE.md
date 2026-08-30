# Q38FN n-gram request pipeline

`common/q38fn_ngram_pipeline.[ch]` is the reusable C11 path for n-gram row
requests. It maintains a bounded ring of slots and persistent workers. A batch
is deduplicated by global packed row ID using a fixed-size open-addressing
table, sorted by split/local row, and coalesced
into bounded spans, read through a source callback, and expanded back to the
original logical order on `q38fn_ngram_wait`.
The producer uses an in-place fixed-width radix sort for larger deduplicated
batches and an insertion-sort fast path for batches of up to 32 rows. The
original request array is retained for output restoration; this removes libc
comparator overhead from larger token windows without paying radix counter
passes for the common single-token 16-row request.
Each slot keeps its 512-entry dedup index with generation tags, so submitting
another ticket reuses the table without clearing it; the original request
buffer remains untouched until result materialization.
Ready slots are dispatched through a bounded index ring, avoiding an O(depth)
scan by every worker.
Slot reservation is kept under the pipeline mutex, but row-ID deduplication and
sorting run in the reserved slot's BUILDING state after releasing that mutex.
This allows workers to process earlier READY tickets while the producer builds
the next token window; the slot remains unavailable until its batch is queued.
`q38fn_ngram_wait` similarly marks a completed slot CONSUMING, releases the
state mutex while materializing the output rows, and only then returns the slot
to the free pool.
The Fugaku pipeline probe accepts `Q38FN_PIPELINE_MAX_SPAN=1..32`; the adapter
currently transfers at most 32 rows per request. On the current random
eight-token workload, raising the probe span from 16 to 32 did not reduce the
number of spans because the sorted rows are distributed across shards.
On A64FX, result materialization uses SVE row copies; other targets use the
portable `memcpy` fallback. Fully identity-mapped direct spans use one bulk
copy at completion; reordered batches retain per-row restoration.

For live token execution, `q38fn_ngram_submit_token` applies the checkpoint's
16-head hash function and submits the complete batch. For deeper memory-level
parallelism, `q38fn_ngram_submit_token_window` hashes up to 16 token contexts
into one ticket, deduplicating rows across the whole window and returning
results in token/head order. The caller can also retain multiple tickets and
wait in token order, overlapping token hashing and PLE preparation with worker
and transport queues. Fully contiguous cache-miss spans use a direct read into
the result slab, avoiding the scratch-to-result copy; gapped or partially
cached spans retain the bounded scratch/reorder path. When the optional cache
is disabled, its mutex is also bypassed so the default remote pipeline has no
cache-lock overhead.

When enabled, the cache uses 64 striped locks keyed by set rather than one
global lock, allowing unrelated remote-row sets to be probed and filled by
different workers concurrently.

The source callback is transport-neutral. Local safetensors use
`q38fn_ngram_fd_read_span`; a 4-node deployment assigns 32 splits per owner
and supplies a thread-safe uTofu request/response callback for remote splits.
That callback should use registered per-peer slabs, sequence numbers, credits,
and completion polling. MPI is needed only for startup topology exchange.

`common/q38fn_ngram_transport.h` supplies the adapter wrapper. Its
`fetch_span` callback is where the Fugaku implementation submits a registered
request descriptor, issues a uTofu Put to the owner, polls the TCQ, and copies
the completed response into the supplied destination. The wrapper keeps
`utofu.h` out of the portable pipeline while persistent workers provide
bounded parallel outstanding operations.

The owner runs one response-service thread per peer, allowing independent HBM
copies to overlap while Put submission remains serialized per VCQ.

`a64fx/q38fn/ngram_utofu_source.[ch]` is the concrete Fugaku adapter. It uses
the same coordinate/VCQ/STADD convention as `a64fx/utofu-tests`, registers a
small request/response region on independent client and service VCQs, and runs
one service thread per peer. Each peer has four bounded request/response
slots, allowing bounded transfers to the same owner; every slot is padded to
an A64FX cache-line multiple so independent remote writes do not share a
line. `Q38FN_UTOFU_OWNER_CREDITS` defaults to two transactions per remote owner;
set it to 1 for conservative mode, while values 3--4 enable controlled
multi-slot experiments. Service threads are pinned to distinct allowed CPUs and interleaved
across four CMGs by default; set `Q38FN_UTOFU_CMG_COUNT` for a different
affinity topology or `Q38FN_UTOFU_PIN_SERVICE=0` to disable pinning. The owner
service keeps response acknowledgments pending while it scans; it remains in
user-space polling and yields only once per 64 scans. This avoids scheduler
overhead under load while retaining cooperative behavior when idle. It keeps
responses pending while scanning other slots, retransmitting periodically
instead of blocking the whole rank on one response. Pending responses expire
after the same 10-second bound used by the requester, allowing abandoned slots
to be reclaimed. The client sends one acknowledgment Put by default; set
`Q38FN_UTOFU_ACK_PUTS=2..8` only when testing a less reliable transport path.
Request Puts request and drain completion notices; response and acknowledgment
Puts omit completion notices and use nonblocking submission, polling only on
TCQ backpressure. Requests and
acknowledgments use the client VCQ while response Puts use the service VCQ,
preventing shared-TCQ completion
notices from being consumed by the wrong progress path. The client response
poller also yields only once per 64 polls. A rank owns
`shard % nranks`; pipeline spans are fragmented into reliable 32-row
(10,240-byte) uTofu transfers, and the
owner reads its resident shard through the caller-supplied
`q38fn_utofu_read_fn` before returning one response Put. Build-check it with:

```bash
module load lang/tcsds-1.2.43
make -C q38fn CC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/fccpx utofu_source
make -C q38fn CC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/fccpx utofu_probe
mpiexec -np 4 q38fn/ngram_utofu_probe tofu_topo.txt 4 256 100 8
```

For multi-rank Fugaku runs, build and launch the MPI-linked variant instead;
the MPI bootstrap is required for `mpiexec` to create one process per rank:

```bash
make -C q38fn MPICC=mpiclang utofu_probe_mpi
mpiexec -np 4 q38fn/ngram_utofu_probe_mpi tofu_topo.txt 4 16 1000 8
```

Use a shared filesystem for `Q38FN_RESULT_PREFIX` when collecting rank files;
`/local` is private to each node. The four-rank native run completed all 1,000
requests with the default implicit acknowledgment protocol.

For an end-to-end persistent-pipeline measurement (rather than raw span
latency), build and run `utofu_pipeline_probe`:

```bash
make -C q38fn CC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/fccpx utofu_pipeline_probe
mpiexec -np 4 q38fn/ngram_utofu_pipeline_probe \
  ~/models/q38fn/bf16 tofu_topo.txt 4 1000 8 8 resident
```

The arguments select iterations, pipeline workers, token-window size, and
optional HBM residency. The probe keeps four times the worker count of token
windows in flight and validates the first returned row against safetensors.

For real checkpoint payload validation, build `utofu_real_probe` and pass the
shared model directory. Each rank reads the next rank's packed shard through
uTofu and compares it with a direct safetensors slice:

```bash
make -C q38fn CC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/fccpx utofu_real_probe
mpiexec -np 4 q38fn/ngram_utofu_real_probe \
  ~/models/q38fn/bf16 tofu_topo.txt 4 16 100 8
```

Append `resident` to load the 32 rank-owned shards into anonymous memory
before serving requests. This is the performance configuration; it requires
about 25.6 GiB per rank and should be launched with the job's 48-core NUMA
affinity settings. The loader drops each source-file chunk from page cache
immediately after copying it into the resident allocation, avoiding a second
25.6-GiB cache footprint during startup. It also requests transparent
hugepages for the repeatedly scanned resident image; set
`Q38FN_HBM_NO_HUGEPAGE=1` when the allocation policy requires base pages:

```bash
mpiexec -np 4 q38fn/ngram_utofu_real_probe \
  ~/models/q38fn/bf16 tofu_topo.txt 4 16 100 resident 8
```

Set `Q38FN_NGRAM_STAGE_BASE=/local/$USER/q38fn-ngram-rank` to cache
each owner’s 32 payloads on node-local SSD. Staging is resumable by exact-size
file; subsequent resident runs read the local SSD and avoid the slow shared
filesystem path. Use a distinct stage directory per rank/job so two owners do
not truncate each other’s files; the adapter appends the rank to the base.

When staged files are available, resident loading uses parallel positional reads.
`Q38FN_HBM_LOAD_THREADS` controls the number of loader threads (1--48, default
4); each thread reads independent 16-MiB chunks, so it does not share a file
offset or require a loader mutex. On one Fugaku node, the 800-MiB shard probe
took 15.83 s with one thread, 0.98 s with four, and 1.33 s with eight (the
effective rates were 0.06, 1.89, and 1.01 GB/s respectively). Four threads
was retained as the default. The shared-filesystem fallback remains serialized
to avoid multiplying metadata and storage pressure before staging completes.

Owner response copies use the same small lookahead only for spans up to 16 rows;
32-row responses disable the software hints because the A64FX hardware stream
prefetcher is faster for that already-contiguous transfer. The fixed five-vector
SVE copier has measured 8.7--10.83 GB/s for 32-row copies and 1.03--5.13 GB/s
for eight-row copies across native runs, with identical checksums. These
owner-response figures are separate from the pure HBM stream ceiling (about
99.9% of the A64FX peak); they include response-copy and service-path effects.

To populate the cache without allocating HBM, build `ngram_stage` and launch:

```bash
make -C q38fn CC=fcc ngram_stage
mpiexec -np 4 -of-proc q38fn/stage.log q38fn/ngram_stage \
  ~/models/q38fn/bf16 /local/$USER/q38fn-ngram-rank 4
```

For a multi-node allocation, use the MPI-linked stager so every node receives
its own rank-owned files (and use `Q38FN_NGRAM_STAGE_LIMIT=4` for the focused
four-shard smoke test):

```bash
make -C q38fn MPICC=mpiclang ngram_stage_mpi
mpiexec -np 4 q38fn/ngram_stage_mpi \
  ~/models/q38fn/bf16 /local/$USER/q38fn-ngram-rank 4
```

Run that `mpiexec` as the allocation's PJM-launched command (for example from
the body of a dedicated `pjsub` script).  Do not invoke it from a
bash-over-HTTP server shell: that shell is already a PJM/`plexec` rank, and
Fugaku rejects nested launches with `PLE 0008 plexec must be started
sequentially`.  The same restriction applies to the four-rank pipeline probe;
the HTTP bridge is suitable for single-rank probes and control, while the
multi-rank executable must be started by PJM.

The MPI stager obtains its rank from `MPI_Comm_rank()` when Fugaku does not
export a rank environment variable. `/local` remains node-private; the base
directory is suffixed with the rank on each node.

The adapter expects a rank-ordered `tofu_topo.txt` in the launch directory.
Regenerate it inside every allocation with
`mpiexec -np N a64fx/utofu-tests/tofu_topo_helper`; a topology from another
allocation can parse successfully but route Puts incorrectly. Fujitsu MPI may
export no `OMPI_COMM_WORLD_RANK`/`PMI_RANK`; the native pipeline probe then maps
rank from `utofu_query_my_coords()` against this topology.
Use `q38fn_ngram_set_cache_remote_only(p, rank)` when local shards are HBM
resident, keeping the small row cache for uTofu traffic. The adapter is a
transport probe boundary; `q38fn_runner` still needs the model HBM shard
loader and forward path wired to it.

An earlier four-node A64FX allocation passed the synthetic probe on all four
ranks for 100 remote requests per rank (16 rows/request):

```text
payload_GB_s = 0.140--0.141 per rank
requests_s   = 27,301--27,515 per rank
```

The final worker-count argument starts concurrent requester threads per rank;
the adapter has four per-peer slots and a configurable one-to-four owner-credit
limit. Sweep worker counts from 1 to 16/32 to find the
uTofu saturation point. The probe uses Open MPI's `-of-proc` option because Fujitsu MPI does not
reliably forward rank stdout through the HTTP shell. Long runs should retain
the adapter's 10-second response timeout and inspect per-rank files; startup
ordering and uTofu transfer-size limits are allocation-dependent.

The default protocol uses an implicit acknowledgment: publishing the next
request in a slot acknowledges the previous response, eliminating one network
Put per request. A four-rank, 1,000-request native A/B measured 0.630--0.659
GB/s/rank with implicit acknowledgment versus 0.512--0.589 GB/s/rank with the
explicit acknowledgment protocol; all runs completed with `rc=0`. Set
`Q38FN_UTOFU_IMPLICIT_ACK=0` to retain explicit acknowledgments for compatibility
or fault-injection tests.

The response payload is cache-line aligned within every registered slot. The
header padding is included through `offsetof(data)` when calculating the Put
length, so the 64-byte alignment change does not truncate the final row or
increase the rounded transfer size for the 1--32 row protocol.
The client receive copier uses the same five-vector SVE specialization when
both endpoints are 64-byte aligned and retains a predicate-loop fallback for
arbitrary public destinations.
The request lifetime is configurable with `Q38FN_UTOFU_TIMEOUT_MS`; keep the
default 10,000 ms for resident-HBM runs, but use 30,000 ms when deliberately
testing nonresident shared-FS reads, whose service thread can be delayed by
filesystem contention. A four-rank synthetic run with eight workers and
`Q38FN_UTOFU_TIMEOUT_MS=30000` completed all ranks with `rc=0`.

For reliable per-rank collection, set `Q38FN_RESULT_PREFIX` to a writable
shared path. The pipeline probe writes one file named `<prefix>.<rank>` after
each run, independent of MPI stdout forwarding. A four-rank nonresident
validation with eight workers and a four-token window produced four files,
all `rc=0`, identical checksum `12337086998269232710`, and 122.32--133.14
logical rows/s. Use these files for comparisons between request-preparation
and transport revisions.

Those measurements predate the bounded-slot, nonblocking-acknowledgment
transport refactor. Re-run `utofu_probe` and `utofu_real_probe` on a fresh
allocation before comparing throughput; the portable tests and target-ISA
syntax checks do not validate uTofu runtime routing or completion ordering.

Build the deep-outstanding local benchmark:

```bash
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -pthread -Icommon \
  q38fn/ngram_pipeline_probe.c common/q38fn_ngram_pipeline.c \
  -o q38fn/ngram_pipeline_probe
q38fn/ngram_pipeline_probe ~/models/q38fn/bf16 /local/q38fn \
  10000 0 4 64 32 2 0
```

Arguments after storage are `iterations partition workers queue_depth
max_span_rows max_gap_rows duplicate_period`. Results include logical and
unique rows/s, useful and physical GB/s, deduplication, direct versus
scratch/reorder span counts, average span size, and wait time.
`duplicate_period=4` isolates deduplication; vary span and gap to measure
contiguous coalescing.

The runner integration should hash the next token window, submit its 16 rows,
continue the current layer while the ticket is pending, and wait only when PLE
consumes a missing result. Queue depth and result slabs remain bounded.

`Q38FN_PIPELINE_PIN_WORKERS=1` pins pipeline workers round-robin across the
allocation's allowed CPUs for locality experiments. It is disabled by default:
on the current remote uTofu workload, pinning eight workers/rank reduced
throughput to roughly 84--91 rows/s/rank because it competed with transport
progress threads.

The resident-HBM probe models the same overlap with an eight-token window: it
deduplicates row IDs across all 8 x 16 heads, prefetches every unique 320-byte
row before copying payloads, and reports both logical-token and unique-row
throughput. The window exposes independent HBM misses without requiring a
large scratch allocation.

### A64FX measurements

On the 4-node interactive allocation's A64FX local storage, the persistent
pipeline measured 2,977 random logical batches/s with four workers and depth
64. Increasing to eight workers and depth 256 reached 4,089 batches/s; 16
workers reached 4,078 batches/s, so eight workers is the current local-read
saturation point. A duplicate-period-4 workload reached 712,044 logical
batches/s at 178,011 unique batches/s, with 120,000 rows removed by dedup.

These filesystem numbers are separate from the resident-HBM stream result:
the optimized direct-read mmap/affinity probe measured 921.79--922.32
GB/s/node, or 99.93--99.98% of the 922.47 GB/s calibration ceiling, while
eliminating its tensor-sized intermediate buffer.

The HBM lookup probe copies complete 320-byte rows and prefetches all unique
rows before consuming the batch. Its default lookup window is four and its
three prefetch hints target row
cache lines 0, 2, and 4. Bounded asynchronous prefetching now issues those
hints in 16-row groups; repeated native window-4 runs reached 104.49--107.23
GB/s useful payload. Random rows remain latency-bound, while the 95%
hardware efficiency target applies to the contiguous/coalesced stream path.

The resident-shard uTofu owner retains the SVE streaming copy for the response
and interleaves a four-row-ahead low-locality prefetch with short (≤16-row)
spans. Full 32-row responses disable the software hints because the A64FX
hardware stream prefetcher is faster for that contiguous transfer. Set
`Q38FN_HBM_PREFETCH_ROWS=0,2,4,8` to sweep the short-span look-ahead; the
default is 4 and values above 64 are ignored.

The owner path was compiled with `fcc -Nclang` and passed bounded resident
real-payload checks for both one-row and 32-row responses on all four ranks
(`rc=0`, with direct safetensors comparison on the first response). The
four-rank test intentionally limits residency to one shard per rank so it can
validate response copying without staging the complete 102.4-GB table.

For a reproducible native sweep of prefetch and CPU pinning, build the pipeline
probe and run `a64fx/q38fn/sweep_ngram_hbm.sh MODEL_DIR TOPOLOGY RANKS`; it
tests prefetch distances 0/2/4/8 with `Q38FN_UTOFU_PIN_SERVICE` enabled and
disabled. Keep ranks, workers, window, and iterations fixed when comparing
results.

The prefetch-enabled probe was validated across four ranks: random useful
payload was 2.399--2.447 GB/s/node, and contiguous stream efficiency was
98.06--98.46% of the 922.47 GB/s calibration on hosts a31-4009c,
a31-4200c, a31-4001c, and a31-4208c.

The optional bounded cross-batch cache was measured with eight workers and a
256-slot queue on the same A64FX node. For duplicate-period 4, an 8,192-row
four-way cache served 39,996 rows from cache with only 4 physical misses.
Power-of-two cache geometries use shift/mask set and way selection to avoid
division in the lookup path; arbitrary cache sizes retain the general modulo
path.
Throughput was 1.476M logical batches/s versus 1.602M without the cache on
this page-cached local workload; the extra tag probes are not free. The cache
is intended primarily to remove remote/uTofu operations, where avoiding a
transport miss matters more than this local page-cache overhead.

For a distributed run, call `q38fn_ngram_set_cache_remote_only(p, rank)` after
initialization and set each source's `owner` field to its node rank. Local rows
then bypass the software cache, while rows fetched through remote source
callbacks can be retained in the bounded cache.
