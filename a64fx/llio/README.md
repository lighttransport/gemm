# LLIO file-I/O benchmarks (A64FX / Fugaku)

Measure the cost of:

1. `cp` from shared storage (`$HOME/models/qwen35/9b` on `/vol0006`) into the
   per-node LLIO local cache `$PJM_LOCALTMP` (~87 GiB).
2. Reading the copied files via `fopen()`/`fread()` and `mmap()` from C
   (fcc native) and Python.

The 9B model set lives on `/vol0006` (global Lustre via NFS-like mount) and
is slow to first-touch (smoke test: ~50 MB/s on a cold mmproj read).
LLIO localtmp is node-local SSD-backed cache and should be hundreds of
MB/s for cold and multi-GB/s for warm. These benchmarks let you put
numbers on that gap so the VLM/LLM loaders can make sensible decisions
about staging weights before inference.

## Files

| file | what |
|------|------|
| `bench_fopen.c` | C: `fopen` + chunked `fread` with XOR checksum |
| `bench_mmap.c`  | C: `mmap` + sequential/per-page touch with XOR checksum |
| `bench_fopen.py`| Python equivalent (`os.read` or buffered `open`) |
| `bench_mmap.py` | Python equivalent (`mmap` module) |
| `Makefile`      | fcc native (default) and fccpx cross builds |
| `copy_bench.sh` | times `cp` of each gguf file into `$PJM_LOCALTMP/llio_bench/` |
| `run_bench.sh`  | runs all four read benches against a single file |
| `pjsub_run.sh`  | pjsub wrapper that requests 87 GiB LLIO and runs everything |
| `llio_transfer_bench.sh` | times `llio_transfer --sync` (L2 common-file cache) and reads from the *original* Lustre path |
| `probe_sharedtmp_limit.sh` | sweeps synthetic file sizes to find the largest one `llio_transfer --sync` accepts for the current `sharedtmp-size` |

## Summary — pick a staging mechanism

| | `cp → $PJM_LOCALTMP` | `llio_transfer --sync` |
|---|---|---|
| max single-file size | up to `localtmp-size` (87 GiB cap) | ~`sharedtmp-size` / 6 (measured: 7 GiB in 40 GiB) |
| extra alloc per file | none | ~6× the file size |
| app-visible path | changes to `/local/...` | unchanged (`$HOME/...`) |
| read bandwidth | ~0.4–0.5 GB/s | ~0.4–0.5 GB/s (equivalent) |

With the default 40 GiB sharedtmp, `cp` works for every gguf in the
9B set; `llio_transfer` only handles files ≤ ~7 GiB. To stage larger
files via `llio_transfer`, request `--llio sharedtmp-size=<~6×>`
(e.g. ≥96 GiB for the 17 GiB BF16 weight). See `RESULTS.md` for the
full numbers.

## Quick start (already inside a job with LLIO localtmp)

This directory lives on the shared filesystem; build native and run
in-place:

```bash
cd a64fx/llio
make                                  # fcc native
./copy_bench.sh                       # stage gguf files into $PJM_LOCALTMP
./run_bench.sh                        # bench the Q4 file by default
./run_bench.sh "$PJM_LOCALTMP/llio_bench/Qwen3.5-9B-BF16.gguf"
```

## Quick start (from login/cross-compile node)

```bash
pjsub a64fx/llio/pjsub_run.sh
```

`pjsub_run.sh` requests:

```
#PJM -L "rscgrp=small,node=1,elapse=00:60:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --llio localtmp-size=87Gi
#PJM --llio sharedtmp-size=80Gi
```

`localtmp-size=87Gi` is the documented per-node maximum; trim it if the
job manager rejects the request. `sharedtmp-size` is **required** for
the `llio_transfer` step — common files are cached in the shared-tmp
allocation; without it `llio_transfer --sync` fails with "Not enough
disk space".

## What each bench measures

### `bench_fopen <file> [chunk=1MiB] [passes=2]`

Per pass: open, loop `fread(buf, 1, chunk, fp)` to EOF, XOR all bytes,
close. Prints per-pass time and bandwidth (`bytes_read / dt`). Pass 0 is
"cold", subsequent passes are "warm" (page cache hot). `setvbuf(_IONBF)`
disables stdio's own buffering so each call hits `read(2)` directly.

### `bench_mmap <file> [mode=seq|seq8|page] [passes=2] [populate=0|1]`

Per pass: walk the mapped region according to `mode`:

- `seq8` — contiguous 8-byte XOR (compiler will vectorize on A64FX)
- `seq`  — one byte per 64-byte cache line
- `page` — one byte per 4 KiB page (isolates fault cost from read cost)

`populate=1` adds `MAP_POPULATE` so the kernel prefaults the whole file
at `mmap()` time (lets you separate fault cost from scan cost).
`madvise(MADV_SEQUENTIAL)` is always set.

### `llio_transfer_bench.sh [SRC_DIR] [file1 file2 ...]`

Wraps Fugaku's `llio_transfer` tool, which distributes a file from
the global filesystem (vol0006/Lustre) into the I/O node's SSD
("L2 common-file cache"). After `llio_transfer --sync`, the file's
**original path** still works — `fopen("$HOME/models/.../foo.gguf")`
reads now hit the SSD cache instead of crossing the network, so apps
don't have to be modified to use a staged path. Contrast with the
`cp → $PJM_LOCALTMP` approach in `copy_bench.sh`, where the path
changes.

Two preconditions, both per `llio_transfer(1)`:
- **Do not open / cp / stat the file in the job before `--sync`** —
  pre-existing page-cache data conflicts with the common-file cache
  and the transfer fails.
- **Allocate sharedtmp.** Common-file cache space is drawn from
  `PJM_LLIO_SHAREDTMP_SIZE`. With `sharedtmp-size=0Gi` (the value
  used in earlier runs) `--sync` returns `[ERR.] LLIO 2450 Not
  enough disk space.` immediately.

The script auto-`--purge`s before and after, times `--sync`, and then
runs `bench_fopen`/`bench_mmap` against the original path.

### Python variants

Same XOR checksums so a cross-language sanity check is possible: a cold
C `fopen` and a cold Python `mmap` of the same file should produce the
same `xor=...` value. Python throughput is much lower (per-byte work
runs in the interpreter), but the *I/O* component is comparable.

## Reading the output

Each pass prints:

```
pass 0  cold  read=5.555 GiB  time=12.345 s  bw=0.483 GB/s (0.450 GiB/s)  xor=...
```

`cold` = pass 0 (page cache may be empty after copy); `warm` = subsequent
pass (page cache hot, dominated by memory bandwidth + interpreter
overhead for Python). For LLIO localtmp on Fugaku you should see cold
hundreds of MB/s and warm in the multi-GB/s range; for NFS-mounted
`/vol0006` cold is ~50 MB/s.

## Caveats

- `cp` time in `copy_bench.sh` includes a `sync` after each file so the
  number reflects "data on destination" not just "data in page cache".
- `bench_*` does not call `posix_fadvise(DONTNEED)` between passes — to
  re-measure cold cost you must re-run the program (a fresh `open()`
  doesn't drop pages) or re-stage the file.
- On the Fugaku interactive shell `PJM_LLIO_LOCALTMP_SIZE=0` means no
  localtmp was allocated; `copy_bench.sh` falls back to `/tmp` with a
  warning. Use `pjsub_run.sh` to get the real measurement.
