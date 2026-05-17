# Results — A64FX / Fugaku LLIO localtmp file I/O (2026-05-17)

Single compute node, interactive session with
`--llio localtmp-size=87Gi`. `$PJM_LOCALTMP=/local`, source on
`/vol0006` (Lustre-backed NFS).

Host: `a25-0101c` · freq=2000 · eco_state=0 · A64FX native (fcc).

## 1. `cp` — `$HOME/models/qwen35/9b` → `/local/llio_bench`

NFS source is the bottleneck; LLIO write throughput is not visible
here.

| file | size | time (s) | MB/s |
|---|---:|---:|---:|
| mmproj-F32.gguf           | 1.7 GiB | 46.1  | 39.6 |
| mmproj-BF16.gguf          | 0.86 GiB | 20.9 | 44.1 |
| mmproj-F16.gguf           | 0.86 GiB | 18.5 | 49.8 |
| Qwen3.5-9B-UD-Q4_K_XL.gguf| 5.6 GiB | 130.6 | 45.7 |
| Qwen3.5-9B-UD-Q8_K_XL.gguf| 12.1 GiB | 296.5 | 43.8 |
| Qwen3.5-9B-BF16.gguf      | 16.7 GiB | 622.7 | 28.8 |
| **TOTAL**                 | **37.7 GiB** | **1135.2** | **35.7** |

Total: 40.5 GB / ~19 min for the full 9B set. Stage once per job.

## 2. Read — `Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.56 GiB) from `/local`

| bench                       | cold (GB/s) | warm (GB/s) | notes |
|---|---:|---:|---|
| C `fopen` chunk=1 MiB       | 0.452 | 0.521 | |
| C `fopen` chunk=16 MiB      | 0.542 | 0.569 | larger chunk helps slightly |
| C `mmap` seq8               | 0.422 | 0.477 | contiguous 8 B XOR |
| C `mmap` page-touch         | 0.411 | 0.488 | one byte per 4 KiB |
| C `mmap` seq8 MAP_POPULATE  | 0.437 | 0.487 | +11.0 s prefault cost |
| Python `os.read` 1 MiB      | 0.031 | 0.034 | interpreter-bound |
| Python `mmap` page-touch    | 0.698 | 0.718 | only touches 1 B/page |

## 3. Read — `Qwen3.5-9B-UD-Q8_K_XL.gguf` (12.08 GiB) from `/local`

| bench                       | cold (GB/s) | warm (GB/s) |
|---|---:|---:|
| C `fopen` chunk=1 MiB       | 0.672 | 0.673 |
| C `fopen` chunk=16 MiB      | 0.697 | 0.710 |
| C `mmap` seq8               | 0.686 | 0.699 |
| C `mmap` page-touch         | 0.659 | 0.674 |
| C `mmap` seq8 MAP_POPULATE  | 0.704 | 0.693 (+18.0 s prefault) |

XOR checksums matched across C and Python implementations and across
`fopen` vs `mmap` — sanity-checks correctness of every reader.

## Takeaways

1. **Staging is mandatory.** NFS source ≈ 35–50 MB/s; LLIO localtmp
   ≈ 600–700 MB/s — a 14–19× gap. First-touch from `/vol0006` will
   dominate every prefill if you skip staging.
2. **fopen and mmap give the same bandwidth on LLIO.** Pick whichever
   the loader prefers; on A64FX the loader is rarely the bottleneck
   once data lives in `$PJM_LOCALTMP`.
3. **Cold vs warm is ~5–10 % on LLIO** — the node-local cache stays
   hot across runs within a job, so the first inference pass and
   subsequent ones see similar I/O cost.
4. **Larger `fread` chunks help slightly** (1 MiB → 16 MiB: +5 %).
   Loaders that already read in big blocks won't gain from tuning.
5. **`MAP_POPULATE` is not free.** It moves the cost to mmap-time
   (e.g. 18 s for 12 GiB) without speeding the subsequent scan; only
   use it if you specifically want to eliminate page-fault jitter
   from the steady-state path.
6. **Python is 15–20× slower than C** for byte-level work — never use
   pure Python to read weights at decode time. (Numpy `np.fromfile`
   or `np.memmap` would close the gap but isn't measured here.)

## 4. `llio_transfer` (L2 common-file cache) — Q4 file

Re-launched with `--llio localtmp-size=40Gi sharedtmp-size=40Gi`,
fresh job (no prior file access). Q4 only — Q8 (12 GiB) repeatedly
hit `LLIO 2450 Not enough disk space.` even with 40 GiB allocated
and 70 s wait after `--purge`; common-file accounting reserves more
space than the raw file size and/or releases evicted entries
asynchronously. Allocate more sharedtmp headroom if staging larger
files (~2× the file size as a working rule).

`Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.56 GiB), file path **unchanged**
after `llio_transfer --sync`:

| step | time (s) | bw |
|---|---:|---:|
| `llio_transfer --sync` (Lustre → L2 SSD) | 105.2 | 56.7 MB/s |
| `bench_fopen` chunk=16 MiB cold | 16.73 | 0.357 GB/s |
| `bench_fopen` chunk=16 MiB warm | 12.08 | 0.494 GB/s |
| `bench_mmap` seq8 cold           | 12.22 | 0.488 GB/s |
| `bench_mmap` seq8 warm           | 11.24 | 0.531 GB/s |

For apples-to-apples comparison in the same job:

| stage | source → cache | fopen 16M cold | warm | mmap seq8 cold | warm |
|---|---:|---:|---:|---:|---:|
| `cp` Q4 → `/local` (this session) | 101.7 s (58.7 MB/s) | — | — | — | — |
| `cp` Q4 → `/local` (prior session) | 130.6 s (45.7 MB/s) | 0.542 | 0.569 | 0.422 | 0.477 |
| `llio_transfer --sync` Q4         | 105.2 s (56.7 MB/s) | 0.357 | 0.494 | 0.488 | 0.531 |

**Takeaway:** `llio_transfer` and `cp → $PJM_LOCALTMP` give equivalent
steady-state read bandwidth (~0.4–0.5 GB/s). They also take about
the same time to populate the cache (both bottlenecked on the Lustre
read). The user-facing difference is path: `llio_transfer` keeps the
**original path**, so apps that hard-code `$HOME/models/...` paths
work without modification. `cp` requires the loader to read from
`/local/...`.

Per `llio_transfer(1)`: do **not** open / cp / stat the file in the
same job before `--sync` — pre-existing first-layer cache data
conflicts and `--sync` returns "Not enough disk space" or similar.

## Reproducing

```bash
# inside a job with --llio localtmp-size=87Gi
cd a64fx/llio
make
./copy_bench.sh
./run_bench.sh                              # Q4 + Python
BENCH_PY=0 ./run_bench.sh /local/llio_bench/Qwen3.5-9B-UD-Q8_K_XL.gguf
```
