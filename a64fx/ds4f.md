# DeepSeek-V4-Flash (DS4F) on 11× A64FX (Fugaku) — build & run repro

EP-only inference harness for DeepSeek-V4-Flash (284B total / 13B activated MoE,
43 MoE layers, hidden 4096, vocab 129280). 256 routed experts are EP-sharded
~23–24/node; dense (attn / shared expert / router / embed / head) is replicated and
computed redundantly per node. Weights stay quantized in HBM with on-demand dequant
(dense FP8 e4m3fn, experts MXFP4) to fit ~25 GB usable HBM2/node.

This file is the **operational repro** for the real-weight Tier-B1 run. The science /
status lives in the auto-memory `project_ds4f_flash_ep_harness`; the design rationale
in `~/.claude/plans/plan-deepseek-v4-flash-iterative-prism.md`.

## TL;DR

```sh
# inside a live 12-node alloc (shape 2x3x2; 1 node reserved for agent/claude control),
# from a64fx/llm:
./run_ds4f_stage_11n.sh                                  # 1. shard weights -> each node's /local (~22 GB/node)
DS4F_REAL=1 DS4F_EXACT=1 ./run_ds4f_11n.sh                # 2. exact Tier-B1 forward on the 11 EP nodes
DS4F_REAL=1 DS4F_TIERB2=1 ./run_ds4f_11n.sh              # 2b. Tier-B2 compressor/indexer (reuses blobs)
DS4F_REAL=1 DS4F_FP8_BF16=1 ./run_ds4f_11n.sh           # 2c. bf16-pv dense promote (fast decode)
./run_ds4f_longctx_11n.sh                                # 2d. long-ctx Tier-B2 bench (O(topk) payoff)
```

All launchers default to `NP=11`, `EXCLUDE=0,0,0` (drops the agent/claude-colocated
node — the 12th node is reserved for agent control and would OOM under the ~22 GB load),
and regenerate `vcoord_ds4f.txt` + the Tofu topology for the *current* alloc. Steps 2b–2d
reuse the Step-1 blobs (no re-stage). Last Tier-B1 validation (job 49092345, 2026-06-05):
`rc=0`, NaNs=0, argmax=122293, prefill 10.44 / decode 10.16 tok/s, arena 22.18 /
RSS 21.81 GB/node. Per-config numbers are in Steps 2–2d.

## Files (branch `ds4f`)

| File | Role |
|---|---|
| `common/ds4f.h` | self-contained model: config / alloc / forward / loaders / dequant matvecs |
| `a64fx/llm/ds4f_runner.c` | single-node driver (no uTofu) — validation vehicle |
| `a64fx/llm/ds4f_ep_runner.c` | 11-node uTofu EP driver (links `-ltofucom`) |
| `a64fx/llm/ds4f_stage.c` | sharded safetensors → per-node `/local/ds4f/rank<rr>.blob` stager |
| `a64fx/llm/run_ds4f_stage_11n.sh` | stage launcher (mpiexec, PMIX_RANK self-ID) |
| `a64fx/llm/run_ds4f_11n.sh` | run launcher (vcoord + topo + mpiexec) |
| `a64fx/llm/run_ds4f_longctx_11n.sh` | long-ctx Tier-B2 bench wrapper (ctx-warm + sentinel) |
| `a64fx/utofu-tests/tofu_topo_helper` | MPI program that writes `tofu_topo.txt` (rank→coords) |

Build is native `fcc`/`FCC` (NOT `fccpx`/`FCCpx`); binaries run directly, **no `pjsub`**
for the run itself (this is a native A64FX node). The launchers rebuild the runner via
`make ds4f_ep_runner CC=fcc OPENMP=1`.

## Prerequisites

- A live allocation: `pjsub -L "node=12,..."` shape **2×3×2** (12 nodes; 1 is the
  claude/login node, excluded → 11 EP nodes). Confirm with `env | grep PJM_MPI_SHAPE`
  (`X=2 Y=3 Z=2`) and `echo $PJM_SUBJOBID`.
- Real weights at `$HOME/models/ds4f` (config.json + 46 safetensors shards).
- PATH must **PREPEND** `/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin` (the
  launchers do this) so `mpiclang` (for the topo helper) and `fcc` both stay resolvable.

## Step 1 — stage weights to node-local `/local`

```sh
cd a64fx/llm
DS4F_STAGE_FLUSH_GB=2 ./run_ds4f_stage_11n.sh
```

Each rank self-identifies via `PMIX_RANK` (own_mpi exports it to non-MPI binaries) and
streams **only its owned tensors** (dense → all ranks; expert `e` → rank `e%11`) file→file
from the read-only safetensors mmap into `/local/ds4f/rank<rr>.blob` + a text manifest.
RSS stays ~tens of MB (it is NOT loading weights). `DS4F_STAGE_FLUSH_GB=2` bounds the
HBM `/local` dirty-page cache (fdatasync + `fadvise(DONTNEED)` every 2 GB) — without it
the blob's dirty pages pile up in HBM (~7 GB) and OOM-segfault the stage.

Done signal: 11× `a64fx/llm/ds4f_stage_rank*.txt` on the shared FS. ~1270 s wall
(shared-FS read-bound, ~22 GB/node). The launcher prints `OK: all 11 ranks staged`.

## Step 2 — run the exact Tier-B1 forward

```sh
DS4F_REAL=1 DS4F_EXACT=1 ./run_ds4f_11n.sh
```

The launcher: regenerates `vcoord_ds4f.txt` (excludes relative `(0,0,0)`), runs
`tofu_topo_helper` → `a64fx/llm/tofu_topo.txt` for **this** alloc, builds the runner,
then `mpiexec -np 11 -vcoordfile vcoord_ds4f.txt build/ds4f_ep_runner`. Each rank mmaps
its `/local` blob, copies tensors into a ~22 GB arena (FP8 e4m3fn dense + MXFP4 expert
nibble-repack), and runs prefill 8 + decode 16. Output: per-rank
`ds4f_ep_{perf,load}_rank*.txt` + `ds4f_ep_rank00.txt` summary.

`DS4F_REAL=1` pins dense to FP8 (real e4m3fn bytes). `DS4F_EXACT=1` swaps the dense
stand-ins for the exact DeepSeek math (RoPE/YaRN, per-head q-norm, MQA sliding-window +
sink attn, grouped low-rank o-proj, sqrtsoftplus gate, swiglu clamp); `exact=0` is
byte-identical to the stand-in path.

## Step 2b — Tier-B2 (compressor/indexer) real-weight run

```sh
DS4F_REAL=1 DS4F_TIERB2=1 ./run_ds4f_11n.sh      # implies EXACT; reuses the SAME blobs
```

No re-stage needed: the compressor/indexer tensors (`layers.L.attn.compressor.*`,
`.indexer.*`) are classified `CLS_DENSE` by the stager, so they are already in every
rank's `/local` blob. `ds4f_load_real` loads them by name and widens to f32 at load
time (BF16→f32 for the compressor/`weights_proj`/indexer-compressor weights, FP8
e4m3fn + E8M0 128² dequant for `indexer.wq_b`, F32 `ape` + BF16 `norm` direct). The
forward then runs the stateful 21-CSA + 20-HCA compressor/indexer path.

Validated (job 49092345, 2026-06-05): `rc=0`, 11/11, **NaNs=0**, argmax=98198
(lockstep), RSS 21.82 GB/node. **Perf:** decode 0.56 / prefill 0.60 tok/s — attention
is 94.8 % because the f32 compressor/indexer matvecs are pure overhead at short ctx
(prefill 8 + decode 16 = 24 positions). Tier-B2 only amortizes past ~`index_topk=512`
context (its point is O(topk) long-context attention); the kernels are f32 reference
implementations, not yet SVE/quant-optimized (the speed follow-up).

## Step 2c — real-weight bf16-pv dense promote (fast decode)

```sh
DS4F_REAL=1 DS4F_FP8_BF16=1 ./run_ds4f_11n.sh    # predequant replicated dense FP8 -> bf16-pv
```

No re-stage needed (same FP8 blobs). At load, the 9 replicated dense tensors (`head`,
attn `wq_a/wq_b/wkv/wo_a/wo_b`, router `gate`, shared-expert `sh_w1/w3/w2`) are requant'd
FP8→bf16 in the pair-interleaved `BF16_PV` layout via `ds4f_load_dense`; routed experts
stay MXFP4. **FP8→bf16 is lossless** (e4m3's 3-bit mantissa × 2^k block scale fits bf16's
7 mantissa bits with no rounding) ⇒ a pure speed win, no accuracy cost. The default
(no `DS4F_FP8_BF16`) takes the same-dtype fast path and is byte-identical to the FP8 run.
Set `DS4F_BF16_PV=0` to force plain (non-pv) bf16. Costs ~+6 GB (FP8 ~12 GB dense → bf16
~18 GB); safe at the harness's short ctx, watch the budget past ~8 K (the +6 GB pushes a
128 K KV cache over HBM).

Validated (reused `/local` blobs, EXCLUDE=0,0,0, prefill 8 + decode 16): `rc=0`, 11/11,
**NaNs=0**, argmax=60574 (lockstep), RSS **27.50 GB/node** (arena 26.02 GB, fits < 32 GB
HBM2). **Perf:** decode **16.73 tok/s** (+65 % over FP8's 10.16) / prefill **17.27**
(+63 % over 10.60) — near the synthetic bf16-pv 17.49 ceiling. The profile flips to
matvec-bound (o_proj 29 % + qkv 19 % + shared 11 %); comm ~20 % is the EP all-reduce
Amdahl floor (same as synthetic). MXFP4-dense is NOT promotable from real FP8 (lossy) —
`ds4f_load_dense` aborts on an MXFP4 dense dest.

## Step 2d — long-context Tier-B2 run (the O(topk) payoff)

```sh
./run_ds4f_longctx_11n.sh                       # ctx=16384, combined bf16-pv + Tier-B2
DS4F_CTX_WARM=8192 ./run_ds4f_longctx_11n.sh    # a lower curve point
```

The wrapper sets `DS4F_CTX_WARM=N`: after the (short) prefill, `ds4f_warm_kv(m,N)` +
`ds4f_warm_tb2(m,N)` fill the synthetic KV + compressed (`cmp_kv` / `idx_kv`) caches to
`N` positions, then decode runs from `pos=N`. Both fills are **deterministic** (fixed
per-layer splitmix64 seeds) and **rank-independent** (local caches only, no all-reduce)
⇒ lockstep is preserved across all 11 ranks. This measures decode/attn cost at long
context **without** the O(ctx²) cost of a real prefill of that length. `DS4F_MAXPOS`
auto-sizes to `CTX_WARM + MAXGEN + 256` (the runner asserts `ctx_warm + maxgen ≤ max_pos`).

Validated (combined `DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_TIERB2=1`, reused blobs,
EXCLUDE=0,0,0, prefill 8 + decode 16, all rc=0 / 11/11 / NaNs=0 / argmax lockstep):

| ctx | attn ms | tb2prep ms | qkv | o_proj | decode tok/s | RSS GB | argmax |
|---|---|---|---|---|---|---|---|
| 24 (baseline) | 4.11 | 16.86 | 18.71 | 29.48 | 10.10 | 27.51 | 98198 |
| 8192 | 88.97 | 113.96 | 18.76 | 28.58 | 3.56 | 28.23 | 16 |
| 16384 | 95.96 | 159.68 | 18.85 | 28.56 | 3.00 | 28.95 | 10973 |
| 32768 | — OOM (`rc=137`) — | | | | | | |

- **O(topk) payoff (real weights):** doubling ctx 8192→16384 grew `attn` only **+7.9 %**.
  The CSA compressed-attn term is capped at `index_topk=512` in both (T = ctx/4 = 2048 →
  4096, both > 512) ⇒ CSA attn flat; the +7.9 % is the HCA(128) portion (ctx/128). Had
  attn scaled with T (no cap), ctx=16384 would be ~2800 ms (4.11 × 680) — **the cap saves
  ~29×**.
- **The new long-ctx ceiling is the index *scan*, not attention.** `tb2prep` grew
  **+40 %** per 2× ctx (≈linear) and is now **41–48 % of decode** — `index_score` scans
  all T = ctx/4 compressed `idx_kv` positions to rank the top-512. It reads *activations*,
  so it is immune to both the topk cap and the bf16 weight-quant (Step 2 follow-up). The
  weight-bound qkv/o_proj are perfectly ctx-flat. Decode dropped only −16 % for a 2× ctx
  step (sub-linear; a dense O(ctx) model would roughly halve).
- **HBM ceiling (f32 KV):** `kv_cache = max_pos · 512 · 4B · 43 layers`. ctx=16384
  (RSS 28.95 GB) is the safe top; **ctx=32768 OOM-kills** (`rc=137`) — the load completes
  @ 27.51 GB (lazy pages), then the warm faults the full ~2.9 GB KV + compressed caches
  past usable HBM. An **f16 KV cache** would double the headroom to reach 32K+ (a TODO).

## OPS gotchas (each cost a real cycle)

1. **A job restart wipes `/local` AND moves the alloc.** A new `pjsub`/restart →
   new `$PJM_SUBJOBID` →
   - `/local` is cleared → the staged blobs are GONE → `DS4F_REAL=1` opens a missing
     `/local/ds4f/rank<rr>.blob` → fast failure/segfault. **Re-run Step 1.**
   - the physical Tofu group changes (e.g. `12 19 15` → `7 19 0`) → any saved
     `tofu_topo.txt` is stale → `mpiexec` can't place ranks on nonexistent coords →
     instant `rc=1` (~2 s, "exceed limit on virtual coordinate"). **Never pass
     `SKIP_TOPO=1` across jobs** — let the launcher regenerate.

2. **There are two `tofu_topo.txt` files.** The run launcher reads the one in its CWD
   (`a64fx/llm/tofu_topo.txt`). Running the helper from `a64fx/utofu-tests/` writes a
   *different* file and does not update what the runner reads. Pre-verify from `llm/`
   or just trust the launcher's own regeneration (the default).

3. **Excluding the claude node.** claude's interactive node is the alloc node whose
   intra-group coord is `a=b=c=0`. Probe it: `utofu_query_my_coords()` prints
   `x y z 0 0 0`. own_mpi maps that to vcoord `(0,0,0)`, so `EXCLUDE=0,0,0` (the
   launcher default) drops it. **Verify** after topo regen: the placed
   `tofu_topo.txt` must NOT contain the `x y z 0 0 0` row.

   ```sh
   # one-shot probe of this node's coord (compile once):
   cat > /tmp/mycoord.c <<'EOF'
   #include <stdio.h>
   #include <utofu.h>
   #ifndef TOFU_NCOORDS
   #define TOFU_NCOORDS 6
   #endif
   int main(void){uint8_t c[TOFU_NCOORDS]={0};
     if(utofu_query_my_coords(c))return 1;
     printf("%u %u %u %u %u %u\n",c[0],c[1],c[2],c[3],c[4],c[5]);return 0;}
   EOF
   fcc -Nclang -O3 -march=armv8.2-a+sve -o /tmp/mycoord /tmp/mycoord.c -ltofucom && /tmp/mycoord
   # -> e.g. "7 19 0 0 0 0"; then after the run, confirm:
   grep -c "7 19 0 0 0 0" a64fx/llm/tofu_topo.txt   # must print 0
   ```
   Running the heavy ~22 GB load on the claude node OOM-kills the claude session (shared
   cgroup). Staging on it is safe (file→file, bounded RSS) but pointless.

4. **Orphaned `org/mpiexec` (PPID=1)** from a killed run holds ranks → later
   "exceed limit on virtual coordinate". Check `pgrep -af 'org/mpiexec|ds4f_ep_runner'`
   and kill stale ones (use a pattern that does NOT self-match the killing shell).

5. **stdout is not forwarded** from `mpiexec`'d ranks — only file writes survive. Wait on
   the 11× per-rank files, NOT a stdout banner.

## Output discipline (keep agent context small)

Chain stage→run **detached** (`setsid nohup …`), funnel all output to log files, and
read only a compact sentinel — never stream the 11× per-rank dumps into the agent. A
single background waiter that polls the sentinel for an end marker (e.g. `RUN2_END`)
beats echo-ping polling. Pattern: write `STAGE done=N/11`, `RUN rc=…`, topo group,
crash grep, and the ~10-line `ds4f_ep_rank00.txt` to one `/tmp/*sentinel.txt`.

## Validated result (job 49092345, 2026-06-05)

```
STAGE done=11/11  wall=1272 s  (no OOM — bounded /local cache held)
RUN   rc=0  11/11 perf + 11/11 load  topo group 7 19 0 (claude excluded)
NaNs=0  argmax=122293  (deterministic; identical to the first validated run)
prefill  8 tok   95.8 ms/tok  10.44 tok/s  comm 13.3%  ar_calls=344
decode  16 tok   98.4 ms/tok  10.16 tok/s  comm 12.4%  72.9 GB/s-weights
arena 22.18 GB   RSS 21.81 GB / node   (fits ~25 GB usable HBM2)
per-phase decode: o_proj 33.4% + qkv 27.8% + shared 14.1% = ~75% replicated dense FP8;
                  experts 5.2%, attn 3.9%, head 2.2%, router 0.6%.
```

The dominator is replicated dense FP8 gather (same finding as the synthetic Stage 2/3
and the EP-runner). `DS4F_FP8_BF16=1` (Step 2c) promotes that dense to bf16-pv at load
and lifts real-weight decode to **16.73 tok/s** (+65 %); the remaining lever is batched
decode M>1 to amortize the 43 per-layer all-reduces.

## Flag reference (env)

| Flag | Default | Effect |
|---|---|---|
| `NP` | 11 | rank count |
| `EXCLUDE` | `0,0,0` | relative vcoord to drop (claude node); `none` = whole alloc (DANGER) |
| `DS4F_REAL` | 0 | 1 = load real weights from `/local` blobs (else synthetic fill) |
| `DS4F_EXACT` | 0 | 1 = exact dense Tier-B1 math (else shape-correct stand-in) |
| `DS4F_TIERB2` | 0 | 1 = stateful compressor/indexer (CSA/HCA) decode; implies `EXACT`. Real weights load by name (validated, job 49092345) |
| `DS4F_FP8_BF16` | 0 | 1 = predequant replicated dense FP8→bf16 at load (faster decode, +~6 GB); lossless (Step 2c, real-weight validated). Default = byte-identical FP8 |
| `DS4F_BF16_PV` | (auto) | with `DS4F_FP8_BF16=1`: empty = pair-interleaved pv (fastest); `0` = plain bf16; `1` = force pv |
| `DS4F_MHC` | 0 | 1 = exact manifold-constrained hyper-connections |
| `DS4F_STAGE_FLUSH_GB` | 2 | stager HBM dirty-cache flush granularity |
| `DS4F_STAGE_DIR` | `/local/ds4f` | per-node blob dir |
| `DS4F_PREFILL` / `DS4F_MAXGEN` | 8 / 16 | prefill / decode token counts |
| `DS4F_CTX_WARM` | 0 | >0 = warm synthetic KV+compressed caches to this ctx, decode from there (long-ctx bench, Step 2d); deterministic + rank-independent (lockstep-safe) |
| `DS4F_MAXPOS` | 4096 | KV/compressed cache capacity; must exceed `CTX_WARM+MAXGEN` (f32 KV: ~ctx16384 fits, 32768 OOMs) |
| `DS4F_LAYERS` | 0 (=43) | layer count (small = smoke) |
| `LLM_THREADS` | 48 | OpenMP threads/node |
| `SKIP_TOPO` | unset | 1 = reuse existing `tofu_topo.txt` (**never across jobs**) |

## Current state (2026-06-05)

Everything below is **DONE + validated** on the 11 EP nodes (real DeepSeek-V4-Flash
weights), committed on branch `ds4f` (latest `037a2d1`) except where noted:

- **Tier-B1 exact dense forward** (Step 2): RoPE/YaRN, per-head q-norm, MQA sliding-window
  + sink attn, grouped low-rank o-proj, sqrtsoftplus gate, swiglu clamp — argmax=122293,
  10.16 tok/s FP8 dense.
- **Tier-B2 compressor/indexer** (Step 2b): stateful 21-CSA + 20-HCA decode path, real
  weights loaded by name; bf16 weight-quant (`037a2d1`, −37 % tb2prep, argmax-exact);
  pooled+SVE compressor/indexer kernels (`f648b79`, 14× over the f32-reference path).
- **bf16-pv dense promote** (Step 2c): lossless FP8→bf16-pv at load, decode 16.73 tok/s
  (+65 %).
- **Long-ctx Tier-B2** (Step 2d): O(topk) payoff confirmed — attn +7.9 % per 2× ctx
  (capped at `index_topk=512`); the index *scan* is the new long-ctx ceiling. ctx-warm
  infra (`ds4f_ep_runner.c` `DS4F_CTX_WARM` + `run_ds4f_longctx_11n.sh`) is the only
  **uncommitted** piece as of this writing → committed alongside this doc.

## Remaining work (priority order)

1. **SVE/quant kernels for the `index_score` scan** — the confirmed long-ctx bottleneck
   (`tb2prep` 41–48 % of decode at ctx ≥ 8K, O(ctx/ratio), reads `idx_kv` *activations*
   so weight-quant can't help). Currently pooled-f32 over `idx_kv`; quantize `idx_kv` +
   SVE-dot the scan to lift long-ctx decode.
2. **f16 KV cache** — halve the KV footprint (`kv_cache` is f32 today) to reach ctx ≥ 32K
   without OOM; the f32 build caps at ~ctx16384 (RSS 28.95 GB).
3. **Batched decode / prefill M > 1** — amortize the 43 per-layer EP all-reduces
   (comm ~20 % Amdahl floor at M=1). Needs packed-B GEMM kernels; note the known A64FX
   batched-prefill regression (`project_batched_prefill_finding`).
4. **fp4 compressor/indexer weights** — LOSSY, low value (the bf16 weights are already
   bit-exact and the `index_score`/RoPE/fp4 overhead, not the weights, is the `tb2prep`
   floor). Only if HBM pressure justifies.
