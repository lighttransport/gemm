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
- **HBM ceiling — it is the *weight* footprint, not KV.** The bf16 KV cache (Step 2e) is
  now landed and halves `kv_cache` to `max_pos · 512 · 2B · 43L`, yet **ctx=32768 still
  OOM-kills** under the gen config. The dominant resident cost is the **27.5 GB replicated
  dense weights** (inflated +6 GB by `DS4F_FP8_BF16=1`'s FP8→bf16 predequant), leaving only
  ~2.5 GB for KV+compressed caches. Halving a ~1.4 GB KV term cannot offset a 27.5 GB base.
  The real long-ctx levers are **shrinking the weights** (`DS4F_FP8_BF16=0` keeps dense FP8,
  −6 GB, slower decode) and **int8 KV** — see Step 2e and Remaining work.

## Step 2e — gen-quality + decode tok/s at 10k+ (real coding task, mHC + bf16 KV)

End-to-end greedy generation on **real** DeepSeek-V4-Flash weights, production sparse path
(`DS4F_REAL=1 DS4F_TIERB2=1 DS4F_FP8_BF16=1` + **`DS4F_MHC=1`**), via a pure-Python
byte-level BPE tokenizer (`tools/ds4f_tokenizer.py`) and gen-mode in `ds4f_ep_runner.c`
(`DS4F_PROMPT_IDS` / `DS4F_GEN_OUT` / `DS4F_MAX_NEW`, greedy argmax feedback, eos=1).

```sh
./run_ds4f_gen_11n.sh                          # built-in quicksort code-completion prompt
PROMPT_FILE=my.txt MAX_NEW=200 ./run_ds4f_gen_11n.sh
```

**`DS4F_MHC=1` is REQUIRED for coherent output** — the model ships per-layer hyper-connection
weights (`hc_attn_*`/`hc_ffn_*` + global `hc_head_*`, `hc_mult=4`, `hc_sinkhorn_iters=20`);
the forward only uses them when `m->mhc` is set. With mHC **off** the residual collapses to a
single stream → architecturally wrong → garbage tokens (the "argmax-exact vs batched
reference" check does **not** catch this; it only verifies EP-path == single-node-path). The
cheap reference-free gate is teacher-forcing next-token accuracy (`DS4F_TF_CHECK=1`):

| config | TF next-token acc | gen output |
|---|---|---|
| mHC **off** | 0/23 = 0 % | `ايره peeled… 哈哈哈` (garbage) |
| mHC **on** (`DS4F_MHC=1`) | 16/23 = **69.6 %** | correct recursive quicksort (below) |

```
def quicksort(arr):
    """Sort a list of numbers in ascending order using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left  = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x >  pivot]
    return quicksort(left) + [pivot] + quicksort(right)         # ← model-generated, NaNs=0
```

**Decode tok/s at long context** (gen config: mHC + bf16 KV, prefill 8 + decode 16, all
rc=0 / 11/11 / NaNs=0 / argmax lockstep). These are the *heavy* (mHC-on) numbers — the
Step 2d O(topk) table above is the lighter mHC-off sparse-path characterization:

| ctx | decode tok/s | tb2prep | attn | RSS GB | arena GB/node | status |
|---|---|---|---|---|---|---|
| 24 (gen, real prompt) | 5.93 | — | — | 27.51 | — | coherent quicksort, mHC |
| 10240 | **2.99** | 38 % (126 ms) | 27 % (91 ms) | 27.96 | 26.11 | rc=0 |
| 16384 | **2.68** | 43 % (160 ms) | 27 % | 28.22 | 26.36 | rc=0 |
| 32768 | — | | | | | **OOM-kill** (sig=9 during warm-fill) |
| 65536 | — | | | | | died pre-load (PMIx — 32768-OOM aftermath, see OPS note) |

- Real-model decode is **~2.7–3 tok/s at 10–16K ctx** — below the 10–15 tok/s target. The
  dominant cost is **`tb2prep`** (the compressor/indexer `index_score` O(ctx/ratio) scan,
  38–43 %) then **attn** (~27 %); both read *activations*, so weight-quant can't help them.
  This is the same long-ctx ceiling Step 2d identified, now measured under the full mHC path.
- **bf16 KV did not raise the ctx ceiling much** — the blocker is the 27.5 GB weight base
  (see the HBM note above), not KV precision. ctx ≤ ~16K is the safe top under
  `DS4F_FP8_BF16=1`; reaching 32K needs `DS4F_FP8_BF16=0` (FP8 dense, −6 GB) and/or int8 KV.

## Step 2f — single-node decode roofline + FP8 prefill tile-dequant (alloc-free)

When the 11-node alloc is PMIx-degraded, this single-node, pthread-pinned bench
(`a64fx/llm/ds4f_decode_bw_bench.c`, `make ds4f_decode_bw`, runs directly on-node, pins
cores 12–59, leaves 0–11 + memory for the agent) rooflines every weight rep in the M=1
cold-HBM decode regime. Two confounds in the old `ds4f_fp8_fast_bench.c` were fatal and
are fixed here: **NUMA first-touch** (fill *through* the pool via `mmap`/`munmap` per pool
— glibc `aligned_alloc` recycles already-faulted pages, pinning the whole pool to CMG0 and
collapsing the read ceiling to single-CMG ~90 GB/s) and **cold cache** (a pool of P
distinct matrices cycled per sweep). Trust shapes ≥16 MB only; smaller ones are
barrier/L2-overhead-skewed. Gate = argmax+top5 vs an own-repr f32 reference (never bit-eq).

**Node read ceiling R ≈ 720 GB/s** (1T 42 → 12T 466 → 24T 714 → 48T 719). The decisive,
non-obvious result (it **inverts** the "FP8 = faster *and* −6 GB" premise):

| 48T, ≥16 MB shapes | bf16 | fp8-gather | fp8-magic | scalar | neon |
|---|---|---|---|---|---|
| wo_a[8192,4096] 32 MB | **610** (85 % R) | 64 | 106 | 23 | 34 |
| wo_b[4096,4096] 16 MB | **579** (93 % R) | 103 | 128 | 20 | 32 |
| bigK[4096,8192] 32 MB | **307** (42 % R) | 85 | 73 | 18 | 32 |
| wq_b[32768,1024] 32 MB | **373** (52 % R) | 50 | 56 | 13 | 27 |

- **FP8 on-demand decode is dequant/issue-bound, not BW-bound** (8–20 % of R, still scaling
  past 24T); **bf16-predequant is BW-bound** (~85 % of R) and **2.1–3.3× faster per dense
  matvec** despite 2× the bytes. On A64FX, FP8 trades 2× memory for 2–3× slower decode —
  speed vs memory is a fundamental tension, not a free win.
- **magic ≥ gather** within FP8 on the dominant dense shapes (wo_a 1.66×, wo_b 1.24×);
  gather wins only large-K (bigK 0.85×). magic flushes E4M3 subnormals via FTZ → argmax-exact
  on synthetic, but a real-weight argmax check gates flipping `DS4F_FP8_MAGIC`'s default.
- **scalar/NEON-128 refuted** for the matvec (3–7× slower than SVE-512) — the "free cores in
  decode → short-vector lower latency" idea loses; the 48T matvec is decode-op-throughput bound.

**Landed (validated single-node, argmax-exact): FP8→bf16 FUSED tile-dequant prefill GEMM.**
`ds4f_gemm_worker` (`common/ds4f_impl.h`) now batches **FP8 dense** (the memory-lean default)
instead of falling back to per-token matvec. The **fused** kernel: for each 8-row group it
dequants one `TILE_K`-wide FP8 sub-tile (SVE LUT-gather + per-128-block E8M0 scale + `>>16`,
bit-identical to the load-time predequant) **into a tiny 8 KB L1 PV pair-buffer** (`uzp1`+`zip1`
interleave → the same `p_odd` layout the bf16 path uses), then immediately consumes it across all
M tokens with the peak `matvec_bf16_8x3_pv_acc` microkernel, accumulating over K tiles. **No bf16
HBM round-trip** and it reuses the exact bf16 microkernel — so the dequant hides behind the FMA
pipeline. `DS4F_EXACT=1 DS4F_FP8_BF16=0 DS4F_PREFILL_BATCH=64 DS4F_PREFILL_CHECK=1` → argmax
**0/64** mismatch, rel 8.5e-6 (K-tile reassoc; FP8→bf16 itself lossless).

*Evolution of this lever:* the first version expanded the whole row-slice to a bf16 scratch then
ran the plain `gemm_bf16_f32_tokmajor` — two penalties (bf16 HBM/L2 round-trip + the slower
non-PV kernel) capped it at **~53% of the +6 GB bf16-resident path** (18.4 vs 39 tok/s). Fusing
the dequant into the K-tile and switching to the PV 8×3 kernel removed both: **FP8 batched
prefill is now ~88% of bf16-resident** (A/B at L=16, P=384: fused-FP8 ~85–92 vs bf16-resident
~96–106 tok/s; equal within noise at L=8) — **with no +6 GB**. Decode (M=1) path untouched. The
residual ~12% is the gather-bound dequant compute not 100% hidden; closing it would need the
gather-free *magic* decode (real-weight-argmax-validation-blocked, FTZ subnormals).

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

6. **An OOM-kill (sig=9) degrades the remote PMIx service.** When a rank is `SIGKILL`ed
   (HBM OOM during warm-fill, e.g. ctx=32768), it cannot finalize MPI cleanly, leaving the
   `plexec`/PMIx daemon on that node in a bad state. **Every subsequent launch then dies
   pre-load in ~3 s** with `[ERR.] PLE 0080 plexec PMIx service error occurred.(nid=…)` and
   `rc=255`, `perf=0/11 load=0/11` — looks like a fresh crash but is aftermath. A short wait
   (45 s, even 180 s) does **not** reliably clear it; recovery usually requires recycling the
   alloc (new `pjsub` → `/local` wiped → re-stage Step 1, ~21 min). **Practical rule: don't
   probe ctx past the known ceiling (~16K) on a shared alloc you don't want to lose** — one
   OOM costs the whole allocation. The longctx wrapper now flags this (`rc≠0` with
   `load=0/11` ⇒ pre-load failure) instead of reporting `crash_hits=0`.

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
| `DS4F_MHC` | 0 | 1 = exact manifold-constrained hyper-connections (`hc_mult=4`). **REQUIRED for coherent real generation** (Step 2e); gen wrapper defaults it on |
| `DS4F_PROMPT_IDS` / `DS4F_GEN_OUT` / `DS4F_MAX_NEW` | — | gen-mode (Step 2e): prompt id file in, generated id file out, greedy decode budget (stops on eos=1) |
| `DS4F_TF_CHECK` | 0 | 1 = teacher-forcing next-token accuracy gate (~0 % = broken forward, ~50–80 % = working); the cheap reference-free correctness check |
| `DS4F_STAGE_FLUSH_GB` | 2 | stager HBM dirty-cache flush granularity |
| `DS4F_STAGE_DIR` | `/local/ds4f` | per-node blob dir |
| `DS4F_PREFILL` / `DS4F_MAXGEN` | 8 / 16 | prefill / decode token counts |
| `DS4F_CTX_WARM` | 0 | >0 = warm synthetic KV+compressed caches to this ctx, decode from there (long-ctx bench, Step 2d); deterministic + rank-independent (lockstep-safe) |
| `DS4F_MAXPOS` | 4096 | KV/compressed cache capacity; must exceed `CTX_WARM+MAXGEN`. KV is **bf16** (Step 2e); ctx ≤ ~16K fits, 32768 OOMs (weight-footprint-bound, not KV) |
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
- **Gen-quality + decode tok/s** (Step 2e): real coding-task generation produces coherent
  code (recursive quicksort) with **`DS4F_MHC=1`** (the key finding — mHC is required;
  without it, garbage). bf16 KV cache landed (halves KV). Measured decode **2.99 tok/s @
  ctx10240, 2.68 @ ctx16384** (mHC path; below the 10–15 target — `tb2prep`/attn bound).
  Two bugs fixed en route: the **`s_tb2_sel` heap overflow** (`index_topk=512 > max_pos`
  corrupted compressor ring-state → NaNs; size to `max(max_pos, index_topk)`), and the
  mHC-off architecture gap. Gen-mode + tokenizer + bf16 KV + these fixes are **uncommitted**
  as of this writing.

## Remaining work (priority order)

1. **SVE/quant kernels for the `index_score` scan** — the confirmed long-ctx bottleneck
   (`tb2prep` 41–48 % of decode at ctx ≥ 8K, O(ctx/ratio), reads `idx_kv` *activations*
   so weight-quant can't help). Currently pooled-f32 over `idx_kv`; quantize `idx_kv` +
   SVE-dot the scan to lift long-ctx decode.
2. **256k decode without degradation (10–15 tok/s)** — the original long-ctx target, NOT
   reached this session. Two independent blockers, both characterized:
   - **Memory:** bf16 KV (done, Step 2e) was not enough — ctx=32768 still OOMs because the
     27.5 GB replicated dense weights (inflated +6 GB by `DS4F_FP8_BF16=1`) dominate, not KV.
     Levers: `DS4F_FP8_BF16=0` (FP8 dense, −6 GB, slower) and **int8 KV** (quarter the f32
     footprint). int8 KV + FP8 dense should clear 32K–64K; 256k needs int8 weights too.
     *int8-KV scheme de-risked* (`tools/int8kv_probe.c`, single-process, no alloc): the kv
     latent's massive-activation channels (~1e3–1e5, the same ones that forced bf16-not-fp16)
     make naive **per-token int8 catastrophic** (one sink dim sets the scale → 99 % of the
     O(1) dims collapse to 0). The viable layout is a **static per-channel scale calibrated
     on the first ~256 positions** (S5): L1-rel 0.0032 ≈ bf16, only 2.7 % of |·|>0.1 dims
     >5 % off, and **streaming-decode-compatible** (sink channels are positionally consistent,
     so early calibration holds for all later positions). 1 B/elem (+ a 512-float per-layer
     per-channel scale, amortized). NOTE: probe uses a *representative synthetic* latent
     distribution — the per-channel scheme + calibration window still need argmax-exact
     confirmation on real latents (alloc-blocked). Implementation: `kv_cache` uint16→int8 +
     `kv_scale[layer][512]` (calibrate during prefill/warm), dequant at the latent-read sites.
   - **Speed:** even at 10–16K, decode is ~2.7–3 tok/s (target 10–15), bound by `tb2prep`
     (the `index_score` O(ctx/ratio) scan, 38–43 %, item 1) + attn (~27 %). At 256k the scan
     dominates outright — int8/SVE `index_score` is the prerequisite for the tok/s target.
3. **Batched decode / prefill M > 1** — amortize the 43 per-layer EP all-reduces
   (comm ~20 % Amdahl floor at M=1). Needs packed-B GEMM kernels; note the known A64FX
   batched-prefill regression (`project_batched_prefill_finding`).
4. **fp4 compressor/indexer weights** — LOSSY, low value (the bf16 weights are already
   bit-exact and the `index_score`/RoPE/fp4 overhead, not the weights, is the `tb2prep`
   floor). Only if HBM pressure justifies.
