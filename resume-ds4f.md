# DS4F single-node serving — status and hand-off

Hand-off for the DS4F (DeepSeek-V4-Flash) HTTP serving runner on the
Threadripper 1950X + RX 9070 XT host.  Supersedes all earlier content in this
file, including the old "40 prompt tok/s / 9 decode tok/s acceptance reference"
— that decode figure was never this runner's; see *Ceiling* below.

Last updated 2026-08-13, at commit `1a1a4690` (branch `ds4f`, unpushed).

---

## Resuming prompt

> Continue DS4F serving work in `/mnt/nvme02/work/gemm/ds4f` (branch `ds4f`).
> Read this file first, then `a64fx/llm/ds4f_serve.md` and `AGENTS.md`.
>
> Ground rules:
> - Do not modify or optimise llama.cpp. It is a behavioural/perf reference only.
> - Production tuning is explicit runner arguments, never hidden env defaults.
> - Any change that alters greedy token IDs stays **opt-in and default-off**,
>   and must be recorded with the fixture it diverged on.
> - Measure with `a64fx/llm/ds4f_serve_bench.py` (in-process, low noise), not the
>   HTTP path, and use **paired adjacent runs** — see *Measuring* below.
> - Do not push.
>
> Current state: decode 5.02 tok/s at defaults, 5.69 with the two opt-in flags,
> from 3.93 at the start of the previous session. Every phase that has been
> profiled is at or near its measured roofline. Read *Closed* before proposing
> work — several obvious ideas are already measured negative.

---

## System under test

```text
HTTP client -> a64fx/llm/ds4f_serve.py -> cooperative Unix-socket runner
            -> libds4f_serve.so (common/ds4f_impl.h + hetero/ds4f/hip_ds4f_dense.c)
```

| | |
|---|---|
| CPU | Threadripper 1950X, 16C/32T, Zen1 (AVX2, no AVX-512), 188 GiB DDR4, 1 NUMA node |
| GPU | RX 9070 XT, gfx1201, 16.3 GiB, PCIe Gen5 x16 — the runner's only GPU |
| GPU | RTX 5060 Ti, 16 GiB, PCIe Gen3 x8 — **reserved for llama.cpp**, unused by the runner |
| Model | DeepSeek-V4-Flash: 43 layers, hidden 4096, MLA (64 heads, kv_lora 512, window 128 + sink), **256 experts top-6 + 1 shared**, mHC (4 streams, 20 sinkhorn iters), Tier-B2 compressor + lightning indexer (top-512), DSpark MTP (3 stages, block 5) |
| Weights | `/mnt/nvme02/models/ds4f-0731/*.safetensors` (48 shards). Experts are **natively FP4**, dense is FP8-e4m3 — this *is* the unquantised release; there is no BF16 copy |
| Staged | `/tmp/ds4f_nocopy_stage_mtp` (NOCOPY manifest, 72317 tensors incl. MTP, 156.02 GB, ~65 s load) |
| Tokenizer | `/mnt/nvme02/models/ds4f-0731/tokenizer.json` |

Architecture constants are **hardcoded** in `ds4f_default_config()`
(`common/ds4f.h`), not read from the checkpoint's `config.json`.

---

## Build and launch

```sh
sh a64fx/llm/build_ds4f_serve.sh
python3 -m py_compile a64fx/llm/ds4f_serve.py a64fx/llm/ds4f_serve_runner.py \
  a64fx/llm/ds4f_serve_bench.py a64fx/llm/bench_ds4f_http.py

DS4F_STAGE_DIR=/tmp/ds4f_nocopy_stage_mtp \
TOK=/mnt/nvme02/models/ds4f-0731/tokenizer.json \
DS4F_SERVE_USE_HIP=1 DS4F_MAXPOS=16384 PORT=8080 \
  sh a64fx/llm/run_ds4f_single_serve.sh \
    --agent-cache-max-tokens 14336 --single-prefill-quantum-tokens 1024 \
    --prefill-quantum-tokens 32 --decode-quantum-tokens 4

curl -fsS http://127.0.0.1:8080/health
```

The runner prints the configuration that **actually took effect** at startup —
several flags are silently downgraded during load, so always check this line
rather than trusting the arguments:

```
[serve] forward: exact=1 mhc=1 tierb2=1 sparse=0 mtp=0 mxfp4_raw=1 w4a8=0
        group_split=1 int8_kv=0 int8_cmp=0 max_pos=16384 threads=16 spec=0
```

---

## Current performance

`ds4f_serve_bench.py --prompt-tokens 1024 --warm-decode 16 --decode-tokens 64`,
16 threads:

| config | prefill tok/s | decode tok/s |
|---|---:|---:|
| start of previous session | 7.56 | 3.93 |
| **current defaults (bit-exact)** | **8.81** | **5.02** |
| `--mxfp4-w4a8 1 --hip-tb2-decode 1` | 14.22 | **5.69** |

llama.cpp on this host, same full-MXFP4 weights: **7.33 tok/s**
(`/mnt/nvme02/work/llama.cpp/da4f.md`). Its widely-quoted 9–10 tok/s is a
**Q3/IQ3** result (ROCm 8.97–9.22, CUDA 9.69–9.98), and that file states 10 tok/s
was not reached inside the Q3 quality budget. Note llama.cpp binaries here must
be rebuilt to load this model — the installed ones predate `LLM_ARCH_DEEPSEEK4`.

### Decode phase shares (de-nested, tuned config)

routed experts ~50%, qkv+o_proj ~16%, tb2prep ~10%, mHC ~7%, head ~4%,
attention proper ~3%, shared FFN ~4%, router ~1%.

### Achieved vs roofline — everything profiled is at its limit

| phase | achieved | ceiling |
|---|---:|---:|
| routed experts (CPU, W4A8) | 41.4 GB/s | 37.8–40.5 GB/s (`bench_expert_bw --i8seq`, 40–100 GiB) |
| routed experts (CPU, f32 acts) | 32.3 GB/s | 31.0 GB/s standalone |
| `tb2lcmp` (520 MB/token) | 45.3 GB/s | 48.1 GB/s (S0 read-only) |
| `tb2qproj` (352 MB/token) | 46.8 GB/s | 48.1 GB/s |
| qkv/o_proj (GPU FP8 matvec, M=1) | ~200 GB/s | ~640 GB/s peak — **the one real gap** |

---

## Runner flags that matter

| flag | default | quality | note |
|---|---|---|---|
| `--mv-group-split` | 1 | **bit-exact** | splits a fused group's concatenated row space; worth ~0 without W4A8 |
| `--hc-parallel` | 1 (library) | **bit-exact** | pool-parallel mHC collapse/expand |
| `--hip-block-threads` | 64 | **bit-exact** | +2.7% over 128; deviates from AGENTS.md's prefill-tuned profile deliberately |
| `--threads` | 16 | — | 16 is best *with* group split; without it 12–14 won |
| `--mxfp4-w4a8` | 0 | **CHANGES TOKENS** | int8 expert activations; +30% decode, +60% prefill |
| `--hip-tb2-decode` | 0 | **CHANGES TOKENS** | Tier-B2 projections on GPU; +5.8% decode |
| `--hip-decode-qkv-fuse` | 0 | — | measured −19%, leave off |
| `--hip-decode-attn-oproj` | 0 | — | measured −39%, leave off |
| `--speculative-tokens` | 0 | — | leave at 0, see *Closed* |

Diagnostics (env only, never production tuning): `DS4F_PROF=1` phase profile,
`DS4F_EXPERT_BW=1` measured expert GB/s, `DS4F_SERVE_TIME=1` per-token forward ms.
Speculation acceptance telemetry prints at close whenever speculation ran.

---

## Ceiling — read before promising a number

Decode reads **3.44 GB of routed-expert weights per token** (13.35 MiB/expert ×
6 × 43 layers). Against the measured ~40 GB/s CPU read ceiling that is an
~85 ms/token floor for the expert phase alone, i.e. **~12 tok/s even if every
other phase were free** — and experts are only half of decode.

**18–20 tok/s is not reachable on this host with full-precision experts.** It
requires reading fewer expert bytes per token — sub-FP4 weights or a smaller
top-k — which changes model output. GPU residency does not rescue this: 16 GiB
of VRAM against a 148 GB expert bank is ~5% coverage, and PCIe streaming is far
below DRAM bandwidth.

---

## Measuring — the traps that cost the last session hours

1. **Use `ds4f_serve_bench.py`, not HTTP.** The HTTP path varies ±30% run to
   run. The serving layer itself is only ~9% overhead (measured), so it is not
   what you are debugging.
2. **Use paired adjacent runs** (A,B,A,B) — absolute numbers drift ~10% between
   runs, but paired deltas are stable. Treat anything under ~3% as noise.
3. **`DS4F_PROF=1` inflates decode ~30%.** Its phase *shares* are usable;
   absolute GB/s computed from its timings is **not** (this produced a wrong
   "21.6 GB/s expert" claim once). Use `DS4F_EXPERT_BW=1` for real bandwidth.
4. **Greedy decode is deterministic** run to run (verified byte-identical), so
   any output difference is a real behavioural change, not noise.
5. Model load is ~65 s. Do not mistake it for inference time.

---

## Quality gate

Two fixtures, greedy (`temperature 0`, `top_p 1`), 64 tokens: a ~1145-token C
code-review prompt and a ~1200-token long-range-recall prompt (a fact at
position 0, asked at position ~1200 — it catches attention regressions).

Run the same fixtures before and after, diff the decoded text, and record the
first divergence. A change that moves tokens is not forbidden — it is
**default-off and documented**. The recall fixture must never regress.

---

## TODO, in expected-value order

1. **Improve the M=1 FP8 GPU matvec.** qkv+o_proj is ~16% of decode running at
   ~200 GB/s of ~640 GB/s peak — the only profiled phase with real headroom, and
   worth up to ~10% decode. This is gfx1201 kernel work (`ds4f_dense_fp8_matvec`
   in `hetero/ds4f/hip_ds4f_kernels.h`): load width, waves per CU, per-row
   parallelism at M=1. Flag tuning is exhausted.
2. **Decide the W4A8 / GPU-tb2 quality question.** Together they are worth +13%
   decode and +60% prefill but both move greedy tokens. Someone with authority
   over model quality should run a real eval (not two fixtures) and either
   promote them to default or close them permanently. This is the largest
   available win and it is a *judgement*, not an engineering task.
3. **Re-tune the hot-expert VRAM cache.** `--hip-expert-cache-mb auto` admits
   ~490–515 bundles; the adaptive refresh (`--hip-adaptive-cache-period`)
   measured net-zero at M=1 long ago and has not been revisited since the expert
   path got 28% faster. Cheap to A/B.
4. **`attn_gemm hit=0 miss=6888`** — the fast 8-head-blocked attention path is
   gated off on every call (`ds4f_attn_tb2_gemm`, `common/ds4f_impl.h`). Worth
   little (attention is ~3% of decode) but it is a real unexplained gate.
5. **Prefill at defaults is 8.81 tok/s vs 14.22 with W4A8.** Prefill has had far
   less attention than decode; if prefill matters for the coding-agent workload,
   profile it separately — the phase mix is different (experts dominate less).
6. **Full `resume-ds4f.md` test matrix** — context switching, durable agent-prefix
   cache hit/miss/fallback, contention, error recovery. The previous session
   focused entirely on throughput and did **not** re-run these.

---

## Closed — do not retry without new information

| idea | verdict |
|---|---|
| **Batched speculative verification** | Measured negative and structurally impossible. A K=4 block costs 193 ms/verified position vs 181 ms for a standalone M=1 forward — **no expert amortisation**, because top-6-of-256 makes a block's expert set nearly disjoint (one token per bucket). Verify cost linear in K bounds speculation at 1 forward/token, so even perfect acceptance only returns to baseline. Sequential verifier measured 4.34 vs 5.54 without. Implementation removed; acceptance telemetry kept. |
| **Expert bandwidth tuning** | Done — 41.4 GB/s, at/above the standalone kernel roofline. Further gains need a cheaper kernel (fewer ops per weight byte), not better streaming or scheduling. |
| **tb2prep CPU optimisation** | Done — both projections at 45–47 GB/s vs a 48.1 GB/s ceiling. |
| **`--hip-decode-qkv-fuse` / `--hip-decode-attn-oproj`** | Now wired (they were unreachable dead code) and measured −19% / −39%. The generic `ds4f_matvec_multi` route with `DS4F_MV_FUSE` + `DS4F_MV_ASYNC` is better. |
| **`DS4F_HC_RMSPAR`** | +1% decode, −19% prefill, and not bit-exact. |
| **`HSA_ENABLE_INTERRUPT=0`, `GPU_MAX_HW_QUEUES=1`** | Noise on decode; the first costs 22% of prefill. |
| **GPU Tier-B2 attention** (`DS4F_ATTN_HYBRID_GPU`) | ~3× slower per position — one GPU round trip per sequential position. |
| **Whole-layer expert streaming to GPU** | Prefill collapsed to 0.85 tok/s. |
| **Pinned expert staging** | 9.66 vs 11.60 tok/s prefill. |
| **SMT / 32 threads** | Worse than 16 (42.7 vs 47.6 GB/s read). |

### The pattern worth knowing

Five separate wins last session were **features that already existed and were
switched off**: W4A8 activations, the group-split opportunity it unlocked,
`idx_wq_b`'s GPU binding, the decode gates, and `DS4F_HC_PAR`. Before writing a
kernel, check whether the code is already there and defaulted off — and check
the `[serve] forward:` line to confirm what actually ran.

### One correction carried forward

A runner change briefly wrote argparse defaults into the library's environment
variables unconditionally, clobbering operator-set values and making env-based
A/B a silent no-op. Fixed (those flags default to `None`). If you read results
from that window in git history, re-verify the configuration before trusting the
label.
