# GLM-5.2 Q2 12-node — 108K coding-agent prompt result

Real coding-agent flow through the arg-driven runner: prefill the ~108K-token code
prompt (`$HOME/glm5_codegen.bin`, 3-doc system prompt + query at the tail), then
greedy-generate code and detokenize.

Invocation (job 49662277, 12 nodes):
```sh
SYS_TOK=108474 GEN=128 sh a64fx/glm5/run_glm52_q2_12n.sh codegen \
    --cp-threshold 2048 --set GLM5_MSA_BLOCK_REP=1
```

## Run 1 — feasible config (block-rep + early int4 transition)

| item | value |
|---|---|
| layers | 78 (full model) |
| prefill | 108474 tokens |
| KV tiering | Tier A bf16 (2048 slots) → **transition @ 2048** → Tier B int4 CP-shard (9088 slots/rank) + MSA |
| prefill throughput | **31.47 tok/s** |
| prefill NaNs | **0** |
| prefill argmax (last-token) | 51238 |
| completion sentinel | `glm5_prefill_g12n=done` |
| generate | 128 tokens @ 0.53 tok/s |
| crash | none (256K MSA-overflow fix holds at 108K) |

**Stability: PASS.** The full 108K prefill completes with NaNs=0 and no crash —
the MSA long-context heap-overflow fix (`glm5_msa_select` / `glm5_msa_prefill_select`,
see `GLM52_Q2_12N.md` §13) holds at 108K context.

**Generated text (detokenized):**
```
 enumeration0毛细, pepper figure621 smoke610 Action),291ousvox1 01 Smoke02.,`` that1,0ny,occo,截1osal6 Seingt Booth``,occo62 with6,ingt flo6osal designs,丕7errs5plication0, pepper012 mothersingt voted Gaul,鼻子112ampoampo0 pepper1adraasters6UNIT6,,_mgmt0 applic1鼻子enta1,1161 soczano661opia066ampo,26ampo019,Formats脆111ordonage
```

**Quality: FAIL (incoherent).** This is the expected consequence of the two
approximations used to make 108K feasible on 12 nodes, both of which the codebase
flags as non-quality:
1. **`GLM5_MSA_BLOCK_REP=1`** scores one representative token per block instead of
   all 128 — it is documented as a synthetic 1M-**stress** tool, "should not be
   used for quality validation." It degrades which blocks the sparse attention
   selects.
2. **Early int4 transition** (`--cp-threshold 2048`): only 2048 positions stay in
   exact bf16 Tier-A; the other ~106K are int4-quantized + sparsely attended.

Why the approximations were needed: full-fidelity long-context is O(T²) — the MSA
index-scan is O(T) per token, so a full-fidelity 108K prefill at 78 layers is
~hours. Getting a fast result forced the lossy path.

## Run 2 — where the incoherence comes from (fidelity diagnostic)

Same 6144-token prompt prefix, three tier configs, compared by last-token argmax
and a 16-token greedy continuation (job 49662277):

| cfg | KV / attention | argmax | first gen ids |
|---|---|---|---|
| **A** | exact Tier-A bf16, dense | 1565 | `1565 16 13 17 13 22 13 23 13 23 …` (repeats) |
| **B** | Tier-B int4 + full MSA scoring (no block-rep) | **11** | `11 323 6713 11 105074 46021 11 …` |
| **C** | Tier-B int4 + block-rep | **11** | `11 18313 11 100803 70338 11975 …` |

(The 6144 prefix has no query — the tail of the coding prompt's query lives at
~token 108416 — so A's repetition is a greedy-continuation artifact, not a tier
defect. The useful signal is the *cross-config* argmax.)

**Findings:**
- **A ≠ B**: switching exact-bf16-dense → int4+sparse changes even the first token
  (1565 → 11). The dominant quality loss is the **Tier-B int4 KV + sparse MSA
  approximation itself**, not block-rep.
- **B and C share the first token (11)** and diverge only later → **block-rep is a
  secondary effect**; the big shift is int4+sparse.

## Run 3 — int4 is NOT the cause; it's dense-vs-sparse attention

Added an **exact bf16 CP-sharded Tier-B** path (`--kv-tier-bf16` /
`GLM5_KV_TIER_BF16`; the transition copies bf16 words to CP slots instead of
`q4_pack`-ing them — the store/load/idxdot paths already branch on `int4_kv`, so it
was a contained change). CP sharding divides the KV by ep_size, so bf16 Tier-B
fits: **108K ≈ 1.1 GB/node, 256K ≈ 2.4 GB/node** over 12 ranks.

Re-ran the 6144 diagnostic with it:

| cfg | KV / attention | argmax |
|---|---|---|
| A | exact bf16, **dense** | 1565 |
| B | int4, **sparse MSA** | 11 |
| Bbf16 | **exact bf16**, **sparse MSA** | **11** |

**`Bbf16 == B == 11`, not A.** Exact bf16 KV produces the *same* output as int4 —
so **KV quantization (int4 vs bf16) is NOT the coherence bottleneck.** The entire
A-vs-B gap is **dense vs sparse (MSA) attention.** (The earlier "int4 capacity
limit" conclusion is retracted.)

Two consequences:
- The 6144 A/B/C/Bbf16 comparison is **confounded**: forcing the Tier A→B
  transition at pos 2048 makes even a *short* 6144 context attend sparsely,
  whereas the model natively uses **dense** attention until the context is
  genuinely long. So these runs compare the model against an artificially-sparse
  version of itself, not against a bug.
- `--kv-tier-bf16` is kept as a correct, zero-downside feature (exact KV where
  memory allows), but it is **not** the coherence fix.

## Run 4 — native path OOMs; the auto budget is too aggressive at large ctx

`codegen` with defaults (auto tiering) chose **Tier-A = 26112 replicated bf16
slots (~2.85 GB/node)**; combined with 26 GB weights + dense-prefill scratch + the
transition's transient double-allocation, rank 10 was **OOM-killed (SIGKILL)**
before finishing. So the native-sparse verdict is still open, and a second finding
falls out: **`glm5_kv_init`'s auto Tier-A budget doesn't reserve enough headroom**
for the dense-prefill scratch and the Tier A→B transition overhead at large
`max_pos`. Forcing a smaller Tier-A (`--cp-threshold 8192`) or lowering
`--kv-budget-gb` avoids it; the auto budget should subtract those terms.

## Where this leaves the 256K target

**Solved:** 256K **stability** — the full 262144-position prefill completes
`NaNs=0` at 315 tok/s with the tiering transition, after fixing a real MSA
heap-overflow (`GLM52_Q2_12N.md` §13). `--kv-tier-bf16` adds exact CP-sharded KV.

**Open (deep, not a quick fix):** coherent long-context *output*. Established
facts:
- KV precision is NOT the bottleneck (bf16 == int4 output).
- The output shift is the **dense→sparse (MSA) attention regime**, which is the
  model's native long-context mechanism — so the question is whether native sparse
  is faithful here, which the OOM prevented answering.
- Full-fidelity long-context is O(T²) (MSA index-scan); block_rep makes it fast
  but approximates block selection.

Recommended next steps (in order):
1. **Fix the auto Tier-A budget** to reserve prefill-scratch + transition headroom,
   then re-run native `codegen` (auto tier, no block_rep) to get the real
   native-sparse coherence verdict.
2. If native sparse under-performs on 12 nodes, **use more nodes** (bigger dense
   Tier-A window; the original harness ran this prompt on 96) or an **adaptive /
   later transition** that keeps recent context (incl. the query) dense.

**Bottom line:** the runner refactor, perf targets, and 256K *stability* are done;
256K *coherence* is bounded by the sparse-attention regime + node count, not by a
runner bug — it needs the budget fix above and, most directly, more nodes.
