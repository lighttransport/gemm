# GLM-5.2 INT8 on A64FX — Session Report (2026-06-30)

## Headline
**The GLM-5.2 int8 (compressed-tensors w8a16) model could not generate text — root-caused to a one-tensor loader bug and fixed.** Also added a decode sampler (temperature / top-p / repetition penalty). Both validated end-to-end on Fugaku A64FX and committed (`db56d8b9`, branch `glm5-2`).

---

## 1. The int8 generation bug — root cause & fix

**Symptom:** int8 produced `0,0,0…` garbage; teacher-forcing next-token accuracy **5.2%** (bf16 = 42.7%). Prior int8 work only validated *prefill* (tok/s, NaN, KV-cache) — generation coherence was never checked.

**Root cause:** the MoE router weight `mlp.gate.weight` is stored **F32** in the int8 checkpoint (it is the *one* non-quantized tensor kept at higher precision; bf16 in the bf16 checkpoint). The loader assumed bf16 for all non-quantized tensors and raw-copied the bytes, so the matvec read a **4-byte/element F32 array as 2-byte bf16** → a scrambled, ~10× weaker router → every expert scored ~0.5 → **flat / arbitrary expert selection** → broken MoE → broken forward.

Why it hid so long: only routing broke (dense, attention, shared-expert, routed-matvec were all fine — they just ran on the *wrong* experts). Offline weight checks passed because the *file* is correct; only the *loader* misread the dtype. (`mlp.gate.weight` + `mlp.gate.e_score_correction_bias` are the only F32 tensors; the bias is used as `float*` so it was already correct.)

**Fix** (`common/glm5_impl.h`, gate load): convert F32→bf16 (round-to-nearest-even) when `e->f32` is set, instead of raw-copying. `glm5_ent` already tracks `f32` (parsed from the dtype string); staging preserves it.

**Verified:** post-fix int8 router output (`||rl||` 9.40 vs bf16 9.40), raw sum, and **selected experts (250,186,174,72) match bf16 exactly**; full-model int8 generation is coherent.

### How it was localized (cheap 12-node probes)
Added two gated diagnostics to the runner: `GLM5_TF_CHECK` (teacher-forcing accuracy) and `GLM5_DUMP_NORMS` (per-layer / per-component norm dumps; since trimmed out post-fix). Bisection chain:
1. dense + attention norms match bf16 → diverge at the **first MoE layer**
2. shared expert matches, **routed experts under-contribute** (~4×)
3. routed-matvec *inputs* (gs, scale, bytes) correct → **downstream**
4. router *input* (`h2`) matches but *output* (`rl`) collapses 10× → the **gate**
5. gate is bf16-typed but `||rl||` 10× weak → checkpoint dtype is **F32**, loaded as bf16

Supporting checks ruled out the kernel (`a64fx/glm5/int8_consistency_test.c`, native 1-node, ≤2.4e-3) and the weights (`~/int8_cmp.py`, cosine ≥0.9999 — note cosine is scale-invariant, so it could not have caught a uniform-scale loader bug; the norm probes did).

---

## 2. Decode sampling (new feature)
Temperature + top-p (nucleus) + repetition penalty (per-unique, CTRL/HF-style) added to the decode loop. **Lockstep-safe:** under a vocab-sharded lm_head it gathers the full logit vector via the EP all-reduce, then samples with a shared RNG seed so every rank picks the same token. Env: `GLM5_TEMP`, `GLM5_TOPP`, `GLM5_REP_PEN`, `GLM5_SEED` (all off by default). A self-inflicted per-occurrence rep-penalty bug (caused counting degeneration) was caught in validation and fixed to per-unique.

---

## 3. Validation & node-hour campaigns
- **bf16 sampling**: coherent output after the rep-penalty fix.
- **int8 end-to-end (96n)**: coherent generation, was `0,0,0` before.
- **Best-of-N eval (10 × 96n × 5.25h ≈ 5040 node-h)**: 3000 coherent answers across 8 frontier tasks (coding/debugging/algorithm/math/systemdesign/logic/physics/analysis), distinct seeds. Math samples are correct rigorous induction proofs. Corpus at `a64fx/glm5/bestof_run_4939059*/`.
- **Follow-on (9 × 96n × ~0.85h ≈ 780 node-h)**: additional best-of-N samples, `bestof1h_run_*`.

---

## 4. Known issues / future work
- **int8 crashes at ≥256 nodes** (silent abort after model setup; ran fine at 12n/96n). Separate, un-fixed — limits int8 to ≤96n today. Likely an EP / zero-owner-rank or memory issue at large node counts.
- **Decode throughput** is single-stream-bound: ~0.16–0.25 tok/s at 96–384 nodes (per-token all-reduce dominates). Generation-heavy runs should use few nodes or the batched/cbatch path; 384n is for prefill, not decode.
- **int8 alignment quirk**: occasional over-refusal of benign tasks + concise answers — a quantization-alignment trait (cf. FP8 instruction-following), not a generation defect.

## 5. Commit
`db56d8b9` on `glm5-2`: `glm5: fix int8 router (F32 gate) + add decode sampling` — `common/glm5.h`, `common/glm5_impl.h`, `a64fx/glm5/glm5_ep_runner.c`, `a64fx/glm5/int8_consistency_test.c`. Not pushed.
