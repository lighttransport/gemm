# GLM5.2 kernel fragments — clair-compiled, qlair-simulated

Small self-contained C fragments of GLM5.2 building blocks, each with an inline
self-check (`ok=1` on exact-equality recompute), compiled with the clair C
compiler and cycle-simulated with qlair.

| kernel | GLM5.2 role | dims | result |
|---|---|---|---|
| `rmsnorm.c` | pre-attn / pre-MoE norm | hidden=6144, f32 | ok=1, ~1.23M cyc |
| `swiglu.c` | MoE FFN activation `SiLU(gate)*up` | moe_inter=2048 | ok=1 |
| `int8_dequant.c` | w8a16 dequant `(q-128)*scale` (offset-binary, group-128) | C=512 | ok=1 |
| `router_topk.c` | MoE router softmax + top-8 of 256 (argmax-critical) | E=256, K=8 | ok=1 (desc+distinct) |

## Compute/comm mixture (per-layer step)
`moe_layer_mix.c` and `attn_layer_mix.c` run **compute + uTofu comm in one qlair
simulation** and split the cost via `CNTVCT` (needs `--cores 2` for the Tofu sim):

| layer step | compute (CNTVCT) | comm (N=4→8) | comm frac |
|---|---|---|---|
| **MoE** (RMSNorm+router+SwiGLU / dispatch+combine a2a, 6 KB) | 1.61M | 18.8K → 38.2K | 1–2% |
| **attention** (RMSNorm+MLA proj / all-reduce, 2 KB) | 25.8M* | 6.8K → 13.8K | ~0% |

\* attention projection is a down-scaled scalar-float matvec (`/32` reduce) so it
fits the sim budget; full size is ~800M.

**Reading it — and why the real decode inverts to comm-bound.** These are the
*hardware floors*: compute here is **scalar float** (an upper bound — real GLM5.2
uses int8 SDOT / SVE, ~50–100× faster), and comm here is the **uTofu wire floor**
(a lower bound — the measured decode all-reduce is ~26 ms, dominated by the
robust-completion *drain*, ~4 orders above the µs wire cost). So the raw ratio
(compute-dominated) flips: with optimized int8 compute ↓ and the AR drain ↑, real
decode is comm-bound — consistent with `../decode_sim.py` and `../MOE_COMM_RESULTS.md`.
The value here is that both halves run in one cycle-accurate + calibrated-uTofu
simulation, so the *structure* (which ops, how many transfers, payload sizes) is real.

## Build & run
```
~/work/clair/clair/build/clair -t arm64 -f bin rmsnorm.c -o r.elf
~/work/clair/clair/build/qlair -P -p r.elf
```

## IMPORTANT: use `-O0` (no `-O`) for float kernels
Two clair `-O` codegen defects were found while writing these (both **only** at
`-O`; `-O0` is correct, and qlair simulates identically):

1. **`-O` float codegen** — went from "every float is 0" to running real
   float array kernels, across clair `1c956cc0` → `93fb5ca4` → `e3d53986` →
   `7bef4c3f` → `e49b5f26` → `1836cbf8`:
   - float/double **globals** emitted as `.long 0` (root cause) — fixed;
     float-array **int-literal** inits (`float p[]={1,2,3,4}`) emitted as raw
     ints (≈0 denormals) — now stored as IEEE754 bits.
   - float32 register-width bugs (FPTOSI/SITOFP/FCMP/float-literal/**fmadd**
     used `dN` for single) — fixed.
   - **ternary `?:`** wasn't lowered (emitted no code) — now lowers to SELECT,
     plus the `fcsel` assembler encoding it needs.
   - **mixed int/float** ops (`x*10`, `y<0`) built `fmul s,s,x` — now promote
     the int operand via SITOFP.
   - **float array subscript** `a[i]` typed its load with the raw array type
     (`float[4]`), so elements landed in GP registers → invalid `fadd s,s,x`;
     and a **global array** base loaded element 0 instead of decaying to its
     address. Both fixed (read and compound-assign `p[i]*=s` paths).
   - **FMA fusion use-after-free**: `acc += a*b` fused `FADD(FMUL)`→`FMA` and
     freed the old FADD without redirecting a loop-carried PHI's `phi_incoming`
     → SIGSEGV. Now rewires all uses; dot-product / sum-of-squares works.
   Verified at `-O`: `a+b*c=14`, float-array dot=240, `int8_dequant` **ok=1**,
   `swiglu` compiles (see below).
   **Remaining `-O` blockers** (kernels still use `-O0`; qlair identical):
   - a **cross-block global-address clobber** — a global's address register
     (e.g. `x22`) is cached across basic blocks but a loop condition's
     `cset x22` overwrites it, so the post-loop read hits ~0. Re-materialising
     the `adrp` per use fixes swiglu/`p[i]*=s` but exposes a register-allocator
     **over-subscription** (the re-emit clobbers a live temp), regressing
     `int8_dequant` — so it's not yet landed; the allocator fix is the gate.
   - a **`float` `if`-inside-loop** (argmax/max: `if(a[i]>mx) mx=a[i]`)
     infinite-loops at `-O`. Traced: the for-increment's `i++` load resolves
     its address to a stale i32 *value* register (the body's `&a[i]` GEP)
     instead of `&i`, so `i++` writes to `&a[i]` and `i` never advances. The
     **int** form (`if2`) is well-formed and works — it's specific to float
     `if`-bodies. `sqrtf`/`expf` at `-O` still open. Together these stall
     `rmsnorm`/`router_topk` at `-O`.

2. **`long += (long)(float)` loop-accumulate is wrong** (both backends): summing
   `float→long` conversions into a `long` accumulator over an array injects a
   spurious high bit (bit 40) each step; the same loop with an `int` accumulator
   is correct. The kernels here use `int` checksums to avoid it — the actual
   kernel math (validated by the exact-equality `ok` flag) is unaffected.

Neither defect blocks the kernels: their compute is correct at `-O0`, and qlair
gives cycle-accurate counts either way.

3. **`&global` as a call output arg fails on the 2nd consecutive call** — in the
   mixture kernels, `utofu_reg_mem(H1, rb, ..., &S1)` left the global `S1 = 0`
   while the first call's `&S0` worked. Workaround (used here): take the stadd
   into a **local**, then copy to the global (`utofu_stadd_t s1; reg_mem(...,&s1);
   S1 = s1;`). Same class as the earlier arg-lowering issues.
