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

## Build & run
```
~/work/clair/clair/build/clair -t arm64 -f bin rmsnorm.c -o r.elf
~/work/clair/clair/build/qlair -P -p r.elf
```

## IMPORTANT: use `-O0` (no `-O`) for float kernels
Two clair `-O` codegen defects were found while writing these (both **only** at
`-O`; `-O0` is correct, and qlair simulates identically):

1. **`-O` float codegen is broken** — every floating-point result is 0. `-O`
   routes ARM64 through the IR-pipeline backend (`src/ir/ir-codegen-arm64.cc`),
   whose float value-loading is non-functional (even `a+b`, and a `float`
   ternary, return 0); integer code is fine. `-O0` uses the working
   AssemblyEmitter backend (`arm64-codegen-visitor.cc`). A related
   use-after-free crash exists in the `optimizeFMA` pass: it rewrites
   `a + b*c` into a new FMA instruction but never redirects other references
   to the deleted FADD — a loop-carried PHI's `phi_incoming` (which is not in
   `operands`) then dangles → SIGSEGV. (Both are large fixes in the `-O`
   backend, kept separate.)

2. **`long += (long)(float)` loop-accumulate is wrong** (both backends): summing
   `float→long` conversions into a `long` accumulator over an array injects a
   spurious high bit (bit 40) each step; the same loop with an `int` accumulator
   is correct. The kernels here use `int` checksums to avoid it — the actual
   kernel math (validated by the exact-equality `ok` flag) is unaffected.

Neither defect blocks the kernels: their compute is correct at `-O0`, and qlair
gives cycle-accurate counts either way.
