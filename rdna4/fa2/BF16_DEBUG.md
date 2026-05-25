# bf16 flash-attn debug log

bf16 typed flash attention produces near-zero output (cos≈0, maxd≈0.01) only in
the full BQ=256 / 16-wave kernel. f16 and fp8 share the identical kernel structure
and hit cos≈0.9996. Run down of what was ruled out, in order:

| layer | test | bf16 result |
|---|---|---|
| WMMA instruction | `wmma_min_*` 16×16×16, HIPRTC **and** amdclang++ AOT | L1err=0 ✓ |
| compiler/opt | f16-type fa2 at -O2 / no-ffast | cos 0.99 ✓ |
| QK+CVT+V (1 wave) | `wmma_mid_rtc` hd128, P-store CVT, V-MMA | bf16==f16 ✓ |
| host upload | `qb=ne*(dt?1:2)` uploaded HALF the bf16 | **real bug, fixed** → dt==1?1:2 |
| CVT rounding | `+0x8000>>16` vs trunc `>>16` | neutral |
| P-store reload | v8 vector vs scalar gather | neutral |
| full 16-wave | bench_fa2 b16 / b16s | cos≈0, b16s sync=700 |

Conclusion: not the toolchain, not the instruction, not the ops. The instruction
is good under standalone HIPRTC — earlier "HIPRTC miscompile" call was wrong. One
real bug found (half-upload). Residual is a bf16-type × 16-wave interaction the
1-wave mid harness doesn't reach; needs the mid test grown to BQ=256/16-wave to
localize. bf16 ≈ 18.5 TF/s = f16, so no perf lost: f16 is the 16-bit path. open.

Tests: `make tests` builds wmma_min_rtc / wmma_min_aot_{f16,bf16} / wmma_mid_rtc;
each takes `f16`|`bf16`, prints L1err.
