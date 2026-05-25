# fa2 findings

Goal: port gau-nernst.github.io/fa-5090 (94% on 5090) to RDNA4. Best lever
that transfers is the LDS swizzle. bf16 27.7→36.0 TF/s (+30%, 19% of 190 peak).
fp8 29 TF/s (8%), matching fa3 ceiling. The 80–90% target needs async-MMA + big
register file (Hopper/5090); RDNA4 synchronous WMMA + thin BKV=32 tiles cap us.
Don't chase fp8>30. f16 swizzle+db is the keeper (37 TF/s, db real +5%).

bf16 path: __builtin_amdgcn_wmma_..._bf16 crashes sync=700 (illegal addr) at
S>=512 / grid.y>1, zeros at S=256; identical f16 module clean to 4096. So it's
the bf16 builtin under HIPRTC -O3 -ffast-math, not the logic. b16* run fp16 type.
NOTE: 700 wedges the GPU queue — needs reset before reruns.

UPDATE: bf16 chased to ground, still broken — f16 is the keeper 16-bit path.
Tried: -O2 + no ffast-math (cos=0, no crash), -O3 ffast (sync=700), signed vs
unsigned short, explicit f32→bf16 CVT for P-store, flux2's (short)(b>>16) form.
CORRECTION: builtin is NOT broken. wmma_min_{rtc,aot}: one 16x16x16 bf16 WMMA
passes L1err=0 under BOTH HIPRTC and amdclang++ AOT at -O2/-O3. f16-type fa2 at
-O2/no-ffast also cos=0.99, so opt-level/exp2 are clear. Failure is bf16 *type*
only inside the full fa2 kernel (Q/K/V/P LDS round-trip), not the toolchain, not
the instruction. f16 is the 16-bit perf proxy. fp8 d256 still keeper.
Tests: wmma_min.h + _rtc/_aot.cpp (16x16x16, pass), wmma_mid_rtc (hd128 QK+CVT+V
1-wave, bf16==f16). REAL BUG FOUND: run() qb=ne*(dt?1:2) uploaded half the bf16
(dt2 truthy); fixed to dt==1?1:2. CVT trunc vs round, O2/O3, all neutral. Residual:
b16 cos~0 maxd~0.01 (near-zero noise) only in full 16-wave kernel; mid 1-wave exact.
bf16 perf 18.5 TF/s ~= f16, so no perf lost keeping f16; bf16 accuracy still open.
Fat-matrix (--dim): fp8 d128→256 = 20.5→22 TF/s, d512 regresses (11.5, LDS-bound);
16-bit OOMs LDS past d128 (smK/smV+smP>64KB). fp8 d256 is the keeper.
