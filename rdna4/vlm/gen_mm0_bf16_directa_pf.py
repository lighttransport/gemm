#!/usr/bin/env python3
"""Direct-A/LDS-B BF16 mm0 kernel WITH explicit X prefetch hoisted out of kk0.

Variant of gen_mm0_bf16_directa.py: hoists all 8 X fragments to top of K-iter
and adds a prefetch ring for K+1 X alongside W prefetch.  The intent is to
expose more ILP to the LLVM scheduler than the original directa kernel, which
emits per-quad s_wait_loadcnt that stalls each 4-WMMA group.

Baseline directa (pre-load inside kk0) = 65 TFLOP/s.
This swap-by-copy variant            = 115 TFLOP/s (measured).
Tried array-indexed ring                = 38 TFLOP/s (LDS-spilled — DON'T).
"""

from __future__ import annotations

import argparse
from pathlib import Path


ACC = [(mi, ni) for mi in range(4) for ni in range(4)]


def wmma_block_at(half: int, indent: str = "            ") -> str:
    lines: list[str] = []
    for mi, ni in ACC:
        lines.append(
            f"{indent}cv{mi}{ni}=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12("
            f"a{mi}{half},b{ni}{half},cv{mi}{ni});"
        )
    return "\n".join(lines)


def emit_kernel() -> str:
    acc_decl = (
        "    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;\n"
        "    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;"
    )
    stores = []
    for mi, ni in ACC:
        stores.append(
            f"    store_acc8_f32(Y, bias, cv{mi}{ni}, row + {mi * 16:2d}, "
            f"wave_n0 + {ni * 16:2d} + idx, 4608);"
        )

    initial_x_loads = "\n".join(
        f"    bf16x8 a{mi}{half} = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + {mi * 16:2d} + idx) * 4608 + 0 + {half * 16} + k_off));"
        for mi in range(4) for half in range(2)
    )
    next_x_loads = "\n".join(
        f"            na{mi}{half} = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + {mi * 16:2d} + idx) * 4608 + nk + {half * 16} + k_off));"
        for mi in range(4) for half in range(2)
    )
    next_x_decl = "        bf16x8 " + ", ".join(
        f"na{mi}{half}" for mi in range(4) for half in range(2)
    ) + ";"
    swap_x = "\n".join(
        f"            a{mi}{half} = na{mi}{half};"
        for mi in range(4) for half in range(2)
    )

    return f"""#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

typedef unsigned short bf16_raw;
typedef unsigned short bf16x8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void store_acc8_f32(float *Y, const float *bias, float8 acc, int row0, int col, int ld) {{
    float bv = bias ? bias[col] : 0.0f;
    Y[(size_t)(row0 + 0) * ld + col] = acc[0] + bv;
    Y[(size_t)(row0 + 1) * ld + col] = acc[1] + bv;
    Y[(size_t)(row0 + 2) * ld + col] = acc[2] + bv;
    Y[(size_t)(row0 + 3) * ld + col] = acc[3] + bv;
    Y[(size_t)(row0 + 4) * ld + col] = acc[4] + bv;
    Y[(size_t)(row0 + 5) * ld + col] = acc[5] + bv;
    Y[(size_t)(row0 + 6) * ld + col] = acc[6] + bv;
    Y[(size_t)(row0 + 7) * ld + col] = acc[7] + bv;
}}

__device__ __forceinline__ void lds_barrier_signal() {{ __asm__ __volatile__("s_barrier_signal -1" ::: "memory"); }}
__device__ __forceinline__ void lds_barrier_wait() {{ __asm__ __volatile__("s_barrier_wait 0xffff" ::: "memory"); }}
__device__ __forceinline__ void lds_barrier() {{ lds_barrier_signal(); lds_barrier_wait(); }}

extern "C" __global__ __launch_bounds__(128, 1)
void gemm_mm0_bf16_asm(float *Y, const bf16_raw *W, const bf16_raw *X,
                       const float *bias) {{
    int tid = threadIdx.x;
    int wave_id = tid >> 5;
    int lane = tid & 31;
    int wM = wave_id & 1;
    int wN = wave_id >> 1;
    int half = lane >> 4;
    int idx = lane & 15;
    int k_off = half * 8;
    int cta_m0 = blockIdx.y * 128;
    int cta_n0 = blockIdx.x * 128;
    __shared__ bf16x8 smB8[2*128*4];

    int e0 = tid * 4 + 0;
    int e1 = tid * 4 + 1;
    int e2 = tid * 4 + 2;
    int e3 = tid * 4 + 3;
    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;
    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;

    smB8[(e0&3)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));
    smB8[(e1&3)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));
    smB8[(e2&3)*128+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));
    smB8[(e3&3)*128+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));
    lds_barrier();

    float8 z = {{0,0,0,0,0,0,0,0}};
{acc_decl}

    int buf = 0;
    int a_base = wM * 64;
    int b_base = wN * 64;

    // Pre-load X for the first K-iter (k=0).
{initial_x_loads}

    for (int k = 0; k < 4608; k += 32) {{
        bf16x8 nb0,nb1,nb2,nb3;
{next_x_decl}
        int has_next = k + 32 < 4608;
        if (has_next) {{
            int nk = k + 32;
            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));
            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));
            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));
            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));
{next_x_loads}
        }}
        int base = buf * 512;
        // K-half 0 WMMAs
        {{
            int kslot0 = (0 + k_off) >> 3;
            bf16x8 b00 = smB8[base+kslot0*128+(b_base+0 +idx)];
            bf16x8 b10 = smB8[base+kslot0*128+(b_base+16+idx)];
            bf16x8 b20 = smB8[base+kslot0*128+(b_base+32+idx)];
            bf16x8 b30 = smB8[base+kslot0*128+(b_base+48+idx)];
{wmma_block_at(0)}
        }}
        // K-half 1 WMMAs
        {{
            int kslot1 = (16 + k_off) >> 3;
            bf16x8 b01 = smB8[base+kslot1*128+(b_base+0 +idx)];
            bf16x8 b11 = smB8[base+kslot1*128+(b_base+16+idx)];
            bf16x8 b21 = smB8[base+kslot1*128+(b_base+32+idx)];
            bf16x8 b31 = smB8[base+kslot1*128+(b_base+48+idx)];
{wmma_block_at(1)}
        }}
        if (has_next) {{
            int nb = 1 - buf;
            int nbase = nb * 512;
            smB8[nbase+(e0&3)*128+r0] = nb0;
            smB8[nbase+(e1&3)*128+r1] = nb1;
            smB8[nbase+(e2&3)*128+r2] = nb2;
            smB8[nbase+(e3&3)*128+r3] = nb3;
            lds_barrier_signal();
            buf = nb;
            lds_barrier_wait();
{swap_x}
        }}
    }}

    int wave_m0 = cta_m0 + wM * 64;
    int wave_n0 = cta_n0 + wN * 64;
    int row = wave_m0 + half * 8;
{chr(10).join(stores)}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="rdna4/vlm/generated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mm0_bf16_directa_pf.hip").write_text(emit_kernel(), encoding="utf-8")
    print(out_dir / "mm0_bf16_directa_pf.hip")


if __name__ == "__main__":
    main()
