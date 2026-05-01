#!/usr/bin/env python3
"""Generate a direct-A/LDS-B BF16 mm0 kernel for RDNA4 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


ACC = [(mi, ni) for mi in range(4) for ni in range(4)]


def wmma_block(indent: str = "            ") -> str:
    lines: list[str] = []
    for mi, ni in ACC:
        lines.append(
            f"{indent}cv{mi}{ni}=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12("
            f"a{mi},b{ni},cv{mi}{ni});"
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
    for (int k = 0; k < 4608; k += 32) {{
        bf16x8 nb0,nb1,nb2,nb3;
        int has_next = k + 32 < 4608;
        if (has_next) {{
            int nk = k + 32;
            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));
            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));
            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));
            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));
        }}
        int base = buf * 512;
        for (int kk0 = 0; kk0 < 32; kk0 += 16) {{
            int kslot = (kk0 + k_off) >> 3;
            int kcol = k + kk0 + k_off;
            bf16x8 a0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + kcol));
            bf16x8 a1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + kcol));
            bf16x8 a2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + kcol));
            bf16x8 a3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + kcol));
            bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];
            bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];
            bf16x8 b2 = smB8[base+kslot*128+(b_base+32+idx)];
            bf16x8 b3 = smB8[base+kslot*128+(b_base+48+idx)];
{wmma_block()}
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
        }}
    }}

    int wave_m0 = cta_m0 + wM * 64;
    int wave_n0 = cta_n0 + wN * 64;
    int row = wave_m0 + half * 8;
{chr(10).join(stores)}
}}
"""


def emit_schedule() -> str:
    return """# Generated RDNA4 mm0 BF16 direct-A schedule

This diagnostic keeps operand A in VGPRs loaded directly from global memory and
double-buffers only operand B through LDS.  It intentionally spends more global
bandwidth on A inside the CTA to reduce LDS traffic and wait pressure, matching
the major pattern observed in the hipBLASLt mm0 dump.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="rdna4/vlm/generated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mm0_bf16_directa.hip").write_text(emit_kernel(), encoding="utf-8")
    (out_dir / "mm0_bf16_directa_schedule.md").write_text(emit_schedule(), encoding="utf-8")
    print(out_dir / "mm0_bf16_directa.hip")
    print(out_dir / "mm0_bf16_directa_schedule.md")


if __name__ == "__main__":
    main()
