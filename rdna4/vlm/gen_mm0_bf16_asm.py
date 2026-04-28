#!/usr/bin/env python3
"""
Generate a constrained RDNA4 mm0 BF16 GEMM kernel for external-code-object
benchmarking.

The current emitter intentionally produces HIP source with a generated
mainloop/schedule and a stable external symbol:

    gemm_mm0_bf16_asm(float *Y, const bf16_raw *W, const bf16_raw *X,
                      const float *bias)

This keeps the benchmark harness independent from HIPRTC while we iterate
toward a pure GCN `.s` emitter.  The schedule model is explicit here: register
families, LDS double buffering, WMMA order, and handoff points are generated
from tables rather than open-coded in the benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ACC = [(mi, ni) for mi in range(4) for ni in range(4)]


def wmma_block(a_name: str = "a", b_name: str = "b", indent: str = "            ") -> str:
    lines: list[str] = []
    for mi, ni in ACC:
        lines.append(
            f"{indent}cv{mi}{ni}=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12("
            f"{a_name}{mi},{b_name}{ni},cv{mi}{ni});"
        )
    return "\n".join(lines)


def emit_kernel(
    global_layout: str = "row",
    lds_stride: int = 128,
    lds_stride_a: int | None = None,
    lds_stride_b: int | None = None,
) -> str:
    if global_layout == "row":
        e_defs = """    int e0 = tid * 4 + 0;
    int e1 = tid * 4 + 1;
    int e2 = tid * 4 + 2;
    int e3 = tid * 4 + 3;"""
        layout_note = "row-per-thread"
    elif global_layout == "coalesced":
        e_defs = """    int e0 = tid +   0;
    int e1 = tid + 128;
    int e2 = tid + 256;
    int e3 = tid + 384;"""
        layout_note = "coalesced-row-chunks"
    else:
        raise ValueError(global_layout)
    stride_a = lds_stride if lds_stride_a is None else lds_stride_a
    stride_b = lds_stride if lds_stride_b is None else lds_stride_b
    if stride_a < 128:
        raise ValueError("lds_stride_a must be at least 128")
    if stride_b < 128:
        raise ValueError("lds_stride_b must be at least 128")

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
#include <string.h>

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
    __shared__ bf16x8 smA8[2*{stride_a}*4];
    __shared__ bf16x8 smB8[2*{stride_b}*4];

    // Global staging layout: {layout_note}.
{e_defs}
    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;
    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;

    smA8[(e0&3)*{stride_a}+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));
    smA8[(e1&3)*{stride_a}+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));
    smA8[(e2&3)*{stride_a}+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));
    smA8[(e3&3)*{stride_a}+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));
    smB8[(e0&3)*{stride_b}+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));
    smB8[(e1&3)*{stride_b}+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));
    smB8[(e2&3)*{stride_b}+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));
    smB8[(e3&3)*{stride_b}+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));
    lds_barrier();

    float8 z = {{0,0,0,0,0,0,0,0}};
{acc_decl}

    int buf = 0;
    for (int k = 0; k < 4608; k += 32) {{
        bf16x8 na0,na1,na2,na3,nb0,nb1,nb2,nb3;
        int has_next = k + 32 < 4608;
        if (has_next) {{
            int nk = k + 32;
            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));
            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));
            na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));
            na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));
            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));
            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));
            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));
            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));
        }}
        int base_a = buf * {stride_a * 4};
        int base_b = buf * {stride_b * 4};
        int a_base = wM * 64;
        int b_base = wN * 64;
        for (int kk0 = 0; kk0 < 32; kk0 += 16) {{
            int kslot = (kk0 + k_off) >> 3;
            bf16x8 a0 = smA8[base_a+kslot*{stride_a}+(a_base+0 +idx)];
            bf16x8 a1 = smA8[base_a+kslot*{stride_a}+(a_base+16+idx)];
            bf16x8 a2 = smA8[base_a+kslot*{stride_a}+(a_base+32+idx)];
            bf16x8 a3 = smA8[base_a+kslot*{stride_a}+(a_base+48+idx)];
            bf16x8 b0 = smB8[base_b+kslot*{stride_b}+(b_base+0 +idx)];
            bf16x8 b1 = smB8[base_b+kslot*{stride_b}+(b_base+16+idx)];
            bf16x8 b2 = smB8[base_b+kslot*{stride_b}+(b_base+32+idx)];
            bf16x8 b3 = smB8[base_b+kslot*{stride_b}+(b_base+48+idx)];
{wmma_block()}
        }}
        if (has_next) {{
            int nb = 1 - buf;
            int nbase_a = nb * {stride_a * 4};
            int nbase_b = nb * {stride_b * 4};
            smA8[nbase_a+(e0&3)*{stride_a}+r0] = na0; smA8[nbase_a+(e1&3)*{stride_a}+r1] = na1; smA8[nbase_a+(e2&3)*{stride_a}+r2] = na2; smA8[nbase_a+(e3&3)*{stride_a}+r3] = na3;
            smB8[nbase_b+(e0&3)*{stride_b}+r0] = nb0; smB8[nbase_b+(e1&3)*{stride_b}+r1] = nb1; smB8[nbase_b+(e2&3)*{stride_b}+r2] = nb2; smB8[nbase_b+(e3&3)*{stride_b}+r3] = nb3;
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


def emit_schedule(lds_stride_a: int = 128, lds_stride_b: int = 128) -> str:
    lds_kib = (2 * 4 * (lds_stride_a + lds_stride_b) * 16) // 1024
    return f"""# Generated RDNA4 mm0 BF16 schedule

Tile:
- CTA: 128x128x32
- waves: 4
- wave tile: 64x64
- LDS: {lds_kib} KiB total, transposed as kslot*stride+row
- LDS A stride: {lds_stride_a}
- LDS B stride: {lds_stride_b}

Register constraints:
- accumulators: 16 float8 vectors per wave
- A fragments: a0..a3
- B fragments: b0..b3
- next global fragments: na0..na3, nb0..nb3

Current generated backend:
- HIP source with generated WMMA order and explicit split barriers.
- External symbol: gemm_mm0_bf16_asm

Target handwritten backend:
1. Keep the same ABI and symbol.
2. Replace generated HIP mainloop with GCN:
   - global_load_b128 next A/B fragments early.
   - ds_store_b128 into inactive LDS buffer only after enough load latency.
   - ds_load_b128 small fragment groups.
   - v_wmma_f32_16x16x16_bf16 interleaved with ds_load and stores.
   - s_barrier_signal/wait only at buffer handoff.
3. Preserve 238-ish VGPR budget and zero spills.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="rdna4/vlm/generated")
    parser.add_argument("--basename", default="mm0_bf16_asm")
    parser.add_argument("--global-layout", choices=["row", "coalesced"], default="row")
    parser.add_argument("--lds-stride", type=int, default=128)
    parser.add_argument("--lds-stride-a", type=int, default=None)
    parser.add_argument("--lds-stride-b", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.basename}.hip").write_text(
        emit_kernel(args.global_layout, args.lds_stride, args.lds_stride_a, args.lds_stride_b),
        encoding="utf-8",
    )
    stride_a = args.lds_stride if args.lds_stride_a is None else args.lds_stride_a
    stride_b = args.lds_stride if args.lds_stride_b is None else args.lds_stride_b
    (out_dir / f"{args.basename}_schedule.md").write_text(
        emit_schedule(stride_a, stride_b), encoding="utf-8"
    )
    print(out_dir / f"{args.basename}.hip")
    print(out_dir / f"{args.basename}_schedule.md")


if __name__ == "__main__":
    main()
