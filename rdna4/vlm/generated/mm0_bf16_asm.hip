#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef unsigned short bf16_raw;
typedef unsigned short bf16x8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void store_acc8_f32(float *Y, const float *bias, float8 acc, int row0, int col, int ld) {
    float bv = bias ? bias[col] : 0.0f;
    Y[(size_t)(row0 + 0) * ld + col] = acc[0] + bv;
    Y[(size_t)(row0 + 1) * ld + col] = acc[1] + bv;
    Y[(size_t)(row0 + 2) * ld + col] = acc[2] + bv;
    Y[(size_t)(row0 + 3) * ld + col] = acc[3] + bv;
    Y[(size_t)(row0 + 4) * ld + col] = acc[4] + bv;
    Y[(size_t)(row0 + 5) * ld + col] = acc[5] + bv;
    Y[(size_t)(row0 + 6) * ld + col] = acc[6] + bv;
    Y[(size_t)(row0 + 7) * ld + col] = acc[7] + bv;
}

__device__ __forceinline__ void lds_barrier_signal() { __asm__ __volatile__("s_barrier_signal -1" ::: "memory"); }
__device__ __forceinline__ void lds_barrier_wait() { __asm__ __volatile__("s_barrier_wait 0xffff" ::: "memory"); }
__device__ __forceinline__ void lds_barrier() { lds_barrier_signal(); lds_barrier_wait(); }

extern "C" __global__ __launch_bounds__(128, 1)
void gemm_mm0_bf16_asm(float *Y, const bf16_raw *W, const bf16_raw *X,
                       const float *bias) {
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
    __shared__ bf16x8 smA8[2*144*4];
    __shared__ bf16x8 smB8[2*144*4];

    // Global staging layout: row-per-thread.
    int e0 = tid * 4 + 0;
    int e1 = tid * 4 + 1;
    int e2 = tid * 4 + 2;
    int e3 = tid * 4 + 3;
    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;
    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;

    smA8[(e0&3)*144+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));
    smA8[(e1&3)*144+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));
    smA8[(e2&3)*144+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));
    smA8[(e3&3)*144+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));
    smB8[(e0&3)*144+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));
    smB8[(e1&3)*144+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));
    smB8[(e2&3)*144+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));
    smB8[(e3&3)*144+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));
    lds_barrier();

    float8 z = {0,0,0,0,0,0,0,0};
    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;
    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;

    int buf = 0;
    for (int k = 0; k < 4608; k += 32) {
        bf16x8 na0,na1,na2,na3,nb0,nb1,nb2,nb3;
        int has_next = k + 32 < 4608;
        if (has_next) {
            int nk = k + 32;
            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));
            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));
            na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));
            na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));
            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));
            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));
            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));
            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));
        }
        int base_a = buf * 576;
        int base_b = buf * 576;
        int a_base = wM * 64;
        int b_base = wN * 64;
        for (int kk0 = 0; kk0 < 32; kk0 += 16) {
            int kslot = (kk0 + k_off) >> 3;
            bf16x8 a0 = smA8[base_a+kslot*144+(a_base+0 +idx)];
            bf16x8 a1 = smA8[base_a+kslot*144+(a_base+16+idx)];
            bf16x8 a2 = smA8[base_a+kslot*144+(a_base+32+idx)];
            bf16x8 a3 = smA8[base_a+kslot*144+(a_base+48+idx)];
            bf16x8 b0 = smB8[base_b+kslot*144+(b_base+0 +idx)];
            bf16x8 b1 = smB8[base_b+kslot*144+(b_base+16+idx)];
            bf16x8 b2 = smB8[base_b+kslot*144+(b_base+32+idx)];
            bf16x8 b3 = smB8[base_b+kslot*144+(b_base+48+idx)];
            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);
            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);
            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);
            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);
            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);
            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);
            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);
            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);
            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);
            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);
            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);
            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);
            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);
            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);
            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);
            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);
        }
        if (has_next) {
            int nb = 1 - buf;
            int nbase_a = nb * 576;
            int nbase_b = nb * 576;
            smA8[nbase_a+(e0&3)*144+r0] = na0; smA8[nbase_a+(e1&3)*144+r1] = na1; smA8[nbase_a+(e2&3)*144+r2] = na2; smA8[nbase_a+(e3&3)*144+r3] = na3;
            smB8[nbase_b+(e0&3)*144+r0] = nb0; smB8[nbase_b+(e1&3)*144+r1] = nb1; smB8[nbase_b+(e2&3)*144+r2] = nb2; smB8[nbase_b+(e3&3)*144+r3] = nb3;
            lds_barrier_signal();
            buf = nb;
            lds_barrier_wait();
        }
    }

    int wave_m0 = cta_m0 + wM * 64;
    int wave_n0 = cta_n0 + wN * 64;
    int row = wave_m0 + half * 8;
    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);
    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);
    store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, 4608);
    store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, 4608);
    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);
    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);
    store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, 4608);
    store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, 4608);
    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);
    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);
    store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, 4608);
    store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, 4608);
    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);
    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);
    store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, 4608);
    store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, 4608);
}
