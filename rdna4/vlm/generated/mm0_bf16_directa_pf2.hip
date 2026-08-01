#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

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

    float8 z = {0,0,0,0,0,0,0,0};
    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;
    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;

    int buf = 0;
    int a_base = wM * 64;
    int b_base = wN * 64;

    // Pre-load X for k=0 (a) and k=32 (na).
    bf16x8 a00 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + 0 + 0 + k_off));
    bf16x8 a01 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + 0 + 16 + k_off));
    bf16x8 a10 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + 0 + 0 + k_off));
    bf16x8 a11 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + 0 + 16 + k_off));
    bf16x8 a20 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + 0 + 0 + k_off));
    bf16x8 a21 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + 0 + 16 + k_off));
    bf16x8 a30 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + 0 + 0 + k_off));
    bf16x8 a31 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + 0 + 16 + k_off));
    bf16x8 na00 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + 32 + 0 + k_off));
    bf16x8 na01 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + 32 + 16 + k_off));
    bf16x8 na10 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + 32 + 0 + k_off));
    bf16x8 na11 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + 32 + 16 + k_off));
    bf16x8 na20 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + 32 + 0 + k_off));
    bf16x8 na21 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + 32 + 16 + k_off));
    bf16x8 na30 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + 32 + 0 + k_off));
    bf16x8 na31 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + 32 + 16 + k_off));

    // K loop unrolled by 2.  4608 / 64 = 72 iters.  Inside each iter:
    //   - First half uses `a` (k slot), prefetches NA for k+64.
    //   - Second half uses `na` (k+32 slot), prefetches A for k+96.
    for (int k = 0; k < 4608; k += 64) {
        // === First half: K = k ===
        bf16x8 nb0,nb1,nb2,nb3;
        int has_next1 = k + 32 < 4608;
        if (has_next1) {
            int nk = k + 32;
            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));
            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));
            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));
            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));
        }
        // Prefetch X for k+64 into `na` (overwriting the values consumed at k+32).
        if (k + 64 < 4608) {
            na00 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + (k+64) + 0 + k_off));
            na01 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + (k+64) + 16 + k_off));
            na10 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + (k+64) + 0 + k_off));
            na11 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + (k+64) + 16 + k_off));
            na20 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + (k+64) + 0 + k_off));
            na21 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + (k+64) + 16 + k_off));
            na30 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + (k+64) + 0 + k_off));
            na31 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + (k+64) + 16 + k_off));
        }
        int base = buf * 512;
        {
            int kslot0 = (0 + k_off) >> 3;
            bf16x8 b00 = smB8[base+kslot0*128+(b_base+0 +idx)];
            bf16x8 b10 = smB8[base+kslot0*128+(b_base+16+idx)];
            bf16x8 b20 = smB8[base+kslot0*128+(b_base+32+idx)];
            bf16x8 b30 = smB8[base+kslot0*128+(b_base+48+idx)];
            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a00,b00,cv00);
            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a00,b10,cv01);
            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a00,b20,cv02);
            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a00,b30,cv03);
            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a10,b00,cv10);
            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a10,b10,cv11);
            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a10,b20,cv12);
            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a10,b30,cv13);
            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a20,b00,cv20);
            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a20,b10,cv21);
            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a20,b20,cv22);
            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a20,b30,cv23);
            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a30,b00,cv30);
            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a30,b10,cv31);
            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a30,b20,cv32);
            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a30,b30,cv33);
        }
        {
            int kslot1 = (16 + k_off) >> 3;
            bf16x8 b01 = smB8[base+kslot1*128+(b_base+0 +idx)];
            bf16x8 b11 = smB8[base+kslot1*128+(b_base+16+idx)];
            bf16x8 b21 = smB8[base+kslot1*128+(b_base+32+idx)];
            bf16x8 b31 = smB8[base+kslot1*128+(b_base+48+idx)];
            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a01,b01,cv00);
            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a01,b11,cv01);
            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a01,b21,cv02);
            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a01,b31,cv03);
            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a11,b01,cv10);
            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a11,b11,cv11);
            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a11,b21,cv12);
            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a11,b31,cv13);
            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a21,b01,cv20);
            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a21,b11,cv21);
            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a21,b21,cv22);
            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a21,b31,cv23);
            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a31,b01,cv30);
            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a31,b11,cv31);
            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a31,b21,cv32);
            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a31,b31,cv33);
        }
        if (has_next1) {
            int nb = 1 - buf;
            int nbase = nb * 512;
            smB8[nbase+(e0&3)*128+r0] = nb0;
            smB8[nbase+(e1&3)*128+r1] = nb1;
            smB8[nbase+(e2&3)*128+r2] = nb2;
            smB8[nbase+(e3&3)*128+r3] = nb3;
            lds_barrier_signal();
            buf = nb;
            lds_barrier_wait();
        }

        // === Second half: K = k+32 ===
        if (!has_next1) break;  // last K block; only first half ran.
        bf16x8 mb0,mb1,mb2,mb3;
        int has_next2 = k + 64 < 4608;
        if (has_next2) {
            int nk2 = k + 64;
            mb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk2 + c0));
            mb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk2 + c1));
            mb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk2 + c2));
            mb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk2 + c3));
        }
        // Prefetch X for k+96 into `a` (overwriting the values consumed at k).
        if (k + 96 < 4608) {
            a00 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + (k+96) + 0 + k_off));
            a01 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base +  0 + idx) * 4608 + (k+96) + 16 + k_off));
            a10 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + (k+96) + 0 + k_off));
            a11 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 16 + idx) * 4608 + (k+96) + 16 + k_off));
            a20 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + (k+96) + 0 + k_off));
            a21 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 32 + idx) * 4608 + (k+96) + 16 + k_off));
            a30 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + (k+96) + 0 + k_off));
            a31 = *((const bf16x8 *)(X + (size_t)(cta_m0 + a_base + 48 + idx) * 4608 + (k+96) + 16 + k_off));
        }
        base = buf * 512;
        {
            int kslot0 = (0 + k_off) >> 3;
            bf16x8 b00 = smB8[base+kslot0*128+(b_base+0 +idx)];
            bf16x8 b10 = smB8[base+kslot0*128+(b_base+16+idx)];
            bf16x8 b20 = smB8[base+kslot0*128+(b_base+32+idx)];
            bf16x8 b30 = smB8[base+kslot0*128+(b_base+48+idx)];
            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na00,b00,cv00);
            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na00,b10,cv01);
            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na00,b20,cv02);
            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na00,b30,cv03);
            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na10,b00,cv10);
            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na10,b10,cv11);
            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na10,b20,cv12);
            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na10,b30,cv13);
            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na20,b00,cv20);
            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na20,b10,cv21);
            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na20,b20,cv22);
            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na20,b30,cv23);
            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na30,b00,cv30);
            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na30,b10,cv31);
            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na30,b20,cv32);
            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na30,b30,cv33);
        }
        {
            int kslot1 = (16 + k_off) >> 3;
            bf16x8 b01 = smB8[base+kslot1*128+(b_base+0 +idx)];
            bf16x8 b11 = smB8[base+kslot1*128+(b_base+16+idx)];
            bf16x8 b21 = smB8[base+kslot1*128+(b_base+32+idx)];
            bf16x8 b31 = smB8[base+kslot1*128+(b_base+48+idx)];
            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na01,b01,cv00);
            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na01,b11,cv01);
            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na01,b21,cv02);
            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na01,b31,cv03);
            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na11,b01,cv10);
            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na11,b11,cv11);
            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na11,b21,cv12);
            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na11,b31,cv13);
            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na21,b01,cv20);
            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na21,b11,cv21);
            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na21,b21,cv22);
            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na21,b31,cv23);
            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na31,b01,cv30);
            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na31,b11,cv31);
            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na31,b21,cv32);
            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(na31,b31,cv33);
        }
        if (has_next2) {
            int nb = 1 - buf;
            int nbase = nb * 512;
            smB8[nbase+(e0&3)*128+r0] = mb0;
            smB8[nbase+(e1&3)*128+r1] = mb1;
            smB8[nbase+(e2&3)*128+r2] = mb2;
            smB8[nbase+(e3&3)*128+r3] = mb3;
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
