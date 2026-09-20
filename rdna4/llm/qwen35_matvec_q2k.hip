/* Q2_K x Q8_1 MMVQ arithmetic from llama.cpp 1859b5209, gfx1201.
 * Copyright (c) 2023-2026 The ggml authors, MIT license (see
 * reference_sampler.cpp for the complete retained license).
 * Compile without fast-math. One 256-thread block computes one output row,
 * matching the reference's eight-warp reduction order. */
extern "C" __global__ void qwen35_quantize_q81(
        signed char *q, float *scales, const float *x, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= n) return;
    float v = x[i], a = fabsf(v);
    for (int offset = 16; offset; offset >>= 1)
        a = fmaxf(a, __shfl_xor(a, offset, 32));
    float d = a / 127.0f;
    q[i] = a == 0.0f ? 0 : (signed char)roundf(v / d);
    if (threadIdx.x == 0) scales[blockIdx.x] = (float)(_Float16)d;
}

extern "C" __global__ void qwen35_argmax_parts(const float *x, int n,
                                               float *scores, int *indices) {
    __shared__ float values[256];
    __shared__ int ids[256];
    int t=threadIdx.x, id=0, begin=blockIdx.x*4096;
    float value=-3.402823466e38f;
    for (int i=begin+t; i<n && i<begin+4096; i+=256)
        if (x[i]>value || (x[i]==value && i<id)) { value=x[i]; id=i; }
    values[t]=value; ids[t]=id;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (t<s && (values[t+s]>values[t] || (values[t+s]==values[t] && ids[t+s]<ids[t]))) {
            values[t]=values[t+s]; ids[t]=ids[t+s];
        }
        __syncthreads();
    }
    if (!t) { scores[blockIdx.x]=values[0]; indices[blockIdx.x]=ids[0]; }
}

extern "C" __global__ void qwen35_argmax_finish(const float *scores,
        const int *indices, int n, int *out) {
    __shared__ float values[256];
    __shared__ int ids[256];
    int t=threadIdx.x;
    values[t]=t<n ? scores[t] : -3.402823466e38f;
    ids[t]=t<n ? indices[t] : 0;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (t<s && (values[t+s]>values[t] || (values[t+s]==values[t] && ids[t+s]<ids[t]))) {
            values[t]=values[t+s]; ids[t]=ids[t+s];
        }
        __syncthreads();
    }
    if (!t) *out=ids[0];
}

extern "C" __global__ void qwen35_matvec_q2k(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = tid & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float sum = 0.0f;
    for (int b = tid / 16; b < blocks; b += 16) {
        const unsigned char *p = w + b * 84;
        const unsigned char *s = p + (iqs & ~7) + (iqs & 7) / 4;
        int v = ((const int *)(p + 16))[iqs];
        float sd = 0.0f, sm = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qb = b * 8 + 4 * (iqs / 8) + j;
            int u = ((const int *)(q + qb * 32))[iqs & 7];
            int z = __builtin_amdgcn_sudot4(true, (v >> (2*j)) & 0x03030303,
                                          true, u, 0, false);
            sd += scales[qb] * (z * (s[2*j] & 15));
            int m = s[2*j] >> 4;
            m |= m << 8; m |= m << 16;
            int zm = __builtin_amdgcn_sudot4(true, m, true, u, 0, false);
            sm += scales[qb] * zm;
        }
        float d = (float)*(const _Float16 *)(p + 80);
        float dm = (float)*(const _Float16 *)(p + 82);
        sum += d * sd - dm * sm;
    }
    __shared__ float partial[7][32];
    if (warp > 0) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int widx = 0; widx < 7; ++widx) sum += partial[widx][lane];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}

/* One warp per row, evaluated as eight virtual reference warps. Keeping
 * their accumulators separate preserves the eight-warp reduction order.
 * This improves tall 5120-column projections; the original kernel remains
 * preferable for the 17408-column down projections. */
extern "C" __global__ void qwen35_matvec_q2k_rows(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = lane & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float parts[8] = {};
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        for (int b = lane / 16 + 2 * vw; b < blocks; b += 16) {
            const unsigned char *p = w + b * 84;
            const unsigned char *s = p + (iqs & ~7) + (iqs & 7) / 4;
            int v = ((const int *)(p + 16))[iqs];
            float sd = 0.0f, sm = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int qb = b * 8 + 4 * (iqs / 8) + j;
                int u = ((const int *)(q + qb * 32))[iqs & 7];
                int z = __builtin_amdgcn_sudot4(true, (v >> (2*j)) & 0x03030303,
                                              true, u, 0, false);
                sd += scales[qb] * (z * (s[2*j] & 15));
                int m = s[2*j] >> 4;
                m |= m << 8; m |= m << 16;
                int zm = __builtin_amdgcn_sudot4(true, m, true, u, 0, false);
                sm += scales[qb] * zm;
            }
            float d = (float)*(const _Float16 *)(p + 80);
            float dm = (float)*(const _Float16 *)(p + 82);
            parts[vw] += d * sd - dm * sm;
        }
    }
    float sum = parts[0];
#pragma unroll
    for (int vw = 1; vw < 8; ++vw) sum += parts[vw];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}

template<int MAX_COUNT>
__device__ __forceinline__ void qwen35_matvec_q2k_multi(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    int lane=threadIdx.x&31, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    int blocks=cols/256, iqs=lane&15;
    const unsigned char *w=weights+(size_t)row*blocks*84;
    float sum[MAX_COUNT]={};
#pragma unroll
    for (int vw=0; vw<8; ++vw) {
        float partial[MAX_COUNT]={};
        for (int b=lane/16+2*vw; b<blocks; b+=16) {
            const unsigned char *p=w+b*84;
            const unsigned char *s=p+(iqs&~7)+(iqs&7)/4;
            int v=((const int *)(p+16))[iqs];
            float sd[MAX_COUNT]={}, sm[MAX_COUNT]={};
#pragma unroll
            for (int j=0; j<4; ++j) {
                int qb=b*8+4*(iqs/8)+j;
                int low=s[2*j]&15, high=(s[2*j]>>4)*0x01010101;
#pragma unroll
                for (int t=0; t<MAX_COUNT; ++t) if (t<count) {
                    int u=((const int *)(q+(size_t)t*cols+qb*32))[iqs&7];
                    int z=__builtin_amdgcn_sudot4(true,(v>>(2*j))&0x03030303,true,u,0,false);
                    int zm=__builtin_amdgcn_sudot4(true,high,true,u,0,false);
                    float scale=scales[(size_t)t*(cols/32)+qb];
                    sd[t]+=scale*(z*low); sm[t]+=scale*zm;
                }
            }
            float d=(float)*(const _Float16 *)(p+80), dm=(float)*(const _Float16 *)(p+82);
#pragma unroll
            for (int t=0; t<MAX_COUNT; ++t) if (t<count) partial[t]+=d*sd[t]-dm*sm[t];
        }
#pragma unroll
        for (int t=0; t<MAX_COUNT; ++t) if (t<count) sum[t]+=partial[t];
    }
#pragma unroll
    for (int t=0; t<MAX_COUNT; ++t) if (t<count) {
        for (int offset=16; offset; offset>>=1) sum[t]+=__shfl_xor(sum[t],offset,32);
        if (lane==0) out[(size_t)t*rows+row]=sum[t];
    }
}

extern "C" __global__ void qwen35_matvec_q2k_multi4(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_q2k_multi<4>(out,weights,q,scales,rows,cols,count);
}

extern "C" __global__ void qwen35_matvec_q2k_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_q2k_multi<8>(out,weights,q,scales,rows,cols,count);
}

/* The K=7 verifier evaluates exactly eight rows.  Keep the eight reference
 * virtual-warp partials, but finish one query's four dot products before
 * starting the next query.  This avoids keeping sixteen sd/sm accumulators
 * live across the inner loop while still decoding each weight block once. */
extern "C" __global__ void qwen35_matvec_q2k_fixed8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    (void)count;
    int lane=threadIdx.x&31, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    int blocks=cols/256, iqs=lane&15;
    const unsigned char *w=weights+(size_t)row*blocks*84;
    float sum[8]={};
#pragma unroll
    for (int vw=0; vw<8; ++vw) {
        float partial[8]={};
        for (int b=lane/16+2*vw; b<blocks; b+=16) {
            const unsigned char *p=w+b*84;
            const unsigned char *s=p+(iqs&~7)+(iqs&7)/4;
            int v=((const int *)(p+16))[iqs];
            float d=(float)*(const _Float16 *)(p+80);
            float dm=(float)*(const _Float16 *)(p+82);
#pragma unroll
            for (int t=0; t<8; ++t) {
                float sd=0.0f, sm=0.0f;
#pragma unroll
                for (int j=0; j<4; ++j) {
                    int qb=b*8+4*(iqs/8)+j;
                    int u=((const int *)(q+(size_t)t*cols+qb*32))[iqs&7];
                    int low=s[2*j]&15;
                    int high=(s[2*j]>>4)*0x01010101;
                    int z=__builtin_amdgcn_sudot4(true,
                        (v>>(2*j))&0x03030303,true,u,0,false);
                    int zm=__builtin_amdgcn_sudot4(true,high,true,u,0,false);
                    float scale=scales[(size_t)t*(cols/32)+qb];
                    sd+=scale*(z*low);
                    sm+=scale*zm;
                }
                partial[t]+=d*sd-dm*sm;
            }
        }
#pragma unroll
        for (int t=0; t<8; ++t) sum[t]+=partial[t];
    }
#pragma unroll
    for (int t=0; t<8; ++t) {
        for (int offset=16; offset; offset>>=1)
            sum[t]+=__shfl_xor(sum[t],offset,32);
        if (lane==0) out[(size_t)t*rows+row]=sum[t];
    }
}
