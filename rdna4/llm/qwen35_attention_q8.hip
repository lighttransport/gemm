/* Q8/Q8 D=256 decode arithmetic from llama.cpp 1859b5209 fattn-vec.cuh.
 * Copyright (c) 2023-2026 The ggml authors. MIT license; see
 * reference_sampler.cpp for the complete license.
 * The runner stores Q8 codes and half-rounded scales in separate arrays.
 * Compile without fast-math. The reference compiler contracts the half2 V
 * accumulation; spell that FMA explicitly to prevent contracting rescale.
 */
typedef _Float16 q8_half;
typedef q8_half q8_half2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ q8_half2 q8_hpair(float x, float y) {
    return (q8_half2){(q8_half)x, (q8_half)y};
}
__device__ __forceinline__ q8_half2 q8_hsplat(float x) { return q8_hpair(x, x); }
__device__ __forceinline__ q8_half2 q8_hmul(q8_half2 x, q8_half2 y) {
    q8_half2 result;
    asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(result) : "v"(x), "v"(y));
    return result;
}
__device__ __forceinline__ q8_half2 q8_hfma(q8_half2 x, q8_half2 y, q8_half2 z) {
    q8_half2 result;
    asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(result) : "v"(x), "v"(y), "v"(z));
    return result;
}

__device__ __forceinline__ int q8_attention_splits(int length, int heads,
                                                 int nsm, int occupancy) {
    int tiles = (length + 255) / 256;
    if (length >= 16384)
        return min(128, max(occupancy, (length + 511) / 512));
    int result = min(occupancy, tiles), best = 0, waves_best = 0;
    int blocks_per_wave = nsm * occupancy;
    for (int trial = result; trial <= tiles; ++trial) {
        int blocks = heads * trial;
        int waves = (blocks + blocks_per_wave - 1) / blocks_per_wave;
        int efficiency = 100 * blocks / (waves * blocks_per_wave);
        if (best >= 95 && waves > waves_best) break;
        if (efficiency > best) {
            best = efficiency; waves_best = waves; result = trial;
        }
    }
    return result;
}

__device__ __forceinline__ float q8_attention_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor(x, offset, 32);
    return x;
}

extern "C" __global__ __launch_bounds__(128, 1)
void qwen35_attention_q8_decode(float *out, float *parts, float2 *meta,
        const float *q, const signed char *kc, const signed char *vc,
        const float *ks, const float *vs, const int *position,
        int heads, int kv_heads, int nsm, int occupancy, int forced_splits,
        int queries, int position_start) {
    int query = queries == 1 ? 0 : blockIdx.x;
    int length = position_start < 0 ? *position + 1 : position_start + query + 1;
    int splits = forced_splits ? forced_splits : q8_attention_splits(length, heads, nsm, occupancy);
    int head, split;
    if (queries == 1) {
        int flat = blockIdx.x;
        if (flat >= heads * splits) return;
        if (length >= 16384) { head = flat % heads; split = flat / heads; }
        else { head = flat / splits; split = flat % splits; }
    } else {
        split = blockIdx.y; head = blockIdx.z;
        if (split >= splits) return;
    }
    int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    int kh = head / (heads / kv_heads);
    int stride = kv_heads * 256, scale_stride = kv_heads * 8;
    q += (query * heads + head) * 256;
    __shared__ int qi[64];
    __shared__ float qd[8];
    __shared__ q8_half probabilities[1024];
    __shared__ float maxima[32], sums[32];
    if (warp == 0) {
        #pragma unroll
        for (int base = 0; base < 64; base += 32) {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = .0625f * q[4 * (base + lane) + i];
            float amax = fabsf(v[0]);
            #pragma unroll
            for (int i = 1; i < 4; ++i) amax = fmaxf(amax, fabsf(v[i]));
            #pragma unroll
            for (int offset = 4; offset; offset >>= 1)
                amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
            float d = amax / 127;
            int packed = 0;
            signed char *bytes = (signed char *)&packed;
            if (d != 0.0f) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) bytes[i] = roundf(v[i] / d);
            }
            qi[base + lane] = packed;
            if (lane % 8 == 0) qd[(base + lane) / 8] = d;
        }
    }
    __syncthreads();
    int qr[32]; float dr[8];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        qr[i] = qi[2 * i + lane % 2];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) dr[i] = qd[i];
    q8_half2 acc[4] = {};
    float maximum = -3.402823466e+38f / 2, sum = 0;
    int padded = (length + 255) / 256 * 256;
    for (int base = split * 128; base < padded; base += splits * 128) {
        float score = 0, next_max = maximum;
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            int row = base + warp * 32 + (lane & ~1) + pair;
            float dot = 0;
            if (row < length) {
                const int *k = (const int *)(kc + (size_t)row * stride + kh * 256);
                const float *d = ks + (size_t)row * scale_stride + kh * 8;
                float scale[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) scale[i] = d[i] * dr[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int word = 2 * i + lane % 2;
                    int product = __builtin_amdgcn_sudot4(true, k[word], true, qr[i], 0, false);
                    dot += scale[i / 4] * (float)product;
                }
            }
            dot += __shfl_xor(dot, 1, 32);
            if (row >= length) dot = -__builtin_inff();
            next_max = fmaxf(next_max, dot + (3.0f * .6931f));
            if (lane % 2 == pair) score = dot;
        }
        #pragma unroll
        for (int offset = 2; offset < 32; offset <<= 1)
            next_max = fmaxf(next_max, __shfl_xor(next_max, offset, 32));
        float rescale = expf(maximum - next_max);
        maximum = next_max;
        score = expf(score - maximum);
        sum = sum * rescale + score;
        probabilities[tid] = (q8_half)score;
        q8_half2 rescale_h = q8_hsplat(rescale);
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = q8_hmul(acc[i], rescale_h);
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int col = 0; col < 32; ++col) {
            int row = base + warp * 32 + col;
            q8_half2 p = q8_hsplat(probabilities[warp * 32 + col]);
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                int dim = tile * 128 + lane * 4;
                q8_half2 v0 = {}, v1 = {};
                if (row < length) {
                    const signed char *v = vc + (size_t)row * stride + kh * 256 + dim;
                    unsigned int packed = *(const unsigned int *)v;
                    q8_half2 d = q8_hsplat(vs[(size_t)row * scale_stride + kh * 8 + dim / 32]);
                    v0 = q8_hmul(d, q8_hpair((signed char)packed, (signed char)(packed >> 8)));
                    v1 = q8_hmul(d, q8_hpair((signed char)(packed >> 16), (signed char)(packed >> 24)));
                }
                acc[2 * tile] = q8_hfma(v0, p, acc[2 * tile]);
                acc[2 * tile + 1] = q8_hfma(v1, p, acc[2 * tile + 1]);
            }
        }
    }
    if (warp == 0) { maxima[lane] = -3.402823466e+38f / 2; sums[lane] = 0; }
    __syncthreads();
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    float next_max = maxima[lane];
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        next_max = fmaxf(next_max, __shfl_xor(next_max, offset, 32));
    float rescale = expf(maximum - next_max);
    q8_half2 rescale_h = q8_hsplat(rescale);
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        q8_half2 *p = (q8_half2 *)(probabilities + warp * 256 + tile * 128 + lane * 4);
        p[0] = q8_hmul(acc[2 * tile], rescale_h);
        p[1] = q8_hmul(acc[2 * tile + 1], rescale_h);
    }
    sum = q8_attention_sum(sum * rescale);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    sum = q8_attention_sum(sums[lane]);
    #pragma unroll
    for (int base = 0; base < 256; base += 128) {
        float value = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) value += (float)probabilities[w * 256 + base + tid];
        if (splits == 1) out[(query * heads + head) * 256 + base + tid] = value / sum;
        else parts[((query * heads + head) * splits + split) * 256 + base + tid] = value;
    }
    if (tid == 0 && splits > 1) meta[(query * heads + head) * splits + split] = make_float2(next_max, sum);
    (void)queries;
}

extern "C" __global__ __launch_bounds__(256, 1)
void qwen35_attention_q8_combine(float *out, const float *parts, const float2 *meta,
        const int *position, int heads, int nsm, int occupancy, int forced_splits) {
    int splits = forced_splits ? forced_splits : q8_attention_splits(*position + 1, heads, nsm, occupancy);
    if (splits == 1) return;
    int head = blockIdx.y * heads + blockIdx.x, tid = threadIdx.x;
    parts += head * splits * 256; meta += head * splits;
    extern __shared__ float2 shared_meta[];
    for (int i = tid; i < 2 * splits; i += 256) ((float *)shared_meta)[i] = ((const float *)meta)[i];
    __syncthreads();
    float maximum = shared_meta[0].x;
    for (int i = 1; i < splits; ++i) maximum = fmaxf(maximum, shared_meta[i].x);
    float numerator = 0, denominator = 0;
    for (int i = 0; i < splits; ++i) {
        float scale = expf(shared_meta[i].x - maximum);
        numerator += scale * parts[i * 256 + tid];
        denominator += scale * shared_meta[i].y;
    }
    out[head * 256 + tid] = numerator / denominator;
}
