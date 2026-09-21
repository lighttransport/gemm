/* Q8/Q8 D=256 decode arithmetic from llama.cpp 1859b5209 fattn-vec.cuh.
 * Copyright (c) 2023-2026 The ggml authors. MIT license; see
 * reference_sampler.cpp for the complete license.
 * The runner stores Q8 codes and half-rounded scales in separate arrays.
 * Compile without fast-math. The reference compiler contracts the half2 V
 * accumulation; spell that FMA explicitly to prevent contracting rescale.
 */
typedef _Float16 q8_half;
typedef q8_half q8_half2 __attribute__((ext_vector_type(2)));
typedef q8_half q8_half8 __attribute__((ext_vector_type(8)));
typedef float q8_float8 __attribute__((ext_vector_type(8)));
typedef int q8_int8 __attribute__((ext_vector_type(8)));
typedef unsigned int q8_uint2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ unsigned int q8_pack4(signed char a, signed char b,
                                                  signed char c, signed char d) {
    return (unsigned int)(unsigned char)a |
           ((unsigned int)(unsigned char)b << 8) |
           ((unsigned int)(unsigned char)c << 16) |
           ((unsigned int)(unsigned char)d << 24);
}
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
    if (length >= 16384) {
        int long_splits = (length + 511) / 512;
        /* On gfx1201, the 64K window is faster with fewer, larger K/V
         * partitions: the saved split/combine work outweighs the lower
         * block count while each partition still fits the long-context
         * register budget. Keep the 128-way schedule outside the validated
         * 64K serving window. */
        if (length >= 65536 && length < 98304)
            long_splits = min(long_splits, 64);
        return min(128, max(occupancy, long_splits));
    }
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

/* Long-context prefill kernel for gfx1201.  Eight pairs of waves process eight
 * 16-query tiles while sharing each 16-row K/V tile.  The two waves in a pair
 * split D=256 in half.  This keeps the Q and output accumulators below the
 * register cliff of the older one-wave D=256 kernel and reads the resident
 * Q8/Q8 cache directly, so no O(context) unpack pass is needed per layer.
 *
 * Q uses the same per-32-value Q8_1 quantization contract as the runner's
 * packed-F16 flash path. K/V scales were rounded to F16 when the cache row was
 * written; casting dequantized V to F16 therefore matches that input contract.
 * Grid: (heads, ceil(queries/128)), block: 512. */
extern "C" __global__ __launch_bounds__(512, 1)
void qwen35_attention_q8_prefill_wmma(float *out, const float *q,
        const signed char *kc, const signed char *vc,
        const q8_half *ks, const q8_half *vs, int heads, int kv_heads,
        int queries, int position_start) {
    constexpr int BQ = 128, BKV = 16, WMMA_K = 16, HD = 256, HALF_HD = 128;
    int head = blockIdx.x;
    int q0 = blockIdx.y * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int group = wid >> 1;
    int part = wid & 1;
    int half = lid >> 4;
    int idx = lid & 15;
    int query = q0 + group * 16 + idx;
    int dbase = part * HALF_HD;
    int kh = head / (heads / kv_heads);
    int stride = kv_heads * HD;
    int scale_stride = kv_heads * (HD / 32);

    __shared__ signed char sm_k[BKV * HD];
    __shared__ q8_half sm_v[HD * BKV];
    __shared__ float sm_score[16 * WMMA_K * WMMA_K];
    __shared__ q8_half sm_p[8 * WMMA_K * WMMA_K];

    q8_uint2 qcode[8];
    float qscale[4];
    const float *qh = q + ((size_t)query * heads + head) * HD + dbase;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float x[2][8];
        signed char code[2][8];
        float amax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int d = g * 32 + k * 16 + half * 8 + i;
                float value = query < queries ? .0625f * qh[d] : 0.0f;
                x[k][i] = value;
                amax = fmaxf(amax, fabsf(value));
            }
        }
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        float d = amax / 127.0f;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int quant = d != 0.0f ? (int)roundf(x[k][i] / d) : 0;
                quant = max(-127, min(127, quant));
                code[k][i] = (signed char)quant;
            }
            qcode[2 * g + k][0] = q8_pack4(code[k][0], code[k][1], code[k][2], code[k][3]);
            qcode[2 * g + k][1] = q8_pack4(code[k][4], code[k][5], code[k][6], code[k][7]);
        }
        qscale[g] = d;
    }

    /* llama.cpp's Q8 value path also accumulates PV in packed F16.  Using the
     * native F16-accumulate WMMA form halves the accumulator register file. */
    q8_half8 acc[8];
    #pragma unroll
    for (int kb = 0; kb < 8; ++kb)
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[kb][i] = (q8_half)0.0f;
    float maximum[8], sum[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        maximum[i] = -3.402823466e+38f / 2;
        sum[i] = 0.0f;
    }

    int max_length = position_start + min(queries, q0 + BQ);
    int tiles = (max_length + BKV - 1) / BKV;
    for (int tile = 0; tile < tiles; ++tile) {
        int kv0 = tile * BKV;
        #pragma unroll
        for (int e = tid; e < BKV * HD; e += 512) {
            int row = e / HD;
            int dim = e - row * HD;
            int pos = kv0 + row;
            signed char kval = 0;
            float vval = 0.0f;
            if (pos < max_length) {
                size_t code_index = (size_t)pos * stride + kh * HD + dim;
                size_t scale_index = (size_t)pos * scale_stride + kh * 8 + dim / 32;
                kval = kc[code_index];
                vval = (float)vc[code_index] * vs[scale_index];
            }
            sm_k[row * HD + dim] = kval;
            sm_v[dim * BKV + row] = (q8_half)vval;
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < BKV / WMMA_K; ++sub) {
            q8_float8 partial = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                q8_int8 integer = {0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    q8_uint2 key;
                    int key_base = (sub * WMMA_K + idx) * HD + dbase +
                                   g * 32 + k * 16 + half * 8;
                    key[0] = q8_pack4(sm_k[key_base + 0], sm_k[key_base + 1],
                                      sm_k[key_base + 2], sm_k[key_base + 3]);
                    key[1] = q8_pack4(sm_k[key_base + 4], sm_k[key_base + 5],
                                      sm_k[key_base + 6], sm_k[key_base + 7]);
                    integer = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, qcode[2 * g + k], true, key, integer, false);
                }
                int kpos = kv0 + sub * WMMA_K + idx;
                float kd = kpos < max_length ?
                    ks[(size_t)kpos * scale_stride + kh * 8 + part * 4 + g] : 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float qd = __shfl(qscale[g], half * 8 + i, 32);
                    partial[i] += (float)integer[i] * qd * kd;
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                sm_score[wid * 256 + lid * 8 + i] = partial[i];
            __syncthreads();

            q8_float8 score;
            int score_base = (group * 2) * 256 + lid * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float value = sm_score[score_base + i] + sm_score[score_base + 256 + i];
                int qrow = q0 + group * 16 + half * 8 + i;
                int kpos = kv0 + sub * WMMA_K + idx;
                if (qrow >= queries || kpos >= max_length ||
                    kpos > position_start + qrow) value = -__builtin_inff();
                score[i] = value;
            }

            q8_half8 alpha_h;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float row_max = score[i];
                row_max = fmaxf(row_max, __shfl_xor(row_max, 1, 32));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 2, 32));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 4, 32));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 8, 32));
                float next_max = fmaxf(maximum[i], row_max);
                float alpha = maximum[i] < -1.0e30f ? 0.0f :
                    __expf(maximum[i] - next_max);
                float probability = __expf(score[i] - next_max);
                float tile_sum = probability;
                tile_sum += __shfl_xor(tile_sum, 1, 32);
                tile_sum += __shfl_xor(tile_sum, 2, 32);
                tile_sum += __shfl_xor(tile_sum, 4, 32);
                tile_sum += __shfl_xor(tile_sum, 8, 32);
                sum[i] = sum[i] * alpha + tile_sum;
                maximum[i] = next_max;
                score[i] = probability;
                alpha_h[i] = (q8_half)alpha;
            }
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) acc[kb] *= alpha_h;
            if (part == 0) {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    sm_p[group * 256 + (half * 8 + i) * 16 + idx] =
                        (q8_half)score[i];
            }
            __syncthreads();
            q8_half8 probability;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                probability[i] = sm_p[group * 256 + idx * 16 + half * 8 + i];
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                q8_half8 value;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    value[i] = sm_v[(dbase + kb * 16 + idx) * BKV +
                                        sub * WMMA_K + half * 8 + i];
                acc[kb] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                    probability, value, acc[kb]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int qrow = q0 + group * 16 + half * 8 + i;
            int dim = dbase + kb * 16 + idx;
            if (qrow < queries)
                out[((size_t)qrow * heads + head) * HD + dim] =
                    (float)acc[kb][i] /
                    (sum[i] > 0.0f ? sum[i] : 1.0f);
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 1)
void qwen35_attention_q8_decode(float *out, float *parts, float2 *meta,
        const float *q, const signed char *kc, const signed char *vc,
        const q8_half *ks, const q8_half *vs, const int *position,
    int heads, int kv_heads, int nsm, int occupancy, int forced_splits,
    int queries, int position_start) {
    int query = queries == 1 ? 0 : blockIdx.x;
    int length = position_start < 0 ? position[query] + 1 :
                                      position_start + query + 1;
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
                const q8_half *d = ks + (size_t)row * scale_stride + kh * 8;
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

/* Bitwise-equivalent 6:1 GQA scheduling.  Three query heads that share a KV
 * head occupy one 12-wave block; each head still has the reference kernel's
 * four physical waves, accumulator order, and shared reduction layout. */
extern "C" __global__ __launch_bounds__(384, 1)
void qwen35_attention_q8_decode_gqa3(float *out, float *parts, float2 *meta,
        const float *q, const signed char *kc, const signed char *vc,
        const q8_half *ks, const q8_half *vs, const int *position,
        int heads, int kv_heads, int nsm, int occupancy, int forced_splits,
        int queries, int position_start, int adaptive_long_context) {
    if (queries != 1 || heads != 6 * kv_heads) return;
    int length = position_start < 0 ? position[0] + 1 : position_start + 1;
    if (adaptive_long_context && length >= 16384) return;
    int splits = forced_splits ? forced_splits :
        q8_attention_splits(length, heads, nsm, occupancy);
    int flat = blockIdx.x, group, split;
    if (flat >= 2 * kv_heads * splits) return;
    if (length >= 16384) {
        group = flat % (2 * kv_heads);
        split = flat / (2 * kv_heads);
    } else {
        group = flat / splits;
        split = flat % splits;
    }
    int kh = group >> 1, head_base = kh * 6 + (group & 1) * 3;
    int lane = threadIdx.x, wave_id = threadIdx.y;
    int slot = wave_id >> 2, warp = wave_id & 3;
    int tid = warp * 32 + lane, head = head_base + slot;
    int stride = kv_heads * 256, scale_stride = kv_heads * 8;
    q += head * 256;
    __shared__ int qi[3][64];
    __shared__ float qd[3][8];
    __shared__ q8_half probabilities[3][1024];
    __shared__ float maxima[3][32], sums[3][32];
    if (warp == 0) {
        #pragma unroll
        for (int base = 0; base < 64; base += 32) {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = .0625f * q[4 * (base + lane) + i];
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
            qi[slot][base + lane] = packed;
            if (lane % 8 == 0) qd[slot][(base + lane) / 8] = d;
        }
    }
    __syncthreads();
    int qr[32];
    float dr[8];
    #pragma unroll
    for (int i = 0; i < 32; ++i) qr[i] = qi[slot][2 * i + lane % 2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) dr[i] = qd[slot][i];
    q8_half2 acc[4] = {};
    float maximum = -3.402823466e+38f / 2, sum = 0.0f;
    int padded = (length + 255) / 256 * 256;
    for (int base = split * 128; base < padded; base += splits * 128) {
        float score = 0.0f, next_max = maximum;
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            int row = base + warp * 32 + (lane & ~1) + pair;
            float dot = 0.0f;
            if (row < length) {
                const int *k = (const int *)(kc + (size_t)row * stride + kh * 256);
                const q8_half *d = ks + (size_t)row * scale_stride + kh * 8;
                float scale[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) scale[i] = d[i] * dr[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int word = 2 * i + lane % 2;
                    int product = __builtin_amdgcn_sudot4(
                        true, k[word], true, qr[i], 0, false);
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
        probabilities[slot][tid] = (q8_half)score;
        q8_half2 rescale_h = q8_hsplat(rescale);
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = q8_hmul(acc[i], rescale_h);
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int col = 0; col < 32; ++col) {
            int row = base + warp * 32 + col;
            q8_half2 p = q8_hsplat(probabilities[slot][warp * 32 + col]);
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                int dim = tile * 128 + lane * 4;
                q8_half2 v0 = {}, v1 = {};
                if (row < length) {
                    const signed char *v = vc + (size_t)row * stride + kh * 256 + dim;
                    unsigned int packed = *(const unsigned int *)v;
                    q8_half2 d = q8_hsplat(
                        vs[(size_t)row * scale_stride + kh * 8 + dim / 32]);
                    v0 = q8_hmul(d, q8_hpair(
                        (signed char)packed, (signed char)(packed >> 8)));
                    v1 = q8_hmul(d, q8_hpair(
                        (signed char)(packed >> 16), (signed char)(packed >> 24)));
                }
                acc[2 * tile] = q8_hfma(v0, p, acc[2 * tile]);
                acc[2 * tile + 1] = q8_hfma(v1, p, acc[2 * tile + 1]);
            }
        }
    }
    if (warp == 0) {
        maxima[slot][lane] = -3.402823466e+38f / 2;
        sums[slot][lane] = 0.0f;
    }
    __syncthreads();
    if (lane == 0) maxima[slot][warp] = maximum;
    __syncthreads();
    float next_max = maxima[slot][lane];
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        next_max = fmaxf(next_max, __shfl_xor(next_max, offset, 32));
    float rescale = expf(maximum - next_max);
    q8_half2 rescale_h = q8_hsplat(rescale);
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        q8_half2 *p = (q8_half2 *)(probabilities[slot] +
            warp * 256 + tile * 128 + lane * 4);
        p[0] = q8_hmul(acc[2 * tile], rescale_h);
        p[1] = q8_hmul(acc[2 * tile + 1], rescale_h);
    }
    sum = q8_attention_sum(sum * rescale);
    if (lane == 0) sums[slot][warp] = sum;
    __syncthreads();
    sum = q8_attention_sum(sums[slot][lane]);
    #pragma unroll
    for (int base = 0; base < 256; base += 128) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w)
            value += (float)probabilities[slot][w * 256 + base + tid];
        if (splits == 1) out[head * 256 + base + tid] = value / sum;
        else parts[(head * splits + split) * 256 + base + tid] = value;
    }
    if (tid == 0 && splits > 1)
        meta[head * splits + split] = make_float2(next_max, sum);
}

/* Bitwise-equivalent 3-head GQA kernel with explicit K/V reuse.  Four waves
 * process the reference kernel's four row stripes for three heads at once.
 * Each head retains the scalar dot, online-softmax, packed-F16 accumulation,
 * and reduction order, while every K/V word is loaded only once. */
extern "C" __global__ __launch_bounds__(128, 1)
void qwen35_attention_q8_decode_gqa3_reuse(float *out, float *parts,
        float2 *meta, const float *q, const signed char *kc,
        const signed char *vc, const q8_half *ks, const q8_half *vs,
        const int *position, int heads, int kv_heads, int nsm,
        int occupancy, int forced_splits, int queries,
        int position_start, int adaptive_long_context) {
    if (queries != 1 || heads != 6 * kv_heads) return;
    int length = position_start < 0 ? position[0] + 1 : position_start + 1;
    if (adaptive_long_context && length < 16384) return;
    int splits = forced_splits ? forced_splits :
        q8_attention_splits(length, heads, nsm, occupancy);
    int flat = blockIdx.x, group, split;
    if (flat >= 2 * kv_heads * splits) return;
    if (length >= 16384) {
        group = flat % (2 * kv_heads);
        split = flat / (2 * kv_heads);
    } else {
        group = flat / splits;
        split = flat % splits;
    }
    int kh = group >> 1, head_base = kh * 6 + (group & 1) * 3;
    int lane = threadIdx.x, warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int stride = kv_heads * 256, scale_stride = kv_heads * 8;
    __shared__ int qi[3][64];
    __shared__ float qd[3][8];
    __shared__ q8_half probabilities[3][1024];
    __shared__ float maxima[3][32], sums_shared[3][32];
    if (warp < 3) {
        const float *qh = q + (head_base + warp) * 256;
        #pragma unroll
        for (int base = 0; base < 64; base += 32) {
            float value[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                value[i] = .0625f * qh[4 * (base + lane) + i];
            float amax = fabsf(value[0]);
            #pragma unroll
            for (int i = 1; i < 4; ++i)
                amax = fmaxf(amax, fabsf(value[i]));
            #pragma unroll
            for (int offset = 4; offset; offset >>= 1)
                amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
            float d = amax / 127;
            int packed = 0;
            signed char *bytes = (signed char *)&packed;
            if (d != 0.0f) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    bytes[i] = roundf(value[i] / d);
            }
            qi[warp][base + lane] = packed;
            if (lane % 8 == 0) qd[warp][(base + lane) / 8] = d;
        }
    }
    __syncthreads();

    q8_half2 acc[3][4] = {};
    float maximum[3], sum[3];
    #pragma unroll
    for (int h = 0; h < 3; ++h) {
        maximum[h] = -3.402823466e+38f / 2;
        sum[h] = 0.0f;
    }
    int padded = (length + 255) / 256 * 256;
    for (int base = split * 128; base < padded; base += splits * 128) {
        float score[3] = {};
        float next_max[3];
        #pragma unroll
        for (int h = 0; h < 3; ++h) next_max[h] = maximum[h];
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            int row = base + warp * 32 + (lane & ~1) + pair;
            float dot[3] = {};
            if (row < length) {
                const int *key = (const int *)(kc +
                    (size_t)row * stride + kh * 256);
                const q8_half *key_scale = ks +
                    (size_t)row * scale_stride + kh * 8;
                #pragma unroll
                for (int g = 0; g < 8; ++g) {
                    float scale[3];
                    #pragma unroll
                    for (int h = 0; h < 3; ++h)
                        scale[h] = key_scale[g] * qd[h][g];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int word = 8 * g + 2 * j + lane % 2;
                        int key_word = key[word];
                        #pragma unroll
                        for (int h = 0; h < 3; ++h) {
                            int product = __builtin_amdgcn_sudot4(
                                true, key_word, true, qi[h][word], 0, false);
                            dot[h] += scale[h] * (float)product;
                        }
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < 3; ++h) {
                dot[h] += __shfl_xor(dot[h], 1, 32);
                if (row >= length) dot[h] = -__builtin_inff();
                next_max[h] = fmaxf(next_max[h], dot[h] + (3.0f * .6931f));
                if (lane % 2 == pair) score[h] = dot[h];
            }
        }
        #pragma unroll
        for (int h = 0; h < 3; ++h) {
            #pragma unroll
            for (int offset = 2; offset < 32; offset <<= 1)
                next_max[h] = fmaxf(next_max[h],
                    __shfl_xor(next_max[h], offset, 32));
            float rescale = expf(maximum[h] - next_max[h]);
            maximum[h] = next_max[h];
            score[h] = expf(score[h] - maximum[h]);
            sum[h] = sum[h] * rescale + score[h];
            probabilities[h][tid] = (q8_half)score[h];
            q8_half2 rescale_h = q8_hsplat(rescale);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[h][i] = q8_hmul(acc[h][i], rescale_h);
        }
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int col = 0; col < 32; ++col) {
            int row = base + warp * 32 + col;
            q8_half2 probability[3];
            #pragma unroll
            for (int h = 0; h < 3; ++h)
                probability[h] = q8_hsplat(
                    probabilities[h][warp * 32 + col]);
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                int dim = tile * 128 + lane * 4;
                q8_half2 v0 = {}, v1 = {};
                if (row < length) {
                    const signed char *value = vc +
                        (size_t)row * stride + kh * 256 + dim;
                    unsigned int packed = *(const unsigned int *)value;
                    q8_half2 d = q8_hsplat(vs[
                        (size_t)row * scale_stride + kh * 8 + dim / 32]);
                    v0 = q8_hmul(d, q8_hpair((signed char)packed,
                        (signed char)(packed >> 8)));
                    v1 = q8_hmul(d, q8_hpair((signed char)(packed >> 16),
                        (signed char)(packed >> 24)));
                }
                #pragma unroll
                for (int h = 0; h < 3; ++h) {
                    acc[h][2 * tile] = q8_hfma(
                        v0, probability[h], acc[h][2 * tile]);
                    acc[h][2 * tile + 1] = q8_hfma(
                        v1, probability[h], acc[h][2 * tile + 1]);
                }
            }
        }
    }
    if (warp == 0) {
        #pragma unroll
        for (int h = 0; h < 3; ++h) {
            maxima[h][lane] = -3.402823466e+38f / 2;
            sums_shared[h][lane] = 0.0f;
        }
    }
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int h = 0; h < 3; ++h) maxima[h][warp] = maximum[h];
    }
    __syncthreads();
    float final_max[3];
    #pragma unroll
    for (int h = 0; h < 3; ++h) {
        final_max[h] = maxima[h][lane];
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            final_max[h] = fmaxf(final_max[h],
                __shfl_xor(final_max[h], offset, 32));
        float rescale = expf(maximum[h] - final_max[h]);
        q8_half2 rescale_h = q8_hsplat(rescale);
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            q8_half2 *p = (q8_half2 *)(probabilities[h] +
                warp * 256 + tile * 128 + lane * 4);
            p[0] = q8_hmul(acc[h][2 * tile], rescale_h);
            p[1] = q8_hmul(acc[h][2 * tile + 1], rescale_h);
        }
        sum[h] = q8_attention_sum(sum[h] * rescale);
        if (lane == 0) sums_shared[h][warp] = sum[h];
    }
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < 3; ++h) {
        sum[h] = q8_attention_sum(sums_shared[h][lane]);
        #pragma unroll
        for (int base = 0; base < 256; base += 128) {
            float value = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w)
                value += (float)probabilities[h][w * 256 + base + tid];
            int head = head_base + h;
            if (splits == 1)
                out[head * 256 + base + tid] = value / sum[h];
            else
                parts[(head * splits + split) * 256 + base + tid] = value;
        }
        if (tid == 0 && splits > 1) {
            int head = head_base + h;
            meta[head * splits + split] = make_float2(final_max[h], sum[h]);
        }
    }
}

/* Exact verifier kernel for up to eight adjacent queries. Four waves share
 * every K/V row across the queries while retaining the scalar kernel's dot,
 * online-softmax, packed-F16 value accumulation, and final reduction order
 * independently for each query. Grid: (1, splits, heads), block: (32, 4). */
extern "C" __global__ __launch_bounds__(128, 1)
void qwen35_attention_q8_decode_reuse8(float *out, float *parts, float2 *meta,
        const float *q, const signed char *kc, const signed char *vc,
        const q8_half *ks, const q8_half *vs, const int *position,
        int heads, int kv_heads, int nsm, int occupancy, int forced_splits,
        int queries, int position_start) {
    if (queries < 2 || queries > 8) return;
    int lane = threadIdx.x, warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int split = blockIdx.y, head = blockIdx.z;
    int length[8], padded[8], max_padded = 0;
    #pragma unroll
    for (int query = 0; query < 8; ++query) {
        if (query < queries) {
            length[query] = position_start < 0 ? position[query] + 1 :
                              position_start + query + 1;
            padded[query] = (length[query] + 255) / 256 * 256;
            max_padded = max(max_padded, padded[query]);
        } else {
            length[query] = padded[query] = 0;
        }
    }
    int splits = forced_splits ? forced_splits :
        q8_attention_splits(length[queries - 1], heads, nsm, occupancy);
    if (split >= splits) return;
    int kh = head / (heads / kv_heads);
    int stride = kv_heads * 256, scale_stride = kv_heads * 8;
    __shared__ int qi[8][64];
    __shared__ float qd[8][8];
    __shared__ q8_half probabilities[8][1024];
    __shared__ float maxima[8][32], sums[8][32];

    for (int query = warp; query < queries; query += 4) {
        const float *query_data = q +
            ((size_t)query * heads + head) * 256;
        #pragma unroll
        for (int base = 0; base < 64; base += 32) {
            float value[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                value[i] = .0625f * query_data[4 * (base + lane) + i];
            float amax = fabsf(value[0]);
            #pragma unroll
            for (int i = 1; i < 4; ++i)
                amax = fmaxf(amax, fabsf(value[i]));
            #pragma unroll
            for (int offset = 4; offset; offset >>= 1)
                amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
            float d = amax / 127;
            int packed = 0;
            signed char *bytes = (signed char *)&packed;
            if (d != 0.0f) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    bytes[i] = roundf(value[i] / d);
            }
            qi[query][base + lane] = packed;
            if (lane % 8 == 0) qd[query][(base + lane) / 8] = d;
        }
    }
    __syncthreads();

    q8_half2 acc[8][4] = {};
    float maximum[8], sum[8];
    #pragma unroll
    for (int query = 0; query < 8; ++query) {
        maximum[query] = -3.402823466e+38f / 2;
        sum[query] = 0.0f;
    }
    for (int base = split * 128; base < max_padded;
         base += splits * 128) {
        float score[8] = {};
        float next_max[8];
        #pragma unroll
        for (int query = 0; query < 8; ++query)
            next_max[query] = maximum[query];
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            int row = base + warp * 32 + (lane & ~1) + pair;
            float dot[8] = {};
            if (row < length[queries - 1]) {
                const int *key = (const int *)(kc +
                    (size_t)row * stride + kh * 256);
                const q8_half *key_scale = ks +
                    (size_t)row * scale_stride + kh * 8;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int word = 2 * i + lane % 2;
                    int key_word = key[word];
                    #pragma unroll
                    for (int query = 0; query < 8; ++query) {
                        if (query < queries && row < length[query]) {
                            int product = __builtin_amdgcn_sudot4(
                                true, key_word, true,
                                qi[query][2 * i + lane % 2], 0, false);
                            float scale = key_scale[i / 4] * qd[query][i / 4];
                            dot[query] += scale * (float)product;
                        }
                    }
                }
            }
            #pragma unroll
            for (int query = 0; query < 8; ++query) {
                if (query < queries) {
                    dot[query] += __shfl_xor(dot[query], 1, 32);
                    if (row >= length[query]) dot[query] = -__builtin_inff();
                    next_max[query] = fmaxf(next_max[query],
                                             dot[query] + (3.0f * .6931f));
                    if (lane % 2 == pair) score[query] = dot[query];
                }
            }
        }
        #pragma unroll
        for (int query = 0; query < 8; ++query) {
            if (query < queries) {
                #pragma unroll
                for (int offset = 2; offset < 32; offset <<= 1)
                    next_max[query] = fmaxf(next_max[query],
                        __shfl_xor(next_max[query], offset, 32));
                float rescale = expf(maximum[query] - next_max[query]);
                maximum[query] = next_max[query];
                score[query] = expf(score[query] - maximum[query]);
                sum[query] = sum[query] * rescale + score[query];
                probabilities[query][tid] = (q8_half)score[query];
                q8_half2 rescale_h = q8_hsplat(rescale);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc[query][i] = q8_hmul(acc[query][i], rescale_h);
            }
        }
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int col = 0; col < 32; ++col) {
            int row = base + warp * 32 + col;
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                int dim = tile * 128 + lane * 4;
                q8_half2 value0 = {}, value1 = {};
                if (row < length[queries - 1]) {
                    const signed char *value = vc +
                        (size_t)row * stride + kh * 256 + dim;
                    unsigned int packed = *(const unsigned int *)value;
                    q8_half2 d = q8_hsplat(vs[(size_t)row * scale_stride +
                                                  kh * 8 + dim / 32]);
                    value0 = q8_hmul(d, q8_hpair(
                        (signed char)packed, (signed char)(packed >> 8)));
                    value1 = q8_hmul(d, q8_hpair(
                        (signed char)(packed >> 16),
                        (signed char)(packed >> 24)));
                }
                #pragma unroll
                for (int query = 0; query < 8; ++query) {
                    if (query < queries) {
                        q8_half2 probability = q8_hsplat(
                            probabilities[query][warp * 32 + col]);
                        acc[query][2 * tile] = q8_hfma(
                            value0, probability, acc[query][2 * tile]);
                        acc[query][2 * tile + 1] = q8_hfma(
                            value1, probability, acc[query][2 * tile + 1]);
                    }
                }
            }
        }
    }
    if (warp == 0) {
        #pragma unroll
        for (int query = 0; query < 8; ++query) {
            maxima[query][lane] = -3.402823466e+38f / 2;
            sums[query][lane] = 0.0f;
        }
    }
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int query = 0; query < 8; ++query)
            if (query < queries) maxima[query][warp] = maximum[query];
    }
    __syncthreads();
    float final_max[8], final_sum[8];
    #pragma unroll
    for (int query = 0; query < 8; ++query) {
        if (query < queries) {
            final_max[query] = maxima[query][lane];
            #pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                final_max[query] = fmaxf(final_max[query],
                    __shfl_xor(final_max[query], offset, 32));
            float rescale = expf(maximum[query] - final_max[query]);
            q8_half2 rescale_h = q8_hsplat(rescale);
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                q8_half2 *probability = (q8_half2 *)(
                    probabilities[query] + warp * 256 +
                    tile * 128 + lane * 4);
                probability[0] = q8_hmul(acc[query][2 * tile], rescale_h);
                probability[1] = q8_hmul(acc[query][2 * tile + 1], rescale_h);
            }
            final_sum[query] = q8_attention_sum(sum[query] * rescale);
            if (lane == 0) sums[query][warp] = final_sum[query];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int query = 0; query < 8; ++query) {
        if (query < queries) {
            final_sum[query] = q8_attention_sum(sums[query][lane]);
            #pragma unroll
            for (int base = 0; base < 256; base += 128) {
                float result = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w)
                    result += (float)probabilities[query][
                        w * 256 + base + tid];
                if (splits == 1)
                    out[((size_t)query * heads + head) * 256 +
                        base + tid] = result / final_sum[query];
                else
                    parts[(((size_t)query * heads + head) * splits + split) *
                          256 + base + tid] = result;
            }
            if (tid == 0 && splits > 1)
                meta[((size_t)query * heads + head) * splits + split] =
                    make_float2(final_max[query], final_sum[query]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void qwen35_attention_q8_combine(float *out, const float *parts, const float2 *meta,
        const int *position, int heads, int nsm, int occupancy, int forced_splits) {
    int query = blockIdx.y;
    int splits = forced_splits ? forced_splits :
        q8_attention_splits(position[query] + 1, heads, nsm, occupancy);
    if (splits == 1) return;
    int head = query * heads + blockIdx.x, tid = threadIdx.x;
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
