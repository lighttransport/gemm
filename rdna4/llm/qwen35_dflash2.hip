/* RDNA4 kernels used by the Qwen3.8 DFlash2 sidecar. */

#define DFLASH_ATTN_ROWS_PER_WAVE 4
#define DFLASH_SELECTOR_TOPK 16
#define DFLASH_SELECTOR_RANK 256

extern "C" __global__
void qwen35_dflash2_capture(float *features, const float *hidden,
        int rows, int hidden_size, int tap, int taps) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden_size;
    if (gid >= total) return;
    int row = gid / hidden_size;
    int dim = gid - row * hidden_size;
    features[((size_t)row * taps + tap) * hidden_size + dim] = hidden[gid];
}

extern "C" __global__ __launch_bounds__(256, 1)
void qwen35_dflash2_capture_rmsnorm(float *norm, float *features,
        const float *hidden, const float *weight, int rows, int hidden_size,
        int tap, int taps, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    const float *input = hidden + (size_t)row * hidden_size;
    float *output = norm + (size_t)row * hidden_size;
    extern __shared__ float reduction[];
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float value = input[i];
        sum += value * value;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(reduction[0] / (float)hidden_size + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float value = input[i];
        output[i] = value * scale * weight[i];
        features[((size_t)row * taps + tap) * hidden_size + i] = value;
    }
}

extern "C" __global__
void qwen35_dflash2_conv(float *out, const float *hidden,
        const float *dynamic, const float *base, int rows, int hidden_size,
        int groups, int side) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden_size;
    if (gid >= total) return;
    int row = gid / hidden_size;
    int dim = gid - row * hidden_size;
    int group_size = hidden_size / groups;
    int group = dim / group_size;
    float value = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < 2; ++tap) {
        int source_row = row - tap;
        if (source_row < 0) continue;
        int coeff = group + groups * (tap + 2 * side);
        int base_index = dim + hidden_size * (tap + 2 * side);
        value += (dynamic[(size_t)row * (groups * 4) + coeff] +
                  base[base_index]) *
                 hidden[(size_t)source_row * hidden_size + dim];
    }
    out[gid] = value;
}

/* Stable top-k over each non-anchor draft row.  The (score,id) comparison
 * reproduces the host scan's first-id tie break while 256 threads divide the
 * vocabulary scan. */
extern "C" __global__ __launch_bounds__(256, 1)
void qwen35_dflash2_topk(const float *logits, int *candidates,
        int rows, int vocab) {
    int row = blockIdx.x + 1;
    int tid = threadIdx.x;
    if (row >= rows || tid >= 256) return;
    float best[DFLASH_SELECTOR_TOPK];
    int ids[DFLASH_SELECTOR_TOPK];
    #pragma unroll
    for (int k = 0; k < DFLASH_SELECTOR_TOPK; ++k) {
        best[k] = -3.402823466e+38f;
        ids[k] = vocab;
    }
    const float *input = logits + (size_t)row * vocab;
    for (int i = tid; i < vocab; i += 256) {
        float value = input[i];
        if (value > best[DFLASH_SELECTOR_TOPK-1] ||
            (value == best[DFLASH_SELECTOR_TOPK-1] &&
             i < ids[DFLASH_SELECTOR_TOPK-1])) {
            int k = DFLASH_SELECTOR_TOPK-1;
            while (k > 0 && (value > best[k-1] ||
                   (value == best[k-1] && i < ids[k-1]))) {
                best[k] = best[k-1]; ids[k] = ids[k-1]; --k;
            }
            best[k] = value; ids[k] = i;
        }
    }
    __shared__ float block_scores[256 * DFLASH_SELECTOR_TOPK];
    __shared__ int block_ids[256 * DFLASH_SELECTOR_TOPK];
    #pragma unroll
    for (int k = 0; k < DFLASH_SELECTOR_TOPK; ++k) {
        block_scores[tid * DFLASH_SELECTOR_TOPK + k] = best[k];
        block_ids[tid * DFLASH_SELECTOR_TOPK + k] = ids[k];
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < DFLASH_SELECTOR_TOPK; ++k) {
            best[k] = -3.402823466e+38f;
            ids[k] = vocab;
        }
        for (int i = 0; i < 256 * DFLASH_SELECTOR_TOPK; ++i) {
            float value = block_scores[i];
            int id = block_ids[i];
            if (value > best[DFLASH_SELECTOR_TOPK-1] ||
                (value == best[DFLASH_SELECTOR_TOPK-1] &&
                 id < ids[DFLASH_SELECTOR_TOPK-1])) {
                int k = DFLASH_SELECTOR_TOPK-1;
                while (k > 0 && (value > best[k-1] ||
                       (value == best[k-1] && id < ids[k-1]))) {
                    best[k] = best[k-1]; ids[k] = ids[k-1]; --k;
                }
                best[k] = value; ids[k] = id;
            }
        }
        #pragma unroll
        for (int k = 0; k < DFLASH_SELECTOR_TOPK; ++k)
            candidates[row * DFLASH_SELECTOR_TOPK + k] = ids[k];
    }
}

__device__ __forceinline__ float qwen35_dflash2_q4k_value(
        const unsigned char *weights, int token, int index) {
    const unsigned char *p = weights + (size_t)token * 144;
    float d = (float)*(const _Float16 *)p;
    float dmin = (float)*(const _Float16 *)(p + 2);
    const unsigned char *scales = p + 4;
    int group = index >> 6;
    int half = (index >> 5) & 1;
    int si = 2 * group + half;
    int scale, minimum;
    if (si < 4) {
        scale = scales[si] & 63;
        minimum = scales[si + 4] & 63;
    } else {
        scale = (scales[si + 4] & 15) | ((scales[si - 4] >> 6) << 4);
        minimum = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
    }
    int q = p[16 + group * 32 + (index & 31)];
    q = half ? q >> 4 : q & 15;
    return d * (float)(scale * q) - dmin * (float)minimum;
}

/* The selector is sequential across draft positions but tiny: sixteen lanes
 * evaluate the candidates, each retaining the host implementation's ordered
 * 256-term accumulation. */
extern "C" __global__ __launch_bounds__(32, 1)
void qwen35_dflash2_select(const float *logits, const float *selector_gate,
        const unsigned char *predecessor_w, const unsigned char *successor_w,
        const int *candidates, int *drafts, int anchor, int rows, int vocab) {
    int k = threadIdx.x;
    if (blockIdx.x || k >= DFLASH_SELECTOR_TOPK) return;
    __shared__ float scores[DFLASH_SELECTOR_TOPK];
    /* The predecessor is common to every candidate in a row.  Decode its
     * rank vector once cooperatively instead of repeating the same Q4_K
     * lookup in all sixteen candidate lanes. */
    __shared__ float predecessor_values[DFLASH_SELECTOR_RANK];
    __shared__ int predecessor;
    if (k == 0) predecessor = anchor;
    __syncthreads();
    for (int row = 1; row < rows; ++row) {
        for (int j = k; j < DFLASH_SELECTOR_RANK; j += DFLASH_SELECTOR_TOPK)
            predecessor_values[j] = qwen35_dflash2_q4k_value(
                predecessor_w, predecessor, j);
        __syncthreads();
        int token = candidates[row * DFLASH_SELECTOR_TOPK + k];
        float score = logits[(size_t)row * vocab + token];
        const float *gate = selector_gate + (size_t)row * DFLASH_SELECTOR_RANK;
        for (int j = 0; j < DFLASH_SELECTOR_RANK; ++j) {
            float next = qwen35_dflash2_q4k_value(successor_w, token, j);
            score += predecessor_values[j] * gate[j] * next;
        }
        scores[k] = score;
        __syncthreads();
        if (k == 0) {
            int selected = 0;
            for (int i = 1; i < DFLASH_SELECTOR_TOPK; ++i)
                if (scores[i] > scores[selected]) selected = i;
            predecessor = candidates[row * DFLASH_SELECTOR_TOPK + selected];
            drafts[row-1] = predecessor;
        }
        __syncthreads();
    }
}

/* Diagnostic selector geometry: eight candidate groups share one block.
 * Each candidate still has its score accumulated by lane zero in the exact
 * 0..255 order used above; the other lanes only parallelize the Q4_K decode
 * into shared memory. Two batches cover the sixteen candidates without
 * changing the host-visible tie break or predecessor chain. */
extern "C" __global__ __launch_bounds__(256, 1)
void qwen35_dflash2_select_warp(const float *logits,
        const float *selector_gate, const unsigned char *predecessor_w,
        const unsigned char *successor_w, const int *candidates, int *drafts,
        int anchor, int rows, int vocab) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int group = tid >> 5;
    if (blockIdx.x || group >= 8) return;
    __shared__ float predecessor_values[DFLASH_SELECTOR_RANK];
    __shared__ float candidate_values[8][DFLASH_SELECTOR_RANK];
    __shared__ float scores[DFLASH_SELECTOR_TOPK];
    __shared__ int predecessor;
    if (tid == 0) predecessor = anchor;
    __syncthreads();
    for (int row = 1; row < rows; ++row) {
        for (int j = lane; j < DFLASH_SELECTOR_RANK; j += 32)
            predecessor_values[j] = qwen35_dflash2_q4k_value(
                predecessor_w, predecessor, j);
        __syncthreads();
        for (int batch = 0; batch < 2; ++batch) {
            int k = batch * 8 + group;
            int token = candidates[row * DFLASH_SELECTOR_TOPK + k];
            for (int j = lane; j < DFLASH_SELECTOR_RANK; j += 32)
                candidate_values[group][j] = qwen35_dflash2_q4k_value(
                    successor_w, token, j);
            __syncthreads();
            if (lane == 0) {
                float score = logits[(size_t)row * vocab + token];
                const float *gate = selector_gate +
                    (size_t)row * DFLASH_SELECTOR_RANK;
                for (int j = 0; j < DFLASH_SELECTOR_RANK; ++j)
                    score += predecessor_values[j] * gate[j] *
                             candidate_values[group][j];
                scores[k] = score;
            }
            __syncthreads();
        }
        if (tid == 0) {
            int selected = 0;
            for (int i = 1; i < DFLASH_SELECTOR_TOPK; ++i)
                if (scores[i] > scores[selected]) selected = i;
            predecessor = candidates[row * DFLASH_SELECTOR_TOPK + selected];
            drafts[row - 1] = predecessor;
        }
        __syncthreads();
    }
    (void)vocab;
}

/* One wave owns four query rows for one head and one contiguous position
 * split.  Splitting the 2K window supplies enough independent waves to hide
 * cache latency while each partial retains position order.  DFlash2 is a
 * proposal model, so the target verifier remains authoritative if the stable
 * softmax merge below perturbs a proposal score at the last bit. */
extern "C" __global__ __launch_bounds__(32, 8)
void qwen35_dflash2_attention(float *partial, const float *query,
        const float *key_cache, const float *value_cache,
        int rows, int position_start, int n_heads, int n_kv_heads,
        int head_dim, int window, int splits) {
    int head = blockIdx.x;
    int row0 = blockIdx.y * DFLASH_ATTN_ROWS_PER_WAVE;
    int split = blockIdx.z;
    int lane = threadIdx.x;
    if (head >= n_heads || row0 >= rows || split >= splits || lane >= 32) return;
    int ratio = n_heads / n_kv_heads;
    int kv_head = head / ratio;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    int end = position_start + rows;
    int begin = end > window ? end - window : 0;
    int length = end - begin;
    int split_begin = begin + (length * split) / splits;
    int split_end = begin + (length * (split + 1)) / splits;
    float maximum[DFLASH_ATTN_ROWS_PER_WAVE] = {
        -3.402823466e+38f, -3.402823466e+38f,
        -3.402823466e+38f, -3.402823466e+38f};
    float normalizer[DFLASH_ATTN_ROWS_PER_WAVE] = {};
    float accum[DFLASH_ATTN_ROWS_PER_WAVE][4] = {};
    float scale = rsqrtf((float)head_dim);

    for (int pos = split_begin; pos < split_end; ++pos) {
        int slot = pos % window;
        size_t cache = (size_t)slot * kv_stride + kv_head * head_dim;
        float kval[4], vval[4];
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            kval[part] = dim < head_dim ? key_cache[cache + dim] : 0.0f;
            vval[part] = dim < head_dim ? value_cache[cache + dim] : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < DFLASH_ATTN_ROWS_PER_WAVE; ++r) {
            if (row0 + r < rows) {
                size_t qbase = (size_t)(row0+r) * q_stride + head * head_dim;
                float dot = 0.0f;
                #pragma unroll
                for (int part = 0; part < 4; ++part) {
                    int dim = lane + part * 32;
                    if (dim < head_dim) dot += query[qbase + dim] * kval[part];
                }
                dot += __shfl_xor(dot, 16, 32);
                dot += __shfl_xor(dot, 8, 32);
                dot += __shfl_xor(dot, 4, 32);
                dot += __shfl_xor(dot, 2, 32);
                dot += __shfl_xor(dot, 1, 32);
                float score = dot * scale;
                float next_max = fmaxf(maximum[r], score);
                float alpha = maximum[r] < -1.0e30f ? 0.0f :
                              expf(maximum[r] - next_max);
                float probability = expf(score - next_max);
                normalizer[r] = normalizer[r] * alpha + probability;
                #pragma unroll
                for (int part = 0; part < 4; ++part) {
                    accum[r][part] = accum[r][part] * alpha +
                                     probability * vval[part];
                }
                maximum[r] = next_max;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < DFLASH_ATTN_ROWS_PER_WAVE; ++r) {
        if (row0 + r < rows) {
            size_t obase = (((size_t)split * rows + row0+r) * n_heads + head) *
                           (head_dim + 2);
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                int dim = lane + part * 32;
                if (dim < head_dim) {
                    partial[obase + dim] = accum[r][part];
                }
            }
            if (lane == 0) {
                partial[obase + head_dim] = maximum[r];
                partial[obase + head_dim + 1] = normalizer[r];
            }
        }
    }
}

/* Merge contiguous split summaries in increasing position order. */
extern "C" __global__ __launch_bounds__(32, 8)
void qwen35_dflash2_attention_combine(float *out, const float *partial,
        int rows, int n_heads, int head_dim, int splits) {
    int head = blockIdx.x;
    int row = blockIdx.y;
    int lane = threadIdx.x;
    if (head >= n_heads || row >= rows || lane >= 32) return;
    float maximum = -3.402823466e+38f;
    float normalizer = 0.0f;
    float accum[4] = {};
    for (int split = 0; split < splits; ++split) {
        size_t base = (((size_t)split * rows + row) * n_heads + head) *
                      (head_dim + 2);
        float part_max = partial[base + head_dim];
        float part_norm = partial[base + head_dim + 1];
        float next_max = fmaxf(maximum, part_max);
        float alpha = maximum < -1.0e30f ? 0.0f : expf(maximum - next_max);
        float beta = part_max < -1.0e30f ? 0.0f : expf(part_max - next_max);
        normalizer = normalizer * alpha + part_norm * beta;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            if (dim < head_dim)
                accum[part] = accum[part] * alpha + partial[base + dim] * beta;
        }
        maximum = next_max;
    }
    size_t obase = ((size_t)row * n_heads + head) * head_dim;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        int dim = lane + part * 32;
        if (dim < head_dim)
            out[obase + dim] = accum[part] /
                (normalizer > 0.0f ? normalizer : 1.0f);
    }
}

/* DFlash2's production window uses at most twelve partitions.  Keep the
 * split walk and the ordered online-softmax merge in one verifier block for
 * that shape.  One warp owns one partition and four adjacent rows, then warp
 * zero performs the same increasing-split merge as
 * qwen35_dflash2_attention_combine.  The shared tile is bounded by
 * 12 * 4 * (128 + 2) floats (about 25 KiB), so this remains below the RDNA4
 * LDS limit while removing the global partial traffic and a launch.
 */
extern "C" __global__ __launch_bounds__(512, 1)
void qwen35_dflash2_attention_fused(float *out, const float *query,
        const float *key_cache, const float *value_cache,
        int rows, int position_start, int n_heads, int n_kv_heads,
        int head_dim, int window, int splits) {
    int head = blockIdx.x;
    int row0 = blockIdx.y * DFLASH_ATTN_ROWS_PER_WAVE;
    int split = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (head >= n_heads || row0 >= rows || split >= splits || lane >= 32 ||
        splits < 1 || splits > 12) return;
    int ratio = n_heads / n_kv_heads;
    int kv_head = head / ratio;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    int end = position_start + rows;
    int begin = end > window ? end - window : 0;
    int length = end - begin;
    int split_begin = begin + (length * split) / splits;
    int split_end = begin + (length * (split + 1)) / splits;
    float maximum[DFLASH_ATTN_ROWS_PER_WAVE] = {
        -3.402823466e+38f, -3.402823466e+38f,
        -3.402823466e+38f, -3.402823466e+38f};
    float normalizer[DFLASH_ATTN_ROWS_PER_WAVE] = {};
    float accum[DFLASH_ATTN_ROWS_PER_WAVE][4] = {};
    float scale = rsqrtf((float)head_dim);
    for (int pos = split_begin; pos < split_end; ++pos) {
        int slot = pos % window;
        size_t cache = (size_t)slot * kv_stride + kv_head * head_dim;
        float kval[4], vval[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            kval[part] = dim < head_dim ? key_cache[cache + dim] : 0.0f;
            vval[part] = dim < head_dim ? value_cache[cache + dim] : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < DFLASH_ATTN_ROWS_PER_WAVE; ++r) {
            if (row0 + r < rows) {
                size_t qbase = (size_t)(row0 + r) * q_stride + head * head_dim;
                float dot = 0.0f;
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    int dim = lane + part * 32;
                    if (dim < head_dim) dot += query[qbase + dim] * kval[part];
                }
                dot += __shfl_xor(dot, 16, 32);
                dot += __shfl_xor(dot, 8, 32);
                dot += __shfl_xor(dot, 4, 32);
                dot += __shfl_xor(dot, 2, 32);
                dot += __shfl_xor(dot, 1, 32);
                float score = dot * scale;
                float next_max = fmaxf(maximum[r], score);
                float alpha = maximum[r] < -1.0e30f ? 0.0f :
                              expf(maximum[r] - next_max);
                float probability = expf(score - next_max);
                normalizer[r] = normalizer[r] * alpha + probability;
#pragma unroll
                for (int part = 0; part < 4; ++part)
                    accum[r][part] = accum[r][part] * alpha +
                                     probability * vval[part];
                maximum[r] = next_max;
            }
        }
    }
    constexpr int TILE = 128 + 2;
    extern __shared__ float merged[];
    for (int r = 0; r < DFLASH_ATTN_ROWS_PER_WAVE; ++r) {
        if (row0 + r < rows) {
            size_t base = ((size_t)split * DFLASH_ATTN_ROWS_PER_WAVE + r) * TILE;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                int dim = lane + part * 32;
                if (dim < head_dim) merged[base + dim] = accum[r][part];
            }
            if (lane == 0) {
                merged[base + head_dim] = maximum[r];
                merged[base + head_dim + 1] = normalizer[r];
            }
        }
    }
    __syncthreads();
    if (split != 0) return;
    for (int r = 0; r < DFLASH_ATTN_ROWS_PER_WAVE; ++r) {
        if (row0 + r >= rows) continue;
        float final_max = -3.402823466e+38f;
        float final_norm = 0.0f;
        float final_accum[4] = {};
        for (int s = 0; s < 12; ++s) {
            if (s >= splits) break;
            size_t base = ((size_t)s * DFLASH_ATTN_ROWS_PER_WAVE + r) * TILE;
            float part_max = merged[base + head_dim];
            float part_norm = merged[base + head_dim + 1];
            float next_max = fmaxf(final_max, part_max);
            float alpha = final_max < -1.0e30f ? 0.0f :
                          expf(final_max - next_max);
            float beta = part_max < -1.0e30f ? 0.0f :
                         expf(part_max - next_max);
            final_norm = final_norm * alpha + part_norm * beta;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                int dim = lane + part * 32;
                if (dim < head_dim)
                    final_accum[part] = final_accum[part] * alpha +
                                        merged[base + dim] * beta;
            }
            final_max = next_max;
        }
        size_t out_base = ((size_t)(row0 + r) * n_heads + head) * head_dim;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            if (dim < head_dim)
                out[out_base + dim] = final_accum[part] /
                    (final_norm > 0.0f ? final_norm : 1.0f);
        }
    }
}

/* Fixed DFlash2 head shape.  The sidecar is permanently HD=128, so the
 * bounds checks and dynamic stride arithmetic in the generic fused kernel
 * only add control pressure to every cache row.  This keeps the same
 * per-lane dot/reduction order and increasing-split merge, but makes the
 * four 32-value lanes compile-time complete. */
extern "C" __global__ __launch_bounds__(512, 1)
void qwen35_dflash2_attention_fused_128(float *out, const float *query,
        const float *key_cache, const float *value_cache,
        int rows, int position_start, int n_heads, int n_kv_heads,
        int head_dim, int window, int splits) {
    (void)head_dim;
    constexpr int HD = 128;
    constexpr int ROWS = DFLASH_ATTN_ROWS_PER_WAVE;
    constexpr int TILE = HD + 2;
    int head = blockIdx.x;
    int row0 = blockIdx.y * ROWS;
    int split = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (head >= n_heads || row0 >= rows || split >= splits || lane >= 32 ||
        splits < 1 || splits > 12) return;
    int ratio = n_heads / n_kv_heads;
    int kv_head = head / ratio;
    int q_stride = n_heads * HD;
    int kv_stride = n_kv_heads * HD;
    int end = position_start + rows;
    int begin = end > window ? end - window : 0;
    int length = end - begin;
    int split_begin = begin + (length * split) / splits;
    int split_end = begin + (length * (split + 1)) / splits;
    float maximum[ROWS] = {-3.402823466e+38f, -3.402823466e+38f,
                           -3.402823466e+38f, -3.402823466e+38f};
    float normalizer[ROWS] = {};
    float accum[ROWS][4] = {};
    const float scale = rsqrtf((float)HD);

    for (int pos = split_begin; pos < split_end; ++pos) {
        int slot = pos % window;
        size_t cache = (size_t)slot * kv_stride + kv_head * HD;
        float kval[4], vval[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            kval[part] = key_cache[cache + dim];
            vval[part] = value_cache[cache + dim];
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (row0 + r < rows) {
                size_t qbase = (size_t)(row0 + r) * q_stride + head * HD;
                float dot = 0.0f;
#pragma unroll
                for (int part = 0; part < 4; ++part)
                    dot += query[qbase + lane + part * 32] * kval[part];
                dot += __shfl_xor(dot, 16, 32);
                dot += __shfl_xor(dot, 8, 32);
                dot += __shfl_xor(dot, 4, 32);
                dot += __shfl_xor(dot, 2, 32);
                dot += __shfl_xor(dot, 1, 32);
                float score = dot * scale;
                float next_max = fmaxf(maximum[r], score);
                float alpha = maximum[r] < -1.0e30f ? 0.0f :
                              expf(maximum[r] - next_max);
                float probability = expf(score - next_max);
                normalizer[r] = normalizer[r] * alpha + probability;
#pragma unroll
                for (int part = 0; part < 4; ++part)
                    accum[r][part] = accum[r][part] * alpha +
                                     probability * vval[part];
                maximum[r] = next_max;
            }
        }
    }

    extern __shared__ float merged[];
    for (int r = 0; r < ROWS; ++r) {
        if (row0 + r < rows) {
            size_t base = ((size_t)split * ROWS + r) * TILE;
#pragma unroll
            for (int part = 0; part < 4; ++part)
                merged[base + lane + part * 32] = accum[r][part];
            if (lane == 0) {
                merged[base + HD] = maximum[r];
                merged[base + HD + 1] = normalizer[r];
            }
        }
    }
    __syncthreads();
    if (split != 0) return;
    for (int r = 0; r < ROWS; ++r) {
        if (row0 + r >= rows) continue;
        float final_max = -3.402823466e+38f;
        float final_norm = 0.0f;
        float final_accum[4] = {};
        for (int s = 0; s < 12; ++s) {
            if (s >= splits) break;
            size_t base = ((size_t)s * ROWS + r) * TILE;
            float part_max = merged[base + HD];
            float part_norm = merged[base + HD + 1];
            float next_max = fmaxf(final_max, part_max);
            float alpha = final_max < -1.0e30f ? 0.0f :
                          expf(final_max - next_max);
            float beta = part_max < -1.0e30f ? 0.0f :
                         expf(part_max - next_max);
            final_norm = final_norm * alpha + part_norm * beta;
#pragma unroll
            for (int part = 0; part < 4; ++part)
                final_accum[part] = final_accum[part] * alpha +
                                    merged[base + lane + part * 32] * beta;
            final_max = next_max;
        }
        size_t out_base = ((size_t)(row0 + r) * n_heads + head) * HD;
#pragma unroll
        for (int part = 0; part < 4; ++part)
            out[out_base + lane + part * 32] = final_accum[part] /
                (final_norm > 0.0f ? final_norm : 1.0f);
    }
}
