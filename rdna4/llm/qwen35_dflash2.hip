/* RDNA4 kernels used by the Qwen3.8 DFlash2 sidecar. */

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

/* One wave owns one (query, head). Each lane retains four output dimensions
 * and performs an online softmax over the 2048-entry circular draft cache.
 * DFlash2 attention is non-causal inside the current noise block. */
extern "C" __global__ __launch_bounds__(32, 8)
void qwen35_dflash2_attention(float *out, const float *query,
        const float *key_cache, const float *value_cache,
        int rows, int position_start, int n_heads, int n_kv_heads,
        int head_dim, int window) {
    int head = blockIdx.x;
    int row = blockIdx.y;
    int lane = threadIdx.x;
    if (head >= n_heads || row >= rows || lane >= 32) return;
    int ratio = n_heads / n_kv_heads;
    int kv_head = head / ratio;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    int end = position_start + rows;
    int begin = end > window ? end - window : 0;
    float maximum = -3.402823466e+38f;
    float normalizer = 0.0f;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale = rsqrtf((float)head_dim);

    for (int pos = begin; pos < end; ++pos) {
        int slot = pos % window;
        size_t cache = (size_t)slot * kv_stride + kv_head * head_dim;
        size_t qbase = (size_t)row * q_stride + head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            if (dim < head_dim)
                dot += query[qbase + dim] * key_cache[cache + dim];
        }
        dot += __shfl_xor(dot, 16, 32);
        dot += __shfl_xor(dot, 8, 32);
        dot += __shfl_xor(dot, 4, 32);
        dot += __shfl_xor(dot, 2, 32);
        dot += __shfl_xor(dot, 1, 32);
        float score = dot * scale;
        float next_max = fmaxf(maximum, score);
        float alpha = maximum < -1.0e30f ? 0.0f : expf(maximum - next_max);
        float probability = expf(score - next_max);
        normalizer = normalizer * alpha + probability;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            int dim = lane + part * 32;
            if (dim < head_dim)
                accum[part] = accum[part] * alpha +
                              probability * value_cache[cache + dim];
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
