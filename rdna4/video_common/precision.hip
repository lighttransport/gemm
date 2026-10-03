// SPDX-License-Identifier: MIT
// Explicit upstream FP16 boundaries, in F32 storage for the native graph.
// Custom RMSNorm rounds before its affine multiply. Autocast LayerNorm
// stays FP32; modulation rounds only the Half scale-plus-one expression.
// Gate products are Half; residual addition follows the residual dtype.
__device__ float hv15_half_value(float x) { return __half2float(__float2half(x)); }
// Eager RMSNorm rsqrt is correctly rounded; retain the FP32 multiply before Half.
__device__ float hv15_rms_inverse(float value) {
    float inverse = rsqrtf(value);
    float square = inverse * inverse;
    float low = fmaf(inverse, inverse, -square);
    float error = fmaf(-value, square, 1.f) - value * low;
    return fmaf(.5f * inverse, error, inverse);
}
__device__ float hv15_rms_half(float value, float inverse) {
    float normalized = value * inverse;
    // Prevent combining FP32 multiplication and Half conversion at tie boundaries.
    asm volatile("" : "+v"(normalized));
    return hv15_half_value(normalized);
}
// Match PyTorch ROCm vectorized Welford reduction, including FMA and reciprocal.
struct hv15_wf_state {
    float mean, square, count;
};
__device__ hv15_wf_state hv15_wf_add(hv15_wf_state state, float value) {
    float delta = value - state.mean;
    float count = state.count + 1.f;
    float mean = fmaf(delta, __builtin_amdgcn_rcpf(count), state.mean);
    return {mean, fmaf(delta, value - mean, state.square), count};
}
__device__ hv15_wf_state hv15_wf_join(hv15_wf_state first, hv15_wf_state second) {
    float count = first.count + second.count;
    if (count == 0.f)
        return {0, 0, 0};
    float reciprocal = __builtin_amdgcn_rcpf(count);
    float na = second.count * reciprocal, nb = first.count * reciprocal;
    float delta = first.mean - second.mean;
    float mean = fmaf(nb, first.mean, na * second.mean);
    float square = fmaf(delta * delta * second.count, nb, second.square + first.square);
    return {mean, square, count};
}
__device__ void hv15_layer_stats(const float *input, int channels, float &mean, float &inverse,
                                 float eps) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    __shared__ float means[8], squares[8], counts[8];
    hv15_wf_state state{0, 0, 0};
    for (int group = tid; group < channels / 4; group += 256)
        for (int j = 0; j < 4; ++j)
            state = hv15_wf_add(state, input[row * channels + group * 4 + j]);
    for (int offset = 16; offset; offset >>= 1) {
        hv15_wf_state peer{__shfl_down(state.mean, offset), __shfl_down(state.square, offset),
                           __shfl_down(state.count, offset)};
        state = hv15_wf_join(state, peer);
    }
    for (int offset = 4; offset; offset >>= 1) {
        if (lane == 0 && wave >= offset && wave < 2 * offset) {
            means[wave - offset] = state.mean;
            squares[wave - offset] = state.square;
            counts[wave - offset] = state.count;
        }
        __syncthreads();
        if (lane == 0 && wave < offset)
            state = hv15_wf_join(state, {means[wave], squares[wave], counts[wave]});
        __syncthreads();
    }
    if (tid == 0) {
        means[0] = state.mean;
        squares[0] = state.square / float(channels);
    }
    __syncthreads();
    mean = means[0];
    inverse = rsqrtf(squares[0] + eps);
}
extern "C" {
__global__ void hv15_autocast_norm(float *y, const float *x, const float *w, const float *b,
                                   int rows, int channels, int mode, float eps) {
    float mean, inverse;
    hv15_layer_stats(x, channels, mean, inverse, eps);
    int row = blockIdx.x, tid = threadIdx.x;
    for (int c = tid; c < channels; c += 256) {
        float normalized = (x[row * channels + c] - mean) * inverse;
        y[row * channels + c] = w ? fmaf(normalized, w[c], b ? b[c] : 0.f) : normalized;
    }
}

__global__ void hv15_half_element(float *y, const float *x, const float *z, const float *b,
                                  int count, int channels, int mode, float scale, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float v = x[i];
    if (mode == 0)
        v = v * scale + offset;
    if (mode == 1)
        v = v + z[i] * scale;
    if (mode == 2)
        v = v * z[i];
    if (mode == 3)
        v = v / (1.f + expf(-v));
    if (mode == 4)
        v = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    if (mode == 5)
        v = 0.5f * v * (1.f + erff(v * 0.7071067812f));
    if (mode == 6)
        v += b[i % channels];
    if (mode == 7)
        v = hv15_half_value(v * hv15_half_value(1.f + z[i % channels])) + b[i % channels];
    if (mode == 8)
        v = v * z[i % channels];
    if (mode == 9)
        v = fmaxf(v, 0.f);
    y[i] = hv15_half_value(v);
}
__global__ void hv15_round_half(float *x, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        x[i] = hv15_half_value(x[i]);
}
__global__ void hv15_half_norm(float *y, const float *x, const float *w, const float *b, int rows,
                               int channels, int rms, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows)
        return;
    if (channels == 128 && rms == 1) {
        __shared__ float inverse;
        if (tid < 32) {
            float square = 0.f;
            for (int i = 0; i < 4; ++i) {
                float value = x[row * 128 + tid * 4 + i];
                square += value * value;
            }
            for (int offset = 1; offset < 32; offset <<= 1)
                square += __shfl_down(square, offset);
            if (tid == 0)
                inverse = hv15_rms_inverse(square / 128.f + eps);
        }
        __syncthreads();
        if (tid < 128) {
            float value = hv15_rms_half(x[row * 128 + tid], inverse);
            y[row * 128 + tid] = hv15_half_value(value * (w ? w[tid] : 1.f) + (b ? b[tid] : 0.f));
        }
        return;
    }
    __shared__ float s[256], q[256];
    float sum = 0.f, sq = 0.f;
    for (int c = tid; c < channels; c += 256) {
        float v = x[row * channels + c];
        sum += v;
        sq += v * v;
    }
    s[tid] = sum;
    q[tid] = sq;
    __syncthreads();
    for (int n = 128; n; n >>= 1) {
        if (tid < n) {
            s[tid] += s[tid + n];
            q[tid] += q[tid + n];
        }
        __syncthreads();
    }
    float mean = rms ? 0.f : s[0] / channels;
    float inverse = rms == 2 ? sqrtf(float(channels)) / fmaxf(sqrtf(q[0]), 1.e-12f)
                             : rsqrtf(fmaxf(q[0] / channels - mean * mean, 0.f) + eps);
    for (int c = tid; c < channels; c += 256) {
        float v = (x[row * channels + c] - mean) * inverse;
        if (rms)
            v = hv15_half_value(v);
        y[row * channels + c] = hv15_half_value(v * (w ? w[c] : 1.f) + (b ? b[c] : 0.f));
    }
}
__global__ void hv15_autocast_modulate(float *y, const float *x, const float *mod, int rows, int c,
                                       int start) {
    int row = blockIdx.x, tid = threadIdx.x;
    float mean, inverse;
    hv15_layer_stats(x, c, mean, inverse, 1.e-6f);
    for (int j = tid; j < c; j += 256)
        y[row * c + j] =
            (x[row * c + j] - mean) * inverse * hv15_half_value(1.f + mod[start + c + j]) +
            mod[start + j];
}
__global__ void hv15_half_gate(float *y, const float *x, const float *z, const float *mod,
                               int count, int c, int start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = hv15_half_value(x[i] + hv15_half_value(z[i] * mod[start + i % c]));
}
__global__ void hv15_float_gate_half(float *y, const float *x, const float *z, const float *mod,
                                     int count, int c, int start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = x[i] + hv15_half_value(z[i] * mod[start + i % c]);
}
__global__ void hv15_modulate_values(float *y, const float *x, const float *scale,
                                     const float *shift, int count, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = x[i] * hv15_half_value(1.f + scale[i % c]) + shift[i % c];
}
__global__ void hv15_half_qkv(half *q, half *k, half *v, const float *packed, const float *qw,
                              const float *kw, int rows, int height, int width, int image) {
    int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    __shared__ float qa[128], ka[128], qs[128], ks[128];
    int base = row * 6144 + head * 128;
    float a = packed[base + d], b = packed[base + 2048 + d];
    if (d < 32) {
        float qsum = 0.f, ksum = 0.f;
        for (int i = 0; i < 4; ++i) {
            float qvalue = packed[base + d * 4 + i];
            float kvalue = packed[base + 2048 + d * 4 + i];
            qsum += qvalue * qvalue;
            ksum += kvalue * kvalue;
        }
        for (int offset = 1; offset < 32; offset <<= 1) {
            qsum += __shfl_down(qsum, offset);
            ksum += __shfl_down(ksum, offset);
        }
        if (d == 0) {
            qs[0] = hv15_rms_inverse(qsum / 128.f + 1.e-6f);
            ks[0] = hv15_rms_inverse(ksum / 128.f + 1.e-6f);
        }
    }
    __syncthreads();
    qa[d] = hv15_half_value(hv15_rms_half(a, qs[0]) * qw[d]);
    ka[d] = hv15_half_value(hv15_rms_half(b, ks[0]) * kw[d]);
    __syncthreads();
    if (image) {
        int first = d & ~1,
            axis = first < 16   ? 0
                   : first < 72 ? 1
                                : 2,
            axisdim = axis == 0 ? 16 : 56;
        int frequency = (first - (axis == 0 ? 0 : axis == 1 ? 16 : 72)) / 2;
        int position = axis == 0   ? row / (height * width)
                       : axis == 1 ? (row / width) % height
                                   : row % width;
        float angle = position * powf(256.f, -2.f * frequency / axisdim), c = cosf(angle),
              s = sinf(angle);
        a = (d & 1) ? qa[d] * c + qa[d - 1] * s : qa[d] * c - qa[d + 1] * s;
        b = (d & 1) ? ka[d] * c + ka[d - 1] * s : ka[d] * c - ka[d + 1] * s;
    } else {
        a = qa[d];
        b = ka[d];
    }
    int out = (head * rows + row) * 128 + d;
    q[out] = __float2half(a);
    k[out] = __float2half(b);
    v[out] = __float2half(packed[base + 4096 + d]);
}
}

// Use the fixed CPU FP32 frequency basis and GPU libm, as the upstream embedder.
extern "C" __global__ void hv15_time_embedding(float *out, const float *frequency, float time) {
    int i = threadIdx.x;
    if (i < 128) {
        float phase = time * frequency[i];
        out[i] = cosf(phase);
        out[i + 128] = sinf(phase);
    }
}
