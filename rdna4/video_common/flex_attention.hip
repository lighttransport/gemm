// SPDX-License-Identifier: MIT
// WMMA implementation of the pinned FlexAttention forward schedule.
// Unlike the normal tile, QK is transposed so each lane owns 32 consecutive
// score registers for one query, matching the reference's row reduction.
extern "C" __global__ __launch_bounds__(256, 1) void video_attention_flex_f16(
    float *out, const float *q, const float *k, const float *v, int rows, int heads,
    int dim, float scale, int kv_heads, int causal) {
    (void)dim;
    constexpr int HD = 128, BK = 64;
    int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    int half_lane = lane >> 4, col = lane & 15;
    int head = blockIdx.x, q0 = blockIdx.y * 128 + wave * 16;
    int kh = head / (heads / kv_heads);
    __shared__ va_h8 keys[16 * BK];
    __shared__ _Float16 values[HD * (BK + 4)];
    __shared__ _Float16 probabilities[8 * 16 * (BK + 4)];
    auto *prob = probabilities + wave * 16 * (BK + 4);
    va_h8 qr[8];
    va_f8 acc[8];
#pragma unroll
    for (int a = 0; a < 8; ++a) {
        acc[a] = va_f8{};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int row = q0 + col, d = a * 16 + half_lane * 8 + j;
            qr[a][j] = _Float16(row < rows ? q[((long)row * heads + head) * HD + d] : 0.f);
        }
    }
    float maximum = -__builtin_inff(), denominator = 0.f;
    for (int base = 0; base < rows; base += BK) {
        for (int i = tid; i < BK * 16; i += 256) {
            int r = i / 16, d = (i % 16) * 8, row = base + r;
            va_h8 key;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                long at = ((long)row * kv_heads + kh) * HD + d + j;
                key[j] = _Float16(row < rows ? k[at] : 0.f);
                values[(d + j) * (BK + 4) + r] = _Float16(row < rows ? v[at] : 0.f);
            }
            keys[(d / 8) * BK + r] = key;
        }
        __syncthreads();
        va_f8 scores[4];
#pragma unroll
        for (int b = 0; b < 4; ++b)
            scores[b] = va_f8{};
#pragma unroll
        for (int a = 0; a < 8; ++a) {
#pragma unroll
            for (int b = 0; b < 4; ++b)
                scores[b] = va_mma<false>(keys[(a * 2 + half_lane) * BK + b * 16 + col],
                                         qr[a], scores[b]);
        }
        float next = maximum;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int key_row = base + b * 16 + half_lane * 8 + j;
                float score = scores[b][j] * scale;
                score *= 1.4426950408889634f;
                if (key_row >= rows || (causal && key_row > q0 + col))
                    score = -__builtin_inff();
                scores[b][j] = score;
                next = fmaxf(next, score);
            }
        }
        next = fmaxf(next, __shfl_xor(next, 16));
        float alpha = __builtin_amdgcn_exp2f(maximum - next);
        float sum = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float p = __builtin_amdgcn_exp2f(scores[b][j] - next);
                sum += p;
                prob[col * (BK + 4) + b * 16 + half_lane * 8 + j] = _Float16(p);
            }
        }
        sum += __shfl_xor(sum, 16);
        denominator = fmaf(denominator, alpha, sum);
        maximum = next;
#pragma unroll
        for (int a = 0; a < 8; ++a) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[a][j] *= __shfl(alpha, half_lane * 8 + j);
        }
        __syncthreads();
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            va_h8 p;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                p[j] = prob[col * (BK + 4) + b * 16 + half_lane * 8 + j];
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                va_h8 value;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    value[j] = values[(a * 16 + col) * (BK + 4) + b * 16 + half_lane * 8 + j];
                acc[a] = va_mma<false>(p, value, acc[a]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int row = q0 + half_lane * 8 + j;
        float sum = __shfl(denominator, half_lane * 8 + j);
#pragma unroll
        for (int a = 0; a < 8; ++a)
            if (row < rows)
                out[((long)row * heads + head) * HD + a * 16 + col] =
                    sum > 0.f ? acc[a][j] / sum : 0.f;
    }
}
