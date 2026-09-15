/* Warp-sized dense MMA tiles. All 32 lanes participate, including tails.
 * A and B are row-major [16, K] (B is logically transposed). Acc is [16,16]. */
__device__ unsigned short px_pack(float f, int p) {
    return p == 1 ? __float_as_uint(round_bfloat(f)) >> 16 : __half_as_ushort(__float2half_rn(f));
}
__device__ void px_mma16(const unsigned short *a, const unsigned short *b, float *out, int stride,
                         int precision) {
    int lane = threadIdx.x & 31;
#ifdef PX_HIP
#if defined(__gfx1200__) || defined(__gfx1201__)
    using U8 = unsigned short __attribute__((ext_vector_type(8)));
    using H8 = _Float16 __attribute__((ext_vector_type(8)));
    using F8 = float __attribute__((ext_vector_type(8)));
    U8 av, bv;
    F8 cv;
    for (int j = 0; j < 8; ++j) {
        av[j] = a[(lane & 15) * stride + (lane >> 4) * 8 + j];
        bv[j] = b[(lane & 15) * stride + (lane >> 4) * 8 + j];
        cv[j] = out[((lane >> 4) * 8 + j) * 16 + (lane & 15)];
    }
    if (precision == 1)
        cv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, cv);
    else
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(H8, av),
                                                              __builtin_bit_cast(H8, bv), cv);
    for (int j = 0; j < 8; ++j)
        out[((lane >> 4) * 8 + j) * 16 + (lane & 15)] = cv[j];
#endif
#else
    int row = lane >> 2, col = (lane & 3) * 2;
    unsigned av[4];
    for (int j = 0; j < 4; ++j) {
        int r = row + (j & 1) * 8, k = col + (j >> 1) * 8;
        av[j] = unsigned(a[r * stride + k]) | (unsigned(a[r * stride + k + 1]) << 16);
    }
    for (int half = 0; half < 2; ++half) {
        unsigned bv[2];
        for (int j = 0; j < 2; ++j) {
            int r = row + half * 8, k = col + j * 8;
            bv[j] = unsigned(b[r * stride + k]) | (unsigned(b[r * stride + k + 1]) << 16);
        }
        float c0 = out[row * 16 + col + half * 8], c1 = out[row * 16 + col + half * 8 + 1];
        float c2 = out[(row + 8) * 16 + col + half * 8], c3 = out[(row + 8) * 16 + col + half * 8 + 1];
        if (precision == 1) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                         "{%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
        } else {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                         "{%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
        }
        out[row * 16 + col + half * 8] = c0;
        out[row * 16 + col + half * 8 + 1] = c1;
        out[(row + 8) * 16 + col + half * 8] = c2;
        out[(row + 8) * 16 + col + half * 8 + 1] = c3;
    }
#endif
}

__global__ void px_gemm_mma(float *out, const unsigned short *x, const unsigned short *w, int n, int co,
                            int ci, int precision) {
    __shared__ unsigned short a[16 * 16], b[16 * 16];
    __shared__ float accum[16 * 16];
    int lane = threadIdx.x, row = blockIdx.y * 16, col = blockIdx.x * 16;
    for (int j = lane; j < 256; j += 32)
        accum[j] = 0;
    for (int start = 0; start < ci; start += 16) {
        for (int j = lane; j < 256; j += 32) {
            int r = j / 16, k = j % 16;
            a[j] = (row + r < n && start + k < ci) ? x[size_t(row + r) * ci + start + k] : 0;
            b[j] = (col + r < co && start + k < ci) ? w[size_t(col + r) * ci + start + k] : 0;
        }
        __syncthreads();
        px_mma16(a, b, accum, 16, precision);
        __syncthreads();
    }
    for (int j = lane; j < 256; j += 32)
        if (row + j / 16 < n && col + j % 16 < co)
            out[size_t(row + j / 16) * co + col + j % 16] = accum[j];
}

/* Online softmax, 16 queries x 16 keys. No N x N allocation. */
__global__ void px_flash_mma(float *out, const float *q, const float *k, const float *v, int n, int m,
                             int heads, int precision) {
    __shared__ unsigned short a[256], b[256], p[256];
    __shared__ float scores[256], pv[256], accum[16 * 128], maximum[16], denom[16];
    int lane = threadIdx.x, first = blockIdx.x * 16, head = blockIdx.y, c = heads * 128;
    for (int j = lane; j < 16 * 128; j += 32)
        accum[j] = 0;
    if (lane < 16) {
        maximum[lane] = -INFINITY;
        denom[lane] = 0;
    }
    for (int start = 0; start < m; start += 16) {
        for (int j = lane; j < 256; j += 32)
            scores[j] = 0;
        for (int d = 0; d < 128; d += 16) {
            for (int j = lane; j < 256; j += 32) {
                int r = j / 16, z = j % 16;
                a[j] = first + r < n ? px_pack(q[size_t(first + r) * c + head * 128 + d + z], precision) : 0;
                b[j] = start + r < m ? px_pack(k[size_t(start + r) * c + head * 128 + d + z], precision) : 0;
            }
            __syncthreads();
            px_mma16(a, b, scores, 16, precision);
            __syncthreads();
        }
        if (lane < 16) {
            float mx = maximum[lane];
            for (int j = 0; j < 16 && start + j < m; ++j)
                mx = fmaxf(mx, scores[lane * 16 + j] * 0.08838834764831845f);
            float scale = expf(maximum[lane] - mx), sum = 0;
            for (int j = 0; j < 16; ++j) {
                float prob = start + j < m ? expf(scores[lane * 16 + j] * 0.08838834764831845f - mx) : 0;
                p[lane * 16 + j] = px_pack(prob, precision);
                sum += prob;
            }
            for (int d = 0; d < 128; ++d)
                accum[lane * 128 + d] *= scale;
            denom[lane] = denom[lane] * scale + sum;
            maximum[lane] = mx;
        }
        __syncthreads();
        for (int d = 0; d < 128; d += 16) {
            for (int j = lane; j < 256; j += 32) {
                int r = j / 16, z = j % 16;
                b[j] = start + z < m ? px_pack(v[size_t(start + z) * c + head * 128 + d + r], precision) : 0;
                pv[j] = 0;
            }
            __syncthreads();
            px_mma16(p, b, pv, 16, precision);
            __syncthreads();
            for (int j = lane; j < 256; j += 32)
                accum[(j / 16) * 128 + d + j % 16] += pv[j];
            __syncthreads();
        }
    }
    for (int j = lane; j < 16 * 128; j += 32) {
        int r = j / 128, d = j % 128;
        if (first + r < n)
            out[size_t(first + r) * c + head * 128 + d] = round_precision(accum[j] / denom[r], precision);
    }
}
