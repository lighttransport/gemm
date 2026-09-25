/*
 * Small gfx1201 validation for the llama.cpp fattn-vec F16 contract.
 *
 * This intentionally tests the arithmetic/mapping in isolation from the
 * runner.  llama.cpp's HIP path uses eight lanes per KQ group and
 * v_dot2_f32_f16 for each half2 pair.  The production Qwen kernel must not be
 * changed until this test remains finite and agrees with the host reference.
 */
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

static __device__ __forceinline__ void mad_half2(float & acc, const __half2 a,
                                                   const __half2 b) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
                 : "+v"(acc) : "v"(a) , "v"(b));
}

/* One eight-lane group computes one K row.  D=256 and cpy_ne=4 are the
 * gfx1201 values corresponding to llama.cpp's max 16-byte copy. */
__global__ void kq_half2_vec(float *out, const __half *q, const __half *k,
                             int ncols, int D) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 7;
    const int col = tid >> 3;
    if (col >= ncols) return;
    float sum = 0.0f;
    const __half2 *k2 = reinterpret_cast<const __half2 *>(k + (size_t)col * D);
    for (int k0 = 0; k0 < D / 2; k0 += 8 * 4) {
        for (int j = 0; j < 4; ++j) {
            /* Q_reg is thread-private in fattn-vec: its logical index
             * k0/8+j was populated from this lane's source offset. */
            const __half2 *q2 = reinterpret_cast<const __half2 *>(q);
            mad_half2(sum, k2[k0 + lane * 4 + j], q2[k0 + lane * 4 + j]);
        }
    }
    for (int off = 4; off; off >>= 1) sum += __shfl_down(sum, off, 8);
    if (lane == 0) out[col] = sum;
}

/* The V phase in fattn-vec keeps a half2 accumulator per output pair. */
__global__ void v_half2_vec(float *out, const __half *v, const float *p,
                            int ncols, int D) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 7;
    const int pair = tid >> 3;
    if (pair >= D / 2) return;
    __shared__ float partial[128][8][2];
    __half2 acc = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    for (int col = lane; col < ncols; col += 8) {
        const __half2 x = reinterpret_cast<const __half2 *>(v)[(size_t)col * D / 2 + pair];
        const __half2 w = __halves2half2(__float2half(p[col]), __float2half(p[col]));
        acc = __hfma2(x, w, acc);
    }
    partial[pair][lane][0] = __half2float(__low2half(acc));
    partial[pair][lane][1] = __half2float(__high2half(acc));
    __syncthreads();
    if (lane == 0) {
        float a = 0.0f, b = 0.0f;
        for (int l = 0; l < 8; ++l) { a += partial[pair][l][0]; b += partial[pair][l][1]; }
        out[2 * pair + 0] = a;
        out[2 * pair + 1] = b;
    }
}

int main() {
    constexpr int D = 256;
    constexpr int NC = 16;
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<__half> hq(D), hk((size_t)NC * D), hv((size_t)NC * D);
    std::vector<float> hp(NC);
    for (auto &x : hq) x = __float2half(dist(rng));
    for (auto &x : hk) x = __float2half(dist(rng));
    for (auto &x : hv) x = __float2half(dist(rng));
    float psum = 0.0f;
    for (float &x : hp) { x = std::fabs(dist(rng)); psum += x; }
    for (float &x : hp) x /= psum;

    __half *dq = nullptr, *dk = nullptr, *dv = nullptr;
    float *do_kq = nullptr, *do_v = nullptr, *dp = nullptr;
    hipMalloc(&dq, sizeof(__half) * hq.size());
    hipMalloc(&dk, sizeof(__half) * hk.size());
    hipMalloc(&dv, sizeof(__half) * hv.size());
    hipMalloc(&do_kq, sizeof(float) * NC);
    hipMalloc(&do_v, sizeof(float) * D);
    hipMalloc(&dp, sizeof(float) * NC);
    hipMemcpy(dq, hq.data(), sizeof(__half) * hq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dk, hk.data(), sizeof(__half) * hk.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv, hv.data(), sizeof(__half) * hv.size(), hipMemcpyHostToDevice);
    hipMemcpy(dp, hp.data(), sizeof(float) * hp.size(), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(kq_half2_vec, dim3(1), dim3(128), 0, 0,
                       do_kq, dq, dk, NC, D);
    hipLaunchKernelGGL(v_half2_vec, dim3(1), dim3(1024), 0, 0,
                       do_v, dv, dp, NC, D);
    std::vector<float> got_kq(NC), got_v(D);
    hipMemcpy(got_kq.data(), do_kq, sizeof(float) * NC, hipMemcpyDeviceToHost);
    hipMemcpy(got_v.data(), do_v, sizeof(float) * D, hipMemcpyDeviceToHost);
    float max_kq = 0.0f, max_v = 0.0f;
    for (int c = 0; c < NC; ++c) {
        float ref = 0.0f;
        for (int d = 0; d < D; ++d)
            ref += __half2float(hq[d]) * __half2float(hk[(size_t)c * D + d]);
        max_kq = fmaxf(max_kq, fabsf(ref - got_kq[c]));
    }
    for (int d = 0; d < D; ++d) {
        float ref = 0.0f;
        for (int c = 0; c < NC; ++c)
            ref += hp[c] * __half2float(hv[(size_t)c * D + d]);
        max_v = fmaxf(max_v, fabsf(ref - got_v[d]));
    }
    bool finite = true;
    for (float x : got_kq) finite &= std::isfinite(x);
    for (float x : got_v) finite &= std::isfinite(x);
    std::printf("fattn_vec_half2 finite=%d max_kq_abs=%.9g max_v_abs=%.9g\n",
                finite ? 1 : 0, max_kq, max_v);
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(do_kq); hipFree(do_v); hipFree(dp);
    return finite && max_kq < 1.0e-3f && max_v < 1.0e-3f ? 0 : 1;
}
