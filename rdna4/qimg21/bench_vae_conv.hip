// SPDX-License-Identifier: MIT
// Parity/throughput: tiled VAE 3x3 conv (vae_conv_tiled.hip, incl. both packs) vs the per-wave
// vae_conv2d_3x3_wmma_f32 kernel and an exact f32 direct convolution. usage: bench_vae_conv
#include <hip/hip_runtime.h>
#pragma clang diagnostic ignored "-Wunused-result"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "vae_conv_tiled.hip"

/* verbatim logic of vae_conv2d_3x3_wmma_f32 (hip_qimg_kernels.h) */
__global__ void old_conv(float *out, const float *inp, const float *weight, const float *bias, int ci, int h, int w, int co, int pad_replicate) {
    typedef float float8 __attribute__((ext_vector_type(8))); typedef short bf16x8 __attribute__((ext_vector_type(8)));
    int oc_tile = blockIdx.x * 16, sp_tile = blockIdx.y * 16, lane = threadIdx.x, half = lane >> 4, idx = lane & 15, sp = h * w, K_total = ci * 9;
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0}; int oc_a = oc_tile + idx, spi = sp_tile + idx;
    int oy = (spi < sp) ? (spi / w) : 0, ox = (spi < sp) ? (spi % w) : 0;
    for (int k0 = 0; k0 < K_total; k0 += 16) {
        bf16x8 a, b;
        for (int i = 0; i < 8; i++) {
            int k = k0 + half * 8 + i, ic = k / 9, fyx = k - ic * 9, fy = fyx / 3, fx = fyx - fy * 3;
            float wv = 0.f; if (ic < ci && oc_a < co) wv = weight[((oc_a * ci + ic) * 3 + fy) * 3 + fx];
            float xv = 0.f;
            if (ic < ci && spi < sp) {
                int iy = oy + fy - 1, ix = ox + fx - 1;
                if (pad_replicate) { if (iy < 0) iy = 0; if (iy >= h) iy = h - 1; if (ix < 0) ix = 0; if (ix >= w) ix = w - 1; xv = inp[ic * sp + iy * w + ix]; }
                else if (iy >= 0 && iy < h && ix >= 0 && ix < w) xv = inp[ic * sp + iy * w + ix];
            }
            unsigned bw, bx; memcpy(&bw, &wv, 4); memcpy(&bx, &xv, 4); a[i] = (short)(bw >> 16); b[i] = (short)(bx >> 16);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    for (int i = 0; i < 8; i++) { int oc = oc_tile + half * 8 + i; if (oc < co && spi < sp) out[oc * sp + spi] = acc[i] + (bias ? bias[oc] : 0.f); }
}

__global__ void ref_conv(float *out, const float *in, const float *wt, const float *bias, int ci, int h, int w, int co, int rep) {
    int p = blockIdx.x * blockDim.x + threadIdx.x, oc = blockIdx.y, sp = h * w; if (p >= sp) return;
    int oy = p / w, ox = p % w; float acc = bias ? bias[oc] : 0.f;
    for (int ic = 0; ic < ci; ic++) for (int fy = 0; fy < 3; fy++) for (int fx = 0; fx < 3; fx++) {
        int iy = oy + fy - 1, ix = ox + fx - 1;
        if (rep) { iy = iy < 0 ? 0 : (iy >= h ? h - 1 : iy); ix = ix < 0 ? 0 : (ix >= w ? w - 1 : ix); }
        else if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
        acc += in[(long)ic * sp + iy * w + ix] * wt[((long)oc * ci + ic) * 9 + fy * 3 + fx];
    }
    out[(long)oc * sp + p] = acc;
}

int main() {
    int shapes[][5] = {{384, 384, 128, 128, 0}, {384, 384, 256, 256, 1}, {192, 192, 512, 512, 0}, {96, 96, 1024, 1024, 1}, {384, 192, 512, 512, 0}};
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &s : shapes) {
        int ci = s[0], co = s[1], h = s[2], w = s[3], rep = s[4], sp = h * w;
        std::vector<float> hx((size_t)ci * sp), hw((size_t)co * ci * 9), hb(co);
        for (auto &v : hx) v = nd(rng); for (auto &v : hw) v = nd(rng) * 0.05f; for (auto &v : hb) v = nd(rng) * 0.1f;
        float *x, *wt, *b, *y0, *y1, *y2; q21v_u16 *xb, *wb;
        hipMalloc(&x, hx.size() * 4); hipMalloc(&wt, hw.size() * 4); hipMalloc(&b, co * 4);
        hipMalloc(&y0, (size_t)co * sp * 4); hipMalloc(&y1, (size_t)co * sp * 4); hipMalloc(&y2, (size_t)co * sp * 4);
        hipMalloc(&xb, hx.size() * 2); hipMalloc(&wb, hw.size() * 2);
        hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice); hipMemcpy(wt, hw.data(), hw.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(b, hb.data(), co * 4, hipMemcpyHostToDevice);
        ref_conv<<<dim3((sp + 255) / 256, co), 256>>>(y0, x, wt, b, ci, h, w, co, rep);
        auto run_old = [&]() { old_conv<<<dim3(co / 16, sp / 16), 32>>>(y1, x, wt, b, ci, h, w, co, rep); };
        auto run_new = [&]() {
            q21v_pack_hwc<<<dim3((sp + 31) / 32, (ci + 31) / 32), dim3(32, 8)>>>(xb, x, ci, sp);
            q21v_pack_w3x3<<<(unsigned)(((long)co * ci * 9 + 255) / 256), 256>>>(wb, wt, co, ci);
            q21v_conv3x3_tiled<<<dim3((sp + 127) / 128, (co + 127) / 128), 256>>>(y2, xb, wb, b, ci, h, w, co, rep);
        };
        run_old(); run_new(); hipDeviceSynchronize();
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float to = 0, tn = 0;
        hipEventRecord(e0); for (int r = 0; r < 3; r++) run_old(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&to, e0, e1);
        hipEventRecord(e0); for (int r = 0; r < 10; r++) run_new(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&tn, e0, e1);
        to /= 3; tn /= 10;
        std::vector<float> a((size_t)co * sp), o(a.size()), n(a.size());
        hipMemcpy(a.data(), y0, a.size() * 4, hipMemcpyDeviceToHost); hipMemcpy(o.data(), y1, o.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(n.data(), y2, n.size() * 4, hipMemcpyDeviceToHost);
        double dn = 0, dold = 0, nr = 0; long bad = 0;
        for (size_t i = 0; i < a.size(); i++) { dn += (n[i] - a[i]) * (double)(n[i] - a[i]); dold += (o[i] - a[i]) * (double)(o[i] - a[i]); nr += (double)a[i] * a[i]; bad += !std::isfinite(n[i]); }
        double flop = 2.0 * co * sp * ci * 9;
        printf("ci=%3d co=%3d %4dx%-4d rep=%d  old %8.2f ms (%5.1f TF) rel %.5f | tiled %7.2f ms (%5.1f TF) rel %.5f nonfinite %ld | %5.1fx\n",
               ci, co, h, w, rep, to, flop / (to * 1e-3) / 1e12, sqrt(dold / nr), tn, flop / (tn * 1e-3) / 1e12, sqrt(dn / nr), bad, to / tn);
        hipFree(x); hipFree(wt); hipFree(b); hipFree(y0); hipFree(y1); hipFree(y2); hipFree(xb); hipFree(wb);
    }
    return 0;
}
