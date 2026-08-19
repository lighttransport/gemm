// test_conv2d_sve.cpp - correctness + microbenchmark for the fused
// dual-conv2d SVE patch-embedding kernel (a64fx/vlm/kernels/conv2d_sve.c).
//
// Build (A64FX node, native compilers): fcc for the C driver, FCC for C++:
//   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 \
//       -Ikernels -c kernels/conv2d_sve.c -o build/conv2d_sve.o
//   as  -march=armv8.2-a+sve -o build/micro_kernel_conv2d_8x8.o \
//       kernels/micro_kernel_conv2d_8x8.S
//   FCC -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c++17 -fopenmp \
//       -Ikernels -c tools/test_conv2d_sve.cpp -o build/test_conv2d_sve.o
//   FCC -O3 -fopenmp -o build/test_conv2d_sve \
//       build/test_conv2d_sve.o build/conv2d_sve.o \
//       build/micro_kernel_conv2d_8x8.o -lm
// Run:
//   ./build/test_conv2d_sve [width height threads]     # default 384 384 48
//
// Checks the SVE kernel against the scalar dual-conv2d reference
// (same math/order-of-ops family as common/vision_encoder.h patch_embed)
// and reports GFLOP/s for: scalar reference, SVE 1-thread, SVE N-thread.

#include "conv2d_sve.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

/* deterministic LCG so results are reproducible */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static double frand(double lo, double hi) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    double u = (double)(rng_state >> 11) / (double)(1ull << 53);
    return lo + u * (hi - lo);
}

/* Scalar dual-conv2d reference: mirrors common/vision_encoder.h patch_embed. */
static void ref_patch_embed(const float *rgb, int W, int H, int ps, int dim,
                            const float *k0, const float *k1, const float *bias,
                            float *out) {
    int gw = W / ps, gh = H / ps, ks = ps * ps * 3;
    std::vector<float> patch(ks);
    for (int py = 0; py < gh; py++)
        for (int px = 0; px < gw; px++) {
            int p = py * gw + px;
            for (int c = 0; c < 3; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++)
                        patch[c * ps * ps + dy * ps + dx] =
                            rgb[(((py * ps + dy) * W + px * ps + dx) * 3) + c];
            for (int d = 0; d < dim; d++) {
                float s = 0.0f;
                const float *a = k0 + (size_t)d * ks;
                for (int j = 0; j < ks; j++) s += a[j] * patch[j];
                if (k1) {
                    const float *b = k1 + (size_t)d * ks;
                    for (int j = 0; j < ks; j++) s += b[j] * patch[j];
                }
                out[(size_t)p * dim + d] = s + (bias ? bias[d] : 0.0f);
            }
        }
}

static double now_ms() {
    using namespace std::chrono;
    using namespace std::chrono;
    return duration_cast<duration<double, std::ratio<1, 1000>>>(
        steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    int W = 384, H = 384, nthreads = 48;
    if (argc > 1) W = atoi(argv[1]);
    if (argc > 2) H = atoi(argv[2]);
    if (argc > 3) nthreads = atoi(argv[3]);
    if (W % 16 || H % 16) { fprintf(stderr, "W/H must be multiples of ps=16\n"); return 1; }

    const int ps = 16, dim = 1024, ks = ps * ps * 3;
    const int gw = W / ps, gh = H / ps, n_patches = gw * gh;

    fprintf(stderr, "conv2d_sve test: %dx%d image, %d patches, dim=%d ks=%d, %d threads\n",
            W, H, n_patches, dim, ks, nthreads);

    std::vector<float> rgb((size_t)W * H * 3), k0((size_t)dim * ks),
        k1((size_t)dim * ks), bias(dim);
    for (auto &x : rgb)  x = (float)frand(-1.5, 1.5);
    for (auto &x : k0)   x = (float)frand(-0.05, 0.05);
    for (auto &x : k1)   x = (float)frand(-0.05, 0.05);
    for (auto &x : bias) x = (float)frand(-0.05, 0.05);

    /* pack the merged dual-conv weights once (as the VLM cache does) */
    std::vector<float> Wp(conv2d_packed_w_size(ks, dim));
    {
        double t0 = now_ms();
        conv2d_pack_w_sve(k0.data(), k1.data(), ks, dim, Wp.data());
        fprintf(stderr, "pack W (merged, k-major): %7.2f ms  (%.1f MB)\n",
                now_ms() - t0, Wp.size() * 4.0 / 1e6);
    }

    std::vector<float> ref((size_t)n_patches * dim), got((size_t)n_patches * dim);

    /* ── correctness: tile the SVE kernel over the whole grid ── */
    std::memset(got.data(), 0, got.size() * sizeof(float));
    for (int t = 0; t * CONV2D_TILE < n_patches; t++) {
        int p0 = t * CONV2D_TILE;
        int mc = std::min(CONV2D_TILE, n_patches - p0);
        conv2d_patch_tile_sve(rgb.data(), W, H, ps, dim, Wp.data(), bias.data(),
                              p0, mc, got.data());
    }
    double maxerr = 0.0;
    /* reference */
    ref_patch_embed(rgb.data(), W, H, ps, dim, k0.data(), k1.data(), bias.data(), ref.data());
    for (size_t i = 0; i < ref.size(); i++)
        maxerr = std::max(maxerr, (double)std::fabs(ref[i] - got[i]));
    double refscale = 0.0;
    for (size_t i = 0; i < ref.size(); i += 97) refscale = std::max(refscale, (double)std::fabs(ref[i]));
    fprintf(stderr, "max abs error vs scalar dual-conv reference: %.3e (ref scale %.3e)\n",
            maxerr, refscale);
    if (maxerr > 1e-3 * std::max(1.0, (double)refscale)) {
        fprintf(stderr, "FAIL: error too large\n");
        return 2;
    }
    fprintf(stderr, "PASS (tol 1e-3*max(1,|ref|))\n");

    /* ── benchmark ── */
    double flops_dual = 2.0 * n_patches * dim * ks * 2.0; /* both convs, as the profiler counts */
    double flops_fused = 2.0 * n_patches * dim * ks;      /* merged kernel: one GEMM */

    {   /* scalar reference, 1 thread */
        double t0 = now_ms();
        ref_patch_embed(rgb.data(), W, H, ps, dim, k0.data(), k1.data(), bias.data(), ref.data());
        double dt = now_ms() - t0;
        fprintf(stderr, "scalar dual-conv (1T):  %8.2f ms  %8.1f GFLOP/s (dual-count)\n",
                dt, flops_dual / dt / 1e6);
    }
    {   /* SVE fused, 1 thread */
        double t0 = now_ms();
        for (int t = 0; t * CONV2D_TILE < n_patches; t++)
            conv2d_patch_tile_sve(rgb.data(), W, H, ps, dim, Wp.data(), bias.data(),
                                  t * CONV2D_TILE,
                                  std::min(CONV2D_TILE, n_patches - t * CONV2D_TILE),
                                  got.data());
        double dt = now_ms() - t0;
        fprintf(stderr, "SVE fused conv2d   (1T): %8.2f ms  %8.1f GFLOP/s (merged)\n",
                dt, flops_fused / dt / 1e6);
    }
    {   /* SVE fused, N threads via 2D collapse(2) (internal OMP) */
        double t0 = now_ms();
        conv2d_sve_full(rgb.data(), W, H, ps, dim, Wp.data(), bias.data(), got.data());
        double dt = now_ms() - t0;
        fprintf(stderr, "SVE fused conv2d  (%2dT): %8.2f ms  %8.1f GFLOP/s (merged)\n",
                nthreads, dt, flops_fused / dt / 1e6);
    }
    return 0;
}
