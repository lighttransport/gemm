#define _POSIX_C_SOURCE 200809L
#include "bench_common.h"
#include <float.h>
#include <limits.h>

static void cpu_tests(void) {
    REQUIRE(int4_padded_k(0) == 0 && int4_padded_k(-1) == 0 && int4_padded_k(INT_MAX) == 0);
    REQUIRE(int4_padded_k(1) == 64 && int4_padded_k(65) == 128);
    REQUIRE(int4_codes_size(3, 65) == 192 && int4_scales_size(3, 65) == 24);
    REQUIRE(int4_codes_size(-1, 64) == 0);
    const float fp4[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int q = 0; q < 16; q++) {
        float v = q & 8 ? -(float)(q & 7) : (float)(q & 7);
        REQUIRE(int4_code_decode(q, 1) == v);
        REQUIRE(int4_code_decode(q, 0) == ((q & 8) ? -fp4[q & 7] : fp4[q & 7]));
        REQUIRE(int4_code_encode(v, 1, 1) == (unsigned)((q & 7) ? q : 0));
    }
    REQUIRE(signbit(int4_code_decode(8, 1)));
    for (unsigned q = 1; q <= 126; q++) REQUIRE(int4_scale_encode(int4_scale_decode(q)) == q);
    for (unsigned q = 1; q < 126; q++) {
        float midpoint = (int4_scale_decode(q) + int4_scale_decode(q + 1)) * 0.5f;
        REQUIRE(int4_scale_encode(midpoint) == ((q & 1) ? q + 1 : q));
    }
    REQUIRE(int4_code_encode(.5f, 1, 1) == 0 && int4_code_encode(1.5f, 1, 1) == 2);
    REQUIRE(int4_code_encode(2.5f, 1, 1) == 2 && int4_code_encode(3.5f, 1, 1) == 4);
    REQUIRE(int4_code_encode(FLT_MAX, 1.0f / 512, 0) == 7);
    float z[17] = {0}; uint8_t codes[32], scales[4]; int4_quant_stats st;
    REQUIRE(!int4_quantize_cpu(z, 1, 17, codes, scales, 1, &st));
    for (int i = 0; i < 32; i++) REQUIRE(codes[i] == 0);
    for (int i = 0; i < 4; i++) REQUIRE(scales[i] == 0x38);
    z[0] = NAN; REQUIRE(int4_quantize_cpu(z, 1, 17, codes, scales, 1, NULL) == -1);
    puts("CPU: codebook, scale rounding, packing sizes, zero blocks and invalid inputs PASS");
}

static void quant_tests(int4_context *ctx, int short_run) {
    const int ks[] = {1, 15, 16, 17, 63, 64, 65, 127, 129};
    const float edges[] = {0, -0.0f, 1e-30f, -1e-30f, 1e30f, -1e30f, .5f, 1.5f, 2.5f, -3.5f,
        1.0f / 512, 448, 3136, 3360, -3360, FLT_MAX / 4};
    uint32_t seed = 17;
    CUstream stream; CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    for (unsigned s = 0; s < sizeof(ks) / sizeof(ks[0]); s++) for (int format = 0; format < 2; format++) {
        int rows = short_run ? 3 : 65, k = ks[s];
        int4_matrix dst = matrix_alloc(rows, k);
        size_t n = (size_t)rows * k, nc = int4_codes_size(rows, k), ns = int4_scales_size(rows, k);
        float *src = host_alloc(n * 4);
        for (size_t i = 0; i < n; i++) src[i] = i % 7 == 0 ? edges[(i / 7) % 16] : ((int)(random_u32(&seed) % 2049) - 1024) / 64.0f;
        uint8_t *hc = host_alloc(nc), *hs = host_alloc(ns), *gc = host_alloc(nc), *gs = host_alloc(ns);
        int4_quant_stats cpu, gpu;
        REQUIRE(!int4_quantize_cpu(src, rows, k, hc, hs, format, &cpu));
        CUdeviceptr input = device_alloc(n * 4), stats = device_alloc(sizeof(gpu));
        CUDA(cuMemcpyHtoD(input, src, n * 4)); CUDA(cuMemsetD8(stats, 0, sizeof(gpu)));
        CUDA(int4_quantize_f32(ctx, input, dst, format, stats, stream));
        CUDA(cuStreamSynchronize(stream));
        CUDA(cuMemcpyDtoH(gc, dst.codes, nc)); CUDA(cuMemcpyDtoH(gs, dst.scales, ns));
        CUDA(cuMemcpyDtoH(&gpu, stats, sizeof(gpu)));
        REQUIRE(!memcmp(hc, gc, nc) && !memcmp(hs, gs, ns) && !memcmp(&cpu, &gpu, sizeof(cpu)));
        CUDA(cuMemFree(input)); CUDA(cuMemFree(stats)); matrix_free(dst);
        free(src); free(hc); free(hs); free(gc); free(gs);
    }
    CUDA(cuStreamDestroy(stream));
    puts("GPU quantizer: CPU/GPU codes, scales, clipping counters and K tails match PASS");
}

static void gemm_case(int4_context *ctx, int M, int N, int K, int kind, int formats, int all_variants) {
    int4_matrix a = matrix_alloc(M, K), b = matrix_alloc(N, K);
    size_t ac = int4_codes_size(M, K), bc = int4_codes_size(N, K);
    size_t as = int4_scales_size(M, K), bs = int4_scales_size(N, K), count = (size_t)M * N;
    uint8_t *ah = host_alloc(ac), *bh = host_alloc(bc), *ash = host_alloc(as), *bsh = host_alloc(bs);
    uint32_t seed = 7;
    memset(ah, 0, ac); memset(bh, 0, bc);
    for (int r = 0; r < M + N; r++) {
        int is_a = r < M, row = is_a ? r : r - M;
        uint8_t *p = is_a ? ah : bh, *sc = is_a ? ash : bsh;
        for (int k = 0; k < K; k++) {
            unsigned q = kind == 0 ? 7 : kind == 1 ? (unsigned)(row & 15) : random_u32(&seed) & 15;
            p[(size_t)row * (a.kp / 2) + k / 2] |= q << ((k & 1) * 4);
        }
        for (int g = 0; g < a.kp / 16; g++) {
            static const uint8_t extreme_scales[] = {0, 1, 7, 8, 0x38, 0x7e};
            sc[(size_t)row * (a.kp / 16) + g] = kind < 2 ? 0x38 : kind == 3 ?
                extreme_scales[random_u32(&seed) % 6] : 0x20 + random_u32(&seed) % 49;
        }
    }
    CUDA(cuMemcpyHtoD(a.codes, ah, ac)); CUDA(cuMemcpyHtoD(b.codes, bh, bc));
    CUDA(cuMemcpyHtoD(a.scales, ash, as)); CUDA(cuMemcpyHtoD(b.scales, bsh, bs));
    double *ref = host_alloc(count * sizeof(double));
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double sum = 0;
        for (int k = 0; k < K; k++) sum += (double)int4_decode_cpu(ah, ash, m, k, a.kp, formats & 1) *
            int4_decode_cpu(bh, bsh, n, k, a.kp, (formats >> 1) & 1);
        ref[(size_t)m * N + n] = sum;
    }
    float *out = host_alloc((count + 32) * 4);
    CUdeviceptr output = device_alloc((count + 32) * 4);
    CUstream stream; CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    int nv = all_variants ? INT4_VARIANTS : 1;
    for (int v = 0; v < nv; v++) {
        for (size_t i = 0; i < count + 32; i++) out[i] = 123.25f;
        CUDA(cuMemcpyHtoD(output, out, (count + 32) * 4));
        CUDA(int4_gemm(ctx, a, b, output, formats, v, stream));
        CUDA(cuStreamSynchronize(stream));
        CUDA(cuMemcpyDtoH(out, output, (count + 32) * 4));
        double num = 0, den = 0, max_error = 0;
        for (size_t i = 0; i < count; i++) {
            REQUIRE(isfinite(out[i]));
            double error = out[i] - ref[i]; num += error * error; den += ref[i] * ref[i];
            max_error = fmax(max_error, fabs(error));
            if (kind < 2) REQUIRE(out[i] == (float)ref[i]);
        }
        if (!(den ? sqrt(num / den) <= 1e-4 : max_error <= 1e-5)) {
            fprintf(stderr, "%dx%dx%d mode=%d variant=%s rel_L2=%g max_error=%g\n", M, N, K,
                formats, int4_variant_names[v], sqrt(num / (den + 1e-300)), max_error);
            exit(1);
        }
        for (size_t i = count; i < count + 32; i++) REQUIRE(out[i] == 123.25f);
    }
    CUDA(cuStreamDestroy(stream));
    matrix_free(a); matrix_free(b); CUDA(cuMemFree(output));
    free(ah); free(bh); free(ash); free(bsh); free(ref); free(out);
}

int main(int argc, char **argv) {
    int cpu_only = 0, short_run = 0;
    char dir[4096]; default_cubin_dir(dir, sizeof(dir));
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) cpu_only = 1;
        else if (!strcmp(argv[i], "--sanitizer")) short_run = 1;
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 1; }
    }
    cpu_tests(); if (cpu_only) return 0;
    CUcontext cuda = start_cuda(); int4_context ctx;
    REQUIRE(!int4_init(&ctx, dir));
    quant_tests(&ctx, short_run);
    for (int f = 0; f < 4; f++) {
        gemm_case(&ctx, 16, 8, 64, 0, f, 1);
        gemm_case(&ctx, 16, 16, 64, 1, f, 1); /* Every pair of the 16 nibble codes. */
        gemm_case(&ctx, 17, 9, 129, 3, f, 1); /* Zero, subnormal and maximum scales. */
    }
    const int shapes[][3] = {{1,1,1}, {3,7,15}, {17,9,17}, {31,33,63}, {65,129,65}, {67,131,129}, {16,8,256}};
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int fmax = short_run ? 1 : 4;
        for (int f = 0; f < fmax; f++) gemm_case(&ctx, shapes[s][0], shapes[s][1], shapes[s][2], 2,
            short_run ? INT4_E0_E0 : f, 1);
    }
    int4_matrix bad = {0,0,1,64,64};
    REQUIRE(int4_gemm(&ctx, bad, bad, 1, 3, 1, NULL) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(int4_quantize_f32(&ctx, 1, bad, 1, 0, NULL) == CUDA_ERROR_INVALID_VALUE);
    bad.codes = 17; bad.scales = 4;
    REQUIRE(int4_gemm(&ctx, bad, bad, 4, 3, 1, NULL) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(int4_quantize_f32(NULL, 4, bad, 1, 0, NULL) == CUDA_ERROR_INVALID_VALUE);
    puts("GPU GEMM: all 9 variants, 4 formats, asymmetric K16 scales, odd dimensions and output guards PASS");
    int4_destroy(&ctx); CUDA(cuCtxDestroy(cuda));
    return 0;
}
