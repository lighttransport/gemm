/* Standalone RMBG-2.0 Swin-L feature extractor. Raw F32 CHW input/output. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef SWIN_HAVE_AVX2
#include "../ryzen/gemm_avx2.h"
#endif
#ifdef SWIN_CUDA
#include "../cuda/gemm/cuda_linear_f32.h"
#endif
#if defined(SWIN_CUDA) || !defined(SWIN_NO_MAIN)
static int swin_gpu = 0;
#endif

static void *swin_alloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "swin: allocation failed (%zu bytes)\n", bytes); exit(3); }
    return p;
}

/* Untransposed row-major multiplication, also used inside parallel attention. */
static void swin_matmul(float *y, const float *a, const float *b, int m, int n, int k)
{
#ifdef SWIN_HAVE_AVX2
    memset(y, 0, (size_t)m*n*sizeof(float));
    sgemm_avx2(m, n, k, 1, a, k, b, n, 0, y, n);
#else
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float v = 0;
        for (int d = 0; d < k; d++) v += a[(size_t)i*k+d]*b[(size_t)d*n+j];
        y[(size_t)i*n+j] = v;
    }
#endif
}

static void swin_linear(float *y, const float *weight, const float *bias,
                        const float *x, int m, int n, int k)
{
#ifdef SWIN_CUDA
    if (swin_gpu) {
        if (cuda_linear_f32_run(y, weight, bias, x, m, n, k)) {
            fprintf(stderr, "swin: CUDA GEMM failed\n"); exit(4);
        }
        return;
    }
#endif
    /* W X^T fits the repository row-major kernel without transposing weights. */
    float *xt = swin_alloc((size_t)m*k*sizeof(float));
    float *yt = swin_alloc((size_t)m*n*sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) for (int d = 0; d < k; d++) xt[(size_t)d*m+i] = x[(size_t)i*k+d];
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; row += 64) {
        int rows = n-row < 64 ? n-row : 64;
#ifdef SWIN_HAVE_AVX2
        /* Long dot products feed an unusually sensitive channel in Swin-L's
         * 18-block stage. Sum short FP32 GEMM panels in double precision to
         * bound accumulation error while retaining the repository kernel.
         * Attention's K<=144 products use the ordinary FP32 path above. */
        if (k >= 192) {
            size_t count = (size_t)rows*m;
            float *part = swin_alloc(count*sizeof(float));
            double *sum = swin_alloc(count*sizeof(double));
            memset(sum, 0, count*sizeof(double));
            for (int start = 0; start < k; start += 64) {
                int depth = k-start < 64 ? k-start : 64;
                memset(part, 0, count*sizeof(float));
                sgemm_avx2(rows, m, depth, 1, weight+(size_t)row*k+start, k,
                           xt+(size_t)start*m, m, 0, part, m);
                for (size_t i = 0; i < count; i++) sum[i] += part[i];
            }
            for (size_t i = 0; i < count; i++) yt[(size_t)row*m+i] = (float)sum[i];
            free(sum); free(part);
        } else
#endif
        swin_matmul(yt+(size_t)row*m, weight+(size_t)row*k, xt, rows, m, k);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) for (int d = 0; d < n; d++)
        y[(size_t)i*n+d] = yt[(size_t)d*m+i]+(bias ? bias[d] : 0);
    free(xt); free(yt);
}

#define SWIN_ALLOC swin_alloc
#define SWIN_LINEAR swin_linear
#define SWIN_MATMUL swin_matmul
#ifdef SWIN_DIAGNOSTIC_TRACE
static const char *swin_trace_dir = NULL;
static void swin_trace(int s, int j, const float *x, int h, int w, int c)
{
    if (!swin_trace_dir || s != 2) return;
    char path[4096];
    int len = snprintf(path, sizeof(path), "%s/block_%d.f32", swin_trace_dir, j);
    if (len < 0 || (size_t)len >= sizeof(path)) exit(5);
    FILE *f = fopen(path, "wb");
    size_t n = (size_t)h*w*c;
    if (!f || fwrite(x, sizeof(float), n, f) != n || fclose(f)) exit(5);
}
#define SWIN_TRACE swin_trace
#endif
#define SAFETENSORS_IMPLEMENTATION
#include "swin_v1.h"

static int swin_number(const char *s)
{
    char *end; errno = 0;
    long n = strtol(s, &end, 10);
    return errno || end == s || *end || n < 1 || n > 1024 ? -1 : (int)n;
}

#ifndef SWIN_NO_MAIN
int main(int argc, char **argv)
{
    const char *model = NULL, *input = NULL, *output = NULL, *backend = "cpu";
    int h = 0, w = 0, threads = 4, rc = 1, device = 0;
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) goto usage;
        const char *flag = argv[i], *v = argv[++i];
        if (!strcmp(flag, "--model")) model = v;
        else if (!strcmp(flag, "--input")) input = v;
        else if (!strcmp(flag, "--output-dir")) output = v;
        else if (!strcmp(flag, "--height")) h = swin_number(v);
        else if (!strcmp(flag, "--width")) w = swin_number(v);
        else if (!strcmp(flag, "--threads")) threads = swin_number(v);
        else if (!strcmp(flag, "--backend")) backend = v;
        else if (!strcmp(flag, "--device")) device = !strcmp(v, "0") ? 0 : swin_number(v);
#ifdef SWIN_DIAGNOSTIC_TRACE
        else if (!strcmp(flag, "--trace-dir")) swin_trace_dir = v;
#endif
        else goto usage;
    }
    if (!model || !input || !output || h < 1 || w < 1 || threads < 1 || threads > 128 ||
        device < 0 || (strcmp(backend, "cpu") && strcmp(backend, "cuda"))) goto usage;
    swin_gpu = !strcmp(backend, "cuda");
#ifndef SWIN_CUDA
    if (swin_gpu) { fprintf(stderr, "swin: use cuda/rmbg/swin_backbone for CUDA\n"); return 2; }
#endif
    omp_set_num_threads(threads);
    size_t count = (size_t)3*h*w;
    float *chw = swin_alloc(count*sizeof(float));
    FILE *f = fopen(input, "rb");
    if (!f) { perror(input); free(chw); return 2; }
    int bad = fread(chw, sizeof(float), count, f) != count || fgetc(f) != EOF;
    fclose(f);
    for (size_t i = 0; !bad && i < count; i++) if (!isfinite(chw[i])) bad = 1;
    if (bad) { fprintf(stderr, "swin: expected exact finite F32 CHW input\n"); free(chw); return 2; }
    swin_model *m = swin_load(model);
    if (!m) { free(chw); return 3; }
    swin_feature features[4] = {{0}};
#ifdef SWIN_CUDA
    if (swin_gpu && cuda_linear_f32_init(device)) goto done;
#endif
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    swin_predict(m, chw, h, w, features);
    clock_gettime(CLOCK_MONOTONIC, &end);
    for (int s = 0; s < 4; s++) {
        char path[4096];
        int len = snprintf(path, sizeof(path), "%s/feature_%d.f32", output, s);
        if (len < 0 || (size_t)len >= sizeof(path)) goto done;
        count = (size_t)features[s].c*features[s].h*features[s].w;
        for (size_t i = 0; i < count; i++) if (!isfinite(features[s].data[i])) {
            fprintf(stderr, "swin: nonfinite stage %d output\n", s); goto done;
        }
        f = fopen(path, "wb");
        if (!f) { perror(path); goto done; }
        bad = fwrite(features[s].data, sizeof(float), count, f) != count;
        bad |= fclose(f) != 0;
        if (bad) goto done;
    }
    printf("{\"execution\":\"%s\",\"seconds\":%.6f,\"features\":[",
           swin_gpu ? "cuda_gemm_cpu_attention" : "native_cpu",
           end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)*1e-9);
    for (int s = 0; s < 4; s++) printf("%s[%d,%d,%d]", s ? "," : "", features[s].c, features[s].h, features[s].w);
    puts("]}");
    rc = 0;
done:
#ifdef SWIN_CUDA
    cuda_linear_f32_free();
#endif
    for (int s = 0; s < 4; s++) free(features[s].data);
    swin_free(m); free(chw);
    return rc;
usage:
    fprintf(stderr, "swin_backbone --model RMBG.safetensors --input CHW.f32 --height H --width W "
                    "--output-dir EXISTING_DIR [--threads N] [--backend cpu|cuda] [--device N]\n"
                    "Dimensions must be 1..1024; batch one.\n");
    return 2;
}
#endif
