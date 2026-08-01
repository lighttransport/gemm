/* Single-token long-context attention benchmark.  Unlike attn_bench.c, which
 * measures the chunked prefill path, this exercises attention_core exactly as
 * decode does and makes thread-count/load-balance experiments inexpensive.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -DLAGUNA_FP8 -I../../common -I../utofu-tests \
 *          -o build/decode_attn_bench decode_attn_bench.c -lm
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./build/decode_attn_bench 32768 20
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"
#ifdef _OPENMP
#include <omp.h>
#endif

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static float rand_float(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((rng_state >> 33) % 2000) - 1000) / 1000.0f;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int context = argc > 1 ? atoi(argv[1]) : 32768;
    int iterations = argc > 2 ? atoi(argv[2]) : 20;
    int sliding = argc > 3 && strcmp(argv[3], "sliding") == 0;
    if (context < 1 || iterations < 1) {
        fprintf(stderr, "context and iterations must be positive\n");
        return 2;
    }
    int hd = LAGUNA_HEAD_DIM;
    int nh = sliding ? LAGUNA_SLIDING_HEADS : LAGUNA_FULL_HEADS;
    int kv_stride = LAGUNA_KV_HEADS * hd;
    int max_pos = context + 8;

    laguna_model m;
    memset(&m, 0, sizeof(m));
    m.n_layers = 1;
    m.max_pos = max_pos;
    m.n_seq = 1;
    laguna_layer *ly = &m.layers[0];
    ly->num_heads = nh;
    ly->is_sliding = sliding;
    uint16_t *q_norm = malloc((size_t)hd * sizeof(uint16_t));
    uint16_t *k_norm = malloc((size_t)hd * sizeof(uint16_t));
    ly->q_norm = q_norm;
    ly->k_norm = k_norm;
    for (int i = 0; i < hd; ++i) {
        q_norm[i] = laguna_f32_to_bf16(1.0f);
        k_norm[i] = laguna_f32_to_bf16(1.0f);
    }

    int full_half = LAGUNA_ROPE_FULL_DIM / 2;
    int slide_half = LAGUNA_ROPE_SLIDING_DIM / 2;
    m.full_cos = malloc((size_t)max_pos * full_half * sizeof(float));
    m.full_sin = malloc((size_t)max_pos * full_half * sizeof(float));
    m.swa_cos = malloc((size_t)max_pos * slide_half * sizeof(float));
    m.swa_sin = malloc((size_t)max_pos * slide_half * sizeof(float));
    laguna_build_rope_tables(&m);

    m.kv_cap[0] = max_pos;
    size_t kv_elements = (size_t)max_pos * kv_stride;
    m.kcache = aligned_alloc(256, kv_elements * sizeof(uint16_t));
    m.vcache = aligned_alloc(256, kv_elements * sizeof(uint16_t));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < (long)kv_elements; ++i) {
        uint32_t x = (uint32_t)i * 2654435761u;
        m.kcache[i] = laguna_f32_to_bf16(((int)(x & 0xffff) - 32768) / 65536.0f);
        m.vcache[i] = laguna_f32_to_bf16(((int)(x >> 16) - 32768) / 65536.0f);
    }

    laguna_scratch sc;
    scratch_alloc(&sc, max_pos);
    float *q0 = aligned_alloc(256, (size_t)nh * hd * sizeof(float));
    float *k0 = aligned_alloc(256, (size_t)kv_stride * sizeof(float));
    float *v0 = aligned_alloc(256, (size_t)kv_stride * sizeof(float));
    float *g0 = malloc((size_t)nh * sizeof(float));
    float *out = aligned_alloc(256, (size_t)nh * hd * sizeof(float));
    for (int i = 0; i < nh * hd; ++i) q0[i] = rand_float();
    for (int i = 0; i < kv_stride; ++i) {
        k0[i] = rand_float();
        v0[i] = rand_float();
    }
    for (int i = 0; i < nh; ++i) g0[i] = rand_float();

    float *q = aligned_alloc(256, (size_t)nh * hd * sizeof(float));
    float *k = aligned_alloc(256, (size_t)kv_stride * sizeof(float));
    memcpy(q, q0, (size_t)nh * hd * sizeof(float));
    memcpy(k, k0, (size_t)kv_stride * sizeof(float));
    attention_core(&m, ly, &sc, 0, 0, context - 1, nh, q, k, v0, g0, out);

    double start = now_s();
    for (int it = 0; it < iterations; ++it) {
        memcpy(q, q0, (size_t)nh * hd * sizeof(float));
        memcpy(k, k0, (size_t)kv_stride * sizeof(float));
        attention_core(&m, ly, &sc, 0, 0, context - 1, nh, q, k, v0, g0, out);
    }
    double elapsed = now_s() - start;
    double checksum = 0.0;
    for (int i = 0; i < nh * hd; ++i) checksum += out[i];
    printf("context=%d type=%s base_threads=%d %.3f ms/layer checksum=%.9g\n",
           context, sliding ? "sliding" : "full",
#ifdef _OPENMP
           omp_get_max_threads(),
#else
           1,
#endif
           elapsed * 1e3 / iterations, checksum);
    return 0;
}
