/*
 * bench_praxis_synth.c - full-size SYNTHETIC praxis-1 model on one GPU: random weights at the inferred real dims
 * (config from gen_synth_config.py), routed expert banks as random pinned-host memory streamed through a device ring,
 * everything else random bf16 device buffers. Measures VRAM, per-pass / per-call latency, H2D bandwidth, time split.
 *
 *   cuda/praxis/bench_praxis_synth --synthetic $PRAXIS_ROOT/synth [--stream 2] [--K 2] [--S 64] [--h 13 --w 16]
 *        [--cfg-video 4.0] [--cfg-action 1.0] [--steps 2] [--synth-banks 4] [--repeat 2] [--no-call] [--no-h2d] [--profile]
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_PRAXIS_IMPLEMENTATION
#include "cuda_praxis_runner.h"
#include <time.h>
#include "../../common/praxis_paths.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) {   /* approx normal: sum of 4 uniforms */
    float a = 0;
    for (int i = 0; i < 4; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a += (float)(rs >> 40) / 16777216.0f; }
    return (a - 2.0f) * 1.732f;
}
static float *randv(size_t n) { float *p = (float *)malloc(n * 4); for (size_t i = 0; i < n; i++) p[i] = rnd(); return p; }
static void mem_report(const char *tag) {
    size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot);
    printf("[%s] device in use per cuMemGetInfo (incl. CUDA ctx/cuBLAS): %.2f GB of %.2f GB\n", tag, (tot - fr) / 1073741824.0, tot / 1073741824.0);
}
static void stats(praxis_runner *r, const char *tag) {
    praxis_stats s; praxis_stats_get(r, &s);
    printf("[%s] own accounting: peak %.2f GB (now %.2f) = weights %.2f + routed ring %.2f (%d x %.2f GB) + workspace/state %.2f; routed total %.1f GB; cuMemGetInfo peak use %.2f GB\n",
           tag, s.dev_bytes_peak / 1073741824.0, s.dev_bytes_now / 1073741824.0, s.weights_bytes / 1073741824.0, s.routed_dev_bytes / 1073741824.0,
           s.ring_slots, s.ring_slot_bytes / 1073741824.0, (s.dev_bytes_peak - s.weights_bytes - s.routed_dev_bytes) / 1073741824.0,
           s.routed_total_bytes / 1073741824.0, s.free_min_bytes_used / 1073741824.0);
}

int main(int argc, char **argv) {
    const char *dir = px_root("synth");
    int ctx = 0, ring = 2, K = 2, S = 64, h = 13, w = 16, steps = 2, nbanks = 4, repeat = 2, do_call = 1, prof = 0, skip_h2d = 0, exact = 0, pair = 0, skipv = 0;
    double cfg_v = 4.0, cfg_a = 1.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--synthetic") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--stream") && i + 1 < argc) ring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--S") && i + 1 < argc) S = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--synth-banks") && i + 1 < argc) nbanks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-video") && i + 1 < argc) cfg_v = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-action") && i + 1 < argc) cfg_a = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-call")) do_call = 0;
        else if (!strcmp(argv[i], "--no-h2d")) skip_h2d = 1;
        else if (!strcmp(argv[i], "--profile")) prof = 1;
        else if (!strcmp(argv[i], "--exact")) exact = 1;
        else if (!strcmp(argv[i], "--pair")) pair = 1;
        else if (!strcmp(argv[i], "--ctx-cache")) ctx = 1;
        else if (!strcmp(argv[i], "--skip-last-video")) skipv = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (ctx) setenv("PRAXIS_CTX_CACHE", "1", 1);
    if (skip_h2d) setenv("PRAXIS_SKIP_H2D", "1", 1);
    if (prof) setenv("PRAXIS_PROFILE", "1", 1);
    if (exact) setenv("PRAXIS_EXACT", "1", 1);
    double t0 = now();
    praxis_runner *r = praxis_load_synthetic(dir, 0, 1, nbanks);
    if (!r) return 1;
    printf("synthetic model loaded in %.1f s (random pinned banks: %d distinct)\n", now() - t0, nbanks);
    if (praxis_set_stream(r, ring)) return 1;
    int F = praxis_lat_per_chunk(r), vc = praxis_video_c(r), A = praxis_action_dim(r), P = praxis_proprio_dim(r), TC = praxis_text_dim(r);
    size_t nv = (size_t)vc * K * F * h * w, nc = (size_t)vc * (K - 1) * F * h * w, na = (size_t)K * F * A;
    float *text = randv((size_t)S * TC), *text_u = randv((size_t)S * TC), *anchor = randv((size_t)vc * h * w);
    float *noised = randv(nv), *clean = randv(nc ? nc : 1), *proprio = randv((size_t)K * P), *noised_a = randv(na);
    float *noise_v = randv(nv), *noise_a = randv(na);
    float tt[16]; for (int k = 0; k < K; k++) tt[k] = 0.9f - 0.1f * k;
    float *vel = (float *)malloc(nv * 4), *aout = (float *)malloc(na * 4);
    mem_report("after load");
    stats(r, "after load");

    printf("window: K'=%d chunks x %d frames, latent %dx%d (%d enc tokens/frame), text %d, ring=%d%s\n", K, F, h, w, h * w, S, ring, skip_h2d ? " (NO H2D: compute-only)" : "");
    for (int it = 0; it < repeat; it++) {
        double ta = now();
        if (praxis_encoder(r, text, S, anchor, h, w, noised, clean, tt, K)) return 1;
        cuCtxSynchronize();
        double tb = now();
        if (praxis_decoder(r, noised, tt, vel)) return 1;
        double tc = now();
        if (praxis_expert(r, noised_a, tt, proprio, aout)) return 1;
        double td = now();
        printf("pass %d: encoder %.3f s | decoder %.3f s | action expert %.3f s | total %.3f s%s\n", it, tb - ta, tc - tb, td - tc, td - ta, it == 0 ? " (includes first-use allocations)" : "");
    }
    {
        praxis_stats s; praxis_stats_get(r, &s);
        printf("after passes: H2D %.2f GB in %ld loads (%ld prefetch hits)\n", s.h2d_bytes / 1073741824.0, s.n_loads, s.n_prefetch_hits);
    }
    if (prof) {
        praxis_prof_reset(r);
        if (praxis_encoder(r, text, S, anchor, h, w, noised, clean, tt, K)) return 1;
        if (praxis_decoder(r, noised, tt, vel)) return 1;
        if (praxis_expert(r, noised_a, tt, proprio, aout)) return 1;
        printf("profile of one full pass (encoder+decoder+expert):\n");
        praxis_prof_report(r, stdout);
    }
    stats(r, "after passes");
    if (do_call) {
        float *vo = (float *)malloc((size_t)vc * F * h * w * 4), *ao = (float *)malloc((size_t)F * A * 4);
        for (int it = 0; it < repeat; it++) {
            praxis_stats s0, s1; praxis_stats_get(r, &s0);
            double ta = now();
            if (praxis_sample_window_joint_ex(r, text, text_u, S, anchor, h, w, clean, proprio, noise_v, noise_a, K, steps, 12.5, cfg_v, cfg_a, vo, ao, pair, skipv)) return 1;
            double tb = now();
            praxis_stats_get(r, &s1);
            double gb = (s1.h2d_bytes - s0.h2d_bytes) / 1073741824.0;
            printf("window_joint call %d (K'=%d, %d steps, cfg video %.1f action %.1f%s%s): %.3f s | H2D %.2f GB = %.1f GB/s | %ld expert loads\n",
                   it, K, steps, cfg_v, cfg_a, pair ? ", paired encoders" : "", skipv ? (ctx ? ", skip-last-video, ctx-cache" : ", skip-last-video") : (ctx ? ", ctx-cache" : ""), tb - ta, gb, gb / (tb - ta), s1.n_loads - s0.n_loads);
        }
        if (prof) praxis_prof_report(r, stdout);
    }
    stats(r, "final");
    mem_report("final");
    return 0;
}
