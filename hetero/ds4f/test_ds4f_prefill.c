/* Deterministic batched-prefill gate for the AMD/x86 DS4F path.
 *
 * This uses a small synthetic graph so it is safe to run on a workstation. It
 * compares the new M>1 prefill path with the existing exact token path and
 * reports both throughput and greedy-token agreement. Real staged models use
 * the same ds4f_forward_prefill implementation. */

#include "../../common/ds4f.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--batch M] [--iters N] [--threads N] [--max-pos N]\n", prog);
}

static void make_options(ds4f_runtime_options *o, int threads, int max_pos) {
    ds4f_runtime_options_init(o);
    ds4f_config c = ds4f_default_config();
    c.n_layers = 2;
    c.hidden = 64;
    c.vocab = 128;
    c.n_heads = 2;
    c.q_head_dim = 32;
    c.qk_rope_dim = 8;
    c.q_lora = 16;
    c.kv_lora = 32;
    c.o_inter = 64;
    c.o_groups = 2;
    c.o_lora = 32;
    c.n_experts = 4;
    c.n_active = 2;
    c.moe_inter = 32;
    c.shared_inter = 32;
    c.window_size = 16;
    c.max_pos = max_pos;
    for (int i = 0; i < 64; i++) c.compress_ratios[i] = 0;
    o->cfg = c;
    o->n_threads = threads;
    o->n_cmgs = 1;
    o->exact = 1;
    o->dense_bf16 = 0;
}

int main(int argc, char **argv) {
    int batch = 16, iters = 4, threads = 4, max_pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-pos") == 0 && i + 1 < argc) max_pos = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }
    if (batch < 2 || batch > 128 || iters < 1 || threads < 1) {
        usage(argv[0]); return 2;
    }
    if (max_pos < batch * iters + 1) max_pos = batch * iters + 1;

    ds4f_runtime_options ob, os;
    make_options(&ob, threads, max_pos);
    make_options(&os, threads, max_pos);
    ds4f_model *batched = ds4f_alloc_synth_opts(&ob);
    ds4f_model *serial = ds4f_alloc_synth_opts(&os);
    if (!batched || !serial) {
        fprintf(stderr, "prefill gate: synthetic model allocation failed\n");
        ds4f_free(batched); ds4f_free(serial);
        return 1;
    }
    ds4f_alloc_prefill_batch(batched, batch);
    int C = batched->cfg.hidden;
    float *xb = (float *)ds4f_mem_alloc(batched->mem,
                                        (size_t)batch * C * sizeof(float), 256, 0);
    float *xs = (float *)ds4f_mem_alloc(serial->mem, (size_t)C * sizeof(float), 256, 0);
    int *out_b = (int *)ds4f_mem_alloc(batched->mem, (size_t)batch * iters * sizeof(int), 64, 0);
    int *out_s = (int *)ds4f_mem_alloc(serial->mem, (size_t)batch * iters * sizeof(int), 64, 0);
    if (!xb || !xs || !out_b || !out_s) {
        fprintf(stderr, "prefill gate: scratch allocation failed\n");
        ds4f_free(batched); ds4f_free(serial);
        return 1;
    }

    double tb0 = now_seconds();
    for (int it = 0; it < iters; it++) {
        for (int k = 0; k < batch; k++) {
            int pos = it * batch + k;
            float *x = xb + (size_t)k * C;
            for (int d = 0; d < C; d++)
                x[d] = (float)(((pos + 3) * (d + 11)) % 101 - 50) / 37.0f;
        }
        ds4f_forward_prefill(batched, xb, batch, it * batch, out_b + it * batch);
    }
    double tb = now_seconds() - tb0;

    double ts0 = now_seconds();
    for (int it = 0; it < iters; it++) {
        for (int k = 0; k < batch; k++) {
            int pos = it * batch + k;
            for (int d = 0; d < C; d++)
                xs[d] = (float)(((pos + 3) * (d + 11)) % 101 - 50) / 37.0f;
            out_s[it * batch + k] = ds4f_forward_token(serial, xs, pos);
        }
    }
    double ts = now_seconds() - ts0;

    int mismatches = 0;
    for (int i = 0; i < batch * iters; i++) if (out_b[i] != out_s[i]) mismatches++;
    printf("prefill gate: batch=%d tokens=%d threads=%d batched=%.3f tok/s "
           "token=%.3f tok/s speedup=%.3fx argmax_mismatch=%d\n",
           batch, batch * iters, threads, (batch * iters) / tb,
           (batch * iters) / ts, ts / tb, mismatches);

    ds4f_free(batched);
    ds4f_free(serial);
    return mismatches == 0 ? 0 : 1;
}
