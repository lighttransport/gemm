/*
 * test_praxis_shards.c - sharded safetensors loading + host-RSS boundedness test of the praxis runner.
 *
 *   test_praxis_shards --model DIR [--nvfp4] [--calib DIR2] [--check-bound] [--pack-cache FILE]
 *
 * Loads DIR (model.safetensors.index.json + N shards or a single file) and reports peak host RSS (VmHWM) growth of the load.
 * --calib DIR2: load (and free) a small model first so the one-off library/driver RSS (CUDA, cuBLAS, NVRTC) is already in the
 * HWM baseline and the growth measured for DIR is the weights only.
 * --check-bound: with --nvfp4 assert  growth <= packed banks + one bf16 layer (x3 slack) + non-routed weights + 96 MB;
 * the bf16 total of the checkpoint is printed so the bound can be compared with "load everything" (PRAXIS_NO_DROP=1 disables
 * the page dropping and shows the unbounded behaviour: file pages of the whole mapped checkpoint stay resident).
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_PRAXIS_IMPLEMENTATION
#include "cuda_praxis_runner.h"

static size_t kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r"); char line[256]; size_t v = 0, kl = strlen(key);
    while (f && fgets(line, sizeof line, f)) if (!strncmp(line, key, kl)) { v = (size_t)atoll(line + kl + 1); break; }
    if (f) fclose(f);
    return v;
}
int main(int argc, char **argv) {
    const char *model = NULL, *calib = NULL; int nvfp4 = 0, check = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--calib") && i + 1 < argc) calib = argv[++i];
        else if (!strcmp(argv[i], "--nvfp4")) nvfp4 = 1;
        else if (!strcmp(argv[i], "--check-bound")) check = 1;
        else if (!strcmp(argv[i], "--pack-cache") && i + 1 < argc) setenv("PRAXIS_PACK_CACHE", argv[++i], 1);
    }
    if (!model) { fprintf(stderr, "usage: test_praxis_shards --model DIR [--nvfp4] [--calib DIR2] [--check-bound] [--pack-cache FILE]\n"); return 2; }
    if (nvfp4) setenv("PRAXIS_NVFP4", "1", 1);
    if (calib) {
        praxis_runner *c = praxis_load(calib, 0, 0);
        if (!c) return 1;
        praxis_free(c);
    }
    size_t base_hwm = kb("VmHWM:"), base_rss = kb("VmRSS:");
    double t0 = (double)clock() / CLOCKS_PER_SEC; (void)t0;
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    praxis_runner *r = praxis_load(model, 0, 1);
    clock_gettime(CLOCK_MONOTONIC, &b);
    if (!r) return 1;
    praxis_stats s; praxis_stats_get(r, &s);
    size_t hwm = kb("VmHWM:"), rss = kb("VmRSS:");
    double growth_mb = (hwm - base_hwm) / 1024.0;
    printf("shards %d | load %.2f s | routed bytes on host %.1f MB | non-routed weights %.1f MB\n", s.n_shards,
           (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec), s.routed_total_bytes / 1048576.0, s.weights_bytes / 1048576.0);
    printf("host RSS: baseline HWM %.0f MB -> load HWM %.0f MB (growth %.0f MB), final RSS %.0f MB (RSS growth %.0f MB)\n",
           base_hwm / 1024.0, hwm / 1024.0, growth_mb, rss / 1024.0, (rss - base_rss) / 1024.0);
    printf("pack cache: hits %d written %d\n", s.pack_cache_hits, s.pack_cache_written);
    int rc = 0;
    if (check) {
        /* bf16 size of one MoE layer is not stored once NVFP4-packed; recompute from the config-independent packed size:
         * packed ~ 0.2813 x bf16 (E2M1 + e4m3/16 + globals), so bf16 layer ~ packed layer / 0.2813 */
        double packed_mb = s.routed_total_bytes / 1048576.0;
        double nl = (double)s.n_moe_layers_total;
        double bf16_layer_mb = packed_mb / nl / 0.2813;
        double bound = packed_mb + 3 * bf16_layer_mb + s.weights_bytes / 1048576.0 + 96;
        double unbounded = packed_mb / 0.2813 + packed_mb + s.weights_bytes / 1048576.0;
        printf("bound = packed %.0f + 3 x bf16 layer %.0f + weights %.0f + 96 = %.0f MB ; load-everything would be ~%.0f MB\n", packed_mb, 3 * bf16_layer_mb,
               s.weights_bytes / 1048576.0, bound, unbounded);
        if (growth_mb > bound) { printf("FAIL: host RSS growth %.0f MB exceeds bound %.0f MB\n", growth_mb, bound); rc = 1; }
        else printf("PASS: host RSS growth %.0f MB <= bound %.0f MB\n", growth_mb, bound);
    }
    praxis_free(r);
    return rc;
}
