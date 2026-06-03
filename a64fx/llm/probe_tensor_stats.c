/* probe_tensor_stats.c — load a (possibly split) GGUF via gguf_open_multi and
 * dump value statistics for named tensors. Tests whether the 2-file 27B split
 * loader returns SANE weights for the non-tied lm_head (output.weight) and for
 * tensors living in shard 2. A zero / NaN / tiny output.weight => uniform logits
 * => argmax token 0 => the "!!!!" degenerate output.
 *
 * Build (native A64FX):
 *   fcc -Nclang -O2 -march=armv8.2-a+sve -o build/probe_tensor_stats \
 *       probe_tensor_stats.c -lm
 * Run:
 *   ./build/probe_tensor_stats <any-shard.gguf> [tensor_name ...]
 */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) { u = sign; }
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3ff;
            u = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        u = sign | 0x7f800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static void stats_for(const gguf_context *c, const char *want) {
    int idx = -1;
    for (uint64_t i = 0; i < c->n_tensors; i++) {
        const char *nm = gguf_tensor_name(c, i);
        if (nm && strcmp(nm, want) == 0) { idx = (int)i; break; }
    }
    if (idx < 0) { printf("  [MISSING] %s\n", want); return; }
    const gguf_tensor_info *ti = &c->tensors[idx];
    uint64_t ne = 1; for (uint32_t d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    int seg = c->tensor_seg ? c->tensor_seg[idx] : 0;
    void *data = gguf_tensor_data(c, idx);
    printf("  %-28s seg=%d type=%u(%s) ndim=%u dims=[%llu,%llu,%llu,%llu] ne=%llu data=%p\n",
           want, seg, ti->type, ggml_type_name(ti->type), ti->n_dims,
           (unsigned long long)ti->dims[0], (unsigned long long)ti->dims[1],
           (unsigned long long)ti->dims[2], (unsigned long long)ti->dims[3],
           (unsigned long long)ne, data);
    if (!data) { printf("    !! data pointer NULL (split-load failed for this tensor)\n"); return; }
    if (ti->type != GGML_TYPE_F32 && ti->type != GGML_TYPE_F16 && ti->type != GGML_TYPE_BF16) {
        printf("    (quantized type — value stats skipped)\n"); return;
    }
    double sum = 0, sumabs = 0; float mn = INFINITY, mx = -INFINITY;
    uint64_t nz = 0, nnan = 0, ninf = 0;
    const uint16_t *u16 = (const uint16_t *)data;
    const float    *f32 = (const float *)data;
    for (uint64_t i = 0; i < ne; i++) {
        float v;
        if (ti->type == GGML_TYPE_F32) v = f32[i];
        else if (ti->type == GGML_TYPE_BF16) v = bf16_to_f32(u16[i]);
        else v = f16_to_f32(u16[i]);
        if (isnan(v)) { nnan++; continue; }
        if (isinf(v)) { ninf++; continue; }
        if (v == 0.0f) nz++;
        double a = v < 0 ? -v : v;
        sum += v; sumabs += a;
        if (v < mn) mn = v; if (v > mx) mx = v;
    }
    uint64_t finite = ne - nnan - ninf;
    printf("    min=%.6g max=%.6g mean=%.6g abs_mean=%.6g  zeros=%llu/%llu (%.2f%%) nan=%llu inf=%llu\n",
           mn, mx, finite ? sum / (double)finite : 0.0,
           finite ? sumabs / (double)finite : 0.0,
           (unsigned long long)nz, (unsigned long long)ne,
           ne ? 100.0 * (double)nz / (double)ne : 0.0,
           (unsigned long long)nnan, (unsigned long long)ninf);
    /* a few raw samples (head, mid, tail) */
    uint64_t pos[3] = { 0, ne / 2, ne ? ne - 1 : 0 };
    printf("    samples:");
    for (int k = 0; k < 3; k++) {
        uint64_t i = pos[k]; float v;
        if (ti->type == GGML_TYPE_F32) v = f32[i];
        else if (ti->type == GGML_TYPE_BF16) v = bf16_to_f32(u16[i]);
        else v = f16_to_f32(u16[i]);
        printf("  [%llu]=%.6g", (unsigned long long)i, v);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <shard.gguf> [tensor ...]\n", argv[0]); return 2; }
    const char *path = argv[1];
    printf("opening (multi): %s\n", path);
    gguf_context *c = gguf_open_multi(path, /*use_mmap=*/1);
    if (!c) { fprintf(stderr, "FAILED to open %s\n", path); return 1; }
    printf("n_files=%d n_tensors=%llu  (tensor_seg=%s)\n",
           c->n_files, (unsigned long long)c->n_tensors,
           c->tensor_seg ? "present(split)" : "NULL(single)");
    /* segment membership histogram */
    if (c->tensor_seg) {
        int cnt[16] = {0};
        for (uint64_t i = 0; i < c->n_tensors; i++) {
            int s = c->tensor_seg[i]; if (s >= 0 && s < 16) cnt[s]++;
        }
        for (int s = 0; s < c->n_files; s++)
            printf("  seg %d: %d tensors, data_span=%zu bytes, base=%p\n",
                   s, cnt[s], c->seg_data_size[s], (void*)c->seg_data[s]);
    }
    const char *defaults[] = {
        "output.weight", "token_embd.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.63.ffn_norm.weight",
    };
    printf("--- tensor stats ---\n");
    if (argc > 2) {
        for (int a = 2; a < argc; a++) stats_for(c, argv[a]);
    } else {
        for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]); i++)
            stats_for(c, defaults[i]);
    }
    gguf_close(c);
    return 0;
}
