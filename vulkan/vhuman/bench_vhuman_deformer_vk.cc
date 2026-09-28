// Vulkan face deformer benchmark and CPU parity check.
//   ./bench_vhuman_deformer_vk rig_deformer.safetensors [frames]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vhuman_deformer_vk.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s rig_deformer.safetensors [frames]\n", argv[0]);
        return 2;
    }
    size_t frames = argc > 2 ? (size_t)atol(argv[2]) : 1024;
    vh_deformer *d = vh_deformer_load(argv[1]);
    if (!d) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    vh_vk *g = vh_vk_create(d, 0, 1);
    if (!g) { fprintf(stderr, "no Vulkan device\n"); vh_deformer_free(d); return 1; }
    size_t C = vh_deformer_controls(d), V = vh_deformer_vertices(d);
    std::vector<float> x(C * frames), out(V * 3 * frames), ref(V * 3);
    srand(1);
    for (auto &v : x) v = (rand() % 10 == 0) ? (float)rand() / RAND_MAX : 0.f;
    printf("%s: vertices %zu, morphs %zu, frames %zu\n", vh_vk_name(g), V, vh_deformer_morphs(d), frames);
    double ms[4];
    vh_vk_eval_batch(g, x.data(), frames, 1, out.data(), ms);          // warm up
    printf("memory: inputs %s, readback %s\n", (vh_vk_memory_flags(g) & 1) ? "device-local host-visible" : "host",
           (vh_vk_memory_flags(g) & 2) ? "host-cached" : "uncached");
    for (int ml = 1; ml >= 0; --ml) {
        if (vh_vk_eval_batch(g, x.data(), frames, ml, out.data(), ms)) { fprintf(stderr, "dispatch failed\n"); return 1; }
        double err = 0;
        for (size_t f = 0; f < frames; f += frames / 8 + 1) {
            vh_deformer_eval(d, x.data() + f * C, ml, ref.data());
            for (size_t i = 0; i < V * 3; ++i) err = std::fmax(err, std::fabs(ref[i] - out[f * V * 3 + i]));
        }
        printf("ml=%d  dispatch %.3f ms (%.2f us/frame)  host rig %.3f ms  upload %.3f ms  download %.3f ms  max|gpu-cpu| %.2e m\n",
               ml, ms[2], ms[2] * 1e3 / frames, ms[0], ms[1], ms[3], err);
    }
    vh_vk_free(g);
    vh_deformer_free(d);
    return 0;
}
