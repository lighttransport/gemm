#include <math.h>
#include <stdio.h>
#include <string.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "../../common/glm5next_cpu_runtime.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    gguf_shards *m = gguf_open_shards(argv[1], 2); if (!m) return 1;
    glm5next_cpu_runtime r; char err[160];
    if (glm5next_cpu_runtime_init(&r, m, 2, err, sizeof(err)) != 0) {
        fprintf(stderr, "runtime init failed: %s\n", err); gguf_close_shards(m); return 1;
    }
    if (glm5next_cpu_runtime_step(&r, 1, 0) != 0 ||
        glm5next_cpu_runtime_step(&r, 2, 1) != 0) {
        fprintf(stderr, "two-token runtime failed\n"); glm5next_cpu_runtime_free(&r); gguf_close_shards(m); return 1;
    }
    float *nextn = glm5next_cpu_runtime_nextn_logits(&r, 3, 1);
    if (!nextn) { fprintf(stderr, "NextN runtime failed\n"); return 1; }
    for (int i = 0; i < r.config.vocab_size; ++i)
        if (!isfinite(nextn[i])) { fprintf(stderr, "non-finite NextN logit\n"); return 1; }
    for (int i = 0; i < r.config.vocab_size; ++i)
        if (!isfinite(r.logits[i])) { fprintf(stderr, "non-finite logit\n"); return 1; }
    int best = 0; for (int i = 1; i < r.config.vocab_size; ++i) if (r.logits[i] > r.logits[best]) best = i;
    int nextn_best = 0;
    for (int i = 1; i < r.config.vocab_size; ++i) if (nextn[i] > nextn[nextn_best]) nextn_best = i;
    printf("GLM5NEXT_CPU_RUNTIME PASS positions=2 argmax=%d nextn_argmax=%d\n", best, nextn_best);
    glm5next_cpu_runtime_free(&r); gguf_close_shards(m); return 0;
}
