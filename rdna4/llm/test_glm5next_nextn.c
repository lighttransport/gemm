#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "hip_llm_runner.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    gguf_shards *model = gguf_open_shards(argv[1], 1);
    hip_llm_runner *runner = hip_llm_init(0, 1);
    hip_llm_load_options options;
    float *trunk, *draft;
    int best = -1;
    if (!model || !runner) return 1;
    hip_llm_load_options_default(&options);
    options.max_seq_len = 4;
    if (hip_llm_load_weights_sharded(runner, model, &options) != 0) return 1;
    trunk = hip_llm_forward_logits(runner, 87, 0);
    if (!trunk || hip_llm_forward_nextn_logits(runner, 88, 0) != NULL) {
        fprintf(stderr, "GLM5Next NextN accepted an invalid draft position\n");
        hip_llm_free(runner); gguf_close_shards(model); return 1;
    }
    draft = hip_llm_forward_nextn_logits(runner, 88, 1);
    if (!trunk || !draft) {
        fprintf(stderr, "GLM5Next NextN forward failed\n");
        hip_llm_free(runner); gguf_close_shards(model); return 1;
    }
    {
        float *draft2 = hip_llm_forward_nextn_logits(runner, 89, 2);
        if (!draft2) {
            fprintf(stderr, "GLM5Next chained NextN forward failed\n");
            hip_llm_free(runner); gguf_close_shards(model); return 1;
        }
        for (int i = 0; i < hip_llm_n_vocab(runner); ++i) {
            if (!isfinite(draft2[i])) {
                fprintf(stderr, "non-finite chained NextN logit at %d\n", i);
                hip_llm_free(runner); gguf_close_shards(model); return 1;
            }
        }
    }
    best = 0;
    for (int i = 0; i < hip_llm_n_vocab(runner); ++i) {
        if (!isfinite(draft[i])) {
            fprintf(stderr, "non-finite NextN logit at %d\n", i);
            hip_llm_free(runner); gguf_close_shards(model); return 1;
        }
        if (draft[i] > draft[best]) best = i;
    }
    printf("GLM5NEXT_NEXTN PASS draft_argmax=%d\n", best);
    hip_llm_free(runner);
    gguf_close_shards(model);
    return 0;
}
