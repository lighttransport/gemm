/* Metadata/load smoke for Qwen3.8 combined and split GGUFs.  Uses lazy mmap and
 * never executes a weight matrix, so it is safe on a 32 GiB A64FX node. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    setenv("GGUF_LAZY_MMAP", "1", 1);
    gguf_context *g = gguf_open_multi(argv[1], 1);
    if (!g) return 3;
    transformer_model *m = transformer_load(g, 16);
    if (!m) { gguf_close(g); return 4; }
    printf("QWEN38_LOAD tensors=%llu trunk=%d nextn=%d total=%d mtp=%d\n",
           (unsigned long long)g->n_tensors, m->n_layers,
           m->n_nextn_layers, m->n_layers_all, m->nextn.loaded);
    int ok = g->n_tensors == 866 && m->n_layers == 64 &&
             m->n_nextn_layers == 1 && m->nextn.loaded;
    transformer_free(m);
    gguf_close(g);
    return ok ? 0 : 5;
}
