#include "k3_gguf_expert_tp.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (k3_gguf_tp_role("w1") != 1 || k3_gguf_tp_role("w2") != 2 ||
        k3_gguf_tp_role("w3") != 3 || k3_gguf_tp_kind("rows") != 1 ||
        k3_gguf_tp_kind("cols") != 2) return 1;
    if (argc == 1) {
        puts("k3_gguf_expert_tp: PASS (helper vocabulary)");
        return 0;
    }
    if (argc != 5) {
        fprintf(stderr, "usage: %s MANIFEST RANK NODES LAYER\n", argv[0]);
        return 2;
    }
    k3_gguf_tp_manifest m;
    if (k3_gguf_tp_manifest_load(&m, argv[1], atoi(argv[2]), atoi(argv[3]),
                                 atoi(argv[4]), 0)) {
        fprintf(stderr, "k3_gguf_expert_tp: manifest load failed\n");
        return 3;
    }
    if (!m.count || m.blob_bytes == 0) {
        k3_gguf_tp_manifest_free(&m);
        return 4;
    }
    printf("k3_gguf_expert_tp: PASS segments=%zu blob_bytes=%zu\n",
           m.count, m.blob_bytes);
    k3_gguf_tp_manifest_free(&m);
    return 0;
}
