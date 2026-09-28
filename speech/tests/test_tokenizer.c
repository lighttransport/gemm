/* SPDX-License-Identifier: MIT
 * test_tokenizer <model_dir> <text>: prints assistant-formatted token ids. */
#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#include "safetensors.h"
#include "gguf_loader.h"
#include "bpe_tokenizer.h"
#include "qtts_tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    bpe_vocab *v = qtts_tokenizer_load(argv[1]);
    if (!v) return 1;
    char buf[8192];
    snprintf(buf, sizeof(buf), "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", argv[2]);
    int32_t ids[4096];
    int n = bpe_tokenize(v, buf, -1, ids, 4096);
    printf("[");
    for (int i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
    bpe_vocab_free(v);
    return 0;
}
