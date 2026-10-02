/* Tokenizer-only validation for native identity generation; no model runtime. */
#define _GNU_SOURCE
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/bpe_tokenizer.h"
#include <stdio.h>
int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "tokenizer_probe VOCAB.gguf TEXT\n"); return 2; }
    gguf_context *gguf = gguf_open(argv[1], 1);
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) { if (gguf) gguf_close(gguf); return 2; }
    int32_t ids[4096];
    int count = bpe_tokenize(vocab, argv[2], (int)strlen(argv[2]), ids, 4096);
    if (count < 0) { bpe_vocab_free(vocab); gguf_close(gguf); return 2; }
    putchar('[');
    for (int i=0; i<count; i++) printf("%s%d", i ? "," : "", ids[i]);
    puts("]");
    bpe_vocab_free(vocab); gguf_close(gguf); return 0;
}
