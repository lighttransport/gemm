/*
 * Test program for bpe_tokenizer.h
 * Usage: ./test_bpe_tokenizer <model.gguf>
 */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#include <stdio.h>
#include <string.h>

static void test_tokenize(const bpe_vocab *vocab, const char *text) {
    int text_len = (int)strlen(text);
    int32_t tokens[1024];
    int n = bpe_tokenize(vocab, text, text_len, tokens, 1024);

    printf("Text: \"%s\"\n", text);
    printf("Tokens (%d): [", n);
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%d", tokens[i]);
    }
    printf("]\n");

    /* Decode back (GPT-2 byte-level decode) */
    printf("Decoded: \"");
    for (int i = 0; i < n; i++) {
        const char *s = bpe_token_to_str(vocab, tokens[i]);
        if (s) {
            int dec_len;
            char *dec = bpe_byte_decode(s, (int)strlen(s), &dec_len);
            printf("%.*s", dec_len, dec);
            free(dec);
        } else {
            printf("<?>");
        }
    }
    printf("\"\n\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    printf("Loading GGUF: %s\n", argv[1]);
    gguf_context *gguf = gguf_open(argv[1], 1);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF file\n");
        return 1;
    }
    printf("GGUF loaded: %llu KV pairs, %llu tensors\n",
           (unsigned long long)gguf->n_kv, (unsigned long long)gguf->n_tensors);

    printf("Loading vocab...\n");
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "Failed to load vocab\n");
        gguf_close(gguf);
        return 1;
    }
    printf("Vocab loaded: %d tokens\n", vocab->n_tokens);
    printf("Special tokens: bos=%d eos=%d eot=%d pad=%d\n",
           vocab->bos_id, vocab->eos_id, vocab->eot_id, vocab->pad_id);
    printf("\n");

    /* Test strings */
    test_tokenize(vocab, "Hello world!");
    test_tokenize(vocab, " The quick brown fox");
    test_tokenize(vocab, "Hello, 你好世界! 123");
    test_tokenize(vocab, "I'm going to the store, and I'll buy some food.");
    test_tokenize(vocab, "  \n\n  Hello");
    test_tokenize(vocab, "test123abc");

    bpe_vocab_free(vocab);
    gguf_close(gguf);
    return 0;
}
