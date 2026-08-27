#include <stdio.h>
#include <stdlib.h>
#include "../../common/glm5_bpe.h"
#include "../../common/glm5_chat_template.h"

int main(int argc, char **argv) {
    glm5_bpe b;
    char text[4096];
    int ids[1024];
    glm5_chat_message m = {"user", "Hello from the C11 tokenizer", NULL};
    if (argc != 2) { fprintf(stderr, "usage: %s tokenizer.json\n", argv[0]); return 2; }
    int load_rc = glm5_bpe_load(argv[1], &b);
    if (load_rc != 0) { fprintf(stderr, "cannot load tokenizer: %d\n", load_rc); return 1; }
    if (b.gmask < 0 || b.sop < 0 || b.user < 0 || b.assistant < 0 || b.system < 0) {
        fprintf(stderr, "missing GLM5 special tokens\n"); glm5_bpe_free(&b); return 1;
    }
    if (glm5_chat_template_render(&m, 1, "max", 1, text, sizeof(text)) < 0) {
        fprintf(stderr, "template render failed\n"); glm5_bpe_free(&b); return 1;
    }
    int n = glm5_bpe_encode(&b, text, ids, (int)(sizeof(ids)/sizeof(ids[0])));
    if (n < 8) { fprintf(stderr, "too few tokens: %d\n", n); glm5_bpe_free(&b); return 1; }
    printf("vocab=%zu merges=%zu tokens=%d first=%d last=%d\n",
           b.n_vocab, b.n_merges, n, ids[0], ids[n-1]);
    glm5_bpe_free(&b);
    return 0;
}
