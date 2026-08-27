#include <stdio.h>
#include <string.h>
#include "../../common/glm5_chat_template.h"
#include "../../common/glm53f_arch.h"

int main(void) {
    char out[512];
    glm5_chat_message messages[] = {
        {"user", "Hello", NULL},
    };
    glm53f_arch arch = glm53f_arch_default();
    if (arch.n_layers != 45 || arch.hidden_size != 4096 ||
        glm53f_layer_type(3) != GLM53F_SPARSE_ATTENTION ||
        glm53f_layer_type(4) != GLM53F_LINEAR_ATTENTION) return 1;
    int n = glm5_chat_template_render(messages, 1, "high", 1,
                                       out, sizeof(out));
    const char *want = "[gMASK]<sop><|system|>Reasoning Effort: High"
                       "<|user|>Hello<|assistant|><think>";
    if (n < 0 || strcmp(out, want) != 0) {
        fprintf(stderr, "chat template mismatch: %s\n", out);
        return 1;
    }
    puts(out);
    return 0;
}
