#include "k3_gguf_graph_adapter.h"

#include <stdio.h>
#include <string.h>

static int has_tensor(void *opaque, const char *name) {
    (void)opaque;
    /* Exercise the adapter's complete role vocabulary without loading weights. */
    return strstr(name, "blk.") == name &&
           (strstr(name, ".weight") != NULL || strstr(name, ".bias") != NULL ||
            strstr(name, ".ssm_a") != NULL);
}

int main(void) {
    char name[128];
    if (k3_gguf_role_name(name, sizeof name, 0, "q_proj") != 0 ||
        strcmp(name, "blk.0.attn_q.weight") != 0) return 1;
    if (k3_gguf_role_name(name, sizeof name, 3, "kv_b_proj") != 0 ||
        strcmp(name, "blk.3.attn_k_b.weight") != 0) return 2;
    if (k3_gguf_role_name(name, sizeof name, 92, "mla_o_proj") != 0 ||
        strcmp(name, "blk.92.attn_output.weight") != 0) return 3;
    if (!k3_gguf_is_mla(3) || !k3_gguf_is_mla(91) ||
        !k3_gguf_is_mla(92) || k3_gguf_is_mla(90)) return 4;
    if (k3_gguf_validate_graph(NULL, has_tensor, 0, K3_GGUF_LAYERS, NULL) != 0) return 5;
    puts("k3_gguf_adapter: PASS (KDA/MLA role map and 93-layer graph schema)");
    return 0;
}
