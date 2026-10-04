#ifndef GLM53F_RUNTIME_H
#define GLM53F_RUNTIME_H
#include <stdlib.h>
#include <string.h>

/* Startup-only switches shared by generation, benchmarking and verification.
 * Parse before model creation: scratch and communication ownership depend on them. */
static inline int glm53f_runtime_option(int argc, char **argv, int *index) {
    const char *key = argv[*index], *env;
    if (!strcmp(key, "--dense-prefill-tile")) {
        if (*index + 1 >= argc) return -1;
        const char *value = argv[++*index];
        if (strcmp(value, "4") && strcmp(value, "16") &&
            strcmp(value, "32") && strcmp(value, "64")) return -1;
        return setenv("GLM53F_DENSE_PREFILL_TILE", value, 1) ? -1 : 1;
    }
    if (!strcmp(key, "--act-chunk")) {   /* contiguous 32-column blocks per thread in act_prepare_team */
        if (*index + 1 >= argc) return -1;
        const char *value = argv[++*index];
        if (strcmp(value, "0") && strcmp(value, "8") && strcmp(value, "16") && strcmp(value, "32")) return -1;
        return setenv("GLM53F_ACT_CHUNK", value, 1) ? -1 : 1;
    }
    if (!strcmp(key, "--verify-kernel")) env = "GLM53F_VERIFY_GROUPED";
    else if (!strcmp(key, "--decode-executor")) env = "GLM53F_DECODE_EXECUTOR";
    else if (!strcmp(key, "--router-kernel")) env = "GLM53F_ROUTER_FUSE";
    else if (!strcmp(key, "--collective-owner")) env = "GLM53F_COMM_OWNER";
    else if (!strcmp(key, "--moe-scale-kernel")) env = "GLM53F_IQ_SCALE_WORDS";
    else if (!strcmp(key, "--moe-combine-kernel")) env = "GLM53F_MOE_COMBINE";
    else if (!strcmp(key, "--index-kernel")) env = "GLM53F_INDEX_HEADS";
    else if (!strcmp(key, "--pool-selector")) env = "GLM53F_POOL_PARTITION_4K";
    else if (!strcmp(key, "--q8-row-kernel")) env = "GLM53F_NATIVE_Q8_ROWS8";
    else if (!strcmp(key, "--q8-prefill-kernel")) env = "GLM53F_NATIVE_Q8_TILE2X8";
    else if (!strcmp(key, "--embedding-batch-kernel")) env = "GLM53F_EMBED_BATCH_PACKED";
    else if (!strcmp(key, "--verify-head-kernel")) env = "GLM53F_HEAD_VERIFY_SHARED";
    else if (!strcmp(key, "--mla-prefill-heads")) env = "GLM53F_MLA_PREFILL_SPLIT6";
    else if (!strcmp(key, "--moe-router-prefill")) env = "GLM53F_MOE_ROUTER_TILES12";
    else if (!strcmp(key, "--mtp-target-restore")) env = "GLM53F_MTP_REJECTION_RESTORE";
    else if (!strcmp(key, "--mla-value-kernel")) env = "GLM53F_MLA_VALUES_NORMALIZED2";
    else if (!strcmp(key, "--mla-logits-kernel")) env = "GLM53F_MLA_LOGITS_HEADS3";
    else if (!strcmp(key, "--mla-softmax-kernel")) env = "GLM53F_MLA_PARALLEL_SOFTMAX";
    else if (!strcmp(key, "--mla-projection-kernel")) env = "GLM53F_MLA_FUSED_PROJECTION";
    else if (!strcmp(key, "--moe-prefill-layout")) env = "GLM53F_MOE_GU_PAD";
    else if (!strcmp(key, "--mhc-kernel")) env = "GLM53F_MHC_FUSED_SYNC";
    else if (!strcmp(key, "--mhc-verify-kernel")) env = "GLM53F_MHC_BATCH_TEAM";
    else if (!strcmp(key, "--kda-decode-kernel")) env = "GLM53F_KDA_DECODE_COLUMNS";
    else if (!strcmp(key, "--kda-prefill-kernel")) env = "GLM53F_KDA_PREFILL_COLUMNS";
    else if (!strcmp(key, "--mla-kernel")) env = "GLM53F_MLA_REGISTERS";
    else if (!strcmp(key, "--sparse-verify-kernel")) env = "GLM53F_SPARSE_VERIFY_FRONT";
    else if (!strcmp(key, "--moe-verify-router")) env = "GLM53F_MOE_VERIFY_ROUTER";
    else if (!strcmp(key, "--weight-placement")) env = "GLM53F_CMG_PLACE";
    else if (!strcmp(key, "--moe-layer-kernel")) env = "GLM53F_MOE_FUSED_LAYER";
    else if (!strcmp(key, "--kda-out-kernel")) env = "GLM53F_KDA_FUSED_OUT";
    else if (!strcmp(key, "--kda-layer-kernel")) env = "GLM53F_KDA_FUSED_LAYER";
    else if (!strcmp(key, "--sparse-layer-kernel")) env = "GLM53F_SPARSE_FUSED_LAYER";
    else if (!strcmp(key, "--act-header-kernel")) env = "GLM53F_ACT_HEADER_CACHE";
    else if (!strcmp(key, "--kda-quant-kernel")) env = "GLM53F_KDA_TEAM_QUANT";
    else if (!strcmp(key, "--mhc-prefetch")) env = "GLM53F_MHC_PREFETCH_NEXT";
    else if (!strcmp(key, "--kda-verify-kernel")) env = "GLM53F_KDA_VERIFY_COLUMNS";
    else if (!strcmp(key, "--mhc-batch-tail")) env = "GLM53F_MHC_BATCH_TAIL";
    else if (!strcmp(key, "--snapshot-copy")) env = "GLM53F_SNAPSHOT_PARALLEL";
    else return 0;
    if (*index + 1 >= argc) return -1;
    const char *value = argv[++*index];
    int enabled;
    if ((!strcmp(key, "--moe-prefill-layout") && !strcmp(value, "tight")) ||
        (!strcmp(key, "--q8-row-kernel") && !strcmp(value, "rows4")) ||
        (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile4x4")) ||
        (!strcmp(key, "--pool-selector") && !strcmp(value, "heap")) ||
        (!strcmp(key, "--weight-placement") && !strcmp(value, "interleave")) || !strcmp(value, "legacy")) enabled = 0;
    else if ((!strcmp(key, "--moe-prefill-layout") && !strcmp(value, "padded")) ||
             (!strcmp(key, "--q8-row-kernel") && !strcmp(value, "rows8")) ||
             (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile2x8")) ||
             (!strcmp(key, "--pool-selector") && !strcmp(value, "partition4k")) ||
             (!strcmp(key, "--verify-kernel") && !strcmp(value, "grouped")) ||
             (!strcmp(key, "--collective-owner") && !strcmp(value, "serialized")) ||
             (!strcmp(key, "--router-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--mla-projection-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--embedding-batch-kernel") && !strcmp(value, "packed")) ||
             (!strcmp(key, "--verify-head-kernel") && !strcmp(value, "shared")) ||
             (!strcmp(key, "--mla-prefill-heads") && !strcmp(value, "split6")) ||
             (!strcmp(key, "--moe-router-prefill") && !strcmp(value, "tiles12")) ||
             (!strcmp(key, "--mtp-target-restore") && !strcmp(value, "rejection")) ||
             (!strcmp(key, "--mla-value-kernel") && !strcmp(value, "normalized2")) ||
             (!strcmp(key, "--mla-logits-kernel") && !strcmp(value, "heads3")) ||
             (!strcmp(key, "--mla-softmax-kernel") && !strcmp(value, "parallel")) ||
             (!strcmp(key, "--mhc-kernel") && !strcmp(value, "fused-sync")) ||
             (!strcmp(key, "--mhc-verify-kernel") && !strcmp(value, "team")) ||
             (!strcmp(key, "--sparse-verify-kernel") && !strcmp(value, "front")) ||
             (!strcmp(key, "--moe-verify-router") && !strcmp(value, "batch")) ||
             (!strcmp(key, "--weight-placement") && !strcmp(value, "cmg")) ||
             (!strcmp(key, "--moe-layer-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--kda-out-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--kda-layer-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--sparse-layer-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--act-header-kernel") && !strcmp(value, "cached")) ||
             (!strcmp(key, "--kda-quant-kernel") && !strcmp(value, "team")) ||
             (!strcmp(key, "--mhc-prefetch") && !strcmp(value, "next")) ||
             (!strcmp(key, "--kda-verify-kernel") && !strcmp(value, "columns")) ||
             (!strcmp(key, "--mhc-batch-tail") && !strcmp(value, "parallel")) ||
             (!strcmp(key, "--snapshot-copy") && !strcmp(value, "parallel")) ||
             ((!strcmp(key, "--kda-decode-kernel") || !strcmp(key, "--kda-prefill-kernel")) && !strcmp(value, "columns")) ||
             (!strcmp(key, "--moe-scale-kernel") && !strcmp(value, "words")) ||
             (!strcmp(key, "--moe-combine-kernel") && !strcmp(value, "vector")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "heads")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "registers")) ||
             (!strcmp(key, "--decode-executor") && !strcmp(value, "persistent"))) enabled = 1;
    else if ((!strcmp(key, "--moe-router-prefill") && !strcmp(value, "tiles8")) ||
             (!strcmp(key, "--mla-prefill-heads") && !strcmp(value, "values32")) ||
             (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile4x4-asm")) ||
             (!strcmp(key, "--moe-combine-kernel") && !strcmp(value, "overlap")) ||
             (!strcmp(key, "--kda-decode-kernel") && !strcmp(value, "columns16")) ||
             (!strcmp(key, "--weight-placement") && !strcmp(value, "cmg-experts")) ||
             (!strcmp(key, "--mhc-kernel") && !strcmp(value, "local")) ||
             (!strcmp(key, "--sparse-verify-kernel") && !strcmp(value, "front-onecoll")) ||
             (!strcmp(key, "--kda-verify-kernel") && !strcmp(value, "columns64")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "keys4")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "values"))) enabled = 2;
    else if ((!strcmp(key, "--moe-router-prefill") && !strcmp(value, "unroll1")) ||
             (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile2x8-asm")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "fp16-cache")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "replicated-heads")) ||
             (!strcmp(key, "--mhc-kernel") && !strcmp(value, "local-gram"))) enabled = 3;
    else if (!strcmp(key, "--index-kernel") && !strcmp(value, "replicated-keys4")) enabled = 4;
    else return -1;
    return setenv(env, enabled == 4 ? "4" : enabled == 3 ? "3" : enabled == 2 ? "2" : enabled ? "1" : "0", 1) ? -1 : 1;
}
#endif
