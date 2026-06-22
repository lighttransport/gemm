/* test_gemma4_st.c — drive the DIRECT gemma-4 safetensors loader.
 *
 * Loads a gemma-4 "qat-mobile-transformers" checkpoint straight from
 * safetensors (no GGUF conversion), tokenizes with the BPE vocab from a
 * companion GGUF (same base model → identical vocab), and greedily generates.
 *
 * Usage:
 *   ./test_gemma4_st <model_dir_or.safetensors> <tokenizer.gguf> \
 *                    [-t "prompt"] [-n n_tokens] [-s max_seq_len]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* single-header implementations (this TU provides them; runner.c only declares) */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"
#define IMAGE_UTILS_IMPLEMENTATION
#include "../../common/image_utils.h"

#include "cuda_llm_runner.h"

/* print a token's text, converting SentencePiece ▁ (U+2581, E2 96 81) to space */
static void emit_tok(const bpe_vocab *vocab, int id) {
    const char *s = bpe_token_to_str(vocab, id);
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { putchar(' '); p += 3; }
        else putchar(*p++);
    }
}

static int is_stop(const bpe_vocab *vocab, int id) {
    return id == bpe_eos_id(vocab) || id == bpe_eot_id(vocab) || id == 1 || id == 106;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_gguf = NULL;
    const char *prompt = NULL, *image_path = NULL;
    int n_gen = 32, max_seq_len = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) max_seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) image_path = argv[++i];
        else if (argv[i][0] != '-') { if (!model_path) model_path = argv[i]; else tok_gguf = argv[i]; }
    }
    if (!model_path || !tok_gguf) {
        fprintf(stderr, "Usage: %s <model_dir_or.safetensors> <tokenizer.gguf> [-i image] [-t prompt] [-n n] [-s seq]\n", argv[0]);
        return 1;
    }
    if (!prompt) prompt = image_path ? "explain the image" : "The capital of France is";

    fprintf(stderr, "Tokenizer GGUF: %s\n", tok_gguf);
    gguf_context *gguf = gguf_open(tok_gguf, 1);
    if (!gguf) { fprintf(stderr, "failed to open tokenizer gguf\n"); return 1; }
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) { fprintf(stderr, "failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    if (max_seq_len <= 0) max_seq_len = (image_path ? 320 : 64) + n_gen;
    if (max_seq_len < 1024) max_seq_len = 1024;

    fprintf(stderr, "\n=== Init + load (DIRECT safetensors) ===\n");
    cuda_llm_runner *gpu = cuda_llm_init(0, 1);
    if (!gpu) { fprintf(stderr, "init failed\n"); return 1; }
    if (cuda_llm_load_weights_gemma4_safetensors(gpu, model_path, max_seq_len) != 0) {
        fprintf(stderr, "load_weights_gemma4_safetensors FAILED\n");
        cuda_llm_free(gpu);
        return 1;
    }
    cuda_llm_set_debug(gpu, getenv("CUDA_LLM_DEBUG") ? atoi(getenv("CUDA_LLM_DEBUG")) : 0);
    if (cuda_llm_reset_state(gpu) != 0) { fprintf(stderr, "reset failed\n"); return 1; }
    int n_vocab = cuda_llm_n_vocab(gpu);
    int n_embd  = cuda_llm_n_embd(gpu);

    float *logits = NULL;
    int pos = 0;

    if (image_path) {
        /* ---- VLM path: encode image (vision_tower) + splice ---- */
        int iw, ih;
        uint8_t *image = img_load(image_path, &iw, &ih);
        if (!image) { fprintf(stderr, "failed to load image %s\n", image_path); return 1; }
        fprintf(stderr, "Loaded image %s (%dx%d)\n", image_path, iw, ih);
        if (iw != 224 || ih != 224) {
            uint8_t *rs = img_resize_ac(image, iw, ih, 224, 224);
            img_free(image); image = rs; iw = ih = 224;
        }
        fprintf(stderr, "\n=== Vision encode (safetensors vision_tower) ===\n");
        int n_vis = 0, proj_dim = 0;
        float *vembd = cuda_llm_vision_encode_safetensors(gpu, model_path, image, iw, ih, &n_vis, &proj_dim);
        img_free(image);
        if (!vembd) { fprintf(stderr, "vision encode FAILED\n"); return 1; }
        fprintf(stderr, "Vision: %d tokens x %d dim\n", n_vis, proj_dim);
        { float mn=vembd[0],mx=vembd[0],sm=0; int tt=n_vis*proj_dim;
          for(int i=0;i<tt;i++){ if(vembd[i]<mn)mn=vembd[i]; if(vembd[i]>mx)mx=vembd[i]; sm+=vembd[i]; }
          fprintf(stderr, "Vision embedding: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sm/tt); }
        if (proj_dim != n_embd) { fprintf(stderr, "proj_dim %d != n_embd %d\n", proj_dim, n_embd); return 1; }

        /* Optional A/B: compare safetensors vision_tower vs a GGUF mmproj encode. */
        { const char *cmp = getenv("VIS_CMP_MMPROJ");
          if (cmp) {
            uint8_t *img2 = img_load(image_path, &iw, &ih);
            if (iw != 224 || ih != 224) { uint8_t *rs = img_resize_ac(img2, iw, ih, 224, 224); img_free(img2); img2 = rs; iw = ih = 224; }
            gguf_context *gmm = gguf_open(cmp, 1);
            int nv2 = 0, pd2 = 0;
            float *v2 = cuda_llm_vision_encode(gpu, gmm, img2, iw, ih, &nv2, &pd2);
            img_free(img2);
            if (v2 && nv2 == n_vis && pd2 == proj_dim) {
                double dot=0, na=0, nb=0; int tt=n_vis*proj_dim;
                for (int i=0;i<tt;i++){ dot+=(double)vembd[i]*v2[i]; na+=(double)vembd[i]*vembd[i]; nb+=(double)v2[i]*v2[i]; }
                fprintf(stderr, "VIS A/B cosine(safetensors, mmproj) = %.5f\n", dot/(sqrt(na)*sqrt(nb)+1e-9));
            }
            free(v2); gguf_close(gmm);
          } }

        /* gemma4 image prompt: BOS + pre + <image> + post */
        const char *pre = "<|turn>system\n<|think|><turn|>\n<|turn>user\n<|image>";
        char post[512];
        snprintf(post, sizeof(post), "<image|>%s<turn|>\n<|turn>model\n", prompt);
        int32_t pre_tok[256], post_tok[256];
        int n_pre = bpe_tokenize(vocab, pre, -1, pre_tok, 256);
        int n_post = bpe_tokenize(vocab, post, -1, post_tok, 256);

        fprintf(stderr, "\n=== Prefill (BOS + %d pre + %d vision + %d post) ===\n", n_pre, n_vis, n_post);
        cuda_llm_forward(gpu, 2 /*BOS*/, pos++);
        cuda_llm_prefill(gpu, pre_tok, NULL, 0, n_pre, pos); pos += n_pre;
        cuda_llm_prefill(gpu, NULL, vembd, proj_dim, n_vis, pos); pos += n_vis;
        logits = cuda_llm_prefill_logits(gpu, post_tok, NULL, 0, n_post, pos); pos += n_post;
        free(vembd);
        if (!logits) { fprintf(stderr, "prefill_logits NULL\n"); return 1; }

        fprintf(stderr, "\n=== Generate (greedy) ===\nPrompt: \"%s\"\n", prompt);
    } else {
        /* ---- text path ---- */
        int32_t tok[1024]; int n_tok = 0;
        tok[n_tok++] = 2; /* BOS */
        int nt = bpe_tokenize(vocab, prompt, -1, tok + n_tok, 1024 - n_tok);
        if (nt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        n_tok += nt;
        fprintf(stderr, "\n=== Generate (greedy) ===\n");
        for (int i = 0; i < n_tok; i++) {
            logits = cuda_llm_forward_logits(gpu, tok[i], pos++);
            if (!logits) { fprintf(stderr, "forward failed\n"); return 1; }
        }
        printf("%s", prompt);
    }
    fflush(stdout);

    for (int g = 0; g < n_gen; g++) {
        int best = 0; float bv = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (is_stop(vocab, best)) break;
        emit_tok(vocab, best);
        fflush(stdout);
        logits = cuda_llm_forward_logits(gpu, best, pos++);
        if (!logits) { fprintf(stderr, "\nforward failed at gen %d\n", g); break; }
    }
    printf("\n");

    cuda_llm_free(gpu);
    bpe_vocab_free(vocab);
    gguf_close(gguf);
    return 0;
}
