/* cmp_enc.c — validate the HIP GPU Qwen3 text encoder against the CPU encoder.
 * Encodes the same prompt with both flux2_text_enc paths and reports corr/max-diff
 * over the per-token concatenated hidden states [n_tok, 3*n_embd_inner]. */
/* SAFETENSORS_IMPLEMENTATION is provided by hip_llm_runner.o (global symbols). */
#define GGML_DEQUANT_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define FLUX2_TEXT_ENCODER_IMPLEMENTATION
#include "hip_llm_runner.h"   /* defines HIP_LLM_RUNNER_H -> activates the HIP encoder branch */
#include "../../common/flux2_klein_text_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <qwen3.safetensors> <tok.gguf> [prompt]\n", argv[0]); return 1; }
    const char *st = argv[1], *tok = argv[2];
    const char *prompt = (argc > 3) ? argv[3] : "a red apple on a white table";

    fprintf(stderr, "=== CPU encode ===\n");
    flux2_text_enc *cpu = flux2_text_enc_load_safetensors(st, tok);
    if (!cpu) { fprintf(stderr, "cpu load failed\n"); return 1; }
    int n_cpu = 0; float *hc = flux2_text_enc_encode(cpu, prompt, &n_cpu);
    int dim = cpu->n_embd;
    if (!hc) { fprintf(stderr, "cpu encode failed\n"); return 1; }

    fprintf(stderr, "=== GPU encode ===\n");
    flux2_text_enc *gpu = flux2_text_enc_load_gpu(st, tok, 0);
    if (!gpu) { fprintf(stderr, "gpu load failed\n"); return 1; }
    int n_gpu = 0; float *hg = flux2_text_enc_encode(gpu, prompt, &n_gpu);
    if (!hg) { fprintf(stderr, "gpu encode failed\n"); return 1; }

    printf("n_cpu=%d n_gpu=%d dim=%d\n", n_cpu, n_gpu, dim);
    if (n_cpu != n_gpu) { fprintf(stderr, "token count mismatch\n"); return 1; }

    long N = (long)n_cpu * dim;
    double dot = 0, nc = 0, ng = 0, maxd = 0; long maxi = -1;
    for (long i = 0; i < N; i++) {
        float a = hc[i], b = hg[i];
        dot += (double)a * b; nc += (double)a * a; ng += (double)b * b;
        double d = fabs((double)a - b); if (d > maxd) { maxd = d; maxi = i; }
    }
    double corr = dot / (sqrt(nc) * sqrt(ng) + 1e-12);
    printf("corr=%.6f  max_diff=%.5f @ %ld  cpu_norm=%.2f gpu_norm=%.2f\n",
           corr, maxd, maxi, sqrt(nc), sqrt(ng));
    /* per-snapshot-block corr (the 3 concatenated layers) */
    int inner = cpu->n_embd_inner;
    for (int s = 0; s < 3; s++) {
        double d2 = 0, c2 = 0, g2 = 0;
        for (int t = 0; t < n_cpu; t++)
            for (int j = 0; j < inner; j++) {
                float a = hc[(long)t*dim + s*inner + j], b = hg[(long)t*dim + s*inner + j];
                d2 += (double)a*b; c2 += (double)a*a; g2 += (double)b*b;
            }
        printf("  block %d corr=%.6f\n", s, d2/(sqrt(c2)*sqrt(g2)+1e-12));
    }
    return 0;
}
