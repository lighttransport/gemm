/*
 * test_transformer_mpi.c - MPI pipeline-parallel transformer inference
 *
 * Pipeline parallelism: layers are split across MPI ranks.
 * Each rank processes its assigned layers and passes the hidden state
 * to the next rank via MPI_Send/MPI_Recv.
 *
 * All ranks load the full model (simplifies debugging; wasteful on memory).
 *
 * Build:
 *   mpicc -O2 -march=native -o test_transformer_mpi test_transformer_mpi.c -lm -lpthread
 *
 * Run (2-8 processes on localhost):
 *   mpirun -np 2 ./test_transformer_mpi <model.gguf> [prompt] [max_gen_tokens] [n_threads_per_rank]
 *   mpirun -np 4 ./test_transformer_mpi <model.gguf> "Hello" 32 1
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* Timer using MPI_Wtime for consistency across ranks */
static double mpi_wtime(void) { return MPI_Wtime(); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np N %s <model.gguf> [prompt] [max_gen_tokens] [n_threads]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = (argc >= 3) ? argv[2] : "Hello";
    int max_gen = (argc >= 4) ? atoi(argv[3]) : 32;
    int n_threads = (argc >= 5) ? atoi(argv[4]) : 1;

    /* All ranks load the full model and vocab */
    if (rank == 0)
        fprintf(stderr, "MPI: %d ranks, loading model: %s\n", n_ranks, model_path);

    gguf_context *gguf = gguf_open(model_path, 1);
    if (!gguf) {
        fprintf(stderr, "[rank %d] Failed to open GGUF\n", rank);
        MPI_Finalize();
        return 1;
    }

    bpe_vocab *vocab = NULL;
    if (rank == 0) {
        vocab = bpe_vocab_load(gguf);
        if (!vocab) {
            fprintf(stderr, "[rank 0] Failed to load vocab\n");
            gguf_close(gguf);
            MPI_Finalize();
            return 1;
        }
    }

    int max_seq_len = 256;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "[rank %d] Failed to load model\n", rank);
        MPI_Finalize();
        return 1;
    }

    if (n_threads > 1)
        transformer_set_threads(model, n_threads);

    /* Compute layer partition for this rank */
    int n_layers = model->n_layers;
    int n_embd = model->n_embd;
    int layers_per_rank = n_layers / n_ranks;
    int extra_layers = n_layers % n_ranks;
    int layer_start = 0;
    for (int r = 0; r < rank; r++)
        layer_start += layers_per_rank + (r < extra_layers ? 1 : 0);
    int layer_end = layer_start + layers_per_rank + (rank < extra_layers ? 1 : 0);

    fprintf(stderr, "[rank %d] layers [%d, %d) of %d, n_embd=%d, n_threads=%d\n",
            rank, layer_start, layer_end, n_layers, n_embd, n_threads);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Tokenize on rank 0, broadcast token count and tokens */
    int32_t tokens[256];
    int n_tokens = 0;

    if (rank == 0) {
        n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 256);
        fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt, n_tokens);
    }
    MPI_Bcast(&n_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tokens, n_tokens, MPI_INT32_T, 0, MPI_COMM_WORLD);

    if (!model->has_lm_head) {
        if (rank == 0)
            fprintf(stderr, "Model has no LM head, MPI pipeline requires generative model\n");
        transformer_free(model);
        gguf_close(gguf);
        MPI_Finalize();
        return 1;
    }

    /* ---- Prefill ---- */
    if (rank == 0)
        fprintf(stderr, "\n=== Prefill (%d tokens, %d ranks) ===\n", n_tokens, n_ranks);

    double t_prefill_start = mpi_wtime();

    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        /* Rank 0: embed the token */
        if (rank == 0) {
            transformer_embed_token(model, tokens[i]);
        }

        /* Pipeline: each rank processes its layers, then passes hidden to next */
        if (rank > 0) {
            /* Receive hidden state from previous rank */
            MPI_Recv(transformer_get_hidden(model), n_embd, MPI_FLOAT,
                     rank - 1, pos, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Process our layers */
        transformer_forward_partial(model, pos, layer_start, layer_end);

        if (rank < n_ranks - 1) {
            /* Send hidden state to next rank */
            MPI_Send(transformer_get_hidden(model), n_embd, MPI_FLOAT,
                     rank + 1, pos, MPI_COMM_WORLD);
        }

        /* Last rank: compute logits for the last prompt token */
        float *logits = NULL;
        if (rank == n_ranks - 1 && i == n_tokens - 1) {
            logits = transformer_compute_logits(model);
        }

        /* Broadcast argmax from last rank to rank 0 for generation */
        if (i == n_tokens - 1) {
            int32_t argmax = 0;
            if (rank == n_ranks - 1 && logits) {
                for (int j = 1; j < model->n_vocab; j++)
                    if (logits[j] > logits[argmax]) argmax = j;
            }
            MPI_Bcast(&argmax, 1, MPI_INT32_T, n_ranks - 1, MPI_COMM_WORLD);
            if (rank == 0) {
                const char *tok_str = bpe_token_to_str(vocab, argmax);
                fprintf(stderr, "Prefill greedy next: %d \"%s\" (logit from rank %d)\n",
                        argmax, tok_str ? tok_str : "?", n_ranks - 1);
            }
        }

        pos++;
    }

    double t_prefill_end = mpi_wtime();
    if (rank == 0)
        fprintf(stderr, "Prefill time: %.3f s (%.1f ms/tok)\n",
                t_prefill_end - t_prefill_start,
                (t_prefill_end - t_prefill_start) * 1000.0 / n_tokens);

    /* ---- Generation ---- */
    if (rank == 0) {
        fprintf(stderr, "\n=== Generation (max %d tokens) ===\n", max_gen);
        printf("%s", prompt);
        fflush(stdout);
    }

    double t_gen_start = mpi_wtime();
    int gen_count = 0;
    int32_t next_token = -1;

    for (int g = 0; g < max_gen; g++) {
        float *logits = NULL;

        if (g == 0) {
            /* First generation step: logits already computed by last rank during prefill.
             * We need them on the last rank to sample. */
            if (rank == n_ranks - 1)
                logits = model->logits;
        } else {
            /* Rank 0: embed the previously generated token */
            if (rank == 0)
                transformer_embed_token(model, next_token);

            /* Pipeline forward */
            if (rank > 0) {
                MPI_Recv(transformer_get_hidden(model), n_embd, MPI_FLOAT,
                         rank - 1, pos, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            transformer_forward_partial(model, pos, layer_start, layer_end);

            if (rank < n_ranks - 1) {
                MPI_Send(transformer_get_hidden(model), n_embd, MPI_FLOAT,
                         rank + 1, pos, MPI_COMM_WORLD);
            }

            if (rank == n_ranks - 1)
                logits = transformer_compute_logits(model);

            pos++;
        }

        /* Last rank: greedy sample */
        if (rank == n_ranks - 1 && logits) {
            next_token = 0;
            for (int j = 1; j < model->n_vocab; j++)
                if (logits[j] > logits[next_token]) next_token = j;
        }

        /* Broadcast next_token to all ranks */
        MPI_Bcast(&next_token, 1, MPI_INT32_T, n_ranks - 1, MPI_COMM_WORLD);

        /* Check EOS */
        if (rank == 0 && vocab) {
            if (next_token == vocab->eos_id || next_token == vocab->eot_id) {
                fprintf(stderr, "  [EOS token %d]\n", next_token);
                gen_count = g + 1;
                /* Broadcast stop signal */
                int stop = 1;
                MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
                break;
            }

            const char *tok_str = bpe_token_to_str(vocab, next_token);
            if (tok_str) {
                int dec_len;
                char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                fwrite(decoded, 1, dec_len, stdout);
                fflush(stdout);
                free(decoded);
            }

            int stop = 0;
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            int stop = 0;
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (stop) {
                gen_count = g + 1;
                break;
            }
        }

        if (g == 0) pos++;  /* account for first generated token's position */
        gen_count = g + 1;
    }

    double t_gen_end = mpi_wtime();

    if (rank == 0) {
        printf("\n");
        fprintf(stderr, "\n=== Timing Summary ===\n");
        fprintf(stderr, "  Ranks:          %d\n", n_ranks);
        fprintf(stderr, "  Threads/rank:   %d\n", n_threads);
        fprintf(stderr, "  Prefill:        %.3f s (%d tokens, %.1f ms/tok)\n",
                t_prefill_end - t_prefill_start, n_tokens,
                (t_prefill_end - t_prefill_start) * 1000.0 / n_tokens);
        if (gen_count > 0)
            fprintf(stderr, "  Generation:     %.3f s (%d tokens, %.1f ms/tok, %.1f tok/s)\n",
                    t_gen_end - t_gen_start, gen_count,
                    (t_gen_end - t_gen_start) * 1000.0 / gen_count,
                    gen_count / (t_gen_end - t_gen_start));
    }

    /* Cleanup */
    transformer_free(model);
    if (vocab) bpe_vocab_free(vocab);
    gguf_close(gguf);

    MPI_Finalize();
    return 0;
}
