#ifndef SERVER_LLM_H
#define SERVER_LLM_H

#include <stdint.h>
#include <stddef.h>

typedef struct json_val json_val;
typedef struct ds4f_session ds4f_session;
typedef struct ds4f_runtime_options ds4f_runtime_options;

enum {
    LLM_BACKEND_GGUF = 0,
    LLM_BACKEND_DS4F = 1
};

typedef struct {
    int loaded;
    char model_path[1024];
    char mmproj_path[1024];
    int is_vlm;

    void *gguf;
    void *gguf_mmproj;
    void *vocab;
    void *model;
    void *vm;
    int backend;
    ds4f_session *ds4f;

    int n_embd;
    int n_vocab;
    int max_seq_len;
    int eos_id;
    int eot_id;

    int patch_size;
    int spatial_merge;
    int proj_dim;
} llm_state;

int llm_init(llm_state *s, const char *model_path, const char *mmproj_path,
             const ds4f_runtime_options *ds4f_options);
void llm_free(llm_state *s);

char *llm_chat_completion(llm_state *s, const json_val *messages,
                           int max_tokens, float temperature, float top_p, int seed,
                           const json_val *stop_arr,
                           int *status, char *err, size_t err_cap);

int llm_chat_completion_stream(llm_state *s, int fd, const json_val *messages,
                                int max_tokens, float temperature, float top_p, int seed,
                                const json_val *stop_arr,
                                char *err, size_t err_cap);

char *llm_text_completion(llm_state *s, const char *prompt,
                           int max_tokens, float temperature, float top_p, int seed,
                           const json_val *stop_arr,
                           int *status, char *err, size_t err_cap);

int llm_text_completion_stream(llm_state *s, int fd, const char *prompt,
                                int max_tokens, float temperature, float top_p, int seed,
                                const json_val *stop_arr,
                                char *err, size_t err_cap);

#endif
