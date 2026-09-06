#ifndef SERVER_DS4F_H
#define SERVER_DS4F_H

#include <stddef.h>

typedef struct json_val json_val;
typedef struct ds4f_session ds4f_session;
typedef struct ds4f_runtime_options ds4f_runtime_options;

/* Native x86 DS4F session. The model is loaded from a staged safetensors
 * directory; the session is deliberately single-stream because the model's
 * KV state is mutable and server.c already serializes inference. */
ds4f_session *ds4f_session_open(const char *stage_dir, char *err, size_t err_cap);
ds4f_session *ds4f_session_open_opts(const char *stage_dir,
                                     const ds4f_runtime_options *options,
                                     char *err, size_t err_cap);
void ds4f_session_close(ds4f_session *s);

/* Build the plain-text DS4F role prompt used by ds4f_serve.py. */
char *ds4f_chat_prompt(const json_val *messages, char *err, size_t err_cap);
void ds4f_owned_free(void *ptr);

/* Generate with seeded temperature/top-p sampling. A zero temperature selects
 * the exact argmax path. The one-slot prefix cache snapshots logits as well as
 * KV state so cache hits preserve sampling semantics. */
char *ds4f_session_generate(ds4f_session *s, const char *prompt, int max_tokens,
                            float temperature, float top_p, int seed,
                            int *prompt_tokens, int *completion_tokens,
                            char *err, size_t err_cap);

#endif
