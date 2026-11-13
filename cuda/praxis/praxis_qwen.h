#ifndef PRAXIS_QWEN_H
#define PRAXIS_QWEN_H
#include <stdint.h>
typedef struct praxis_qwen praxis_qwen;
praxis_qwen *praxis_qwen_open(const char *dir, int max_seq);
float *praxis_qwen_cond(praxis_qwen *q, const int32_t *ids, int n);   /* malloc'd [ceil32(n)][2048]; first n rows valid */
void praxis_qwen_close(praxis_qwen *q);
typedef struct praxis_tok praxis_tok;
praxis_tok *praxis_tok_load(const char *tokenizer_json);
int praxis_tok_encode(praxis_tok *t, const char *text, int32_t *ids, int max_ids);   /* #tokens, or -1 if > max_ids */
void praxis_tok_free(praxis_tok *t);
#endif
