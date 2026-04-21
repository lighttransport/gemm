/* CLIP BPE tokenizer — minimal HF-compatible implementation for SAM 3.
 *
 * Loads vocab.json + merges.txt from openai/clip-vit-base-patch32.
 * Produces input_ids + attention_mask matching transformers.CLIPTokenizer
 * for simple noun-phrase inputs ("cat", "a dog", etc.).
 *
 * Context length is fixed per call via max_len (SAM 3 uses 32).
 * BOS=49406, EOS=PAD=49407.
 */
#ifndef SAM3_CLIP_BPE_H_
#define SAM3_CLIP_BPE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam3_clip_bpe sam3_clip_bpe;

sam3_clip_bpe *sam3_clip_bpe_load(const char *vocab_json_path,
                                   const char *merges_txt_path);
void           sam3_clip_bpe_free(sam3_clip_bpe *t);

/* Encode `text` into `out_ids` + `out_mask` of length max_len.
 * Returns number of non-pad tokens (including BOS+EOS), -1 on error. */
int sam3_clip_bpe_encode(const sam3_clip_bpe *t, const char *text,
                          int max_len, int32_t *out_ids, int32_t *out_mask);

#ifdef __cplusplus
}
#endif

#endif
