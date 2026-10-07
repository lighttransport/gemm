/* Single-header implementations for libhip_qimg.so (the test drivers provide these themselves). */
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/gguf_loader.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
