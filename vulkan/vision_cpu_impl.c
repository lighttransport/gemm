// C translation unit for CPU reference implementations
// (C headers with IMPLEMENTATION don't compile cleanly as C++)

#include "../common/profiler.h"

#define GGUF_LOADER_IMPLEMENTATION
#include "../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../common/ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "../common/bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "../common/transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../common/vision_encoder.h"
