// C translation unit for vision encoder CPU reference implementation
// (C headers with IMPLEMENTATION don't compile cleanly as C++)

#define GGUF_LOADER_IMPLEMENTATION
#include "../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../common/ggml_dequant.h"

#include "../common/transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../common/vision_encoder.h"
