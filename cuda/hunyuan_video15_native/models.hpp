#ifndef PIXAL3D_HV15N_MODELS_HPP
#define PIXAL3D_HV15N_MODELS_HPP
#include "gpu.hpp"
#include "tokenizer.hpp"
namespace hv15n {
Tensor qwen(Gpu &gpu, Weights &weights, const Tokenizer &tokenizer, const std::string &prompt);
Tensor siglip(Gpu &gpu, Weights &weights, const fs::path &pixels);
Tensor byt5(Gpu &gpu, Weights &weights, const std::string &prompt);
Tensor encoder_block(Gpu &gpu, Weights &weights, const std::string &kind, int index, Tensor input);
std::pair<Tensor, Tensor> dit_block(Gpu &gpu, Weights &weights, int index, Tensor image,
                                  Tensor text, const Tensor &vector, int height, int width);
struct DitState { Tensor img, txt, vec; std::vector<int> shape; };
Tensor dit_finish(Gpu &gpu, Weights &weights, const DitState &state);
void dit_blocks(Gpu &gpu, Weights &weights, std::vector<DitState> &states,
                int first, int count, int height, int width);
Tensor dit(Gpu &gpu, Weights &weights, const Tensor &latent, const Tensor &conditioning, const Tensor &text,
           const Tensor &glyph, const Tensor &vision, float time, float next_time);
std::pair<Tensor,Tensor> dit_pair(Gpu &gpu, Weights &weights, const Tensor &latent,
                                 const Tensor &conditioning, const Tensor &text,
                                 const Tensor &negative, const Tensor &glyph, const Tensor &vision,
                                 float time, float next_time);
Tensor vae(Gpu &gpu, Weights &weights, const Tensor &video, bool encode);
Tensor vae_tiled(Gpu &gpu, Weights &weights, const Tensor &video, bool encode);
} // namespace hv15n
#endif
