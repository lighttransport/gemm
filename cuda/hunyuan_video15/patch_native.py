"""Apply the repo-owned conditioning overlay to the exact pinned native source."""
from pathlib import Path
import shutil
import subprocess
import sys
PIN = "3f8527a46c54ecf4cb4ed6003da8e8982283c73c"
HERE = Path(__file__).resolve().parent

def patch(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise RuntimeError(f"native revision must be {PIN}, found {commit}")
    # This ignored dependency is an owned build tree. Recreate overlay targets from
    # the pinned blobs so repeated setup cannot accumulate partial patches.
    for name in ("src/conditioning/conditioner.hpp", "src/pipeline/model_builders.cpp",
                 "src/pipeline/video.cpp", "src/pipeline/diffusion_engine.cpp",
                 "src/model/diffusion/hunyuan.hpp", "src/stable-diffusion.cpp",
                 "src/model/vae/hunyuan_vae.hpp"):
        original = subprocess.check_output(["git", "-C", str(root), "show", f"{PIN}:{name}"])
        (root / name).write_bytes(original)
    def replace(name, old, new):
        path = root / name
        text = path.read_text()
        if new in text:
            return
        if text.count(old) != 1:
            raise RuntimeError(f"patch context mismatch: {name}")
        path.write_text(text.replace(old, new, 1))
    shutil.copyfile(HERE / "siglip.hpp", root / "src/model/te/hv15_siglip.hpp")
    shutil.copyfile(HERE / "cuda_math.h", root / "src/core/hv15_cuda_math.h")
    replace("src/conditioning/conditioner.hpp", '#include "model/te/clip.hpp"',
            '#include "model/te/clip.hpp"\n#include "model/te/hv15_siglip.hpp"')
    replace("src/conditioning/conditioner.hpp", '    CLIPVisionModelProjection vision_model;\n',
            '    CLIPVisionModelProjection vision_model;\n    std::shared_ptr<HV15SiglipModel> siglip;\n')
    replace("src/conditioning/conditioner.hpp", '        bool proj_in = false;\n        for (const auto& [name, tensor_storage] : tensor_storage_map) {',
            '''        if (tensor_storage_map.count(weight_prefix + ".vision_model.embeddings.patch_embedding.bias")) {
            vision_model.hidden_size = 1152;
            vision_model.image_size = 384;
            siglip = std::make_shared<HV15SiglipModel>();
            siglip->init(params_ctx, tensor_storage_map, weight_prefix + ".vision_model");
            return;
        }
        bool proj_in = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {''')
    replace("src/conditioning/conditioner.hpp", '        vision_model.get_param_tensors(tensors, weight_prefix);',
            '        if (siglip) siglip->get_param_tensors(tensors, weight_prefix + ".vision_model");\n        else vision_model.get_param_tensors(tensors, weight_prefix);')
    replace("src/conditioning/conditioner.hpp", '        ggml_tensor* hidden_states = vision_model.forward(&runner_ctx, pixel_values, return_pooled, clip_skip);',
            '        ggml_tensor* hidden_states = siglip ? siglip->forward(&runner_ctx, pixel_values) : vision_model.forward(&runner_ctx, pixel_values, return_pooled, clip_skip);')
    replace("src/conditioning/conditioner.hpp", '        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, true));',
            '        HV15CudaF32Scope precision(runtime_backend, siglip && ggml_backend_is_cuda(runtime_backend));\n        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, true));')
    replace("src/pipeline/model_builders.cpp", '''        } else if (sd_version_is_hunyuan_video(version)) {
            result.conditioner''', '''        } else if (sd_version_is_hunyuan_video(version)) {
            if (ctx.params.clip_vision_path && *ctx.params.clip_vision_path) {
                result.clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE), tensor_storage_map, weight_manager);
            }
            result.conditioner''')
    replace("src/pipeline/video.cpp", '            latents.concat_latent = sd::ops::concat(concat_latent, concat_mask, 3);',
            '''            latents.concat_latent = sd::ops::concat(concat_latent, concat_mask, 3);
            if (sd->clip_vision != nullptr && sd->clip_vision->siglip) {
                latents.clip_vision_output = start_image.empty()
                    ? sd::zeros<float>({1152, 729, 1})
                    : sd->get_clip_vision_output(start_image, false, -1);
                if (latents.clip_vision_output.empty()) return std::nullopt;
            }''')
    replace("src/pipeline/diffusion_engine.cpp", '        auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);',
            '''        auto pixel_values = clip_vision->siglip
            ? sd::ops::interpolate(image, {384, 384, 3, 1}) * 2.0f - 1.0f
            : clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);''')
    # Match official system message, preserving its spaces. Native strips padding,
    # so compute crop from prefix tokenization rather than hardcoding 98.
    old = '''            prompt_template_encode_start_idx = 98;
            out_layers                       = {26};

            prompt =
                "<|im_start|>system\\nYou are a helpful assistant. Describe the video by detailing the following aspects:\\n"
                "1. The main content and theme of the video.\\n"
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\\n"
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\\n"
                "4. background environment, light, style and atmosphere.\\n"
                "5. camera angles, movements, and transitions used in the video.<|im_end|>\\n"
                "<|im_start|>user\\n";'''
    new = '''            out_layers = {26};
            prompt =
                "<|im_start|>system\\nYou are a helpful assistant. Describe the video by detailing the following aspects:         "
                "1. The main content and theme of the video.         "
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
                "4. background environment, light, style and atmosphere.         "
                "5. camera angles, movements, and transitions used in the video.<|im_end|>\\n"
                "<|im_start|>user\\n";
            std::vector<int> hv15_prefix_tokens;
            if (!tokenizer->encode(prompt, hv15_prefix_tokens, nullptr)) return {};
            prompt_template_encode_start_idx = static_cast<int>(hv15_prefix_tokens.size());
            max_length = 1000 + prompt_template_encode_start_idx;'''
    replace("src/conditioning/conditioner.hpp", old, new)

    shutil.copyfile(HERE / "dump.hpp", root / "src/core/hv15_dump.hpp")
    shutil.copyfile(HERE / "progress.h", root / "src/core/hv15_progress.h")
    for name in ("src/conditioning/conditioner.hpp", "src/pipeline/diffusion_engine.cpp", "src/pipeline/video.cpp", "src/model/diffusion/hunyuan.hpp"):
        path = root / name
        content = path.read_text()
        if '#include "core/hv15_dump.hpp"' not in content:
            path.write_text('#include "core/hv15_dump.hpp"\n' + content)
    replace("src/pipeline/diffusion_engine.cpp", '#include "core/hv15_dump.hpp"',
            '#include "core/hv15_dump.hpp"\n#include "core/hv15_progress.h"')
    replace("src/pipeline/diffusion_engine.cpp", '        pretty_progress(showstep, (int)total_steps, step_seconds);',
            '        hv15_report_sample_progress(showstep, (int)total_steps, step_seconds);')
    replace("src/pipeline/diffusion_engine.cpp", '            pretty_progress(0, (int)steps, 0);',
            '            hv15_report_sample_progress(0, (int)steps, 0);')
    replace("src/conditioning/conditioner.hpp", '        return new_hidden_states;\n',
            '        if (sd_version_is_hunyuan_video(version)) hv15_dump("qwen_hidden", new_hidden_states);\n        return new_hidden_states;\n')
    replace("src/pipeline/diffusion_engine.cpp", '    return output;\n}\n\n// Returns 50 Hz wav2vec2',
            '    if (clip_vision->siglip) hv15_dump("siglip_hidden", output);\n    return output;\n}\n\n// Returns 50 Hz wav2vec2')
    replace("src/model/diffusion/hunyuan.hpp", '            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());',
            '            auto result = restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());\n            hv15_dump("dit_first", result);\n            return result;')
    replace("src/pipeline/video.cpp", '        return latents;\n    }',
            '        if (sd_version_is_hunyuan_video(sd->version) && sd->clip_vision != nullptr &&\n            sd->clip_vision->siglip && latents.clip_vision_output.empty()) {\n            hv15_dump("siglip_hidden", sd::zeros<float>({1152, 729, 1}));\n        }\n        return latents;\n    }')

    shutil.copyfile(HERE / "conditioning_inputs.hpp", root / "src/core/hv15_conditioning_inputs.hpp")
    replace("src/pipeline/diffusion_engine.cpp", '#include "core/hv15_dump.hpp"',
            '#include "core/hv15_dump.hpp"\n#include "core/hv15_conditioning_inputs.hpp"')
    replace("src/pipeline/diffusion_engine.cpp", '? sd::ops::interpolate(image, {384, 384, 3, 1}) * 2.0f - 1.0f',
            '? hv15_siglip_pixels')
    replace("src/stable-diffusion.cpp", '#include "stable-diffusion.h"',
            '#include "stable-diffusion.h"\n#include "core/hv15_conditioning_inputs.hpp"\nextern "C" SD_API bool hv15_set_siglip_pixels(const char *path) { return hv15_load_siglip_pixels(path); }')

    replace("src/pipeline/video.cpp", '        sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd->rng);',
            '        sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd->rng);\n        if (sd_version_is_hunyuan_video(sd->version)) { hv15_override_noise(noise); hv15_dump("noise_input", noise); }')
    replace("src/pipeline/video.cpp", '        int64_t sampling_end = ggml_time_ms();\n        if (final_latent.empty()) {',
            '        int64_t sampling_end = ggml_time_ms();\n        if (sd_version_is_hunyuan_video(sd->version)) hv15_dump("latent_final", final_latent, true);\n        if (final_latent.empty()) {')
    replace("src/pipeline/video.cpp", '                if (encoded.dim() == 4) {',
            '                hv15_dump("vae_encoded", encoded);\n                if (encoded.dim() == 4) {')
    replace("src/pipeline/video.cpp", '        sd::Tensor<float> vid = sd->decode_first_stage(video_latent, true);',
            '        sd::Tensor<float> vid = sd->decode_first_stage(video_latent, true);\n        if (sd_version_is_hunyuan_video(sd->version)) hv15_dump("vae_decoded", vid);')

    # Video VAE attention is causal by frame, but permits every spatial token
    # in the current frame. Persistent host masks survive deferred graph upload.
    replace("src/model/vae/hunyuan_vae.hpp", "    class AttnBlock : public UnaryBlock {\n    protected:\n        int64_t in_channels;",
            "    class AttnBlock : public UnaryBlock {\n    protected:\n        int64_t in_channels;\n        std::map<std::pair<int64_t, int64_t>, std::vector<float>> causal_masks;")
    replace("src/model/vae/hunyuan_vae.hpp", "            x = ggml_ext_attention_ext(ctx, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);  // [b, t*h*w, c]",
            """            ggml_tensor* mask = nullptr;
            if (t > 1) {
                const int64_t area = w * h, length = area * t;
                auto key = std::make_pair(area, t);
                auto& data = causal_masks[key];
                if (data.empty()) {
                    data.resize(length * length);
                    for (int64_t query = 0; query < length; ++query) {
                        const int64_t visible = (query / area + 1) * area;
                        for (int64_t key_index = 0; key_index < length; ++key_index)
                            data[query * length + key_index] = key_index < visible ? 0.f : -INFINITY;
                    }
                }
                mask = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, length, length);
                ctx->bind_backend_tensor_data(mask, data.data());
            }
            x = ggml_ext_attention_ext(ctx, q, k, v, 1, mask, false, ctx->flash_attn_enabled);  // [b, t*h*w, c]""")

    # Use the dense causal VAE attention profile validated by continuous decode.
    replace("src/model/vae/hunyuan_vae.hpp", "            auto runner_ctx     = get_context();",
            "            auto runner_ctx     = get_context();\n            runner_ctx.flash_attn_enabled = false;")

    # The generic backend tiler adjusts tile positions and uses smootherstep.
    # Hunyuan's official VAE uses fixed strides and linear prefix blending.
    shutil.copyfile(HERE / "vae_spatial_tiling.hpp", root / "src/model/vae/hv15_spatial_tiling.hpp")
    replace("src/model/vae/hunyuan_vae.hpp", '#include "model/vae/wan_vae.hpp"',
            '#include "model/vae/wan_vae.hpp"\n#include "model/vae/hv15_spatial_tiling.hpp"')
    replace("src/model/vae/hunyuan_vae.hpp", '        ggml_cgraph* build_graph(const sd::Tensor<float>& input_tensor, bool decode_graph) {',
            '''        sd::Tensor<float> encode(int threads, const sd::Tensor<float>& input,
                                 sd_tiling_params_t params, bool cx=false, bool cy=false) override {
            if (input.dim()!=5 || !params.enabled || cx || cy || params.temporal_tiling ||
                params.rel_size_w || params.rel_size_h || params.tile_size_w!=params.tile_size_h)
                return VAE::encode(threads,input,params,cx,cy);
            auto pixels=input;
            if (scale_input) scale_tensor_to_minus1_1(&pixels);
            auto output=hv15_vae_spatial_tiles(pixels,false,params.tile_size_w>0 ? params.tile_size_w : 256,
                params.target_overlap,[&](const auto& tile) { return _compute(threads,tile,false); });
            runner_end();
            return output;
        }

        sd::Tensor<float> decode(int threads, const sd::Tensor<float>& input,
                                 sd_tiling_params_t params, bool video=false,
                                 bool cx=false, bool cy=false, bool silent=false) override {
            if (input.dim()!=5 || !params.enabled || cx || cy || params.temporal_tiling ||
                params.rel_size_w || params.rel_size_h || params.tile_size_w!=params.tile_size_h)
                return VAE::decode(threads,input,params,video,cx,cy,silent);
            auto output=hv15_vae_spatial_tiles(input,true,params.tile_size_w>0 ? params.tile_size_w : 256,
                params.target_overlap,[&](const auto& tile) { return _compute(threads,tile,true); });
            runner_end();
            if (!output.empty() && scale_input) scale_tensor_to_0_1(&output);
            return output;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& input_tensor, bool decode_graph) {''')

    # Hunyuan's semantic vision projection uses exact erf GELU, not WAN's tanh.
    shutil.copyfile(HERE / "vision_projection.hpp", root / "src/model/diffusion/hv15_vision_projection.hpp")
    replace("src/model/diffusion/hunyuan.hpp", '#include "model/diffusion/wan.hpp"',
            '#include "model/diffusion/wan.hpp"\n#include "model/diffusion/hv15_vision_projection.hpp"')
    replace("src/model/diffusion/hunyuan.hpp", 'std::make_shared<WAN::MLPProj>(this->config.vision_in_dim, this->config.hidden_size)',
            'std::make_shared<HV15VisionProjection>(this->config.vision_in_dim, this->config.hidden_size)')
    replace("src/model/diffusion/hunyuan.hpp", 'std::dynamic_pointer_cast<WAN::MLPProj>(blocks["vision_in"])',
            'std::dynamic_pointer_cast<HV15VisionProjection>(blocks["vision_in"])')

    shutil.copyfile(HERE / "cuda_attention.h", root / "src/core/hv15_cuda_attention.h")
    replace("src/model/diffusion/hunyuan.hpp", '#include "core/hv15_dump.hpp"',
            '#include "core/hv15_dump.hpp"\n#include "core/hv15_cuda_attention.h"')
    replace("src/model/diffusion/hunyuan.hpp", '            LOG_INFO("HunyuanVideo blocks: %d double, %d single", config.depth, config.depth_single_blocks);',
            '            hv15_cuda_attention_install(backend);\n            LOG_INFO("HunyuanVideo blocks: %d double, %d single", config.depth, config.depth_single_blocks);')

    # Every Hunyuan block needs explicit graph cuts for host-weight staging
    # under the consumer GPU budget. Preserve both streams and time embedding.
    replace("src/model/diffusion/hunyuan.hpp", "            for (int i = 0; i < config.depth; i++) {\n                auto block = std::dynamic_pointer_cast<Flux::DoubleStreamBlock>",
            "            sd::ggml_graph_cut::mark_graph_cut(img, \"hv15.prelude\", \"img\");\n            sd::ggml_graph_cut::mark_graph_cut(txt, \"hv15.prelude\", \"txt\");\n            sd::ggml_graph_cut::mark_graph_cut(vec, \"hv15.prelude\", \"vec\");\n            for (int i = 0; i < config.depth; i++) {\n                auto block = std::dynamic_pointer_cast<Flux::DoubleStreamBlock>")
    replace("src/model/diffusion/hunyuan.hpp", "                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]",
            "                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]\n                sd::ggml_graph_cut::mark_graph_cut(img, \"hv15.double.\" + std::to_string(i), \"img\");\n                sd::ggml_graph_cut::mark_graph_cut(txt, \"hv15.double.\" + std::to_string(i), \"txt\");")
    replace("src/model/diffusion/hunyuan.hpp", "                    txt_img    = block->forward(ctx, txt_img, vec, pe, nullptr);",
            "                    txt_img    = block->forward(ctx, txt_img, vec, pe, nullptr);\n                    sd::ggml_graph_cut::mark_graph_cut(txt_img, \"hv15.single.\" + std::to_string(i), \"txt_img\");")

    # Native Euler callbacks use one-based step numbers: sigmas[step] is r.
    replace("src/pipeline/diffusion_engine.cpp", '    size_t steps                = sigmas.size() - 1;',
            '    if (sd_version_is_hunyuan_video(version)) hv15_dump_sigmas(sigmas);\n    size_t steps                = sigmas.size() - 1;')
    replace("src/pipeline/diffusion_engine.cpp", 'if (sd_version_is_hunyuan_video(version) && step + 1 < sigmas.size()) {\n            hunyuan_timestep_r_tensor = sd::Tensor<float>::from_vector({sigmas[step + 1]});',
            'if (sd_version_is_hunyuan_video(version) && step > 0 && static_cast<size_t>(step) < sigmas.size()) {\n            hunyuan_timestep_r_tensor = sd::Tensor<float>::from_vector({sigmas[step]});')
    replace("src/conditioning/conditioner.hpp", '            if (!quoted_texts.empty()) {\n                std::string byt5_text;',
            '            if (quoted_texts.empty()) hv15_dump("byt5_hidden", sd::zeros<float>({1472, 256, 1}));\n            if (!quoted_texts.empty()) {\n                std::string byt5_text;')

if __name__ == "__main__":
    patch(Path(sys.argv[1]))
