#include "../../common/pixal3d.h"
#include "../../common/stb_image.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

static float float_argument(const std::string &key, const std::string &value) {
    size_t end;
    float result = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(result))
        throw std::runtime_error("Invalid finite number for " + key);
    return result;
}

template <typename T> static T integer_argument(const std::string &key, const std::string &value) {
    size_t end;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid nonnegative integer for " + key);
    auto result = std::stoull(value, &end);
    if (end != value.size() || result > std::numeric_limits<T>::max())
        throw std::runtime_error("Integer out of range for " + key);
    return T(result);
}

int main(int argc, char **argv) {
    pixal3d_options options;
    pixal3d_default_options(&options);
    pixal3d_gpu_options gpu_options;
    pixal3d_default_gpu_options(&gpu_options);
    std::string profile;
    pixal3d_camera camera{0, 0, 1};
    std::string input, mask, output, model, dino, naf, dump;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::puts(
                    "Usage: pixal3d --input RGBA.png --output mesh.glb --fov RADIANS\n"
                    "  --backend cpu|cuda|rocm  --device N  --threads N\n"
                    "  --mask MASK.png (required for RGB)  --distance FLOAT  --mesh-scale FLOAT\n"
                    "  --model-dir DIR  --dinov3 FILE  --naf FILE  --seed N\n"
                    "  --vram-budget-mib N (maximum 14336)  --dump-dir DIR\n"
                    "  --gpu-execution legacy|resident  --gpu-kernels auto|blas|mma\n"
                    "  --gpu-flow-precision bf16|fp32|mixed  --profile-json FILE\n"
                    "Single-view Pixal3D main: 1024 cascade, BF16 flow by default, FP16 decoders, 4096 PBR textures.");
                return 0;
            }
            if (++i >= argc)
                throw std::runtime_error("Missing value for " + key);
            std::string value = argv[i];
            if (key == "--input")
                input = value;
            else if (key == "--output")
                output = value;
            else if (key == "--mask")
                mask = value;
            else if (key == "--gpu-execution") {
                if (value == "legacy")
                    gpu_options.execution = PIXAL3D_GPU_LEGACY;
                else if (value == "resident")
                    gpu_options.execution = PIXAL3D_GPU_RESIDENT;
                else
                    throw std::runtime_error("Invalid GPU execution mode");
            } else if (key == "--gpu-kernels") {
                if (value == "auto")
                    gpu_options.kernels = PIXAL3D_KERNEL_AUTO;
                else if (value == "blas")
                    gpu_options.kernels = PIXAL3D_KERNEL_BLAS;
                else if (value == "mma")
                    gpu_options.kernels = PIXAL3D_KERNEL_MMA;
                else
                    throw std::runtime_error("Invalid GPU kernel mode");
            } else if (key == "--gpu-flow-precision") {
                if (value == "bf16")
                    gpu_options.flow_precision = PIXAL3D_FLOW_BF16;
                else if (value == "fp32")
                    gpu_options.flow_precision = PIXAL3D_FLOW_FP32;
                else if (value == "mixed")
                    gpu_options.flow_precision = PIXAL3D_FLOW_MIXED;
                else
                    throw std::runtime_error("Invalid GPU flow precision");
            } else if (key == "--profile-json")
                profile = value;
            else if (key == "--backend") {
                if (value == "cpu")
                    options.backend = PIXAL3D_CPU;
                else if (value == "cuda")
                    options.backend = PIXAL3D_CUDA;
                else if (value == "rocm")
                    options.backend = PIXAL3D_ROCM;
                else
                    throw std::runtime_error("Invalid backend");
            } else if (key == "--fov")
                camera.fov = float_argument(key, value);
            else if (key == "--distance")
                camera.distance = float_argument(key, value);
            else if (key == "--mesh-scale")
                camera.mesh_scale = float_argument(key, value);
            else if (key == "--seed")
                options.seed = integer_argument<uint32_t>(key, value);
            else if (key == "--threads")
                options.threads = integer_argument<int>(key, value);
            else if (key == "--device")
                options.device = integer_argument<int>(key, value);
            else if (key == "--vram-budget-mib")
                options.vram_budget_mib = integer_argument<size_t>(key, value);
            else if (key == "--model-dir")
                model = value;
            else if (key == "--dinov3")
                dino = value;
            else if (key == "--naf")
                naf = value;
            else if (key == "--dump-dir")
                dump = value;
            else
                throw std::runtime_error("Unknown argument: " + key);
        }
        if (input.empty() || output.empty() || camera.fov <= 0)
            throw std::runtime_error("--input, --output and --fov are required; see --help");
        float distance;
        if (pixal3d_camera_distance(camera.fov, camera.mesh_scale, &distance))
            throw std::runtime_error("Invalid camera FOV or mesh scale");
        auto parent = std::filesystem::absolute(output).parent_path();
        std::filesystem::create_directories(parent);
        if (access(parent.c_str(), W_OK) || std::filesystem::is_directory(output))
            throw std::runtime_error("Output path is not writable: " + output);
        if (!model.empty())
            options.model_dir = model.c_str();
        if (!dino.empty())
            options.dinov3_path = dino.c_str();
        if (!naf.empty())
            options.naf_path = naf.c_str();
        if (!dump.empty())
            options.dump_dir = dump.c_str();
        pixal3d_image image{};
        int channels;
        std::unique_ptr<uint8_t, decltype(&stbi_image_free)> pixels(
            stbi_load(input.c_str(), &image.width, &image.height, &channels, 0), stbi_image_free);
        if (!pixels)
            throw std::runtime_error(std::string("Cannot load input: ") + stbi_failure_reason());
        image.channels = channels;
        image.pixels = pixels.get();
        std::unique_ptr<uint8_t, decltype(&stbi_image_free)> mask_pixels(nullptr, stbi_image_free);
        if (!mask.empty()) {
            int w, h, c;
            mask_pixels.reset(stbi_load(mask.c_str(), &w, &h, &c, 1));
            if (!mask_pixels || w != image.width || h != image.height)
                throw std::runtime_error("Mask must match input dimensions");
            image.mask = mask_pixels.get();
        }
        std::unique_ptr<pixal3d_context, decltype(&pixal3d_destroy)> context(pixal3d_create(&options),
                                                                             pixal3d_destroy);
        if (!context)
            throw std::runtime_error(pixal3d_last_error(nullptr));
        gpu_options.profile_json = profile.empty() ? nullptr : profile.c_str();
        if (pixal3d_configure_gpu(context.get(), &gpu_options))
            throw std::runtime_error(pixal3d_last_error(context.get()));
        pixal3d_result result{};
        if (pixal3d_generate(context.get(), &image, &camera, &result))
            throw std::runtime_error(pixal3d_last_error(context.get()));
        int rc = pixal3d_write_glb(output.c_str(), &result);
        std::printf("{\"vertices\":%d,\"triangles\":%d,\"shape_tokens\":%d,\"seconds\":%.3f,\"peak_device_"
                    "bytes\":%zu,\"peak_host_bytes\":%zu}\n",
                    result.vertex_count, result.triangle_count, result.stats.shape_tokens,
                    result.stats.elapsed_seconds, result.stats.peak_device_bytes,
                    result.stats.peak_host_bytes);
        pixal3d_result_free(&result);
        return rc ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Pixal3D: %s\n", e.what());
        return 1;
    }
}
