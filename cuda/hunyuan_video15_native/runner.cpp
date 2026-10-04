#include "gpu.hpp"
#include "hv15_native.h"
#include "kernels.hpp"
#ifdef HV15N_ROCM
#include "../../rdna4/video_common/kernels.hpp"
#endif
#include <csignal>
#include <iostream>
#include <set>
using namespace hv15n;
static volatile sig_atomic_t stopped;
static void signal_handler(int) {
    stopped = 1;
}
struct Output {
    fs::path directory;
};
static int cancelled(void *) {
    return stopped != 0;
}
static void progress(int step, int total, void *) {
    std::cout << "PROGRESS " << step << " " << total << std::endl;
}
static int frame(int index, int width, int height, const unsigned char *rgb, void *user) {
    auto &out = *static_cast<Output *>(user);
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%05d.ppm", index);
    std::ofstream file(out.directory / name, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(rgb), size_t(width) * height * 3);
    return file ? 0 : -1;
}
static long long integer(const std::string &text) {
    size_t consumed = 0;
    auto value = std::stoll(text, &consumed);
    require(consumed == text.size() && value >= 0, "invalid integer: " + text);
    return value;
}
int main(int argc, char **argv) {
    try {
        hv15n_config config;
        hv15n_config_defaults(&config);
        hv15n_request request;
        hv15n_request_defaults(&request);
        std::map<std::string, std::string> options;
        bool validate = false, generate = false, experimental = false;
        const std::set<std::string> keys = {
            "--model",  "--task",          "--preset",     "--prompt",   "--negative-prompt",
            "--image",  "--vision-pixels", "--noise-file", "--dump-dir", "--width",
            "--height", "--frames",        "--seed",       "--device",   "--vram-budget-mib",
            "--gemm",   "--gemm-fallback", "--offload",    "--out-dir"
#ifdef HV15N_ROCM
            , "--aotriton-bridge"
#else
            , "--cudnn-attention", "--cudnn-library"
#endif
        };
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Native repository HunyuanVideo-1.5 runner\n"
                             "--generate --model DIR --task i2v|t2v --preset quality|fast12 --prompt TEXT\n"
                             "model default: " HV15N_DEFAULT_MODEL_DIR "\n"
                             "--image PREPARED.png --vision-pixels CHW.f32 (I2V only)\n"
                             "--width 480 --height 848 --frames 81 --seed N --out-dir EMPTY_DIR\n"
                             "--device 0 --vram-budget-mib 14336 --offload block\n"
#ifdef HV15N_ROCM
                             "--gemm repo|hipblas --gemm-fallback hipblas|error --allow-experimental\n"
                             "--aotriton-bridge LIB (optional standalone FP16 attention)\n"
#else
                             "--gemm repo|cublas --gemm-fallback cublas|error --allow-experimental\n"
                             "--cudnn-attention off|auto|libh3_cudnn.so --cudnn-library libcudnn.so.9\n"
#endif
                             "--validate: request only; --compile-kernels: runtime compiler only\n";
                return 0;
            }
            if (arg == "--compile-kernels") {
                Gpu::compile(ops_source);
                Gpu::compile(Gpu::mma_source());
#ifdef HV15N_ROCM
                Gpu::compile(video_rocm::attention_source);
#endif
                std::cout << "PASS operators and repository GEMM compiled\n";
                return 0;
            }
            if (arg == "--validate") {
                validate = true;
                continue;
            }
            if (arg == "--generate") {
                generate = true;
                continue;
            }
            if (arg == "--allow-experimental") {
                experimental = true;
                continue;
            }
            require(keys.count(arg) && i + 1 < argc, "unknown option or missing value: " + arg);
            require(!options.count(arg), "duplicate option: " + arg);
            options[arg] = argv[++i];
        }
        auto text = [&](const char *key, const char *fallback) {
            auto it = options.find(key);
            return it == options.end() ? fallback : it->second.c_str();
        };
        request.task = text("--task", request.task);
        request.preset = text("--preset", request.preset);
        request.prompt = text("--prompt", request.prompt);
        request.negative_prompt = text("--negative-prompt", request.negative_prompt);
        request.image = text("--image", nullptr);
        request.vision_pixels = text("--vision-pixels", nullptr);
        request.noise_file = text("--noise-file", nullptr);
        request.dump_dir = text("--dump-dir", nullptr);
        config.model_dir = text("--model", config.model_dir);
        config.gemm = text("--gemm", config.gemm);
        config.gemm_fallback = text("--gemm-fallback", config.gemm_fallback);
        auto number = [&](const char *key, int &destination) {
            auto it = options.find(key);
            if (it != options.end()) {
                auto value = integer(it->second);
                require(value <= INT32_MAX, "integer overflow");
                destination = int(value);
            }
        };
        number("--width", request.width);
        number("--height", request.height);
        number("--frames", request.frames);
        number("--device", config.device);
        number("--vram-budget-mib", config.vram_budget_mib);
        if (options.count("--seed"))
            request.seed = integer(options["--seed"]);
        require(std::string(text("--offload", "block")) == "block", "only block offload is supported");
        char error[1024] = {0};
        require(hv15n_validate(&request, error, sizeof(error)) == 0, error);
        require(validate != generate, "select exactly one of --validate or --generate");
        if (validate) {
            std::cout << "PASS native request\n";
            return 0;
        }
        require(experimental, "native GPU parity is unverified; use --allow-experimental");
        require(options.count("--out-dir"), "output directory is required");
        Output out{options["--out-dir"]};
        fs::create_directories(out.directory);
        require(fs::is_empty(out.directory), "output directory must be empty");
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::unique_ptr<hv15n_context, decltype(&hv15n_free)> ctx(hv15n_load(&config, error, sizeof(error)),
                                                                  hv15n_free);
        require(bool(ctx), error);
#ifdef HV15N_ROCM
        if (options.count("--aotriton-bridge"))
            require(hv15n_set_aotriton_bridge(ctx.get(), options["--aotriton-bridge"].c_str(),
                                             error, sizeof(error)) == 0, error);
#else
        require(!options.count("--cudnn-library") || options.count("--cudnn-attention"),
                "--cudnn-library requires --cudnn-attention");
        if (options.count("--cudnn-attention")) {
            int status = hv15n_set_cudnn_attention(
                ctx.get(), options["--cudnn-attention"].c_str(),
                options.count("--cudnn-library") ? options["--cudnn-library"].c_str() : nullptr,
                error, sizeof(error));
            require(status == 0, std::string(error));
        }
#endif
        hv15n_callbacks callbacks{progress, frame, cancelled, &out};
        int rc = hv15n_generate(ctx.get(), &request, &callbacks, error, sizeof(error));
        std::ofstream metrics(out.directory / "runner_metrics.json");
        metrics << hv15n_metrics(ctx.get()) << "\n";
        require(bool(metrics), "metrics write failed");
        require(rc == 0, error);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "hv15n: " << e.what() << "\n";
        return 2;
    }
}
