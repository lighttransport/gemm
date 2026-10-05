// SPDX-License-Identifier: MIT
#include "../../cuda/hunyuan_video15_native/host.hpp"
#include "h3.h"
#include <csignal>
#include <iostream>
#include <set>
using namespace hv15n;
static volatile sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
static int cancelled(void *) { return stopped != 0; }
static void progress(int a, int b, void *) {
    std::cout << "PROGRESS " << a << " " << b << std::endl;
}
static int frame(int i, int w, int h, const unsigned char *rgb, void *user) {
    auto &dir = *static_cast<fs::path *>(user);
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%05d.ppm", i);
    std::ofstream f(dir / name, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char *>(rgb), size_t(w) * h * 3);
    return f ? 0 : -1;
}
int main(int argc, char **argv) {
    try {
        h3_config c;
        h3_config_defaults(&c);
        h3_request r;
        h3_request_defaults(&r);
        std::map<std::string, std::string> args;
        bool validate = false, generate = false, experimental = false;
        std::set<std::string> keys = {
            "--model",           "--prompt",          "--width",        "--height",
            "--frames",          "--steps",           "--seed",         "--device",
            "--vram-budget-mib", "--convrot-hipblas", "--bf16-hipblas", "--fp32-hipblas",
            "--aotriton-bridge", "--vae-hipblas",     "--noise-file",   "--audio-noise-file",
            "--dump-dir",        "--out-dir",         "--cudnn-attention", "--cudnn-library",
            "--variant", "--conditioning"};
        for (int i = 1; i < argc; i++) {
            std::string s = argv[i];
            if (s == "--help") {
                std::cout << "Native MiniMax H3 Ref2VA/FL2VA INT8 ConvRot (CUDA/RDNA4) "
                             "runner\n--generate --allow-experimental --model DIR --prompt TEXT "
                             "--out-dir EMPTY_DIR\n--width 1344 --height 768 --frames 124 --steps "
                             "40 --seed 42\n--device 0 --vram-budget-mib 12288 --noise-file "
                             "NCTHW.f32 --audio-noise-file NC2T.f32\n--dump-dir DIR --validate\n"
                             "--variant ref2va|fl2va --conditioning PREPARED_BUNDLE\n"
                             "--aotriton-bridge libvideo_aotriton.so (HIP attention)\n"
                             "--vae-hipblas 0|1 (optional FP16 decoder GEMM)\n"
                             "--cudnn-attention off|auto|libh3_cudnn.so (CUDA, opt-in DiT SDPA)\n"
                             "--cudnn-library libcudnn.so.9 (default: loader path, system dirs)\n";
                return 0;
            }
            if (s == "--validate") {
                validate = true;
                continue;
            }
            if (s == "--generate") {
                generate = true;
                continue;
            }
            if (s == "--allow-experimental") {
                experimental = true;
                continue;
            }
            require(keys.count(s) && i + 1 < argc && !args.count(s),
                    "unknown, duplicate or missing option: " + s);
            args[s] = argv[++i];
        }
        auto str = [&](const char *k, const char *d) {
            auto it = args.find(k);
            return it == args.end() ? d : it->second.c_str();
        };
        auto number = [&](const char *k, long long d, long long maximum = INT32_MAX) {
            auto it = args.find(k);
            if (it == args.end())
                return d;
            size_t pos = 0;
            auto n = std::stoll(it->second, &pos);
            require(pos == it->second.size() && n >= 0 && n <= maximum, "invalid integer");
            return n;
        };
        c.model_dir = str("--model", c.model_dir);
        c.aotriton_bridge = str("--aotriton-bridge", nullptr);
        c.device = int(number("--device", c.device));
        c.vram_budget_mib = int(number("--vram-budget-mib", c.vram_budget_mib));
        c.bf16_hipblas = int(number("--bf16-hipblas", c.bf16_hipblas, 1));
        int fp32_hipblas = int(number("--fp32-hipblas", 1, 1));
        c.vae_hipblas = int(number("--vae-hipblas", c.vae_hipblas, 1));
        c.convrot_hipblas = int(number("--convrot-hipblas", c.convrot_hipblas, 1));
        r.prompt = str("--prompt", r.prompt);
        r.width = int(number("--width", r.width));
        r.height = int(number("--height", r.height));
        r.frames = int(number("--frames", r.frames));
        r.steps = int(number("--steps", r.steps));
        r.seed = number("--seed", r.seed, INT64_MAX);
        r.noise_file = str("--noise-file", nullptr);
        r.audio_noise_file = str("--audio-noise-file", nullptr);
        r.dump_dir = str("--dump-dir", nullptr);
        char error[8192] = {};
        std::string variant = str("--variant", "ref2va");
        require(variant == "ref2va" || variant == "fl2va", "invalid H3 variant");
        require(h3_validate(&r, error, sizeof(error)) == 0, error);
        if (validate) {
            std::cout << "PASS H3 request validation\n";
            return 0;
        }
        require(generate && experimental, "generation requires --generate --allow-experimental");
        fs::path out = str("--out-dir", "");
        require(!out.empty(), "missing output directory");
        fs::create_directories(out);
        require(fs::is_empty(out), "output directory must be empty");
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        // Call first: the message argument must be read after the API writes it.
        std::unique_ptr<h3_context, decltype(&h3_free)> ctx(h3_load(&c, error, sizeof(error)),
                                                            h3_free);
        require(bool(ctx), std::string(error));
        int status = h3_set_fp32_hipblas(ctx.get(), fp32_hipblas, error, sizeof(error));
        require(status == 0, error);
        status = h3_set_conditioning(ctx.get(), str("--variant", "ref2va"),
                                     str("--conditioning", nullptr), error, sizeof(error));
        require(status == 0, error);
        require(!args.count("--cudnn-library") || args.count("--cudnn-attention"),
                "--cudnn-library requires --cudnn-attention");
        if (args.count("--cudnn-attention"))
            status = h3_set_cudnn_attention(ctx.get(), str("--cudnn-attention", "off"),
                                            str("--cudnn-library", nullptr), error, sizeof(error));
        require(status == 0, error);
        h3_callbacks cb{progress, frame, cancelled, &out};
        status = h3_generate(ctx.get(), &r, &cb, error, sizeof(error));
        require(status == 0, error);
        std::ofstream metrics(out / "metrics.json");
        metrics << h3_metrics(ctx.get()) << "\n";
        require(bool(metrics), "metrics write failed");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "H3: " << e.what() << "\n";
        return 1;
    }
}
