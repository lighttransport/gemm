// SPDX-License-Identifier: MIT
//
// Vulkan RDNA4 cooperative-matrix GEMM experiment.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vulkan-runner.hh"

using vl_cpp::vulkan::VulkanComputeRunner;

struct Shape {
    const char* name;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const Shape kShapes[] = {
    {"qkv",      4096, 3456, 1152},
    {"attn_out", 4096, 1152, 1152},
    {"ffn_up",   4096, 4304, 1152},
    {"ffn_down", 4096, 1152, 4304},
    {"mm0",      1024, 4608, 4608},
    {"mm2",      1024, 5120, 4608},
};

struct Config {
    std::string shape = "mm0";
    std::string dtype = "f16";
    std::string shader_dir = ".";
    int device = 0;
    int iters = 200;
    int warmup = 5;
    int check_samples = 4096;
    bool check = true;
    bool list = false;
    bool check_caps = false;
};

struct Dims {
    uint32_t m, n, k;
    uint32_t mp, np, kp;
};

static uint32_t round_up(uint32_t x, uint32_t q) {
    return ((x + q - 1) / q) * q;
}

static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            exp = exp + (127 - 15);
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        out = sign | (exp << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

static float bf16_to_float(uint16_t h) {
    uint32_t x = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

static void usage(const char* prog) {
    std::cout
        << "usage: " << prog << " [--shape NAME] [--iters N] [--warmup N] [--check|--no-check]\n"
        << "  --shape       qkv|attn_out|ffn_up|ffn_down|mm0|mm2|all (default: mm0)\n"
        << "  --dtype       f16|bf16 (default: f16)\n"
        << "  --iters       timed dispatch count (default: 200)\n"
        << "  --warmup      warmup dispatch count (default: 5)\n"
        << "  --samples     verification sample count (default: 4096)\n"
        << "  --device      Vulkan device index (default: 0)\n"
        << "  --shader-dir  build directory containing shaders/wmma (default: .)\n"
        << "  --list        list devices\n"
        << "  --check-caps  print cooperative-matrix properties\n";
}

static bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need_value = [&](const char* opt) -> const char* {
            if (++i >= argc) {
                std::cerr << "missing value for " << opt << "\n";
                return nullptr;
            }
            return argv[i];
        };

        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--shape") {
            const char* v = need_value("--shape"); if (!v) return false;
            cfg.shape = v;
        } else if (a == "--dtype") {
            const char* v = need_value("--dtype"); if (!v) return false;
            cfg.dtype = v;
            if (cfg.dtype != "f16" && cfg.dtype != "bf16") {
                std::cerr << "unsupported dtype: " << cfg.dtype << "\n";
                return false;
            }
        } else if (a == "--iters") {
            const char* v = need_value("--iters"); if (!v) return false;
            cfg.iters = std::max(1, std::atoi(v));
        } else if (a == "--warmup") {
            const char* v = need_value("--warmup"); if (!v) return false;
            cfg.warmup = std::max(0, std::atoi(v));
        } else if (a == "--samples") {
            const char* v = need_value("--samples"); if (!v) return false;
            cfg.check_samples = std::max(1, std::atoi(v));
        } else if (a == "--device") {
            const char* v = need_value("--device"); if (!v) return false;
            cfg.device = std::atoi(v);
        } else if (a == "--shader-dir") {
            const char* v = need_value("--shader-dir"); if (!v) return false;
            cfg.shader_dir = v;
        } else if (a == "--check") {
            cfg.check = true;
        } else if (a == "--no-check") {
            cfg.check = false;
        } else if (a == "--list") {
            cfg.list = true;
        } else if (a == "--check-caps") {
            cfg.check_caps = true;
        } else {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        }
    }
    return true;
}

static std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(code.data()), bytes);
    return code;
}

static const char* component_name(VkComponentTypeKHR type) {
    switch (type) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "i8";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        default: return "?";
    }
}

static bool print_caps(VulkanComputeRunner& runner) {
    std::vector<VkCooperativeMatrixPropertiesKHR> props;
    if (!runner.getCooperativeMatrixProperties(props)) {
        std::cout << "VK_KHR_cooperative_matrix properties unavailable\n";
        return false;
    }

    bool has_f16 = false;
    std::cout << "cooperative matrix properties:\n";
    for (const auto& p : props) {
        std::cout << "  M=" << p.MSize << " N=" << p.NSize << " K=" << p.KSize
                  << " A=" << component_name(p.AType)
                  << "(" << static_cast<int>(p.AType) << ")"
                  << " B=" << component_name(p.BType)
                  << "(" << static_cast<int>(p.BType) << ")"
                  << " C=" << component_name(p.CType)
                  << "(" << static_cast<int>(p.CType) << ")"
                  << " R=" << component_name(p.ResultType)
                  << "(" << static_cast<int>(p.ResultType) << ")"
                  << " scope=" << (p.scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : "other")
                  << "\n";
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
            p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            p.scope == VK_SCOPE_SUBGROUP_KHR) {
            has_f16 = true;
        }
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
            p.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
            p.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
            p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            p.scope == VK_SCOPE_SUBGROUP_KHR) {
            has_f16 = true;
        }
    }
    return has_f16;
}

class Bench {
public:
    bool init(const Config& cfg);
    bool run_shape(const Shape& shape, const Config& cfg);
    bool list_devices();
    bool check_caps(int device);
    const std::string& error() const { return last_error_; }

private:
    VulkanComputeRunner runner_;
    VulkanComputeRunner::ComputePipeline pipeline_{};
    bool pipeline_created_ = false;
    std::string last_error_;

    void destroy_pipeline();
    bool record_dispatches(uint32_t gx, uint32_t gy, int count, const void* pc, uint32_t pc_size);
};

void Bench::destroy_pipeline() {
    if (pipeline_created_) {
        runner_.destroyComputePipeline(pipeline_);
        pipeline_created_ = false;
    }
}

bool Bench::init(const Config& cfg) {
    if (!vl_cpp::vulkan::InitializeVulkan()) {
        last_error_ = "failed to initialize Vulkan loader";
        return false;
    }
    if (!runner_.initialize(false)) {
        last_error_ = runner_.getLastError();
        return false;
    }
    if (!runner_.selectDevice(static_cast<uint32_t>(cfg.device))) {
        last_error_ = runner_.getLastError();
        return false;
    }

    std::string spv_path = cfg.shader_dir + "/shaders/wmma/gemm_wmma_" + cfg.dtype + ".spv";
    std::vector<uint32_t> spv = load_spirv(spv_path);
    if (spv.empty()) {
        last_error_ = "failed to load shader: " + spv_path;
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (uint32_t i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    if (!runner_.createComputePipelineWithPushConstants(
            spv, bindings, 3 * sizeof(uint32_t), pipeline_)) {
        last_error_ = runner_.getLastError();
        return false;
    }
    pipeline_created_ = true;
    return true;
}

bool Bench::list_devices() {
    if (!vl_cpp::vulkan::InitializeVulkan()) return false;
    if (!runner_.initialize(false)) {
        std::cerr << "initialize failed: " << runner_.getLastError() << "\n";
        return false;
    }
    uint32_t count = runner_.getDeviceCount();
    for (uint32_t i = 0; i < count; i++) {
        std::cout << "[" << i << "] " << runner_.getDeviceName(i);
        if (runner_.deviceSupportsExtension(i, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
            std::cout << "  coopmat";
        }
        std::cout << "\n";
    }
    return true;
}

bool Bench::check_caps(int device) {
    if (!vl_cpp::vulkan::InitializeVulkan()) return false;
    if (!runner_.initialize(false)) return false;
    if (!runner_.selectDevice(static_cast<uint32_t>(device))) return false;
    std::cout << "device: " << runner_.getDeviceName(static_cast<uint32_t>(device)) << "\n";
    bool ok = print_caps(runner_);
    std::cout << "required f16/f16->f32 16x16x16 subgroup tile: "
              << (ok ? "yes" : "no") << "\n";
    return ok;
}

bool Bench::record_dispatches(uint32_t gx, uint32_t gy, int count, const void* pc, uint32_t pc_size) {
    if (!runner_.beginRecording()) return false;
    runner_.bindComputePipeline(pipeline_);
    runner_.bindDescriptorSets(pipeline_);
    runner_.pushConstants(pipeline_, pc, pc_size);
    for (int i = 0; i < count; i++) {
        runner_.dispatch(gx, gy, 1);
    }
    if (!runner_.endRecordingAndSubmit()) return false;
    return runner_.waitForCompletion();
}

bool Bench::run_shape(const Shape& s, const Config& cfg) {
    Dims d{s.m, s.n, s.k, round_up(s.m, 128), round_up(s.n, 64), round_up(s.k, 64)};
    const size_t x_elems = size_t(d.mp) * d.kp;
    const size_t w_elems = size_t(d.np) * d.kp;
    const size_t y_elems = size_t(d.mp) * d.np;

    std::vector<uint16_t> x(x_elems, 0);
    std::vector<uint16_t> w(w_elems, 0);
    std::vector<float> y(y_elems, 0.0f);

    std::mt19937 rng(0x1234abcd);
    const bool is_bf16 = cfg.dtype == "bf16";
    auto to_raw = [&](float v) -> uint16_t {
        return is_bf16 ? float_to_bf16(v) : float_to_half(v);
    };
    auto from_raw = [&](uint16_t v) -> float {
        return is_bf16 ? bf16_to_float(v) : half_to_float(v);
    };

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t m = 0; m < d.m; m++) {
        for (uint32_t k = 0; k < d.k; k++) x[size_t(m) * d.kp + k] = to_raw(dist(rng));
    }
    for (uint32_t n = 0; n < d.n; n++) {
        for (uint32_t k = 0; k < d.k; k++) w[size_t(n) * d.kp + k] = to_raw(dist(rng));
    }

    VulkanComputeRunner::BufferInfo bx{}, bw{}, by{};
    if (!runner_.createDeviceLocalBuffer(x.size() * sizeof(uint16_t), bx) ||
        !runner_.createDeviceLocalBuffer(w.size() * sizeof(uint16_t), bw) ||
        !runner_.createDeviceLocalBuffer(y.size() * sizeof(float), by)) {
        last_error_ = runner_.getLastError();
        return false;
    }

    bool ok = true;
    ok = ok && runner_.uploadToDeviceLocal(bx, x.data(), x.size() * sizeof(uint16_t));
    ok = ok && runner_.uploadToDeviceLocal(bw, w.data(), w.size() * sizeof(uint16_t));
    ok = ok && runner_.updateDescriptorSet(pipeline_, {bx, bw, by});

    struct Push { uint32_t M, N, K; } pc{d.mp, d.np, d.kp};
    if (ok && !runner_.beginRecording()) ok = false;
    if (ok) {
        runner_.bindComputePipeline(pipeline_);
        runner_.bindDescriptorSets(pipeline_);
        runner_.pushConstants(pipeline_, &pc, sizeof(pc));
        for (int i = 0; i < std::max(1, cfg.warmup); i++) runner_.dispatch(d.np / 64, d.mp / 128, 1);
        ok = runner_.endRecordingAndSubmit() && runner_.waitForCompletion();
    }

    double cosine = 0.0;
    double rms = 0.0;
    float max_abs = 0.0f;
    if (ok && cfg.check) {
        ok = runner_.downloadFromDeviceLocal(by, y.data(), y.size() * sizeof(float));
        double dot = 0.0, ng = 0.0, nr = 0.0, se = 0.0;
        const uint64_t total = uint64_t(d.m) * d.n;
        const uint64_t step = std::max<uint64_t>(1, total / uint64_t(cfg.check_samples));
        int samples = 0;
        for (uint64_t idx = 0; idx < total && samples < cfg.check_samples; idx += step, samples++) {
            uint32_t m = uint32_t(idx / d.n);
            uint32_t n = uint32_t(idx - uint64_t(m) * d.n);
            float ref = 0.0f;
            for (uint32_t k = 0; k < d.k; k++) {
                ref += from_raw(x[size_t(m) * d.kp + k]) *
                       from_raw(w[size_t(n) * d.kp + k]);
            }
            float got = y[size_t(m) * d.np + n];
            float diff = std::fabs(got - ref);
            max_abs = std::max(max_abs, diff);
            se += double(diff) * diff;
            dot += double(got) * ref;
            ng += double(got) * got;
            nr += double(ref) * ref;
        }
        rms = std::sqrt(se / std::max(1, samples));
        cosine = (ng > 0.0 && nr > 0.0) ? dot / std::sqrt(ng * nr) : 0.0;
    }

    double ms = 0.0;
    if (ok) {
        auto t0 = std::chrono::steady_clock::now();
        ok = record_dispatches(d.np / 64, d.mp / 128, cfg.iters, &pc, sizeof(pc));
        auto t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / double(cfg.iters);
    }

    runner_.destroyBuffer(bx);
    runner_.destroyBuffer(bw);
    runner_.destroyBuffer(by);
    if (!ok) {
        last_error_ = runner_.getLastError();
        return false;
    }

    double tflops = (2.0 * double(d.m) * double(d.n) * double(d.k)) / (ms * 1.0e-3) * 1.0e-12;
    std::cout << std::fixed << std::setprecision(4)
              << "  [wmma_" << cfg.dtype << "] " << std::left << std::setw(9) << s.name << std::right
              << " M=" << std::setw(4) << d.m << " N=" << std::setw(4) << d.n
              << " K=" << std::setw(4) << d.k << "  " << std::setw(8) << ms << " ms  "
              << std::setprecision(1) << std::setw(7) << tflops << " TFLOP/s";
    if (cfg.check) {
        std::cout << std::setprecision(6) << "  cos=" << cosine
                  << "  rms=" << rms << "  maxd=" << max_abs;
    }
    if (d.n != d.np || d.k != d.kp || d.m != d.mp) {
        std::cout << "  padded=(" << d.mp << "," << d.np << "," << d.kp << ")";
    }
    std::cout << "\n";
    return true;
}

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        usage(argv[0]);
        return 1;
    }

    Bench bench;
    if (cfg.list) return bench.list_devices() ? 0 : 1;
    if (cfg.check_caps) return bench.check_caps(cfg.device) ? 0 : 1;

    if (!bench.init(cfg)) {
        std::cerr << "failed to initialize benchmark: " << bench.error() << "\n";
        return 1;
    }

    std::cout << "vulkan/wmma RDNA4 " << cfg.dtype << " cooperative-matrix GEMM"
              << "  iters=" << cfg.iters
              << "  check=" << (cfg.check ? "sampled" : "off") << "\n";

    int ran = 0;
    for (const Shape& shape : kShapes) {
        if (cfg.shape != "all" && cfg.shape != shape.name) continue;
        if (!bench.run_shape(shape, cfg)) {
            std::cerr << "benchmark failed: " << bench.error() << "\n";
            return 1;
        }
        ran++;
    }
    if (ran == 0) {
        std::cerr << "unknown shape: " << cfg.shape << "\n";
        return 1;
    }
    return 0;
}
