// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Cooperative Matrix Matmul Benchmark
// Targets AMD RDNA4 with VK_KHR_cooperative_matrix extension
//
// Features:
// - Cooperative matrix FP16 matmul (requires VK_KHR_cooperative_matrix)
// - Tiled FP16 matmul fallback
// - Naive FP32 matmul baseline
// - Performance comparison and verification
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "vulkan-runner.hh"

using namespace vl_cpp;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

enum class KernelType {
    COOPMAT_F16,   // Cooperative matrix FP16 (RDNA4)
    TILED_F16,     // Tiled FP16 (fallback)
    NAIVE_F32      // Naive FP32 (baseline)
};

struct Config {
    uint32_t M = 1024;
    uint32_t N = 1024;
    uint32_t K = 1024;
    int device_id = 0;
    int num_iterations = 20;
    int warmup_iterations = 5;
    KernelType kernel = KernelType::TILED_F16;
    bool verify = true;
    bool benchmark = true;
    bool verbose = false;
    bool list_devices = false;
    bool compare_all = false;
    bool check_coopmat = false;
    std::string shader_dir = ".";
    float tolerance = 1e-2f;  // FP16 tolerance
};

const char* kernel_type_name(KernelType k) {
    switch (k) {
        case KernelType::COOPMAT_F16: return "coopmat_f16";
        case KernelType::TILED_F16: return "tiled_f16";
        case KernelType::NAIVE_F32: return "naive_f32";
        default: return "unknown";
    }
}

std::string get_shader_filename(KernelType k) {
    switch (k) {
        case KernelType::COOPMAT_F16: return "shaders/matmul_coopmat_f16.spv";
        case KernelType::TILED_F16: return "shaders/matmul_tiled_f16.spv";
        case KernelType::NAIVE_F32: return "shaders/matmul_naive_f32.spv";
        default: return "shaders/matmul_naive_f32.spv";
    }
}

bool is_fp16_kernel(KernelType k) {
    return k == KernelType::COOPMAT_F16 || k == KernelType::TILED_F16;
}

//------------------------------------------------------------------------------
// FP16 Conversion Helpers
//------------------------------------------------------------------------------

// Simple FP32 <-> FP16 conversion (IEEE 754 half-precision)
uint16_t float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exponent == 0xFF - 127 + 15) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
    } else if (exponent > 30) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            return *reinterpret_cast<float*>(&result);
        }
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x400;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }

    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;
    uint32_t result = sign | (exponent << 23) | mantissa;
    return *reinterpret_cast<float*>(&result);
}

//------------------------------------------------------------------------------
// CPU Reference Implementation
//------------------------------------------------------------------------------

void cpu_matmul_f32(const float* A, const float* B, float* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void cpu_matmul_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                float a = half_to_float(A[i * K + k]);
                float b = half_to_float(B[k * N + j]);
                sum += a * b;
            }
            C[i * N + j] = float_to_half(sum);
        }
    }
}

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Vulkan Cooperative Matrix Matmul Benchmark (RDNA4 Target)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << "  -l, --list              List Vulkan devices and exit\n";
    std::cout << "  --check-coopmat         Check cooperative matrix support and exit\n";
    std::cout << "  -d, --device <id>       Select device (default: 0)\n";
    std::cout << "  -M <size>               Matrix A rows (default: 1024)\n";
    std::cout << "  -N <size>               Matrix B columns (default: 1024)\n";
    std::cout << "  -K <size>               Inner dimension (default: 1024)\n";
    std::cout << "  -k, --kernel <type>     Kernel type (default: tiled_f16)\n";
    std::cout << "                          Types: coopmat_f16, tiled_f16, naive_f32\n";
    std::cout << "  --compare-all           Benchmark all available kernels\n";
    std::cout << "  --no-verify             Skip verification\n";
    std::cout << "  --no-benchmark          Skip benchmark\n";
    std::cout << "  -i, --iterations <n>    Benchmark iterations (default: 20)\n";
    std::cout << "  -w, --warmup <n>        Warmup iterations (default: 5)\n";
    std::cout << "  -s, --shaders <path>    Shader directory (default: .)\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "\nKernel types:\n";
    std::cout << "  coopmat_f16 - Cooperative matrix FP16 (requires VK_KHR_cooperative_matrix)\n";
    std::cout << "  tiled_f16   - Tiled FP16 with register blocking\n";
    std::cout << "  naive_f32   - Naive FP32 baseline\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " -M 2048 -N 2048 -K 2048\n";
    std::cout << "  " << prog << " -k coopmat_f16 --compare-all\n";
    std::cout << "  " << prog << " --check-coopmat\n";
}

KernelType parse_kernel_type(const std::string& name) {
    if (name == "coopmat_f16" || name == "coopmat") return KernelType::COOPMAT_F16;
    if (name == "tiled_f16" || name == "tiled") return KernelType::TILED_F16;
    if (name == "naive_f32" || name == "naive") return KernelType::NAIVE_F32;
    return KernelType::TILED_F16;
}

bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-l" || arg == "--list") {
            cfg.list_devices = true;
        } else if (arg == "--check-coopmat") {
            cfg.check_coopmat = true;
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) { std::cerr << "Missing device ID\n"; return false; }
            cfg.device_id = std::atoi(argv[i]);
        } else if (arg == "-M") {
            if (++i >= argc) { std::cerr << "Missing M value\n"; return false; }
            cfg.M = std::atoi(argv[i]);
        } else if (arg == "-N") {
            if (++i >= argc) { std::cerr << "Missing N value\n"; return false; }
            cfg.N = std::atoi(argv[i]);
        } else if (arg == "-K") {
            if (++i >= argc) { std::cerr << "Missing K value\n"; return false; }
            cfg.K = std::atoi(argv[i]);
        } else if (arg == "-k" || arg == "--kernel") {
            if (++i >= argc) { std::cerr << "Missing kernel type\n"; return false; }
            cfg.kernel = parse_kernel_type(argv[i]);
        } else if (arg == "--compare-all") {
            cfg.compare_all = true;
        } else if (arg == "--no-verify") {
            cfg.verify = false;
        } else if (arg == "--no-benchmark") {
            cfg.benchmark = false;
        } else if (arg == "-i" || arg == "--iterations") {
            if (++i >= argc) { std::cerr << "Missing iterations\n"; return false; }
            cfg.num_iterations = std::atoi(argv[i]);
        } else if (arg == "-w" || arg == "--warmup") {
            if (++i >= argc) { std::cerr << "Missing warmup\n"; return false; }
            cfg.warmup_iterations = std::atoi(argv[i]);
        } else if (arg == "-s" || arg == "--shaders") {
            if (++i >= argc) { std::cerr << "Missing shader path\n"; return false; }
            cfg.shader_dir = argv[i];
        } else if (arg == "-v" || arg == "--verbose") {
            cfg.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    return spirv;
}

bool list_devices() {
    if (!vulkan::InitializeVulkan()) {
        std::cerr << "Failed to initialize Vulkan\n";
        return false;
    }

    vulkan::VulkanComputeRunner runner;
    if (!runner.initialize(false)) {
        std::cerr << "Failed to initialize runner: " << runner.getLastError() << "\n";
        return false;
    }

    uint32_t count = runner.getDeviceCount();
    std::cout << "\n=== Available Vulkan Compute Devices ===\n\n";

    if (count == 0) {
        std::cout << "No devices found.\n";
    } else {
        for (uint32_t i = 0; i < count; i++) {
            std::cout << "  [" << i << "] " << runner.getDeviceName(i) << "\n";
        }
    }
    std::cout << "\n";

    runner.cleanup();
    return true;
}

//------------------------------------------------------------------------------
// Cooperative Matrix Support Check
//------------------------------------------------------------------------------

bool check_cooperative_matrix_support(int device_id) {
    if (!vulkan::InitializeVulkan()) {
        std::cerr << "Failed to initialize Vulkan\n";
        return false;
    }

    std::cout << "\n=== Checking Cooperative Matrix Support ===\n\n";

    // Note: Full cooperative matrix property enumeration requires
    // VK_KHR_cooperative_matrix extension functions which would need
    // to be added to vkew. For now, we just check if the extension
    // is advertised.

    std::cout << "Cooperative matrix support check requires runtime probing.\n";
    std::cout << "The extension VK_KHR_cooperative_matrix provides:\n";
    std::cout << "  - Hardware-accelerated matrix multiply-accumulate\n";
    std::cout << "  - Supported on AMD RDNA3/RDNA4, NVIDIA Ampere+, Intel Arc\n";
    std::cout << "\nTo verify support, try running the coopmat_f16 kernel.\n";
    std::cout << "If the shader compilation fails, the extension is not supported.\n\n";

    return true;
}

//------------------------------------------------------------------------------
// Verification
//------------------------------------------------------------------------------

struct VerifyResult {
    bool passed = false;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float avg_abs_error = 0.0f;
    size_t error_count = 0;
};

template<typename T>
VerifyResult verify_results(const T* gpu_result, const T* cpu_result,
                            size_t size, float tolerance, bool is_fp16) {
    VerifyResult result;
    result.passed = true;
    double sum_abs_error = 0.0;

    for (size_t i = 0; i < size; i++) {
        float gpu_val, cpu_val;
        if (is_fp16) {
            gpu_val = half_to_float(static_cast<uint16_t>(
                *reinterpret_cast<const uint16_t*>(&gpu_result[i])));
            cpu_val = half_to_float(static_cast<uint16_t>(
                *reinterpret_cast<const uint16_t*>(&cpu_result[i])));
        } else {
            gpu_val = static_cast<float>(*reinterpret_cast<const float*>(&gpu_result[i]));
            cpu_val = static_cast<float>(*reinterpret_cast<const float*>(&cpu_result[i]));
        }

        float abs_error = std::abs(gpu_val - cpu_val);
        float rel_error = (cpu_val != 0.0f) ? abs_error / std::abs(cpu_val) : abs_error;

        sum_abs_error += abs_error;
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        if (abs_error > tolerance && rel_error > tolerance) {
            result.error_count++;
            if (result.error_count <= 3) {
                std::cerr << "  Mismatch at [" << i << "]: GPU=" << gpu_val
                          << ", CPU=" << cpu_val << ", diff=" << abs_error << "\n";
            }
        }
    }

    result.avg_abs_error = static_cast<float>(sum_abs_error / size);
    result.passed = (result.error_count == 0);
    return result;
}

//------------------------------------------------------------------------------
// Matmul Runner
//------------------------------------------------------------------------------

class VulkanMatmul {
public:
    VulkanMatmul() = default;
    ~VulkanMatmul() { cleanup(); }

    bool initialize(int device_id, bool verbose);
    bool loadShader(const std::string& spirv_path, KernelType kernel_type);
    bool run_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
                 uint32_t M, uint32_t N, uint32_t K);
    bool run_f32(const float* A, const float* B, float* C,
                 uint32_t M, uint32_t N, uint32_t K);
    void cleanup();

    std::string getLastError() const { return last_error_; }
    std::string getDeviceName() const { return device_name_; }

private:
    vulkan::VulkanComputeRunner runner_;
    vulkan::VulkanComputeRunner::ComputePipeline pipeline_;
    vulkan::VulkanComputeRunner::BufferInfo buf_a_, buf_b_, buf_c_;
    bool initialized_ = false;
    bool pipeline_created_ = false;
    bool buffers_created_ = false;
    KernelType current_kernel_ = KernelType::NAIVE_F32;
    std::string last_error_;
    std::string device_name_;
    uint32_t current_M_ = 0, current_N_ = 0, current_K_ = 0;
    size_t element_size_ = sizeof(float);

    void destroyBuffers();
    bool createBuffers(uint32_t M, uint32_t N, uint32_t K, size_t elem_size);
};

bool VulkanMatmul::initialize(int device_id, bool verbose) {
    if (!vulkan::InitializeVulkan()) {
        last_error_ = "Failed to initialize Vulkan library";
        return false;
    }

    if (!runner_.initialize(false)) {
        last_error_ = "Failed to initialize runner: " + runner_.getLastError();
        return false;
    }

    uint32_t device_count = runner_.getDeviceCount();
    if (device_count == 0) {
        last_error_ = "No Vulkan compute devices found";
        return false;
    }

    if (static_cast<uint32_t>(device_id) >= device_count) {
        last_error_ = "Invalid device ID: " + std::to_string(device_id);
        return false;
    }

    device_name_ = runner_.getDeviceName(device_id);

    if (!runner_.selectDevice(device_id)) {
        last_error_ = "Failed to select device: " + runner_.getLastError();
        return false;
    }

    if (verbose) {
        std::cout << "Selected device: " << device_name_ << "\n";
    }

    initialized_ = true;
    return true;
}

bool VulkanMatmul::loadShader(const std::string& spirv_path, KernelType kernel_type) {
    if (!initialized_) {
        last_error_ = "Not initialized";
        return false;
    }

    std::vector<uint32_t> spirv = load_spirv(spirv_path);
    if (spirv.empty()) {
        last_error_ = "Failed to load SPIR-V from: " + spirv_path;
        return false;
    }

    // Define descriptor set layout bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    // Push constants: M, N, K (3 x uint32_t = 12 bytes)
    uint32_t pushConstantSize = 3 * sizeof(uint32_t);

    if (!runner_.createComputePipelineWithPushConstants(spirv, bindings,
                                                        pushConstantSize, pipeline_)) {
        last_error_ = "Failed to create pipeline: " + runner_.getLastError();
        return false;
    }

    pipeline_created_ = true;
    current_kernel_ = kernel_type;
    element_size_ = is_fp16_kernel(kernel_type) ? sizeof(uint16_t) : sizeof(float);
    return true;
}

void VulkanMatmul::destroyBuffers() {
    if (buffers_created_) {
        runner_.destroyBuffer(buf_a_);
        runner_.destroyBuffer(buf_b_);
        runner_.destroyBuffer(buf_c_);
        buffers_created_ = false;
    }
}

bool VulkanMatmul::createBuffers(uint32_t M, uint32_t N, uint32_t K, size_t elem_size) {
    if (buffers_created_ && current_M_ == M && current_N_ == N &&
        current_K_ == K && element_size_ == elem_size) {
        return true;
    }

    destroyBuffers();

    size_t size_a = M * K * elem_size;
    size_t size_b = K * N * elem_size;
    size_t size_c = M * N * elem_size;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (!runner_.createBuffer(size_a, usage, props, buf_a_)) {
        last_error_ = "Failed to create buffer A";
        return false;
    }
    if (!runner_.createBuffer(size_b, usage, props, buf_b_)) {
        runner_.destroyBuffer(buf_a_);
        last_error_ = "Failed to create buffer B";
        return false;
    }
    if (!runner_.createBuffer(size_c, usage, props, buf_c_)) {
        runner_.destroyBuffer(buf_a_);
        runner_.destroyBuffer(buf_b_);
        last_error_ = "Failed to create buffer C";
        return false;
    }

    current_M_ = M;
    current_N_ = N;
    current_K_ = K;
    element_size_ = elem_size;
    buffers_created_ = true;
    return true;
}

bool VulkanMatmul::run_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
                           uint32_t M, uint32_t N, uint32_t K) {
    if (!initialized_ || !pipeline_created_) {
        last_error_ = "Not ready";
        return false;
    }

    if (!createBuffers(M, N, K, sizeof(uint16_t))) {
        return false;
    }

    // Copy input data
    void* ptr = nullptr;
    if (!runner_.mapBuffer(buf_a_, &ptr)) {
        last_error_ = "Failed to map buffer A";
        return false;
    }
    std::memcpy(ptr, A, M * K * sizeof(uint16_t));
    runner_.unmapBuffer(buf_a_);

    if (!runner_.mapBuffer(buf_b_, &ptr)) {
        last_error_ = "Failed to map buffer B";
        return false;
    }
    std::memcpy(ptr, B, K * N * sizeof(uint16_t));
    runner_.unmapBuffer(buf_b_);

    // Update descriptors
    std::vector<vulkan::VulkanComputeRunner::BufferInfo> buffers = {buf_a_, buf_b_, buf_c_};
    if (!runner_.updateDescriptorSet(pipeline_, buffers)) {
        last_error_ = "Failed to update descriptor sets";
        return false;
    }

    // Record commands
    if (!runner_.beginRecording()) {
        last_error_ = "Failed to begin recording";
        return false;
    }

    runner_.bindComputePipeline(pipeline_);
    runner_.bindDescriptorSets(pipeline_);

    struct PushConstants { uint32_t M, N, K; } pc = {M, N, K};
    runner_.pushConstants(pipeline_, &pc, sizeof(pc));

    // Calculate dispatch dimensions based on kernel type
    uint32_t group_x, group_y;
    if (current_kernel_ == KernelType::COOPMAT_F16) {
        // Cooperative matrix: 128x64 output tile per workgroup (BM=128, BN=64)
        group_x = (N + 63) / 64;
        group_y = (M + 127) / 128;
    } else {
        // Tiled FP16: 32x32 output tile per workgroup
        group_x = (N + 31) / 32;
        group_y = (M + 31) / 32;
    }
    runner_.dispatch(group_x, group_y, 1);

    if (!runner_.endRecordingAndSubmit()) {
        last_error_ = "Failed to submit: " + runner_.getLastError();
        return false;
    }

    if (!runner_.waitForCompletion()) {
        last_error_ = "Failed to wait: " + runner_.getLastError();
        return false;
    }

    // Read back result
    if (!runner_.mapBuffer(buf_c_, &ptr)) {
        last_error_ = "Failed to map buffer C";
        return false;
    }
    std::memcpy(C, ptr, M * N * sizeof(uint16_t));
    runner_.unmapBuffer(buf_c_);

    return true;
}

bool VulkanMatmul::run_f32(const float* A, const float* B, float* C,
                           uint32_t M, uint32_t N, uint32_t K) {
    if (!initialized_ || !pipeline_created_) {
        last_error_ = "Not ready";
        return false;
    }

    if (!createBuffers(M, N, K, sizeof(float))) {
        return false;
    }

    // Copy input data
    void* ptr = nullptr;
    if (!runner_.mapBuffer(buf_a_, &ptr)) {
        last_error_ = "Failed to map buffer A";
        return false;
    }
    std::memcpy(ptr, A, M * K * sizeof(float));
    runner_.unmapBuffer(buf_a_);

    if (!runner_.mapBuffer(buf_b_, &ptr)) {
        last_error_ = "Failed to map buffer B";
        return false;
    }
    std::memcpy(ptr, B, K * N * sizeof(float));
    runner_.unmapBuffer(buf_b_);

    // Update descriptors
    std::vector<vulkan::VulkanComputeRunner::BufferInfo> buffers = {buf_a_, buf_b_, buf_c_};
    if (!runner_.updateDescriptorSet(pipeline_, buffers)) {
        last_error_ = "Failed to update descriptor sets";
        return false;
    }

    // Record commands
    if (!runner_.beginRecording()) {
        last_error_ = "Failed to begin recording";
        return false;
    }

    runner_.bindComputePipeline(pipeline_);
    runner_.bindDescriptorSets(pipeline_);

    struct PushConstants { uint32_t M, N, K; } pc = {M, N, K};
    runner_.pushConstants(pipeline_, &pc, sizeof(pc));

    // Naive: 16x16 workgroup, one output per thread
    uint32_t group_x = (N + 15) / 16;
    uint32_t group_y = (M + 15) / 16;
    runner_.dispatch(group_x, group_y, 1);

    if (!runner_.endRecordingAndSubmit()) {
        last_error_ = "Failed to submit: " + runner_.getLastError();
        return false;
    }

    if (!runner_.waitForCompletion()) {
        last_error_ = "Failed to wait: " + runner_.getLastError();
        return false;
    }

    // Read back result
    if (!runner_.mapBuffer(buf_c_, &ptr)) {
        last_error_ = "Failed to map buffer C";
        return false;
    }
    std::memcpy(C, ptr, M * N * sizeof(float));
    runner_.unmapBuffer(buf_c_);

    return true;
}

void VulkanMatmul::cleanup() {
    destroyBuffers();
    if (pipeline_created_) {
        runner_.destroyComputePipeline(pipeline_);
        pipeline_created_ = false;
    }
    if (initialized_) {
        runner_.cleanup();
        initialized_ = false;
    }
}

//------------------------------------------------------------------------------
// Benchmark
//------------------------------------------------------------------------------

struct BenchmarkResult {
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    double avg_time_ms = 0.0;
    double median_time_ms = 0.0;
    double gflops = 0.0;
    double tflops = 0.0;
};

template<typename T, typename RunFunc>
BenchmarkResult run_benchmark(VulkanMatmul& matmul, RunFunc run_func,
                              const T* A, const T* B, T* C,
                              uint32_t M, uint32_t N, uint32_t K,
                              int warmup, int iterations) {
    BenchmarkResult result;
    std::vector<double> times;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        run_func(matmul, A, B, C, M, N, K);
    }

    // Timed runs
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_func(matmul, A, B, C, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    result.min_time_ms = times.front();
    result.max_time_ms = times.back();
    result.median_time_ms = times[times.size() / 2];
    result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // FLOPS: 2*M*N*K for matmul
    double ops = 2.0 * M * N * K;
    result.gflops = (ops / (result.median_time_ms * 1e-3)) / 1e9;
    result.tflops = result.gflops / 1000.0;

    return result;
}

//------------------------------------------------------------------------------
// Main Test Functions
//------------------------------------------------------------------------------

int run_single_kernel_f16(const Config& cfg, VulkanMatmul& matmul,
                          const std::vector<uint16_t>& A,
                          const std::vector<uint16_t>& B,
                          std::vector<uint16_t>& C_gpu,
                          std::vector<uint16_t>& C_cpu) {
    std::cout << "[3/4] Running GPU matmul (FP16)...\n";
    if (!matmul.run_f16(A.data(), B.data(), C_gpu.data(), cfg.M, cfg.N, cfg.K)) {
        std::cerr << "ERROR: " << matmul.getLastError() << "\n";
        return 1;
    }
    std::cout << "      GPU computation complete\n";

    if (cfg.verify) {
        std::cout << "[4/4] Verifying against CPU...\n";
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul_f16(A.data(), B.data(), C_cpu.data(), cfg.M, cfg.N, cfg.K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        auto vr = verify_results(C_gpu.data(), C_cpu.data(),
                                 static_cast<size_t>(cfg.M) * cfg.N,
                                 cfg.tolerance, true);

        std::cout << "      " << (vr.passed ? "PASSED" : "FAILED")
                  << (vr.passed ? "" : " (" + std::to_string(vr.error_count) + " errors)") << "\n";
        std::cout << "      Max abs error: " << vr.max_abs_error << "\n";
        std::cout << "      Max rel error: " << vr.max_rel_error << "\n";
        std::cout << "      Avg abs error: " << vr.avg_abs_error << "\n";
        std::cout << "      CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " ms\n";

        if (!vr.passed) return 1;
    }

    if (cfg.benchmark) {
        std::cout << "\n=== Benchmark Results ===\n\n";

        auto run_func = [](VulkanMatmul& m, const uint16_t* a, const uint16_t* b,
                          uint16_t* c, uint32_t M, uint32_t N, uint32_t K) {
            m.run_f16(a, b, c, M, N, K);
        };

        auto br = run_benchmark(matmul, run_func, A.data(), B.data(), C_gpu.data(),
                                cfg.M, cfg.N, cfg.K,
                                cfg.warmup_iterations, cfg.num_iterations);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Min time:    " << br.min_time_ms << " ms\n";
        std::cout << "  Max time:    " << br.max_time_ms << " ms\n";
        std::cout << "  Avg time:    " << br.avg_time_ms << " ms\n";
        std::cout << "  Median time: " << br.median_time_ms << " ms\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Throughput:  " << br.gflops << " GFLOPS";
        if (br.tflops >= 1.0) {
            std::cout << " (" << br.tflops << " TFLOPS)";
        }
        std::cout << "\n";
    }

    return 0;
}

int run_single_kernel_f32(const Config& cfg, VulkanMatmul& matmul,
                          const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C_gpu,
                          std::vector<float>& C_cpu) {
    std::cout << "[3/4] Running GPU matmul (FP32)...\n";
    if (!matmul.run_f32(A.data(), B.data(), C_gpu.data(), cfg.M, cfg.N, cfg.K)) {
        std::cerr << "ERROR: " << matmul.getLastError() << "\n";
        return 1;
    }
    std::cout << "      GPU computation complete\n";

    if (cfg.verify) {
        std::cout << "[4/4] Verifying against CPU...\n";
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul_f32(A.data(), B.data(), C_cpu.data(), cfg.M, cfg.N, cfg.K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        auto vr = verify_results(C_gpu.data(), C_cpu.data(),
                                 static_cast<size_t>(cfg.M) * cfg.N,
                                 1e-3f, false);

        std::cout << "      " << (vr.passed ? "PASSED" : "FAILED") << "\n";
        std::cout << "      Max abs error: " << vr.max_abs_error << "\n";
        std::cout << "      CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " ms\n";

        if (!vr.passed) return 1;
    }

    if (cfg.benchmark) {
        std::cout << "\n=== Benchmark Results ===\n\n";

        auto run_func = [](VulkanMatmul& m, const float* a, const float* b,
                          float* c, uint32_t M, uint32_t N, uint32_t K) {
            m.run_f32(a, b, c, M, N, K);
        };

        auto br = run_benchmark(matmul, run_func, A.data(), B.data(), C_gpu.data(),
                                cfg.M, cfg.N, cfg.K,
                                cfg.warmup_iterations, cfg.num_iterations);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Min time:    " << br.min_time_ms << " ms\n";
        std::cout << "  Median time: " << br.median_time_ms << " ms\n";
        std::cout << "  Throughput:  " << std::fixed << std::setprecision(2) << br.gflops << " GFLOPS\n";
    }

    return 0;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.list_devices) {
        return list_devices() ? 0 : 1;
    }

    if (cfg.check_coopmat) {
        return check_cooperative_matrix_support(cfg.device_id) ? 0 : 1;
    }

    std::cout << "=== Vulkan Cooperative Matrix Matmul Benchmark ===\n\n";
    std::cout << "Target: AMD RDNA4 (VK_KHR_cooperative_matrix)\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Matrix size: [" << cfg.M << " x " << cfg.K << "] * ["
              << cfg.K << " x " << cfg.N << "] = [" << cfg.M << " x " << cfg.N << "]\n";
    std::cout << "  Kernel: " << kernel_type_name(cfg.kernel) << "\n";
    std::cout << "  Device: " << cfg.device_id << "\n";
    std::cout << "  Precision: " << (is_fp16_kernel(cfg.kernel) ? "FP16" : "FP32") << "\n";
    std::cout << "\n";

    // Initialize Vulkan
    std::cout << "[1/4] Initializing Vulkan...\n";
    VulkanMatmul matmul;
    if (!matmul.initialize(cfg.device_id, cfg.verbose)) {
        std::cerr << "ERROR: " << matmul.getLastError() << "\n";
        return 1;
    }
    std::cout << "      Device: " << matmul.getDeviceName() << "\n";

    // Load shader
    std::cout << "[2/4] Loading shader...\n";
    std::string shader_name = get_shader_filename(cfg.kernel);
    std::string shader_path = cfg.shader_dir + "/" + shader_name;

    if (!matmul.loadShader(shader_path, cfg.kernel)) {
        std::cerr << "ERROR: " << matmul.getLastError() << "\n";
        if (cfg.kernel == KernelType::COOPMAT_F16) {
            std::cerr << "\nNote: Cooperative matrix requires VK_KHR_cooperative_matrix support.\n";
            std::cerr << "Try using -k tiled_f16 for devices without this extension.\n";
        }
        return 1;
    }
    std::cout << "      Loaded: " << shader_path << "\n";

    int result;
    if (is_fp16_kernel(cfg.kernel)) {
        // FP16 path
        size_t size_a = cfg.M * cfg.K;
        size_t size_b = cfg.K * cfg.N;
        size_t size_c = cfg.M * cfg.N;

        std::vector<uint16_t> A(size_a), B(size_b), C_gpu(size_c), C_cpu(size_c);

        // Initialize with random FP16 data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < size_a; i++) A[i] = float_to_half(dist(rng));
        for (size_t i = 0; i < size_b; i++) B[i] = float_to_half(dist(rng));

        result = run_single_kernel_f16(cfg, matmul, A, B, C_gpu, C_cpu);
    } else {
        // FP32 path
        size_t size_a = cfg.M * cfg.K;
        size_t size_b = cfg.K * cfg.N;
        size_t size_c = cfg.M * cfg.N;

        std::vector<float> A(size_a), B(size_b), C_gpu(size_c), C_cpu(size_c);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < size_a; i++) A[i] = dist(rng);
        for (size_t i = 0; i < size_b; i++) B[i] = dist(rng);

        result = run_single_kernel_f32(cfg, matmul, A, B, C_gpu, C_cpu);
    }

    std::cout << "\n=== Test Complete ===\n";
    return result;
}
