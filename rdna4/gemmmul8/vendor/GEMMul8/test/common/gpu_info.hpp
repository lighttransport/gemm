#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#define GEMMUL8_TEST_STRINGIFY_IMPL(x) #x
#define GEMMUL8_TEST_STRINGIFY(x)      GEMMUL8_TEST_STRINGIFY_IMPL(x)

#ifdef PRINT
    #undef PRINT
#endif

#define PRINT(outFile, LINE)                               \
    do {                                                   \
        (outFile) << std::scientific << LINE << std::endl; \
        std::cout << std::scientific << LINE << std::endl; \
    } while (0)

#if defined(__HIPCC__)
    #define GEMMUL8_GPUINFO_USE_HIP 1
#elif defined(__CUDACC__) || defined(__NVCC__)
    #define GEMMUL8_GPUINFO_USE_CUDA 1
#else
    #error "gpu_info.hpp requires NVCC/CUDACC or HIPCC."
#endif

#if defined(GEMMUL8_GPUINFO_USE_CUDA)

    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cuComplex.h>

#endif

#include "self_hipify.hpp"

#if defined(GEMMUL8_GPUINFO_USE_HIP)
    #define GEMMUL8_GPUINFO_RUNTIME_NAME "HIP"
    #define GEMMUL8_GPUINFO_BLAS_NAME    "hipBLAS(Lt)"
#else
    #define GEMMUL8_GPUINFO_RUNTIME_NAME "CUDA"
    #define GEMMUL8_GPUINFO_BLAS_NAME    "cuBLAS(Lt)"
#endif

#define CHECK_CUDA(x)                                                      \
    do {                                                                   \
        cudaError_t _e = (x);                                              \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s error %s:%d: %s\n",                   \
                         GEMMUL8_GPUINFO_RUNTIME_NAME, __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                          \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(x)                                                                        \
    do {                                                                                       \
        cublasStatus_t _s = (x);                                                               \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "%s error %s:%d: status=%d\n",                                \
                         GEMMUL8_GPUINFO_BLAS_NAME, __FILE__, __LINE__, static_cast<int>(_s)); \
            std::abort();                                                                      \
        }                                                                                      \
    } while (0)

inline int getCurrentDevice() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    return dev;
}

inline std::string sanitizeDeviceName(std::string deviceName) {
    for (char &c : deviceName) {
        if (c == ' ' || c == '/' || c == '\\') c = '_';
    }
    return deviceName;
}

inline std::string getDeviceName(int dev = -1) {
    if (dev < 0) dev = getCurrentDevice();

    cudaDeviceProp deviceProp{};
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));

    return sanitizeDeviceName(std::string(deviceProp.name));
}

inline int2 getComputeCapability(int dev = -1) {
    if (dev < 0) dev = getCurrentDevice();

    cudaDeviceProp deviceProp{};
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));

    return {deviceProp.major, deviceProp.minor};
}

inline std::string getHipArchName(int dev = -1) {
#if defined(GEMMUL8_GPUINFO_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    if (dev < 0) dev = getCurrentDevice();

    cudaDeviceProp deviceProp{};
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.gcnArchName[0] == '\0') return "unknown";
    return std::string(deviceProp.gcnArchName);
#else
    (void)dev;
    return "N/A";
#endif
}

inline std::string formatCudaVersion(int v) {
    if (v == 0) return "0";

    const int major = v / 1000;
    const int minor = (v % 1000) / 10;

    return std::to_string(major) + "." + std::to_string(minor);
}

inline std::string formatHipVersion(int v) {
    if (v == 0) return "0";

#if defined(__HIP_PLATFORM_NVIDIA__)
    return formatCudaVersion(v);
#else
    const int major = v / 10000000;
    const int minor = (v % 10000000) / 100000;
    const int patch = v % 100000;

    return std::to_string(major) + "." +
           std::to_string(minor) + "." +
           std::to_string(patch);
#endif
}

inline std::string formatLibraryVersion(size_t v) {
    if (v == 0) return "0";

    const size_t major = v / 10000;
    const size_t minor = (v % 10000) / 100;
    const size_t patch = v % 100;

    return std::to_string(major) + "." +
           std::to_string(minor) + "." +
           std::to_string(patch);
}

inline std::string getHostCompilerString() {
#if defined(__INTEL_LLVM_COMPILER)
    return "Intel oneAPI DPC++/C++ " +
           std::to_string(__INTEL_LLVM_COMPILER);
#elif defined(__INTEL_COMPILER)
    return "Intel C++ " +
           std::to_string(__INTEL_COMPILER);
#elif defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return "GCC " +
           std::to_string(__GNUC__) + "." +
           std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

inline std::string getHipblasVersionString() {
#if defined(hipblasVersionMajor) && defined(hipblasVersionMinor) && defined(hipblasVersionPatch)
    std::string s =
        std::to_string(hipblasVersionMajor) + "." +
        std::to_string(hipblasVersionMinor) + "." +
        std::to_string(hipblasVersionPatch);

    #if defined(hipblasVersionTweak)
    s += ".";
    s += GEMMUL8_TEST_STRINGIFY(hipblasVersionTweak);
    #endif

    return s;
#else
    return "unknown";
#endif
}

inline std::string getHipblasLtVersionString() {
#if defined(HIPBLASLT_VERSION_MAJOR) && defined(HIPBLASLT_VERSION_MINOR) && defined(HIPBLASLT_VERSION_PATCH)
    std::string s =
        std::to_string(HIPBLASLT_VERSION_MAJOR) + "." +
        std::to_string(HIPBLASLT_VERSION_MINOR) + "." +
        std::to_string(HIPBLASLT_VERSION_PATCH);

    #if defined(HIPBLASLT_VERSION_TWEAK)
    s += ".";
    s += GEMMUL8_TEST_STRINGIFY(HIPBLASLT_VERSION_TWEAK);
    #endif

    return s;
#elif defined(HIPBLASLT_VERSION)
    return formatLibraryVersion(static_cast<size_t>(HIPBLASLT_VERSION));
#else
    return "unknown";
#endif
}

inline std::string printEnvironmentInfo(const std::string &dateTime) {
    const int dev = getCurrentDevice();

    const std::string DeviceName = getDeviceName(dev);
    const std::string fileName =
        std::string("oz2_results_info_") + DeviceName + "_" + dateTime + ".csv";

    std::ofstream outFile(fileName);
    if (!outFile) {
        std::fprintf(stderr, "Failed to open output file: %s\n", fileName.c_str());
        std::abort();
    }

    int driverVersion  = 0;
    int runtimeVersion = 0;

    auto [cc_major, cc_minor] = getComputeCapability(dev);

    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));

    PRINT(outFile, "Device Name: " + DeviceName);
    PRINT(outFile, "Device ID: " + std::to_string(dev));
    PRINT(outFile, "Compute Capability: " +
                       std::to_string(cc_major) + "." + std::to_string(cc_minor));
    PRINT(outFile, "Host Compiler: " + getHostCompilerString());

#if defined(GEMMUL8_GPUINFO_USE_HIP)
    PRINT(outFile, "HIP Arch Name: " + getHipArchName(dev));

    PRINT(outFile, "HIP Driver API: " +
                       formatHipVersion(driverVersion) +
                       "(" + std::to_string(driverVersion) + ")");

    PRINT(outFile, "HIP Runtime: " +
                       formatHipVersion(runtimeVersion) +
                       "(" + std::to_string(runtimeVersion) + ")");

    PRINT(outFile, "hipBLAS macro: " + getHipblasVersionString());
    PRINT(outFile, "hipBLASLt macro: " + getHipblasLtVersionString());
#endif

#if defined(GEMMUL8_GPUINFO_USE_CUDA)
    PRINT(outFile, "CUDA Driver API: " +
                       formatCudaVersion(driverVersion) +
                       "(" + std::to_string(driverVersion) + ")");

    PRINT(outFile, "CUDA Runtime: " +
                       formatCudaVersion(runtimeVersion) +
                       "(" + std::to_string(runtimeVersion) + ")");

    {
        cublasHandle_t handle = nullptr;
        int cublasVersion     = 0;

        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasGetVersion(handle, &cublasVersion));
        CHECK_CUBLAS(cublasDestroy(handle));

        PRINT(outFile, "cuBLAS: " +
                           formatLibraryVersion(static_cast<size_t>(cublasVersion)) +
                           "(" + std::to_string(cublasVersion) + ")");
    }

    {
        const size_t cublasLtVersion = cublasLtGetVersion();

        PRINT(outFile, "cuBLASLt: " +
                           formatLibraryVersion(cublasLtVersion) +
                           "(" + std::to_string(cublasLtVersion) + ")");
    }
#endif

#if defined(GEMMUL8_GPUINFO_USE_CUDA) && defined(CUDART_VERSION)
    PRINT(outFile, "CUDART_VERSION macro: " +
                       formatCudaVersion(CUDART_VERSION) +
                       "(" + std::to_string(CUDART_VERSION) + ")");
#endif

#if defined(GEMMUL8_GPUINFO_USE_CUDA) && defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
    PRINT(outFile, "CUDACC macro: " +
                       std::to_string(__CUDACC_VER_MAJOR__) + "." +
                       std::to_string(__CUDACC_VER_MINOR__) + "." +
                       std::to_string(__CUDACC_VER_BUILD__));
#endif

#if defined(GEMMUL8_GPUINFO_USE_CUDA) && defined(CUBLAS_VER_MAJOR) && defined(CUBLAS_VER_MINOR) && defined(CUBLAS_VER_PATCH)
    PRINT(outFile, "cuBLAS macro: " +
                       std::to_string(CUBLAS_VER_MAJOR) + "." +
                       std::to_string(CUBLAS_VER_MINOR) + "." +
                       std::to_string(CUBLAS_VER_PATCH));
#endif

#if defined(GEMMUL8_GPUINFO_USE_HIP) && defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
    PRINT(outFile, "HIP macro: " +
                       std::to_string(HIP_VERSION_MAJOR) + "." +
                       std::to_string(HIP_VERSION_MINOR) + "." +
                       std::to_string(HIP_VERSION_PATCH));
#endif

    outFile.close();
    return DeviceName;
}

inline std::string getCurrentDateTime(std::chrono::system_clock::time_point &now) {
    now = std::chrono::system_clock::now();

    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");

    return ss.str();
}
