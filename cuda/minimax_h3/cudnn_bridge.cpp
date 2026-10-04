// SPDX-License-Identifier: MIT
// cuDNN frontend SDPA bridge (header-only frontend, dynamic cuDNN/cudart loading).
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include "cudnn_bridge.h"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <glob.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace cudnn_frontend {
void *cudnn_dlhandle = nullptr;
}
namespace fe = cudnn_frontend;

namespace {
std::mutex lock;
cudnnHandle_t handle = nullptr;
std::map<std::tuple<int, int, int>, std::shared_ptr<fe::graph::Graph>> graphs;
std::string loaded;

int fail(char *error, size_t capacity, const std::string &text) {
    if (error && capacity)
        std::snprintf(error, capacity, "%s", text.c_str());
    return -1;
}
std::vector<std::string> candidates(const char *library) {
    if (library && *library)
        return {library};
    std::vector<std::string> out;
    if (const char *env = std::getenv("H3_CUDNN_LIB"))
        out.push_back(env);
    out.push_back("libcudnn.so.9"); // default loader search (LD_LIBRARY_PATH, ldconfig)
    std::vector<std::string> patterns;
    for (const char *var : {"VIRTUAL_ENV", "CONDA_PREFIX"})
        if (const char *root = std::getenv(var))
            patterns.push_back(std::string(root) +
                               "/lib/python3*/site-packages/nvidia/cudnn/lib/libcudnn.so.9");
    if (const char *home = std::getenv("HOME"))
        patterns.push_back(std::string(home) +
                           "/.local/lib/python3*/site-packages/nvidia/cudnn/lib/libcudnn.so.9");
    for (const char *p : {"/usr/local/lib/python3*/dist-packages/nvidia/cudnn/lib/libcudnn.so.9",
                          "/usr/lib/python3/dist-packages/nvidia/cudnn/lib/libcudnn.so.9",
                          "/usr/local/cuda/lib64/libcudnn.so.9",
                          "/usr/lib/x86_64-linux-gnu/libcudnn.so.9"})
        patterns.push_back(p);
    for (auto &pattern : patterns) {
        glob_t g{};
        if (glob(pattern.c_str(), 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; i++)
                out.push_back(g.gl_pathv[i]);
        globfree(&g);
    }
    return out;
}
} // namespace

extern "C" {
int h3_cudnn_bridge_abi(void) { return 1; }

int h3_cudnn_init(const char *library, char *info, size_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    if (handle) {
        std::snprintf(info, capacity, "%s", loaded.c_str());
        return 0;
    }
    // Prefer the driver-matched CUDA 13 runtime when several are installed.
    setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13", 0);
    std::string tried;
    for (auto &path : candidates(library)) {
        // RTLD_GLOBAL lets cuDNN's own sub-library loads resolve against this copy.
        void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            tried += " " + path;
            continue;
        }
        auto version = reinterpret_cast<size_t (*)()>(dlsym(h, "cudnnGetVersion"));
        if (!version || version() < 90000) {
            tried += " " + path + "(not cuDNN 9)";
            dlclose(h);
            continue;
        }
        fe::cudnn_dlhandle = h;
        try {
            if (fe::detail::create_handle(&handle) != CUDNN_STATUS_SUCCESS) {
                handle = nullptr;
                return fail(info, capacity, "cudnnCreate failed for " + path);
            }
        } catch (const std::exception &e) {
            return fail(info, capacity, std::string("cuDNN runtime: ") + e.what());
        }
        Dl_info where{};
        dladdr(reinterpret_cast<void *>(version), &where);
        loaded = std::string(where.dli_fname ? where.dli_fname : path.c_str()) + " (version " +
                 std::to_string(version()) + ")";
        std::snprintf(info, capacity, "%s", loaded.c_str());
        return 0;
    }
    return fail(info, capacity, "cuDNN 9 not found; tried:" + tried);
}

static std::shared_ptr<fe::graph::Graph> build(int rows, int heads, int dim, std::string &error) {
    auto key = std::make_tuple(rows, heads, dim);
    auto it = graphs.find(key);
    if (it != graphs.end())
        return it->second;
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const int64_t s = rows, h = heads, d = dim;
    auto tensor = [&](const char *name, int64_t uid) {
        return g->tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_uid(uid)
                             .set_dim({1, h, s, d})
                             .set_stride({h * s * d, s * d, d, 1}));
    };
    auto q = tensor("Q", 1), k = tensor("K", 2), v = tensor("V", 3);
    auto options = fe::graph::SDPA_attributes()
                       .set_name("h3_sdpa")
                       .set_generate_stats(false)
                       .set_attn_scale(1.f / std::sqrt(float(dim)));
    auto [o, stats] = g->sdpa(q, k, v, options);
    (void)stats;
    o->set_output(true).set_uid(4).set_dim({1, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    for (auto status : {g->validate(), g->build_operation_graph(handle),
                        g->create_execution_plans({fe::HeurMode_t::A}), g->check_support(handle),
                        g->build_plans(handle)})
        if (status.is_bad()) {
            error = "cuDNN SDPA graph: " + status.get_message();
            return nullptr;
        }
    graphs.emplace(key, g);
    return g;
}

long long h3_cudnn_workspace(int rows, int heads, int dim, char *error, size_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    if (!handle)
        return fail(error, capacity, "cuDNN bridge is not initialized");
    std::string message;
    auto g = build(rows, heads, dim, message);
    if (!g)
        return fail(error, capacity, message);
    int64_t bytes = 0;
    if (g->get_workspace_size(bytes).is_bad())
        return fail(error, capacity, "cuDNN workspace query failed");
    return bytes;
}

int h3_cudnn_attention(void *out, const void *q, const void *k, const void *v, int rows, int heads,
                       int dim, float scale, void *workspace, void *stream, char *error,
                       size_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    if (!handle)
        return fail(error, capacity, "cuDNN bridge is not initialized");
    if (std::fabs(scale - 1.f / std::sqrt(float(dim))) > 1e-7f)
        return fail(error, capacity, "cuDNN bridge supports 1/sqrt(dim) scaling only");
    std::string message;
    auto g = build(rows, heads, dim, message);
    if (!g)
        return fail(error, capacity, message);
    if (fe::detail::set_stream(handle, static_cast<cudaStream_t>(stream)) != CUDNN_STATUS_SUCCESS)
        return fail(error, capacity, "cudnnSetStream failed");
    std::unordered_map<int64_t, void *> pack = {{1, const_cast<void *>(q)},
                                                {2, const_cast<void *>(k)},
                                                {3, const_cast<void *>(v)},
                                                {4, out}};
    auto status = g->execute(handle, pack, workspace);
    if (status.is_bad())
        return fail(error, capacity, "cuDNN SDPA execute: " + status.get_message());
    return 0;
}
}
