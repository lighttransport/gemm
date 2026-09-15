/* Reference-test bridge; linked only into libpixal3d_validation.so. */
#include "../../common/trellis2_fdg_mesh.h"
#include "engine.hh"
#include "mesh.hh"
#include <cstdio>
#include <omp.h>
static thread_local std::string last_error;
static thread_local int test_threads = 16;
static thread_local pixal3d_gpu_options test_gpu_options{sizeof(pixal3d_gpu_options), 1, PIXAL3D_GPU_LEGACY,
                                                         PIXAL3D_KERNEL_AUTO, PIXAL3D_FLOW_BF16, nullptr};
extern "C" int px_test_set_gpu(int execution, int kernels) {
    if (execution < 0 || execution > 1 || kernels < 0 || kernels > 2)
        return -1;
    test_gpu_options.execution = pixal3d_gpu_execution(execution);
    test_gpu_options.kernels = pixal3d_gpu_kernels(kernels);
    return 0;
}
extern "C" int px_test_set_gpu_flow_precision(int precision) {
    if (precision < 0 || precision > 2)
        return -1;
    test_gpu_options.flow_precision = pixal3d_flow_precision(precision);
    return 0;
}
extern "C" int px_test_set_threads(int threads) {
    if (threads <= 0)
        return -1;
    test_threads = threads;
    return 0;
}
static pixal3d_options options(int backend) {
    pixal3d_options o{};
    o.backend = pixal3d_backend(backend);
    o.threads = test_threads;
    o.vram_budget_mib = 14336;
    return o;
}
extern "C" const char *px_test_error() { return last_error.c_str(); }
#define TEST_BEGIN                                                                                           \
    try {                                                                                                    \
        px::Engine e(options(backend));                                                                      \
        e.configure(test_gpu_options);
#define TEST_END                                                                                             \
    return 0;                                                                                                \
    }                                                                                                        \
    catch (const std::exception &ex) {                                                                       \
        last_error = ex.what();                                                                              \
        return -1;                                                                                           \
    }
extern "C" int px_test_dino(int backend, const char *path, float *y, const float *x, int size, int blocks) {
    TEST_BEGIN px::Weights w(path);
    auto out = px::dino(e, w, px::Vec(x, x + size_t(size) * size * 3), size, blocks);
    std::copy(out.begin(), out.end(), y);
    TEST_END
}
extern "C" void px_test_free(void *p) { std::free(p); }
extern "C" int px_test_normals(const float *vertices, int nv, const int32_t *faces, int nf, float *out) {
    try {
        px::Mesh mesh;
        mesh.set(vertices, nv, faces, nf);
        px::Vec normals;
        px::vertex_normals(mesh, normals);
        std::copy(normals.begin(), normals.end(), out);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}
extern "C" int px_test_mesh(const float *vertices, int nv, const int32_t *faces, int nf, int resolution,
                            int target, t2_fdg_mesh *result) {
    try {
        omp_set_num_threads(16);
        px::Mesh mesh;
        mesh.set(vertices, nv, faces, nf);
        if (resolution == -1)
            px::clean_for_uv(mesh);
        else if (resolution == -2)
            px::fill_holes(mesh);
        else if (resolution) {
            trellis2::ClosestPointBVH bvh;
            px::require(bvh.build(vertices, nv, faces, nf), "Cannot build validation BVH");
            mesh = px::remesh(mesh, bvh, resolution);
        }
        if (target)
            px::simplify(mesh, target);
        result->n_verts = mesh.numV();
        result->n_tris = mesh.numF();
        result->vertices = static_cast<float *>(std::malloc(mesh.v.size() * sizeof(float)));
        result->triangles = static_cast<int32_t *>(std::malloc(mesh.f.size() * sizeof(int32_t)));
        px::require(result->vertices && result->triangles, "Cannot allocate validation mesh");
        std::copy(mesh.v.begin(), mesh.v.end(), result->vertices);
        std::copy(mesh.f.begin(), mesh.f.end(), result->triangles);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}
extern "C" int px_test_postprocess_dump(const int32_t *coords, const float *shape, const float *texture,
                                        int n, const char *output, const char *dump_dir) {
    pixal3d_result result{};
    try {
        omp_set_num_threads(16);
        pixal3d_options o;
        pixal3d_default_options(&o);
        o.dump_dir = dump_dir;
        px::postprocess(
            {px::Coords(coords, coords + size_t(n) * 4), px::Vec(shape, shape + size_t(n) * 7), 7},
            {px::Coords(coords, coords + size_t(n) * 4), px::Vec(texture, texture + size_t(n) * 6), 6}, o,
            result);
        int rc = pixal3d_write_glb(output, &result);
        pixal3d_result_free(&result);
        px::require(rc == 0, "GLB validation export failed");
        return 0;
    } catch (const std::exception &ex) {
        pixal3d_result_free(&result);
        last_error = ex.what();
        return -1;
    }
}
extern "C" int px_test_postprocess(const int32_t *coords, const float *shape, const float *texture, int n,
                                   const char *output) {
    return px_test_postprocess_dump(coords, shape, texture, n, output, nullptr);
}
extern "C" int px_test_structure(int backend, const char *path, float *y, const float *x, int precision) {
    TEST_BEGIN px::Weights w(path);
    auto out = px::decode_structure(e, w, px::Vec(x, x + 4096 * 8), precision);
    std::copy(out.begin(), out.end(), y);
    TEST_END
}
extern "C" int px_test_decode(int backend, const char *path, float **y, int32_t **out_coords, int *rows,
                              const float *x, const int32_t *coords, int n, int precision,
                              const char *guide_dir) {
    TEST_BEGIN px::Weights w(path);
    std::vector<px::Subdivision> sub;
    std::vector<px::Sparse> logits;
    bool guided = guide_dir && *guide_dir;
    if (guided)
        for (int stage = 0; stage < 4; ++stage) {
            px::Weights guide(std::string(guide_dir) + "/reference_sub_" + std::to_string(stage) +
                              ".safetensors");
            int count = guide.shape("feats")[0];
            const float *values = guide.get("feats");
            const int32_t *positions = static_cast<const int32_t *>(
                safetensors_data(guide.st, safetensors_find(guide.st, "coords")));
            px::Subdivision s;
            for (int p = 0; p < count; ++p)
                for (int child = 0; child < 8; ++child)
                    if (values[p * 8 + child] > 0) {
                        s.parents.push_back(p);
                        s.slots.push_back(child);
                        s.coords.push_back(0);
                        for (int a = 0; a < 3; ++a)
                            s.coords.push_back(positions[p * 4 + a + 1] * 2 + ((child >> a) & 1));
                    }
            sub.push_back(std::move(s));
        }
    auto out = px::decode_sparse(
        e, w, {px::Coords(coords, coords + size_t(n) * 4), px::Vec(x, x + size_t(n) * 32), 32}, false, sub,
        guided, precision, guided && w.has("blocks.0.4.to_subdiv.weight") ? &logits : nullptr);
    if (guided)
        for (size_t i = 0; i < logits.size(); ++i)
            px::dump(guide_dir, "native_sub_" + std::to_string(i), logits[i].feats, 8, logits[i].coords);
    *y = static_cast<float *>(std::malloc(out.feats.size() * sizeof(float)));
    *out_coords = static_cast<int32_t *>(std::malloc(out.coords.size() * sizeof(int32_t)));
    px::require(*y && *out_coords, "Failed to allocate validation result");
    std::copy(out.feats.begin(), out.feats.end(), *y);
    std::copy(out.coords.begin(), out.coords.end(), *out_coords);
    *rows = out.rows();
    TEST_END
}
extern "C" int px_test_naf(int backend, const char *path, float *y, float *guide, const float *x, int size,
                           const float *patches, int grid, int target, const float *xy, int n) {
    TEST_BEGIN px::Weights w(path);
    auto g = px::naf_guide(e, w, px::Vec(x, x + size_t(size) * size * 3), size, target);
    std::copy(g.begin(), g.end(), guide);
    auto p = px::Vec(patches, patches + size_t(grid) * grid * 1024), coords = px::Vec(xy, xy + 2 * n);
    auto out = e.resident() ? px::naf_sample_gpu(e, g, target, p, grid, coords)
                            : px::naf_sample(g, target, p, grid, coords);
    std::copy(out.begin(), out.end(), y);
    TEST_END
}
extern "C" int px_test_gemm(int backend, float *y, const float *x, const float *w, const float *b, int n,
                            int co, int ci, int bf) {
    TEST_BEGIN e.gemm(y, x, w, b, n, co, ci, bf);
    TEST_END
}
extern "C" int px_test_attention(int backend, float *y, const float *q, const float *k, const float *v, int n,
                                 int m, int heads, int d) {
    TEST_BEGIN e.attention(y, q, k, v, n, m, heads, d);
    TEST_END
}
extern "C" int px_test_flow(int backend, const char *path, float *y, const float *x, const int32_t *coords,
                            int n, int ci, const float *global, int gc, const float *proj, int pc, float t,
                            int blocks, int bf) {
    TEST_BEGIN
    px::Weights w(path);
    auto out = px::flow(e, w, px::Vec(x, x + size_t(n) * ci), px::Coords(coords, coords + size_t(n) * 4), t,
                        px::Vec(global, global + 5 * gc), px::Vec(proj, proj + size_t(n) * pc), blocks, bf);
    std::copy(out.begin(), out.end(), y);
    TEST_END
}

// A benchmark session owns both the model mapping and the engine across warm runs.
struct FlowSession {
    px::Engine engine;
    px::Weights weights;
    FlowSession(int backend, const char *path) : engine(options(backend)), weights(path) {
        engine.configure(test_gpu_options);
    }
};
extern "C" void *px_test_flow_open(int backend, const char *path) {
    try {
        return new FlowSession(backend, path);
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return nullptr;
    }
}
extern "C" void px_test_flow_close(void *session) { delete static_cast<FlowSession *>(session); }
extern "C" int px_test_flow_run(void *session, float *y, const float *x, const int32_t *coords, int n, int ci,
                                const float *global, int gc, const float *proj, int pc, float t, int blocks,
                                int precision) {
    try {
        auto &s = *static_cast<FlowSession *>(session);
        auto out = px::flow(s.engine, s.weights, px::Vec(x, x + size_t(n) * ci),
                            px::Coords(coords, coords + size_t(n) * 4), t, px::Vec(global, global + 5 * gc),
                            px::Vec(proj, proj + size_t(n) * pc), blocks, precision);
        std::copy(out.begin(), out.end(), y);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}

extern "C" int px_test_inpaint(uint8_t *pixels, const uint8_t *mask, int size, int channels, int radius) {
    try {
        std::vector<uint8_t> image(pixels, pixels + size_t(size) * size * channels),
            missing(mask, mask + size_t(size) * size);
        px::inpaint(image, channels, missing, size, radius);
        std::copy(image.begin(), image.end(), pixels);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}

extern "C" int px_test_pointwise(float *out, const float *x, const float *h, const float *params, int rows,
                                 int channels, int operation, int precision) {
    try {
        omp_set_num_threads(16);
        px::Vec values(x, x + size_t(rows) * channels);
        if (operation == 0)
            px::round_precision(values, precision);
        else if (operation == 1 || operation == 2)
            px::gelu(values, operation == 1, precision);
        else if (operation == 3)
            px::add_residual(values, px::Vec(h, h + values.size()), params, channels, precision);
        else if (operation == 4)
            px::apply_modulation(values, px::Vec(params, params + 2 * channels), 0, channels, precision);
        else
            throw std::runtime_error("Invalid pointwise test operation");
        std::copy(values.begin(), values.end(), out);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}

extern "C" int px_test_preprocess(const uint8_t *pixels, const uint8_t *mask, int width, int height,
                                  int channels, float *out512, float *out1024) {
    try {
        omp_set_num_threads(8);
        pixal3d_image input{pixels, mask, width, height, channels};
        auto cropped = px::preprocess(input);
        auto small = px::image_float(px::resize(cropped, 512, 512), false, false);
        auto large = px::image_float(px::resize(cropped, 1024, 1024), false, false);
        std::copy(small.begin(), small.end(), out512);
        std::copy(large.begin(), large.end(), out1024);
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return -1;
    }
}
