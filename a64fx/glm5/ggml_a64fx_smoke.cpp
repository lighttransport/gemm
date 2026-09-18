#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 2;

    ggml_init_params p = {
        ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(16, false),
        nullptr, true
    };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return 3;

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * y = ggml_mul_mat(ctx, a, b);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_set_output(y);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 4;
    const float av[6] = {1, 2, 3, 4, 5, 6};
    const float bv[6] = {2, 1, 0, 1, 3, 2};
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;

    float got[4] = {};
    ggml_backend_tensor_get(y, got, 0, sizeof(got));
    /* ggml stores each 2-column matrix column-contiguously. */
    const float want[4] = {4, 13, 13, 31};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-5f) {
            std::fprintf(stderr, "mismatch i=%d got=%g want=%g\n", i, got[i], want[i]);
            return 6;
        }
    }
    std::printf("GGML_A64FX_MATMUL PASS y=[%g,%g,%g,%g]\n",
                got[0], got[1], got[2], got[3]);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
