#include "gemm_debug_common.hpp"

namespace {

struct Runner {
    gemm_debug::Context &ctx;
    gemm_debug::Progress &progress;

    template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
    bool operator()() {
        bool ok                   = true;
        const unsigned num_moduli = gemm_debug::default_num_moduli<TC>();

        const std::vector<gemm_debug::Dim3> dims = {
            {4096, 4097, 4351},
            {4097, 4351, 4096},
            {4351, 4096, 4097},
        };

        const std::vector<gemm_debug::LdExtra> ld_cases = {
            {  0,   0,   0},
            {  1,   0,   0},
            {  0,   1,   0},
            {  0,   0,   1},
            {255, 255, 255},
        };

        for (auto transa : gemm_debug::op_list<TC>(true)) {
            for (auto transb : gemm_debug::op_list<TC>(true)) {
                for (const auto dim : dims) {
                    for (const auto ld_extra : ld_cases) {
                        for (const auto scal : gemm_debug::scalar_cases<TC>()) {
                            ok &= gemm_debug::run_case<TA, TB, TC, backend>(
                                ctx,
                                progress,
                                "test_gemm_params",
                                dim,
                                transa,
                                transb,
                                ld_extra,
                                scal,
                                num_moduli);
                        }
                    }
                }
            }
        }
        return ok;
    }
};

} // namespace

inline constexpr size_t total_tests() {
    constexpr size_t real_type_combos_per_backend    = 8;
    constexpr size_t complex_type_combos_per_backend = 8;

    constexpr size_t real_ops    = 3 * 3;
    constexpr size_t complex_ops = 3 * 3;
    constexpr size_t dims        = 3;
    constexpr size_t ld_cases    = 5;
    constexpr size_t scalars     = 4;

    constexpr size_t int8 =
        (real_type_combos_per_backend * real_ops +
         complex_type_combos_per_backend * complex_ops) *
        dims * ld_cases * scalars *
        gemm_debug::evaluations_per_case<gemmul8::Backend::INT8>();

    constexpr size_t fp8 =
        (real_type_combos_per_backend * real_ops +
         complex_type_combos_per_backend * complex_ops) *
        dims * ld_cases * scalars *
        gemm_debug::evaluations_per_case<gemmul8::Backend::FP8>();

    return int8 + fp8;
}

int main() {
    bool ok = true;
    {
        gemm_debug::Context ctx;
        gemm_debug::Progress progress(total_tests());
        progress.print();

        ok = gemm_debug::run_all_type_backend_cases(Runner{ctx, progress});

        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
