#include "gemm_debug_common.hpp"

namespace {

struct Runner {
    gemm_debug::Context &ctx;
    gemm_debug::Progress &progress;

    template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
    bool operator()() {
        bool ok                   = true;
        const auto scal           = gemm_debug::default_scalar_case<TC>();
        const unsigned num_moduli = gemm_debug::default_num_moduli<TC>();

        const std::vector<gemm_debug::LdExtra> ld_cases = {
            {  0,   0,   0},
            {  1,   0,   0},
            {  0,   1,   0},
            {  0,   0,   1},
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {  1, 255,  17},
            {255,   1, 255},
        };

        for (auto transa : gemm_debug::op_list<TC>(false)) {
            for (auto transb : gemm_debug::op_list<TC>(false)) {
                for (size_t n = 4096; n <= 4351; ++n) {
                    for (const auto ld_extra : ld_cases) {
                        ok &= gemm_debug::run_case<TA, TB, TC, backend>(
                            ctx,
                            progress,
                            "test_gemm_padding",
                            {n, n, n},
                            transa,
                            transb,
                            ld_extra,
                            scal,
                            num_moduli);
                    }
                }
            }
        }
        return ok;
    }
};

} // namespace

inline constexpr size_t total_tests() {
    constexpr size_t real_type_combos_per_backend    = 1; // d,d,d
    constexpr size_t complex_type_combos_per_backend = 1; // z,z,z

    constexpr size_t real_ops    = 2 * 2;
    constexpr size_t complex_ops = 3 * 3;
    constexpr size_t sizes       = 256; // 4096..4351
    constexpr size_t ld_cases    = 9;

    constexpr size_t int8 =
        (real_type_combos_per_backend * real_ops +
         complex_type_combos_per_backend * complex_ops) *
        sizes * ld_cases *
        gemm_debug::evaluations_per_case<gemmul8::Backend::INT8>();

    constexpr size_t fp8 =
        (real_type_combos_per_backend * real_ops +
         complex_type_combos_per_backend * complex_ops) *
        sizes * ld_cases *
        gemm_debug::evaluations_per_case<gemmul8::Backend::FP8>();

    return int8 + fp8;
}

int main() {
    bool ok = true;
    {
        gemm_debug::Context ctx;
        gemm_debug::Progress progress(total_tests());
        progress.print();

        ok = gemm_debug::run_representative_type_backend_cases(Runner{ctx, progress});

        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
