#include "gemm_debug_common.hpp"

namespace {

struct Runner {
    gemm_debug::Context &ctx;
    gemm_debug::Progress &progress;

    template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
    bool operator()() {
        bool ok         = true;
        const auto scal = gemm_debug::default_scalar_case<TC>();
        const gemm_debug::Dim3 dim{4097, 4097, 4097};

        for (auto transa : gemm_debug::op_list<TC>(false)) {
            for (auto transb : gemm_debug::op_list<TC>(false)) {
                for (unsigned num_moduli = gemm_debug::num_moduli_min_for_range<TC>();
                     num_moduli <= gemm_debug::num_moduli_max_for_range<TC>();
                     ++num_moduli) {
                    ok &= gemm_debug::run_case<TA, TB, TC, backend>(
                        ctx,
                        progress,
                        "test_gemm_num_moduli",
                        dim,
                        transa,
                        transb,
                        {0, 0, 0},
                        scal,
                        num_moduli);
                }
            }
        }
        return ok;
    }
};

} // namespace

inline constexpr size_t total_tests() {
    constexpr size_t tc_fp32_type_combos_per_backend = 8; // real 4 + complex 4
    constexpr size_t tc_fp64_type_combos_per_backend = 8; // real 4 + complex 4

    constexpr size_t real_ops    = 2 * 2;
    constexpr size_t complex_ops = 3 * 3;

    constexpr size_t fp32_num_moduli = 13 - 8 + 1;
    constexpr size_t fp64_num_moduli = 20 - 12 + 1;

    constexpr size_t int8 =
        ((4 * real_ops + 4 * complex_ops) * fp32_num_moduli +
         (4 * real_ops + 4 * complex_ops) * fp64_num_moduli) *
        gemm_debug::evaluations_per_case<gemmul8::Backend::INT8>();

    constexpr size_t fp8 =
        ((4 * real_ops + 4 * complex_ops) * fp32_num_moduli +
         (4 * real_ops + 4 * complex_ops) * fp64_num_moduli) *
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
