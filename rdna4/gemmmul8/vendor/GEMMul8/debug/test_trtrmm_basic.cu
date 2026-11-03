#include "trtrmm_debug_common.hpp"

namespace {

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count                    = 0;
    const auto scal                 = debug_common::default_scalar_case<TC>();
    const unsigned num_moduli       = debug_common::default_num_moduli<TC>();
    const std::vector<size_t> sizes = {4096, 4097, 4351};
    for (auto uplo_A : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
        for (auto uplo_B : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
            for (auto trans_A : debug_common::op_list<TC>(false))
                for (auto trans_B : debug_common::op_list<TC>(false))
                    for (auto diag_A : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                        for (auto diag_B : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                            for (auto n : sizes) {
                                count += debug_common::evaluations_per_case<backend>();
                            }
    return count;
}

struct CountRunner {
    size_t count = 0;
    template <typename TA, typename TB, typename TC, gemmul8::Backend backend> bool operator()() {
        count += count_cases<TA, TB, TC, backend>();
        return true;
    }
};
struct Runner {
    debug_common::Context &ctx;
    debug_common::Progress &progress;
    template <typename TA, typename TB, typename TC, gemmul8::Backend backend> bool operator()() {
        bool ok                         = true;
        const auto scal                 = debug_common::default_scalar_case<TC>();
        const unsigned num_moduli       = debug_common::default_num_moduli<TC>();
        const std::vector<size_t> sizes = {4096, 4097, 4351};
        for (auto uplo_A : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
            for (auto uplo_B : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
                for (auto trans_A : debug_common::op_list<TC>(false))
                    for (auto trans_B : debug_common::op_list<TC>(false))
                        for (auto diag_A : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                            for (auto diag_B : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                                for (auto n : sizes) {
                                    ok &= trtrmm_debug::run_case<TA, TB, TC, backend>(
                                        ctx, progress, "test_trtrmm_basic",
                                        debug_common::Dim3{0, n, 0}, uplo_A, uplo_B, trans_A, trans_B, diag_A, diag_B,
                                        debug_common::LdExtra{0, 0, 0}, scal, num_moduli);
                                }
        return ok;
    }
};

} // namespace

int main() {
    CountRunner counter;
    debug_common::run_all_type_backend_cases(counter);
    bool ok = true;
    {
        debug_common::Context ctx;
        debug_common::Progress progress(counter.count);
        progress.print();
        ok = debug_common::run_all_type_backend_cases(Runner{ctx, progress});
        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
