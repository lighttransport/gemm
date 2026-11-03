#include "trmm_debug_common.hpp"

namespace {

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count                               = 0;
    const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
    const std::vector<debug_common::Dim3> dims = {
        {4096, 4097, 0},
        {4097, 4351, 0},
        {4351, 4096, 0}
    };
    const std::vector<debug_common::LdExtra> ld_cases = {
        {  0,   0,   0},
        {  1,   0,   0},
        {  0,   1,   0},
        {  0,   0,   1},
        {255, 255, 255}
    };
    for (auto side : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
        for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
            for (auto trans : debug_common::op_list<TC>(true))
                for (auto diag : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                    for (const auto dim : dims)
                        for (const auto ld : ld_cases)
                            for (const auto alpha : debug_common::alpha_cases<TC>()) {
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
        bool ok                                    = true;
        const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
        const std::vector<debug_common::Dim3> dims = {
            {4096, 4097, 0},
            {4097, 4351, 0},
            {4351, 4096, 0}
        };
        const std::vector<debug_common::LdExtra> ld_cases = {
            {  0,   0,   0},
            {  1,   0,   0},
            {  0,   1,   0},
            {  0,   0,   1},
            {255, 255, 255}
        };
        for (auto side : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
            for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
                for (auto trans : debug_common::op_list<TC>(true))
                    for (auto diag : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT})
                        for (const auto dim : dims)
                            for (const auto ld : ld_cases)
                                for (const auto alpha : debug_common::alpha_cases<TC>()) {
                                    ok &= trmm_debug::run_case<TA, TB, TC, backend>(
                                        ctx, progress, "test_trmm_params",
                                        dim, side, uplo, trans, diag, ld, alpha, num_moduli);
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
