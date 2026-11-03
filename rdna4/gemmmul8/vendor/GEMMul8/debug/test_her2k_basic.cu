#include "rankk_debug_common.hpp"

namespace {

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count = 0;

    const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
    const std::vector<debug_common::Dim3> dims = {
        {0, 4096, 4096},
        {0, 4097, 4097},
        {0, 4351, 4351}
    };
    for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
        for (auto trans : debug_common::her_op_list<TC>())
            for (const auto dim : dims) {
                count += debug_common::evaluations_per_case<backend>();
            }

    return count;
}

struct CountRunner {
    size_t count = 0;
    template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
    bool operator()() {
        count += count_cases<TA, TB, TC, backend>();
        return true;
    }
};

struct Runner {
    debug_common::Context &ctx;
    debug_common::Progress &progress;
    template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
    bool operator()() {
        bool ok = true;

        const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
        const std::vector<debug_common::Dim3> dims = {
            {0, 4096, 4096},
            {0, 4097, 4097},
            {0, 4351, 4351}
        };
        for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
            for (auto trans : debug_common::her_op_list<TC>())
                for (const auto dim : dims) {
                    ok &= rankk_debug::run_case_double_input<
                        gemmul8::Func::her2k, TA, TB, TC, backend>(
                        ctx, progress, "test_her2k_basic",
                        dim, uplo, trans,
                        debug_common::LdExtra{0, 0, 0}, debug_common::default_scalar_case<TC>(), num_moduli);
                }

        return ok;
    }
};

} // namespace

int main() {
    CountRunner counter;
    debug_common::run_all_complex_type_backend_cases(counter);
    bool ok = true;
    {
        debug_common::Context ctx;
        debug_common::Progress progress(counter.count);
        progress.print();
        ok = debug_common::run_all_complex_type_backend_cases(Runner{ctx, progress});
        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
