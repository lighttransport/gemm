#include "rankk_debug_common.hpp"

namespace {

template <typename TA, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count = 0;

    const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
    const std::vector<debug_common::Dim3> dims = {
        {0, 4096, 4097},
        {0, 4097, 4351},
        {0, 4351, 4096}
    };
    const std::vector<debug_common::LdExtra> ld_cases = {
        {  0,   0,   0},
        {  1,   0,   0},
        {  0,   1,   0},
        {  0,   0,   1},
        {255, 255, 255}
    };
    for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
        for (auto trans : debug_common::rk_op_list<TC>(false))
            for (const auto dim : dims)
                for (const auto ld : ld_cases) {
                    if constexpr (rankk_debug::is_hermitian_func<gemmul8::Func::syrk>()) {
                        for (size_t scalar_idx = 0; scalar_idx < debug_common::real_scalar_cases<TC>().size(); ++scalar_idx)
                            count += debug_common::evaluations_per_case<backend>();
                    } else {
                        for (size_t scalar_idx = 0; scalar_idx < debug_common::scalar_cases<TC>().size(); ++scalar_idx)
                            count += debug_common::evaluations_per_case<backend>();
                    }
                }

    return count;
}

struct CountRunner {
    size_t count = 0;
    template <typename TA, typename TC, gemmul8::Backend backend>
    bool operator()() {
        count += count_cases<TA, TC, backend>();
        return true;
    }
};

struct Runner {
    debug_common::Context &ctx;
    debug_common::Progress &progress;
    template <typename TA, typename TC, gemmul8::Backend backend>
    bool operator()() {
        bool ok = true;

        const unsigned num_moduli                  = debug_common::default_num_moduli<TC>();
        const std::vector<debug_common::Dim3> dims = {
            {0, 4096, 4097},
            {0, 4097, 4351},
            {0, 4351, 4096}
        };
        const std::vector<debug_common::LdExtra> ld_cases = {
            {  0,   0,   0},
            {  1,   0,   0},
            {  0,   1,   0},
            {  0,   0,   1},
            {255, 255, 255}
        };
        for (auto uplo : {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER})
            for (auto trans : debug_common::rk_op_list<TC>(false))
                for (const auto dim : dims)
                    for (const auto ld : ld_cases) {
                        if constexpr (rankk_debug::is_hermitian_func<gemmul8::Func::syrk>()) {
                            for (size_t scalar_idx = 0; scalar_idx < debug_common::real_scalar_cases<TC>().size(); ++scalar_idx)
                                ok &= rankk_debug::run_case_single<gemmul8::Func::syrk, TA, TC, backend>(
                                    ctx, progress, "test_syrk_params", dim, uplo, trans, ld, num_moduli, true, scalar_idx);
                        } else {
                            for (size_t scalar_idx = 0; scalar_idx < debug_common::scalar_cases<TC>().size(); ++scalar_idx)
                                ok &= rankk_debug::run_case_single<gemmul8::Func::syrk, TA, TC, backend>(
                                    ctx, progress, "test_syrk_params", dim, uplo, trans, ld, num_moduli, true, scalar_idx);
                        }
                    }

        return ok;
    }
};

} // namespace

int main() {
    CountRunner counter;
    debug_common::run_all_single_input_type_backend_cases(counter);
    bool ok = true;
    {
        debug_common::Context ctx;
        debug_common::Progress progress(counter.count);
        progress.print();
        ok = debug_common::run_all_single_input_type_backend_cases(Runner{ctx, progress});
        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
