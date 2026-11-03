#include "symm_like_debug_common.hpp"

namespace {

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count = 0;

    const auto scal = debug_common::default_scalar_case<TC>();
    for (unsigned num_moduli = debug_common::num_moduli_min_for_range<TC>();
         num_moduli <= debug_common::num_moduli_max_for_range<TC>();
         ++num_moduli) {
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

        const auto scal = debug_common::default_scalar_case<TC>();
        for (unsigned num_moduli = debug_common::num_moduli_min_for_range<TC>();
             num_moduli <= debug_common::num_moduli_max_for_range<TC>();
             ++num_moduli) {
            ok &= symm_like_debug::run_case<gemmul8::Func::symm, TA, TB, TC, backend>(
                ctx, progress, "test_symm_num_moduli",
                debug_common::Dim3{4097, 4097, 0}, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
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
