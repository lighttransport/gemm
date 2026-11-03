#include "rankk_debug_common.hpp"

namespace {

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count = 0;

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

        for (unsigned num_moduli = debug_common::num_moduli_min_for_range<TC>();
             num_moduli <= debug_common::num_moduli_max_for_range<TC>();
             ++num_moduli) {
            ok &= rankk_debug::run_case_double_input<
                gemmul8::Func::her2k, TA, TB, TC, backend>(
                ctx, progress, "test_her2k_num_moduli",
                debug_common::Dim3{0, 4097, 4097}, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
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
