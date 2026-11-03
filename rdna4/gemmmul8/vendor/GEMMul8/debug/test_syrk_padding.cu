#include "rankk_debug_common.hpp"

namespace {

template <typename TA, typename TC, gemmul8::Backend backend>
size_t count_cases() {
    size_t count = 0;

    const unsigned num_moduli                         = debug_common::default_num_moduli<TC>();
    const std::vector<debug_common::LdExtra> ld_cases = {
        {  0,   0,   0},
        {  1,   0,   0},
        {  0,   1,   0},
        {  0,   0,   1},
        {255,   0,   0},
        {  0, 255,   0},
        {  0,   0, 255},
        {  1, 255,  17},
        {255,   1, 255}
    };
    for (size_t n = 4096; n <= 4351; ++n)
        for (const auto ld : ld_cases) {
            count += debug_common::evaluations_per_case<backend>();
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

        const unsigned num_moduli                         = debug_common::default_num_moduli<TC>();
        const std::vector<debug_common::LdExtra> ld_cases = {
            {  0,   0,   0},
            {  1,   0,   0},
            {  0,   1,   0},
            {  0,   0,   1},
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {  1, 255,  17},
            {255,   1, 255}
        };
        for (size_t n = 4096; n <= 4351; ++n)
            for (const auto ld : ld_cases) {
                ok &= rankk_debug::run_case_single<gemmul8::Func::syrk, TA, TC, backend>(
                    ctx, progress, "test_syrk_padding",
                    debug_common::Dim3{0, n, n}, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, ld, num_moduli);
            }

        return ok;
    }
};

} // namespace

int main() {
    CountRunner counter;
    debug_common::run_representative_single_input_type_backend_cases(counter);
    bool ok = true;
    {
        debug_common::Context ctx;
        debug_common::Progress progress(counter.count);
        progress.print();
        ok = debug_common::run_representative_single_input_type_backend_cases(Runner{ctx, progress});
        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
