#include "trsm_debug_common.hpp"

namespace {

template <typename TA, typename TB, gemmul8::Backend backend>
size_t count_cases() {
    size_t count                                      = 0;
    const auto alpha                                  = debug_common::default_alpha_case<TB>();
    const unsigned num_moduli                         = debug_common::default_num_moduli<TB>();
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
    template <typename TA, typename TB, gemmul8::Backend backend> bool operator()() {
        count += count_cases<TA, TB, backend>();
        return true;
    }
};
struct Runner {
    debug_common::Context &ctx;
    debug_common::Progress &progress;
    template <typename TA, typename TB, gemmul8::Backend backend> bool operator()() {
        bool ok                                           = true;
        const auto alpha                                  = debug_common::default_alpha_case<TB>();
        const unsigned num_moduli                         = debug_common::default_num_moduli<TB>();
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
                ok &= trsm_debug::run_case<TA, TB, backend>(
                    ctx, progress, "test_trsm_padding",
                    debug_common::Dim3{n, n, 0}, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ld, alpha, num_moduli);
            }
        return ok;
    }
};

} // namespace

int main() {
    CountRunner counter;
    debug_common::run_representative_trsm_type_backend_cases(counter);
    bool ok = true;
    {
        debug_common::Context ctx;
        debug_common::Progress progress(counter.count);
        progress.print();
        ok = debug_common::run_representative_trsm_type_backend_cases(Runner{ctx, progress});
        progress.finish();
    }
    CHECK_CUDA(cudaDeviceReset());
    return ok ? 0 : 1;
}
