#include "../gemm_impl.hpp"

namespace gemmul8 {

namespace {

#if !defined(GEMMUL8_INST_TYPE_A)
    #error "GEMMUL8_INST_TYPE_A is not defined"
#endif
using TA = GEMMUL8_INST_TYPE_A;

#if !defined(GEMMUL8_INST_TYPE_B)
    #error "GEMMUL8_INST_TYPE_B is not defined"
#endif
using TB = GEMMUL8_INST_TYPE_B;

#if !defined(GEMMUL8_INST_TYPE_C)
    #error "GEMMUL8_INST_TYPE_C is not defined"
#endif
using TC = GEMMUL8_INST_TYPE_C;

#if !defined(GEMMUL8_INST_BACKEND)
    #error "GEMMUL8_INST_BACKEND is not defined"
#endif
inline constexpr Backend BE = Backend::GEMMUL8_INST_BACKEND;

} // namespace

static_assert(BE == Backend::INT8, "BE must be INT8.");

template std::vector<double> gemm<TA, BE, TB, TC>(
    cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    size_t, size_t, size_t,
    const TC *,
    const TA *const, size_t,
    const TB *const, size_t,
    const TC *,
    TC *const, size_t,
    int, bool,
    void *const,
    void *const,
    void *const,
    bool, bool, bool, bool);

} // namespace gemmul8
