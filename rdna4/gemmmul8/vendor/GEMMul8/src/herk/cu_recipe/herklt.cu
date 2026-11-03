#include "../herk_impl.hpp"

namespace gemmul8 {

namespace {

#if !defined(GEMMUL8_INST_TYPE_A)
    #error "GEMMUL8_INST_TYPE_A is not defined"
#endif
using TA = GEMMUL8_INST_TYPE_A;

#if !defined(GEMMUL8_INST_TYPE_C)
    #error "GEMMUL8_INST_TYPE_C is not defined"
#endif
using TC = GEMMUL8_INST_TYPE_C;

#if !defined(GEMMUL8_INST_BACKEND)
    #error "GEMMUL8_INST_BACKEND is not defined"
#endif
inline constexpr Backend BE = Backend::GEMMUL8_INST_BACKEND;

} // namespace

template std::vector<double> herkLt<TA, BE, TC>(
    cublasLtHandle_t,
    cublasFillMode_t, cublasOperation_t,
    size_t, size_t,
    const std::conditional_t<std::is_same_v<TC, cuDoubleComplex>, double, float> *,
    const TA *const, size_t,
    const std::conditional_t<std::is_same_v<TC, cuDoubleComplex>, double, float> *, TC *const, size_t,
    int, bool,
    void *const,
    void *const,
    bool, bool,
    cudaStream_t);

} // namespace gemmul8
