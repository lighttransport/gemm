#include "../trsm_impl.hpp"

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

#if !defined(GEMMUL8_INST_BACKEND)
    #error "GEMMUL8_INST_BACKEND is not defined"
#endif
inline constexpr Backend BE = Backend::GEMMUL8_INST_BACKEND;

} // namespace

template std::vector<double> trsmLt<TA, BE, TB>(
    cublasLtHandle_t,
    cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t,
    size_t, size_t,
    const TB *,
    const TA *const, size_t,
    TB *const, size_t,
    int, bool,
    void *const,
    cudaStream_t);

} // namespace gemmul8
