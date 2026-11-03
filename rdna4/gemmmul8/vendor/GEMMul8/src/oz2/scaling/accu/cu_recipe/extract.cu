#include "../extract.hpp"

namespace gemmul8::scaling::accu {

namespace {

#if !defined(GEMMUL8_INST_TYPE)
    #error "GEMMUL8_INST_TYPE is not defined"
#endif
using T = GEMMUL8_INST_TYPE;

#if !defined(GEMMUL8_INST_BACKEND)
    #error "GEMMUL8_INST_BACKEND is not defined"
#endif
inline constexpr Backend BE = Backend::GEMMUL8_INST_BACKEND;

#if !defined(GEMMUL8_INST_FILLMODE)
    #error "GEMMUL8_INST_FILLMODE is not defined"
#endif
inline constexpr cublasFillMode_t UPLO = GEMMUL8_INST_FILLMODE;

#if !defined(GEMMUL8_INST_DIAG)
    #error "GEMMUL8_INST_DIAG is not defined"
#endif
inline constexpr cublasDiagType_t DIAG = GEMMUL8_INST_DIAG;

} // namespace

static_assert(UPLO != CUBLAS_FILL_MODE_FULL || DIAG == CUBLAS_DIAG_NON_UNIT,
              "When UPLO is CUBLAS_FILL_MODE_FULL, DIAG must be CUBLAS_DIAG_NON_UNIT.");

template void extract<T, BE, UPLO, DIAG>(
    const cudaStream_t,
    const cublasOperation_t, const cublasSideMode_t,
    const unsigned, const unsigned,
    const T *const, const size_t,
    common::matptr_t<common::low_t<BE>, common::isComplex<T>> &,
    const size_t, const size_t,
    int16_t *const);

} // namespace gemmul8::scaling::accu
