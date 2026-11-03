#include "../extract_symm_hemm.hpp"

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

} // namespace

static_assert(common::isComplex<T>, "This instantiation requires complex T.");
static_assert(UPLO == CUBLAS_FILL_MODE_UPPER || UPLO == CUBLAS_FILL_MODE_LOWER,
              "UPLO must be CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER.");

template void extract_hemm<T, BE, UPLO, false>(
    const cudaStream_t,
    const unsigned,
    const T *const, const size_t,
    common::matptr_t<common::low_t<BE>, common::isComplex<T>> &,
    const size_t, const size_t,
    int16_t *const);

template void extract_hemm<T, BE, UPLO, true>(
    const cudaStream_t,
    const unsigned,
    const T *const, const size_t,
    common::matptr_t<common::low_t<BE>, common::isComplex<T>> &,
    const size_t, const size_t,
    int16_t *const);

} // namespace gemmul8::scaling::accu
