#pragma once
#include "hemm_core.hpp"

namespace gemmul8::oz2::hemm {

template <typename THerm, typename TFull, typename TC, Backend BACKEND>
inline std::vector<double> hemm_launch(
    common::Handle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    size_t rowsC, size_t colsC,
    const TC *alpha,
    const THerm *const Herm, size_t ldHerm,
    const TFull *const Full, size_t ldFull,
    const TC *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work, void *const workHerm, void *const workFull,
    bool enable_skip_scalHerm, bool enable_skip_scalFull,
    bool skip_scalHerm, bool skip_scalFull,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return hemm_core<THerm, TFull, TC, BACKEND, 2U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 3: return hemm_core<THerm, TFull, TC, BACKEND, 3U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 4: return hemm_core<THerm, TFull, TC, BACKEND, 4U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 5: return hemm_core<THerm, TFull, TC, BACKEND, 5U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 6: return hemm_core<THerm, TFull, TC, BACKEND, 6U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 7: return hemm_core<THerm, TFull, TC, BACKEND, 7U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 8: return hemm_core<THerm, TFull, TC, BACKEND, 8U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 9: return hemm_core<THerm, TFull, TC, BACKEND, 9U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 10: return hemm_core<THerm, TFull, TC, BACKEND, 10U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 11: return hemm_core<THerm, TFull, TC, BACKEND, 11U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 12: return hemm_core<THerm, TFull, TC, BACKEND, 12U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 13: return hemm_core<THerm, TFull, TC, BACKEND, 13U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 14: return hemm_core<THerm, TFull, TC, BACKEND, 14U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 15: return hemm_core<THerm, TFull, TC, BACKEND, 15U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 16: return hemm_core<THerm, TFull, TC, BACKEND, 16U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 17: return hemm_core<THerm, TFull, TC, BACKEND, 17U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 18: return hemm_core<THerm, TFull, TC, BACKEND, 18U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 19: return hemm_core<THerm, TFull, TC, BACKEND, 19U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    case 20: return hemm_core<THerm, TFull, TC, BACKEND, 20U>(handle, side, uplo, rowsC, colsC, alpha, Herm, ldHerm, Full, ldFull, beta, C, ldc, fastmode, work, workHerm, workFull, enable_skip_scalHerm, enable_skip_scalFull, skip_scalHerm, skip_scalFull, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::hemm
