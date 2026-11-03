#pragma once
#include "trmm_core.hpp"

namespace gemmul8::oz2::trmm {

template <typename TTri, typename TFull, typename TC, Backend BACKEND>
inline std::vector<double> trmm_launch(
    common::Handle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    size_t rowsC, size_t colsC,
    const TC *alpha,
    const TTri *const Tri, size_t ldTri,
    const TFull *const Full, size_t ldFull,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work, void *const workTri, void *const workFull,
    bool enable_skip_scalTri, bool enable_skip_scalFull,
    bool skip_scalTri, bool skip_scalFull,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return trmm_core<TTri, TFull, TC, BACKEND, 2U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 3: return trmm_core<TTri, TFull, TC, BACKEND, 3U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 4: return trmm_core<TTri, TFull, TC, BACKEND, 4U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 5: return trmm_core<TTri, TFull, TC, BACKEND, 5U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 6: return trmm_core<TTri, TFull, TC, BACKEND, 6U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 7: return trmm_core<TTri, TFull, TC, BACKEND, 7U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 8: return trmm_core<TTri, TFull, TC, BACKEND, 8U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 9: return trmm_core<TTri, TFull, TC, BACKEND, 9U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 10: return trmm_core<TTri, TFull, TC, BACKEND, 10U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 11: return trmm_core<TTri, TFull, TC, BACKEND, 11U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 12: return trmm_core<TTri, TFull, TC, BACKEND, 12U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 13: return trmm_core<TTri, TFull, TC, BACKEND, 13U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 14: return trmm_core<TTri, TFull, TC, BACKEND, 14U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 15: return trmm_core<TTri, TFull, TC, BACKEND, 15U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 16: return trmm_core<TTri, TFull, TC, BACKEND, 16U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 17: return trmm_core<TTri, TFull, TC, BACKEND, 17U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 18: return trmm_core<TTri, TFull, TC, BACKEND, 18U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 19: return trmm_core<TTri, TFull, TC, BACKEND, 19U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    case 20: return trmm_core<TTri, TFull, TC, BACKEND, 20U>(handle, side, uplo, trans, diag, rowsC, colsC, alpha, Tri, ldTri, Full, ldFull, C, ldc, fastmode, work, workTri, workFull, enable_skip_scalTri, enable_skip_scalFull, skip_scalTri, skip_scalFull, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::trmm
