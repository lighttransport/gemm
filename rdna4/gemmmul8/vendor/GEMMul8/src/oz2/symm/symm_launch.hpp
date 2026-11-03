#pragma once
#include "symm_core.hpp"

namespace gemmul8::oz2::symm {

template <typename TSym, typename TFull, typename TC, Backend BACKEND>
inline std::vector<double> symm_launch(
    common::Handle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    size_t rowsC, size_t colsC,
    const TC *alpha,
    const TSym *const Sym, size_t ldSym,
    const TFull *const Full, size_t ldFull,
    const TC *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work, void *const workSym, void *const workFull,
    bool enable_skip_scalSym, bool enable_skip_scalFull,
    bool skip_scalSym, bool skip_scalFull,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return symm_core<TSym, TFull, TC, BACKEND, 2U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 3: return symm_core<TSym, TFull, TC, BACKEND, 3U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 4: return symm_core<TSym, TFull, TC, BACKEND, 4U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 5: return symm_core<TSym, TFull, TC, BACKEND, 5U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 6: return symm_core<TSym, TFull, TC, BACKEND, 6U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 7: return symm_core<TSym, TFull, TC, BACKEND, 7U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 8: return symm_core<TSym, TFull, TC, BACKEND, 8U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 9: return symm_core<TSym, TFull, TC, BACKEND, 9U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 10: return symm_core<TSym, TFull, TC, BACKEND, 10U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 11: return symm_core<TSym, TFull, TC, BACKEND, 11U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 12: return symm_core<TSym, TFull, TC, BACKEND, 12U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 13: return symm_core<TSym, TFull, TC, BACKEND, 13U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 14: return symm_core<TSym, TFull, TC, BACKEND, 14U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 15: return symm_core<TSym, TFull, TC, BACKEND, 15U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 16: return symm_core<TSym, TFull, TC, BACKEND, 16U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 17: return symm_core<TSym, TFull, TC, BACKEND, 17U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 18: return symm_core<TSym, TFull, TC, BACKEND, 18U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 19: return symm_core<TSym, TFull, TC, BACKEND, 19U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    case 20: return symm_core<TSym, TFull, TC, BACKEND, 20U>(handle, side, uplo, rowsC, colsC, alpha, Sym, ldSym, Full, ldFull, beta, C, ldc, fastmode, work, workSym, workFull, enable_skip_scalSym, enable_skip_scalFull, skip_scalSym, skip_scalFull, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::symm
