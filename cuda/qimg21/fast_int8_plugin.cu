/* CUTLASS INT8 GEMM with a fused dequantizing epilogue for the fast
 * Qwen-Image 2.1 W8A8 path (Sm80 mma.sync s8, runs on sm_120):
 *
 *   y[m, n] = bf16(float(sum_k x[m, k] w[n, k]) * xs[m] * ws[n])
 *
 * x is [m, k] INT8 activations with per-token scales xs, w is [n, k] INT8
 * weights with per-row scales ws, y is BF16 with row stride ldy. The
 * epilogue (a CUTLASS Sm80 epilogue visitor tree) multiplies in the same order
 * as the runner's standalone dequant kernel, so both paths round identically;
 * this removes the INT32 accumulator round trip through device memory. */
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm_universal.h"
/* The visitor headers rely on the GEMM headers above being included first. */
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using namespace cute;

template <int TM, int TN, int TK, int WM, int WN, int Stages, int Swizzle>
struct q21f_i8_config {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementOut = cutlass::bfloat16_t;
    using ThreadblockShape = cutlass::gemm::GemmShape<TM, TN, TK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, TK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int AlignmentOut = 8;
    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<ThreadblockShape, WarpShape, ElementOut,
                                                               AlignmentOut, 1>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    /* xs[m]: one value per output row. */
    using TokenScale = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, float,
                                                                          Stride<_1, _0, int32_t>>;
    /* ws[n]: one value per output column. */
    using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, float,
                                                                           Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float,
                                                                cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, TokenScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementOut, float,
                                                                cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WeightScale>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementOut, cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<Store, EVT1>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, ElementB,
        cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, ElementOut, cutlass::layout::RowMajor,
        AlignmentOut, int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
        WarpShape, InstructionShape, EVTD, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
        Stages, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static int run(void *y, int ldy, const void *x, const float *xs, const void *w, const float *ws, int m, int n,
                   int k, cudaStream_t stream) {
        typename EVTD::Arguments callbacks{
            {{{}, {xs, 0.0f, {_1{}, _0{}, int32_t(m)}}, {}}, {ws, 0.0f, {_0{}, _1{}, int32_t(n)}}, {}},
            {static_cast<ElementOut *>(y), {int64_t(ldy), _1{}, int64_t(m) * ldy}}};
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, callbacks, x, w,
                                      nullptr, nullptr, int64_t(m) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
        if (Gemm::get_workspace_size(args)) return cudaErrorInvalidValue;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return cudaErrorUnknown;
        if (gemm.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
        return cudaGetLastError();
    }
};

/* Tile configurations for benchmarking; config -1 picks per shape. On the
 * RTX 5060 Ti, 128x256x64 reaches 121-145 TOPS for M >= 4096 on all Qwen block
 * shapes, and 128x128x64 is faster below M = 1024. */
extern "C" int q21f_i8_gemm(void *y, int ldy, const void *x, const float *xs, const void *w, const float *ws, int m,
                            int n, int k, int config, cudaStream_t stream) {
    if (!y || !x || !xs || !w || !ws || m <= 0 || n <= 0 || k <= 0 || k % 16 || n % 8 || ldy < n || ldy % 8)
        return cudaErrorInvalidValue;
    if (config < 0) config = m < 1024 ? 2 : 0;
    switch (config) {
    case 0: return q21f_i8_config<128, 256, 64, 64, 64, 3, 1>::run(y, ldy, x, xs, w, ws, m, n, k, stream);
    case 1: return q21f_i8_config<256, 128, 64, 64, 64, 3, 1>::run(y, ldy, x, xs, w, ws, m, n, k, stream);
    case 2: return q21f_i8_config<128, 128, 64, 64, 64, 4, 1>::run(y, ldy, x, xs, w, ws, m, n, k, stream);
    case 3: return q21f_i8_config<128, 256, 64, 64, 64, 3, 4>::run(y, ldy, x, xs, w, ws, m, n, k, stream);
    case 4: return q21f_i8_config<128, 128, 128, 64, 64, 3, 1>::run(y, ldy, x, xs, w, ws, m, n, k, stream);
    default: return cudaErrorInvalidValue;
    }
}
