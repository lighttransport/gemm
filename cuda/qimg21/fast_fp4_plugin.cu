/* CUTLASS 4.6 block-scaled NVFP4 GEMM for the fast Qwen-Image 2.1 denoiser
 * (sm_120a, the warp-specialized mainloop of CUTLASS example 79a).
 *
 *   D[m, n] = alpha * sum_k A[m, k] B[n, k] + D[m, n]
 *
 * A (activations) and B (weights) are E2M1 packed two per byte, K contiguous;
 * their E4M3 scale factors (one per 16 K) use CUTLASS's interleaved layout
 * (q21f_fp4_sf_offset). alpha is a device scalar (activation scale times weight
 * scale). D is BF16 row-major and already holds the low-rank branch, which the
 * epilogue reads as C (beta = 1) before writing D. */
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32, AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                        CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static void *q21f_workspace;
static size_t q21f_workspace_bytes;

extern "C" int q21f_fp4_gemm(void *d, const void *a, const void *sfa, const void *b, const void *sfb,
                             const float *alpha, int m, int n, int k, cudaStream_t stream) {
    if (!d || !a || !sfa || !b || !sfb || !alpha || m <= 0 || n <= 0 || k <= 0 || k % 64 || n % 8)
        return cudaErrorInvalidValue;
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<const typename ElementA::DataType *>(a), stride_a,
         static_cast<const typename ElementB::DataType *>(b), stride_b,
         static_cast<const typename ElementA::ScaleFactorType *>(sfa), layout_sfa,
         static_cast<const typename ElementB::ScaleFactorType *>(sfb), layout_sfb},
        {{1.0f, 1.0f}, static_cast<const ElementC *>(d), stride_c, static_cast<ElementD *>(d), stride_d}};
    args.epilogue.thread.alpha_ptr = alpha;
    args.epilogue.thread.beta = 1.0f;
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    size_t need = Gemm::get_workspace_size(args);
    if (need > q21f_workspace_bytes) {
        if (q21f_workspace) cudaFree(q21f_workspace);
        q21f_workspace = nullptr;
        q21f_workspace_bytes = 0;
        cudaError_t e = cudaMalloc(&q21f_workspace, need);
        if (e != cudaSuccess) return e;
        q21f_workspace_bytes = need;
    }
    if (gemm.initialize(args, q21f_workspace, stream) != cutlass::Status::kSuccess) return cudaErrorUnknown;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    return cudaGetLastError();
}

/* Byte offset of the scale factor for (row, K group of 16) in CUTLASS's layout
 * for an [rows, k] operand; used to check the runner's closed form. */
extern "C" long q21f_fp4_sf_offset(int rows, int k, int row, int group) {
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(rows, 128, k, 1));
    return (long)layout(make_coord(row, group * 16, 0));
}

/* Bytes needed for one operand's scale factors (rows and k padded to the atom). */
extern "C" long q21f_fp4_sf_bytes(int rows, int k) {
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(rows, 128, k, 1));
    return (long)size(filter_zeros(layout));
}
