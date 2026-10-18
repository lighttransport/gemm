/* Optional CUTLASS SM120 kernels for the verified cublas_fp4_gemm runner.
 * CUTLASS stays an external build dependency; input packing and timing are
 * shared with the cuBLAS benchmark. All kernels use FP32 accumulation. */
#include <cuda_runtime.h>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/version.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifdef FP4_STATIC_SCHEDULER
/* SM120's cooperative kernel already supports the static scheduler interface.
 * Supply the selector explicitly to compare CLC with a fixed persistent grid. */
namespace cutlass::gemm::kernel::detail {
template<class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<StaticPersistentScheduler,arch::Sm120,Tile,Cluster,Stages> {
    using Scheduler = StaticPersistentTileScheduler100;
};
}
#endif

using namespace cute;

#ifndef FP4_OUTPUT_TYPE
#define FP4_OUTPUT_TYPE 14
#endif
using Output = std::conditional_t<FP4_OUTPUT_TYPE == 0,float,
    std::conditional_t<FP4_OUTPUT_TYPE == 2,cutlass::half_t,cutlass::bfloat16_t>>;
#ifdef FP4_NO_SOURCE
using Source = void;
#else
using Source = Output;
#endif

struct kernel_state {
    virtual int run(cudaStream_t stream) = 0;
    virtual ~kernel_state() = default;
};

extern "C" const char *fp4_kernel_build() {
#define STRINGIFY_(X) #X
#define STRINGIFY(X) STRINGIFY_(X)
    return "CUTLASS=" STRINGIFY(CUTLASS_MAJOR) "." STRINGIFY(CUTLASS_MINOR) "." STRINGIFY(CUTLASS_PATCH)
        " output_type=" STRINGIFY(FP4_OUTPUT_TYPE)
#ifdef NDEBUG
        " release=1"
#else
        " release=0"
#endif
#ifdef FP4_STATIC_SCHEDULER
        " scheduler=static"
#else
        " scheduler=CLC"
#endif
#ifdef FP4_NO_SOURCE
        " source=void"
#else
        " source=output_type"
#endif
        ;
#undef STRINGIFY
#undef STRINGIFY_
}

template<int TM, int TN, int TK, int STAGES, bool PINGPONG = false>
struct kernel_impl : kernel_state {
    static_assert(TK >= 128,"This CUTLASS register pipeline requires at least two K64 fragments.");
    using Tile = Shape<Int<TM>,Int<TN>,Int<TK>>;
    using Cluster = Shape<_1,_1,_1>;
    using Pair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using Row = cutlass::layout::RowMajor;
    using Col = cutlass::layout::ColumnMajor;
#ifdef FP4_EPI_M
    using EpiTile = Shape<Int<FP4_EPI_M>,Int<FP4_EPI_N>>;
#else
    using EpiTile = cutlass::epilogue::collective::EpilogueTileAuto;
#endif
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch,Op,Tile,Cluster,EpiTile,
        float,float,Source,Row,128/cutlass::sizeof_bits<Output>::value,
        Output,Row,128/cutlass::sizeof_bits<Output>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using StageCount = std::conditional_t<STAGES == 0,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        cutlass::gemm::collective::StageCount<STAGES>>;
    using Schedule = std::conditional_t<PINGPONG,cutlass::gemm::KernelTmaWarpSpecializedPingpong,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch,Op,Pair,Row,32,Pair,Col,32,float,Tile,Cluster,StageCount,Schedule>::CollectiveOp;
#ifdef FP4_STATIC_SCHEDULER
    using Scheduler = cutlass::gemm::StaticPersistentScheduler;
#else
    using Scheduler = void;
#endif
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,Scheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    Gemm gemm;
    void *workspace = nullptr;
    bool valid = false;

    kernel_impl(int raster, int swizzle, int m, int n, int k,
                void *a, void *b, void *sa, void *sb, void *out) {
        using S = typename Mainloop::Sm1xxBlkScaledConfig;
        auto shape = make_shape(m,n,k,1);
        auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m,k,1});
        auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n,k,1});
        auto dc = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m,n,1});
        auto dd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m,n,1});
        typename Gemm::Arguments args {
            cutlass::gemm::GemmUniversalMode::kGemm,shape,
            {static_cast<cutlass::float_e2m1_t *>(a),da,
             static_cast<cutlass::float_e2m1_t *>(b),db,
             static_cast<cutlass::float_ue4m3_t *>(sa),S::tile_atom_to_shape_SFA(shape),
             static_cast<cutlass::float_ue4m3_t *>(sb),S::tile_atom_to_shape_SFB(shape)},
            {{1.0f,0.0f},static_cast<Source *>(out),dc,static_cast<Output *>(out),dd}
        };
        args.scheduler.raster_order = static_cast<decltype(args.scheduler.raster_order)>(raster);
        args.scheduler.max_swizzle_size = swizzle;
        cudaDeviceGetAttribute(&args.hw_info.sm_count,cudaDevAttrMultiProcessorCount,0);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return;
        size_t bytes = Gemm::get_workspace_size(args);
        if (bytes && cudaMalloc(&workspace,bytes) != cudaSuccess) return;
        valid = gemm.initialize(args,workspace) == cutlass::Status::kSuccess;
    }
    int run(cudaStream_t stream) override { return static_cast<int>(gemm.run(stream)); }
    ~kernel_impl() override { if (workspace) cudaFree(workspace); }
};

/* Auto stages and explicit overrides test the shared-memory/pipeline tradeoff. */
#if defined(FP4_WINNER_ONLY)
#define FP4_KERNELS(X) X(0,256,128,128,0,false)
#elif !defined(FP4_EXTRA_TILES)
#define FP4_KERNELS(X) \
    X(0,128,128,128,0,false) \
    X(1,128,128,256,0,false) \
    X(2,128,256,128,0,false) \
    X(3,128,64,128,0,false) \
    X(4,128,64,256,0,false) \
    X(5,256,64,128,0,false) \
    X(6,256,128,128,0,false) \
    X(7,128,128,128,3,false) \
    X(8,128,128,128,4,false) \
    X(9,128,128,128,2,false) \
    X(10,128,256,128,2,false) \
    X(11,128,128,128,0,true) \
    X(12,128,128,256,0,true) \
    X(13,128,256,128,0,true)
#else
#define FP4_KERNELS(X) \
    X(0,128,128,256,0,false) \
    X(1,128,128,128,0,false) \
    X(2,128,256,128,0,false) \
    X(3,128,256,128,2,false) \
    X(4,128,128,128,3,false) \
    X(5,256,128,128,0,false)
#endif

extern "C" int fp4_kernel_count() {
#if defined(FP4_WINNER_ONLY)
    return 1;
#elif defined(FP4_EXTRA_TILES)
    return 6;
#else
    return 14;
#endif
}
extern "C" const char *fp4_kernel_name(int index) {
    switch (index) {
#define NAME(I,M,N,K,S,P) case I: return #M "x" #N "x" #K " stages=" #S " pingpong=" #P;
        FP4_KERNELS(NAME)
#undef NAME
    }
    return "invalid";
}
extern "C" void *fp4_kernel_create(int index, int raster, int swizzle, int m, int n, int k,
                                   void *a, void *b, void *sa, void *sb, void *out, int type) {
    if (type != FP4_OUTPUT_TYPE) return nullptr;
    switch (index) {
#define CREATE(I,M,N,K,S,P) case I: { \
        auto *state = new kernel_impl<M,N,K,S,P>(raster,swizzle,m,n,k,a,b,sa,sb,out); \
        if (state->valid) return state; delete state; return nullptr; }
        FP4_KERNELS(CREATE)
#undef CREATE
    }
    return nullptr;
}
extern "C" int fp4_kernel_run(void *state) { return static_cast<kernel_state *>(state)->run(nullptr); }
extern "C" int fp4_kernel_run_stream(void *state, void *stream) {
    return static_cast<kernel_state *>(state)->run(static_cast<cudaStream_t>(stream));
}
extern "C" void fp4_kernel_destroy(void *state) { delete static_cast<kernel_state *>(state); }
