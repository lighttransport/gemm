#pragma once
#include "../common/common.hpp"

namespace gemmul8::oz2::core {

template <Func FUNC,
          common::MatStruct STRUCT_A,
          common::MatStruct STRUCT_B,
          cublasFillMode_t UPLO_C>
__host__ __device__ constexpr common::MatMulKind matmul_kind() {
    if constexpr (FUNC == Func::trmm) {
        static_assert(common::is_triangular<STRUCT_A> || common::is_triangular<STRUCT_B>,
                      "trmm requires one triangular operand.");
        static_assert(!(common::is_triangular<STRUCT_A> && common::is_triangular<STRUCT_B>),
                      "trmm must not have two triangular operands. Use trtrmm.");

        if constexpr (common::is_triangular<STRUCT_A>) {
            return common::MatMulKind::TrmmLeft;
        } else {
            return common::MatMulKind::TrmmRight;
        }

    } else if constexpr (FUNC == Func::trtrmm) {
        static_assert(common::is_triangular<STRUCT_A> && common::is_triangular<STRUCT_B>,
                      "trtrmm requires two triangular operands.");
        return common::MatMulKind::Trtrmm;

    } else if constexpr (FUNC == Func::syrkx || FUNC == Func::herkx) {
        static_assert(UPLO_C == CUBLAS_FILL_MODE_UPPER ||
                          UPLO_C == CUBLAS_FILL_MODE_LOWER,
                      "syrkx/herkx requires UPLO_C = UPPER or LOWER.");
        return common::MatMulKind::ATxB;

    } else {
        return common::MatMulKind::Gemm;
    }
}

}
