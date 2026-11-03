#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::fast {

inline constexpr size_t threads_fast             = 256;
inline constexpr unsigned threads_x_findmax_tile = common::TILE_DIM;
inline constexpr unsigned threads_y_findmax_tile = common::TILE_DIM;

inline constexpr unsigned rowwise_sft_col_tile         = 256U;
inline constexpr unsigned rowwise_sft_split_threshold  = 1024U;
inline constexpr unsigned threads_x_rowwise_sft_reduce = 32U;
inline constexpr unsigned threads_y_rowwise_sft_reduce = 8U;

} // namespace gemmul8::scaling::fast
