#ifndef HV15_VAE_SPATIAL_TILING_HPP
#define HV15_VAE_SPATIAL_TILING_HPP
#include <algorithm>
#include <cmath>
#include <vector>
#include "core/tensor.hpp"

/* Match the pinned official VAE: fixed strides, clipped edge tiles, vertical
 * then horizontal linear blending, and keep only the non-overlap prefix.
 * Keep two rows on the host; the preceding row must retain its blended tail. */
template <typename Compute>
sd::Tensor<float> hv15_vae_spatial_tiles(const sd::Tensor<float>& input,
                                       bool decode, int tile_pixels, float overlap,
                                       Compute&& compute) {
    constexpr int scale = 16;
    if (input.dim()!=5 || tile_pixels<64 || tile_pixels%scale ||
        !std::isfinite(overlap) || overlap<0.f || overlap>=1.f) return {};
    const int latent_tile = tile_pixels/scale;
    const float latent_blend = latent_tile*overlap;
    if (latent_blend!=std::floor(latent_blend)) return {};
    const int input_tile = decode ? latent_tile : tile_pixels;
    const int output_tile = decode ? tile_pixels : latent_tile;
    const int stride = static_cast<int>(input_tile*(1.f-overlap));
    const int blend = static_cast<int>(output_tile*overlap);
    const int keep = output_tile-blend;
    if (stride<1 || keep<1) return {};
    if (input.shape()[0]<=input_tile && input.shape()[1]<=input_tile) return compute(input);
    const int64_t width = decode ? input.shape()[0]*scale : input.shape()[0]/scale;
    const int64_t height = decode ? input.shape()[1]*scale : input.shape()[1]/scale;
    if (!decode && (input.shape()[0]%scale || input.shape()[1]%scale)) return {};
    sd::Tensor<float> output;
    std::vector<sd::Tensor<float>> previous;
    int64_t output_y = 0;
    for (int64_t y=0; y<input.shape()[1]; y+=stride) {
        std::vector<sd::Tensor<float>> row;
        int64_t output_x = 0, row_height = 0;
        size_t column = 0;
        for (int64_t x=0; x<input.shape()[0]; x+=stride, ++column) {
            auto tile_input = sd::ops::slice(input,1,y,std::min(y+input_tile,input.shape()[1]));
            tile_input = sd::ops::slice(tile_input,0,x,std::min(x+input_tile,input.shape()[0]));
            auto tile = compute(tile_input);
            if (tile.empty() || tile.dim()!=5) return {};
            const int64_t tw=tile.shape()[0], th=tile.shape()[1];
            if (tw!=(decode ? tile_input.shape()[0]*scale : tile_input.shape()[0]/scale) ||
                th!=(decode ? tile_input.shape()[1]*scale : tile_input.shape()[1]/scale)) return {};
            for (float value : tile.values()) if (!std::isfinite(value)) return {};
            if (output.empty()) {
                auto shape=tile.shape(); shape[0]=width; shape[1]=height;
                output=sd::Tensor<float>::zeros(shape);
            }
            const int64_t planes=tile.numel()/(tw*th);
            if (output.numel()/(width*height)!=planes) return {};
            if (!previous.empty()) {
                if (column>=previous.size()) return {};
                const auto& above=previous[column];
                const int64_t aw=above.shape()[0], ah=above.shape()[1];
                if (aw!=tw || above.numel()/(aw*ah)!=planes) return {};
                const int64_t extent=std::min<int64_t>({blend,th,ah});
                for (int64_t p=0;p<planes;++p) for (int64_t iy=0;iy<extent;++iy) {
                    const float weight=static_cast<float>(iy)/extent;
                    for (int64_t ix=0;ix<tw;++ix) {
                        auto& value=tile.data()[p*tw*th+iy*tw+ix];
                        value=above.data()[p*aw*ah+(ah-extent+iy)*aw+ix]*(1.f-weight)+value*weight;
                    }
                }
            }
            if (!row.empty()) {
                const auto& left=row.back();
                const int64_t lw=left.shape()[0], lh=left.shape()[1];
                if (lh!=th || left.numel()/(lw*lh)!=planes) return {};
                const int64_t extent=std::min<int64_t>({blend,tw,lw});
                for (int64_t p=0;p<planes;++p) for (int64_t iy=0;iy<th;++iy)
                    for (int64_t ix=0;ix<extent;++ix) {
                        const float weight=static_cast<float>(ix)/extent;
                        auto& value=tile.data()[p*tw*th+iy*tw+ix];
                        value=left.data()[p*lw*lh+iy*lw+lw-extent+ix]*(1.f-weight)+value*weight;
                    }
            }
            const int64_t copy_w=std::min<int64_t>(keep,tw), copy_h=std::min<int64_t>(keep,th);
            if (output_x+copy_w>width || output_y+copy_h>height || (row_height && row_height!=copy_h)) return {};
            row_height=copy_h;
            for (int64_t p=0;p<planes;++p) for (int64_t iy=0;iy<copy_h;++iy)
                std::copy_n(tile.data()+p*tw*th+iy*tw,copy_w,
                            output.data()+p*width*height+(output_y+iy)*width+output_x);
            output_x+=copy_w;
            row.push_back(std::move(tile));
        }
        if (output_x!=width) return {};
        output_y+=row_height;
        previous=std::move(row);
    }
    if (output_y!=height) return {};
    return output;
}
#endif
