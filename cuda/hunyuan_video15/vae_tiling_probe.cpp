/* CPU-only fixed-stride/blending oracle input, including edge and corner tiles. */
#include <cstdio>
#include <fstream>
#include "vae_spatial_tiling.hpp"

int main(int argc, char **argv) {
    if (argc!=2) return 2;
    sd::Tensor<float> input({7,9,2,2,1});
    for (size_t i=0;i<input.numel();++i) input.data()[i]=static_cast<float>(i)/256.f;
    int calls=0;
    auto decode=[&](const sd::Tensor<float>& tile) {
        ++calls;
        auto shape=tile.shape(); shape[0]*=16; shape[1]*=16;
        sd::Tensor<float> output(shape);
        const int64_t w=shape[0],h=shape[1],tw=tile.shape()[0],th=tile.shape()[1];
        for (int64_t p=0;p<4;++p) for (int64_t y=0;y<h;++y) for (int64_t x=0;x<w;++x)
            output.data()[p*w*h+y*w+x]=tile.data()[p*tw*th+(y/16)*tw+x/16]+
                static_cast<float>(x+2*y)/4096.f;
        return output;
    };
    auto output=hv15_vae_spatial_tiles(input,true,64,0.25f,decode);
    if (output.shape()!=std::vector<int64_t>({112,144,2,2,1}) || calls!=9) return 1;
    if (!hv15_vae_spatial_tiles(input,true,64,0.3f,decode).empty()) return 1;
    if (!hv15_vae_spatial_tiles(input,true,64,NAN,decode).empty()) return 1;
    std::ofstream file(argv[1],std::ios::binary);
    file.write(reinterpret_cast<const char *>(output.data()),output.numel()*sizeof(float));
    return file ? 0 : 1;
}
