#pragma once
#include "engine.hh"
#include <boost/json.hpp>
#include <random>
namespace px {
using Json = boost::json::value;
struct Image {
    std::vector<uint8_t> pixels;
    int width = 0, height = 0, channels = 0;
};
Image resize(const Image &image, int width, int height);
Image preprocess(const pixal3d_image &image);
Image preprocess_view(const pixal3d_image &image, int size);
Vec image_float(const Image &image, bool normalized, bool chw);
void dump(const std::string &dir, const std::string &name, const Vec &feats, int channels,
          const Coords &coords = {});
Json read_json(const std::string &path);
void postprocess(const Sparse &shape, const Sparse &texture, const pixal3d_options &options,
                 pixal3d_result &result, Engine *profile = nullptr);
} // namespace px
