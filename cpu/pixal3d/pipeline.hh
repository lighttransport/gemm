#pragma once
#include "engine.hh"
#include <boost/json.hpp>
#include <memory>
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
/* The postprocess in two halves: the geometry chain (FDG mesh, holes, BVH,
 * remesh, simplify, UV unwrap, normals) needs only the decoded shape, so it
 * can run on a CPU thread while the GPU samples and decodes the texture;
 * the texture half (raster, bake, inpaint, pack) then joins it. Geometry
 * timings are kept in the stage and merged into the profile by the texture
 * half, on the calling thread. */
struct GeometryStage;
void add_timing(GeometryStage &geometry, const std::string &name, double seconds);
std::shared_ptr<GeometryStage> postprocess_geometry(const Sparse &shape, const pixal3d_options &options,
                                                    const GpuGeometry &gpu = {});
void postprocess_texture(GeometryStage &geometry, const Sparse &shape, const Sparse &texture,
                         const pixal3d_options &options, pixal3d_result &result, Engine *profile = nullptr);
} // namespace px
