#pragma once
#include "../../common/closest_point_bvh.h"
#include "../../common/mesh_ops.h"
#include "pipeline.hh"
namespace px {
using Mesh = trellis2::Mesh;
Mesh remesh(const Mesh &source, const trellis2::ClosestPointBVH &bvh, int resolution = 1024);
void simplify(Mesh &mesh, int target);
void fill_holes(Mesh &mesh);
void vertex_normals(const Mesh &mesh, Vec &normals);
void clean_for_uv(Mesh &mesh);
void unwrap(const Mesh &mesh, Vec &vertices, std::vector<int32_t> &faces, Vec &uv, std::vector<int32_t> &map);
void inpaint(std::vector<uint8_t> &image, int channels, const std::vector<uint8_t> &mask, int size,
             int radius);
} // namespace px
