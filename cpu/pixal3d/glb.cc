#include "pipeline.hh"
#include "../../common/stb_image_write.h"
#include <filesystem>
#include <fstream>

extern "C" int pixal3d_write_glb(const char *path, const pixal3d_result *r) {
    try {
        px::require(path && r && r->vertices && r->normals && r->uvs && r->triangles && r->base_color_rgba &&
                        r->metallic_roughness_rgb,
                    "Missing mesh or PBR textures");
        px::require(r->vertex_count > 0 && r->triangle_count > 0 && r->texture_size > 0,
                    "Invalid GLB dimensions");
        namespace j = boost::json;
        std::vector<uint8_t> binary;
        j::array views, accessors;
        auto append = [&](const void *data, size_t bytes, int target = 0) {
            while (binary.size() % 4)
                binary.push_back(0);
            j::object view{
                {"buffer", 0}, {"byteOffset", uint64_t(binary.size())}, {"byteLength", uint64_t(bytes)}};
            if (target)
                view["target"] = target;
            views.push_back(view);
            const auto *p = static_cast<const uint8_t *>(data);
            binary.insert(binary.end(), p, p + bytes);
            return int(views.size() - 1);
        };
        int nv = r->vertex_count, nf = r->triangle_count;
        int p = append(r->vertices, size_t(nv) * 12, 34962), n = append(r->normals, size_t(nv) * 12, 34962),
            u = append(r->uvs, size_t(nv) * 8, 34962), f = append(r->triangles, size_t(nf) * 12, 34963);
        j::array minimum, maximum;
        for (int c = 0; c < 3; ++c) {
            float low = INFINITY, high = -INFINITY;
            for (int i = 0; i < nv; ++i) {
                float v = r->vertices[3 * i + c];
                px::require(std::isfinite(v), "Non-finite vertex");
                low = std::min(low, v);
                high = std::max(high, v);
            }
            minimum.push_back(low);
            maximum.push_back(high);
        }
        accessors.push_back(j::object{{"bufferView", p},
                                      {"componentType", 5126},
                                      {"count", nv},
                                      {"type", "VEC3"},
                                      {"min", minimum},
                                      {"max", maximum}});
        accessors.push_back(
            j::object{{"bufferView", n}, {"componentType", 5126}, {"count", nv}, {"type", "VEC3"}});
        accessors.push_back(
            j::object{{"bufferView", u}, {"componentType", 5126}, {"count", nv}, {"type", "VEC2"}});
        accessors.push_back(
            j::object{{"bufferView", f}, {"componentType", 5125}, {"count", nf * 3}, {"type", "SCALAR"}});
        auto encode = [&](const uint8_t *pixels, int channels) {
            std::vector<uint8_t> png;
            auto callback = [](void *context, void *data, int length) {
                auto &out = *static_cast<std::vector<uint8_t> *>(context);
                auto *p = static_cast<uint8_t *>(data);
                out.insert(out.end(), p, p + length);
            };
            px::require(stbi_write_png_to_func(callback, &png, r->texture_size, r->texture_size, channels,
                                               pixels, r->texture_size * channels),
                        "PNG encoding failed");
            return append(png.data(), png.size());
        };
        int color = encode(r->base_color_rgba, 4), material = encode(r->metallic_roughness_rgb, 3);
        j::object root{
            {"asset", j::object{{"version", "2.0"}, {"generator", "native Pixal3D"}}},
            {"scene", 0},
            {"scenes", j::array{j::object{{"nodes", j::array{0}}}}},
            {"nodes", j::array{j::object{{"mesh", 0}}}},
            {"meshes", j::array{j::object{
                           {"primitives",
                            j::array{j::object{
                                {"attributes", j::object{{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
                                {"indices", 3},
                                {"material", 0},
                                {"mode", 4}}}}}}},
            {"materials", j::array{j::object{{"name", "Pixal3D PBR"},
                                             {"alphaMode", "OPAQUE"},
                                             {"doubleSided", false},
                                             {"pbrMetallicRoughness",
                                              j::object{{"baseColorFactor", j::array{1, 1, 1, 1}},
                                                        {"baseColorTexture", j::object{{"index", 0}}},
                                                        {"metallicRoughnessTexture", j::object{{"index", 1}}},
                                                        {"metallicFactor", 1},
                                                        {"roughnessFactor", 1}}}}}},
            {"textures", j::array{j::object{{"source", 0}}, j::object{{"source", 1}}}},
            {"images", j::array{j::object{{"bufferView", color}, {"mimeType", "image/png"}},
                                j::object{{"bufferView", material}, {"mimeType", "image/png"}}}},
            {"buffers", j::array{j::object{{"byteLength", uint64_t(binary.size())}}}},
            {"bufferViews", views},
            {"accessors", accessors}};
        std::string json = j::serialize(root);
        while (json.size() % 4)
            json += ' ';
        while (binary.size() % 4)
            binary.push_back(0);
        uint64_t total = 28 + json.size() + binary.size();
        px::require(total < UINT32_MAX, "GLB exceeds 4 GiB");
        std::string temporary = std::string(path) + ".partial";
        std::ofstream output(temporary, std::ios::binary);
        auto word = [&](uint32_t v) {
            uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
            output.write(reinterpret_cast<char *>(b), 4);
        };
        word(0x46546c67);
        word(2);
        word(uint32_t(total));
        word(uint32_t(json.size()));
        word(0x4e4f534a);
        output.write(json.data(), json.size());
        word(uint32_t(binary.size()));
        word(0x004e4942);
        output.write(reinterpret_cast<const char *>(binary.data()), binary.size());
        output.close();
        px::require(bool(output), "Cannot write GLB");
        std::filesystem::rename(temporary, path);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Pixal3D GLB: %s\n", e.what());
        return -1;
    }
}
