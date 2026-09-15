#include "pipeline.hh"
#include "../../common/safetensors_writer.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sys/resource.h>

namespace px {
Json read_json(const std::string &path) {
    std::ifstream f(path);
    require(bool(f), "Cannot open configuration: " + path);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    return boost::json::parse(text);
}
void dump(const std::string &dir, const std::string &name, const Vec &feats, int channels,
          const Coords &coords) {
    if (dir.empty())
        return;
    std::filesystem::create_directories(dir);
    std::unique_ptr<stw_writer, decltype(&stw_destroy)> writer(stw_create(), stw_destroy);
    require(bool(writer), "Cannot create dump writer");
    uint64_t shape[2] = {feats.size() / channels, uint64_t(channels)};
    require(stw_add(writer.get(), "feats", "F32", shape, 2, feats.data(), feats.size() * 4) == 0,
            "Cannot add dump features");
    if (!coords.empty()) {
        uint64_t cs[2] = {coords.size() / 4, 4};
        require(stw_add(writer.get(), "coords", "I32", cs, 2, coords.data(), coords.size() * 4) == 0,
                "Cannot add dump coordinates");
    }
    require(stw_save(writer.get(), (dir + "/" + name + ".safetensors").c_str()) == 0,
            "Cannot save stage dump");
}
static float number(const Json &j) { return float(j.is_int64() ? j.as_int64() : j.as_double()); }
static std::string string(const Json &j) { return std::string(j.as_string()); }
static Vec values(const Json &j) {
    Vec out;
    for (const auto &v : j.as_array())
        out.push_back(number(v));
    return out;
}
static void normalize(Sparse &x, const Json &config, bool inverse) {
    Vec mean = values(config.at("mean")), std = values(config.at("std"));
    require(mean.size() == size_t(x.channels) && std.size() == mean.size(), "Invalid latent normalization");
    for (size_t i = 0; i < x.feats.size(); ++i) {
        int c = i % x.channels;
        require(std[c] > 0, "Invalid latent scale");
        x.feats[i] = inverse ? (x.feats[i] - mean[c]) / std[c] : x.feats[i] * std[c] + mean[c];
    }
}
struct Conditioning {
    Vec global, projected;
};
struct Pipeline {
    Engine &engine;
    const pixal3d_options &options;
    Json config;
    std::mt19937 random;
    std::string dumps;
    std::map<int, std::pair<Image, Vec>> dino_cache;
    Pipeline(Engine &e, const pixal3d_options &o)
        : engine(e), options(o), config(read_json(std::string(o.model_dir) + "/pipeline.json").at("args")),
          random(o.seed), dumps(o.dump_dir ? o.dump_dir : "") {}
    std::string model_path(const std::string &key) {
        return std::string(options.model_dir) + "/" + string(config.at("models").at(key)) + ".safetensors";
    }
    Conditioning conditioning(const Image &image, const Coords &coords, int resolution, int target,
                              const pixal3d_camera &camera, const std::string &stage) {
        auto started = std::chrono::steady_clock::now();
        std::fprintf(stderr, "Pixal3D %s: conditioning %dx%d, %zu tokens\n", stage.c_str(), image.width,
                     image.height, coords.size() / 4);
        Vec features;
        auto cached = dino_cache.find(image.width);
        if (engine.resident() && cached != dino_cache.end() && cached->second.first.pixels == image.pixels)
            features = cached->second.second;
        else {
            Weights weights(options.dinov3_path);
            features = dino(engine, weights, image_float(image, true, true), image.width);
            if (engine.resident())
                dino_cache[image.width] = {image, features};
        }
        Conditioning cond;
        cond.global.assign(features.begin(), features.begin() + 5 * 1024);
        Vec patches(features.begin() + 5 * 1024, features.end()), xy(coords.size() / 2);
        int rows = int(coords.size() / 4), grid = image.width / 16;
        require(pixal3d_project(coords.data(), rows, resolution, image.width, &camera, xy.data()) == 0,
                "Invalid camera projection");
        Vec low(size_t(rows) * 1024);
        require(pixal3d_sample_features(patches.data(), grid, grid, 1024, xy.data(), rows, low.data()) == 0,
                "Invalid conditioning sample");
        if (target) {
            Weights weights(options.naf_path);
            Vec high = naf(engine, weights, image_float(image, false, false), image.width, patches, grid,
                           target, xy);
            cond.projected.resize(size_t(rows) * 2048);
            for (int r = 0; r < rows; ++r) {
                std::copy_n(low.data() + size_t(r) * 1024, 1024, cond.projected.data() + size_t(r) * 2048);
                std::copy_n(high.data() + size_t(r) * 1024, 1024,
                            cond.projected.data() + size_t(r) * 2048 + 1024);
            }
        } else
            cond.projected = std::move(low);
        dump(dumps, stage + "_global", cond.global, 1024);
        dump(dumps, stage + "_projected", cond.projected, target ? 2048 : 1024, coords);
        engine.record(stage + ".conditioning",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        return cond;
    }
    Sparse sample(const std::string &key, const std::string &sampler, const Coords &coords,
                  const Conditioning &cond, const std::string &stage, const Sparse *concat = nullptr) {
        auto started = std::chrono::steady_clock::now();
        Weights weights(model_path(key));
        int channels = weights.shape("out_layer.weight")[0];
        Sparse x{coords, Vec(coords.size() / 4 * channels), channels};
        // A fixed Box-Muller generator makes native seeds portable across all backends.
        for (size_t i = 0; i < x.feats.size(); i += 2) {
            double u = (double(random()) + .5) / 4294967296., v = (double(random()) + .5) / 4294967296.;
            double radius = std::sqrt(-2 * std::log(u)), theta = 6.283185307179586 * v;
            x.feats[i] = float(radius * std::cos(theta));
            if (i + 1 < x.feats.size())
                x.feats[i + 1] = float(radius * std::sin(theta));
        }
        dump(dumps, stage + "_noise", x.feats, channels, coords);
        Json spec = config.at(sampler);
        const Json &params = spec.at("params");
        int steps = int(params.at("steps").as_int64()), blocks = 0;
        while (weights.has("blocks." + std::to_string(blocks) + ".modulation"))
            ++blocks;
        require(blocks == 30 && steps == 12, "Unsupported Pixal3D flow/sampler configuration");
        float rescale_t = number(params.at("rescale_t")), guidance = number(params.at("guidance_strength"));
        float rescale = number(params.at("guidance_rescale")),
              sigma = number(spec.at("args").at("sigma_min"));
        Vec interval = values(params.at("guidance_interval")), zero_global(cond.global.size()),
            zero_proj(cond.projected.size());
        auto time = [&](int step) {
            double t = 1. - double(step) / steps;
            return float(rescale_t * t / (1 + (rescale_t - 1) * t));
        };
        for (int step = 0; step < steps; ++step) {
            float t = time(step), next = time(step + 1);
            Vec input = x.feats;
            if (concat) {
                require(concat->coords == coords && concat->channels == channels,
                        "Invalid texture shape condition");
                input.resize(x.feats.size() * 2);
                for (int r = 0; r < x.rows(); ++r) {
                    std::copy_n(x.feats.data() + size_t(r) * channels, channels,
                                input.data() + size_t(r) * 2 * channels);
                    std::copy_n(concat->feats.data() + size_t(r) * channels, channels,
                                input.data() + size_t(r) * 2 * channels + channels);
                }
            }
            Vec pos = flow(engine, weights, input, coords, t, cond.global, cond.projected, blocks, true);
            float cfg = t >= interval[0] && t <= interval[1] ? guidance : 1.f;
            Vec neg;
            if (cfg != 1)
                neg = flow(engine, weights, input, coords, t, zero_global, zero_proj, blocks, true);
            pixal3d_euler_cfg(x.feats.data(), pos.data(), neg.empty() ? pos.data() : neg.data(),
                              x.feats.size(), t, next, cfg, rescale, sigma);
            require(std::all_of(x.feats.begin(), x.feats.end(), [](float v) { return std::isfinite(v); }),
                    "Non-finite flow sample");
            dump(dumps, stage + "_step_" + std::to_string(step + 1), x.feats, channels, coords);
            std::fprintf(stderr, "Pixal3D %s: step %d/%d\n", stage.c_str(), step + 1, steps);
        }
        engine.record(stage + ".diffusion",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        return x;
    }
    void run(const pixal3d_image &source, const pixal3d_camera &camera, pixal3d_result &result) {
        float distance;
        require(pixal3d_camera_distance(camera.fov, camera.mesh_scale, &distance) == 0,
                "Invalid camera FOV or mesh scale");
        require(std::isfinite(camera.distance), "Camera distance must be finite");
        Image cropped = preprocess(source), image512 = resize(cropped, 512, 512),
              image1024 = resize(cropped, 1024, 1024);
        dump(dumps, "image512", image_float(image512, false, false), 3);
        dump(dumps, "image1024", image_float(image1024, false, false), 3);
        Coords dense(4096 * 4);
        for (int i = 0; i < 4096; ++i) {
            dense[4 * i + 1] = i / 256;
            dense[4 * i + 2] = i / 16 % 16;
            dense[4 * i + 3] = i % 16;
        }
        Sparse structure;
        {
            auto cond = conditioning(image512, dense, 16, 0, camera, "structure");
            structure =
                sample("sparse_structure_flow_model", "sparse_structure_sampler", dense, cond, "structure");
        }
        Vec occupancy;
        {
            Weights weights(model_path("sparse_structure_decoder"));
            auto decoder_start = std::chrono::steady_clock::now();
            occupancy = decode_structure(engine, weights, structure.feats);
            engine.record(
                "structure.decoder",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decoder_start).count());
        }
        dump(dumps, "structure_decoded", occupancy, 1);
        Coords low;
        for (int x = 0; x < 32; ++x)
            for (int y = 0; y < 32; ++y)
                for (int z = 0; z < 32; ++z) {
                    bool occupied = false;
                    for (int dx = 0; dx < 2; ++dx)
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dz = 0; dz < 2; ++dz)
                                occupied |= occupancy[((x * 2 + dx) * 64 + y * 2 + dy) * 64 + z * 2 + dz] > 0;
                    if (occupied)
                        low.insert(low.end(), {0, x, y, z});
                }
        require(!low.empty(), "Structure decoder produced no occupied voxels");
        Sparse shape_low;
        {
            auto cond = conditioning(image512, low, 32, 512, camera, "shape512");
            shape_low = sample("shape_slat_flow_model_512", "shape_slat_sampler", low, cond, "shape512");
        }
        normalize(shape_low, config.at("shape_slat_normalization"), false);
        dump(dumps, "shape512_denormalized", shape_low.feats, 32, shape_low.coords);
        Coords high;
        {
            Weights weights(model_path("shape_slat_decoder"));
            std::vector<Subdivision> subs;
            auto decoder_start = std::chrono::steady_clock::now();
            Sparse up = decode_sparse(engine, weights, shape_low, true, subs, false);
            engine.record(
                "shape512.decoder",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decoder_start).count());
            high.resize(up.coords.size());
            int64_t n = pixal3d_cascade_coords(up.coords.data(), up.rows(), 512, 1024, high.data());
            require(n > 0, "Invalid cascade coordinates");
            high.resize(size_t(n) * 4);
        }
        Sparse shape;
        {
            auto cond = conditioning(image1024, high, 64, 512, camera, "shape1024");
            shape = sample("shape_slat_flow_model_1024", "shape_slat_sampler", high, cond, "shape1024");
        }
        Sparse texture;
        {
            auto cond = conditioning(image1024, high, 64, 1024, camera, "texture");
            texture = sample("tex_slat_flow_model_1024", "tex_slat_sampler", high, cond, "texture", &shape);
        }
        normalize(shape, config.at("shape_slat_normalization"), false);
        normalize(texture, config.at("tex_slat_normalization"), false);
        dump(dumps, "shape_denormalized", shape.feats, 32, shape.coords);
        dump(dumps, "texture_denormalized", texture.feats, 32, texture.coords);
        std::vector<Subdivision> subs;
        Sparse shape_out, texture_out;
        {
            Weights weights(model_path("shape_slat_decoder"));
            auto decoder_start = std::chrono::steady_clock::now();
            shape_out = decode_sparse(engine, weights, shape, false, subs, false);
            engine.record(
                "shape1024.decoder",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decoder_start).count());
        }
        {
            Weights weights(model_path("tex_slat_decoder"));
            auto decoder_start = std::chrono::steady_clock::now();
            texture_out = decode_sparse(engine, weights, texture, false, subs, true);
            engine.record(
                "texture.decoder",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decoder_start).count());
        }
        require(shape_out.coords == texture_out.coords, "Shape and texture coordinates differ");
        for (float &v : texture_out.feats)
            v = std::clamp(v * .5f + .5f, 0.f, 1.f);
        dump(dumps, "shape_decoded", shape_out.feats, 7, shape_out.coords);
        dump(dumps, "texture_decoded", texture_out.feats, 6, texture_out.coords);
        result.stats.shape_tokens = shape.rows();
        engine.clear_weights();
        auto post_started = std::chrono::steady_clock::now();
        postprocess(shape_out, texture_out, options, result);
        engine.record("postprocess",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - post_started).count());
    }
};
} // namespace px
struct pixal3d_context {
    pixal3d_options options;
    std::string model, dino, naf, dump, error;
    std::unique_ptr<px::Engine> engine;
};
extern "C" void pixal3d_default_gpu_options(pixal3d_gpu_options *o) {
    if (o) {
        *o = {};
        o->struct_size = sizeof(*o);
        o->version = 1;
    }
}
extern "C" int pixal3d_configure_gpu(pixal3d_context *c, const pixal3d_gpu_options *o) {
    if (!c || !o)
        return -1;
    try {
        c->engine->configure(*o);
        return 0;
    } catch (const std::exception &e) {
        c->error = e.what();
        return -1;
    }
}
static thread_local std::string creation_error;
extern "C" void pixal3d_default_options(pixal3d_options *o) {
    if (!o)
        return;
    *o = {};
    o->threads = 16;
    o->vram_budget_mib = 14336;
    o->model_dir = "/mnt/disk2/models/Pixal3D";
    o->dinov3_path = "/mnt/disk2/models/dinov3-vitl16/model.safetensors";
    o->naf_path = "ref/pixal3d/weights/naf_release.safetensors";
    o->texture_size = 4096;
    o->decimation_target = 1000000;
}
extern "C" pixal3d_context *pixal3d_create(const pixal3d_options *o) {
    try {
        px::require(o && o->model_dir && o->dinov3_path && o->naf_path, "Missing Pixal3D model paths");
        px::require(o->device >= 0, "Device ordinal must be nonnegative");
        px::require(o->texture_size == 4096 && o->decimation_target > 0,
                    "Expected 4096 texture and positive face target");
        px::require(o->vram_budget_mib > 512 && o->vram_budget_mib <= 14336,
                    "GPU budget must be in (512,14336] MiB");
        auto c = std::make_unique<pixal3d_context>();
        c->options = *o;
        c->model = o->model_dir;
        c->dino = o->dinov3_path;
        c->naf = o->naf_path;
        c->dump = o->dump_dir ? o->dump_dir : "";
        c->options.model_dir = c->model.c_str();
        c->options.dinov3_path = c->dino.c_str();
        c->options.naf_path = c->naf.c_str();
        c->options.dump_dir = c->dump.c_str();
        for (const auto &p : {c->model + "/pipeline.json", c->dino, c->naf})
            px::require(std::filesystem::is_regular_file(p), "Missing model/configuration: " + p);
        auto config = px::read_json(c->model + "/pipeline.json").at("args");
        for (const char *key : {"sparse_structure_flow_model", "sparse_structure_decoder",
                                "shape_slat_flow_model_512", "shape_slat_flow_model_1024",
                                "tex_slat_flow_model_1024", "shape_slat_decoder", "tex_slat_decoder"}) {
            auto path =
                c->model + "/" + std::string(config.at("models").at(key).as_string()) + ".safetensors";
            px::require(std::filesystem::is_regular_file(path), "Missing checkpoint: " + path);
        }
        c->engine = std::make_unique<px::Engine>(c->options);
        creation_error.clear();
        return c.release();
    } catch (const std::exception &e) {
        creation_error = e.what();
        return nullptr;
    }
}
extern "C" const char *pixal3d_last_error(const pixal3d_context *c) {
    return c ? c->error.c_str() : creation_error.c_str();
}
extern "C" int pixal3d_generate(pixal3d_context *c, const pixal3d_image *image, const pixal3d_camera *camera,
                                pixal3d_result *result) {
    if (!c || !image || !camera || !result) {
        (c ? c->error : creation_error) = "Missing generation context, image, camera or result";
        return -1;
    }
    *result = {};
    c->error.clear();
    auto start = std::chrono::steady_clock::now();
    try {
        c->engine->begin_profile();
        px::Pipeline(*c->engine, c->options).run(*image, *camera, *result);
        result->stats.elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result->stats.peak_device_bytes = c->engine->peak();
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        result->stats.peak_host_bytes = size_t(usage.ru_maxrss) * 1024;
        result->stats.vertices = result->vertex_count;
        result->stats.triangles = result->triangle_count;
        c->engine->record("generate", result->stats.elapsed_seconds);
        c->engine->write_profile();
        return 0;
    } catch (const std::exception &e) {
        c->error = e.what();
        pixal3d_result_free(result);
        return -1;
    }
}
extern "C" void pixal3d_result_free(pixal3d_result *r) {
    if (!r)
        return;
    std::free(r->vertices);
    std::free(r->normals);
    std::free(r->uvs);
    std::free(r->triangles);
    std::free(r->base_color_rgba);
    std::free(r->metallic_roughness_rgb);
    *r = {};
}
extern "C" void pixal3d_destroy(pixal3d_context *c) { delete c; }
