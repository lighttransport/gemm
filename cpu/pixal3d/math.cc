#include "../../common/pixal3d.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

extern "C" int pixal3d_camera_distance(float fov, float scale, float *distance) {
    if (!distance || !std::isfinite(fov) || fov <= 0 || fov >= 3.14159265358979323846f ||
        !std::isfinite(scale) || scale <= 0)
        return -1;
    *distance = 0.5f / (std::tan(fov * 0.5f) * scale);
    return 0;
}

extern "C" int pixal3d_project(const int32_t *coords, size_t count, int grid, int image,
                               const pixal3d_camera *camera, float *xy) {
    if (!coords || !xy || !camera || grid < 2 || image <= 0)
        return -1;
    float distance;
    if (pixal3d_camera_distance(camera->fov, camera->mesh_scale, &distance))
        return -1;
    if (camera->distance > 0)
        distance = camera->distance;
    if (!std::isfinite(camera->distance) || !std::isfinite(distance))
        return -1;
    const float focal = image * 0.5f / std::tan(camera->fov * 0.5f);
    for (size_t i = 0; i < count; ++i) {
        if (coords[4 * i] != 0)
            return -1;
        float p[3];
        for (int a = 0; a < 3; ++a) {
            int c = coords[4 * i + 1 + a];
            if (c < 0 || c >= grid)
                return -1;
            p[a] = (2.f * c / (grid - 1) - 1.f) / (2.f * camera->mesh_scale);
        }
        // ProjGrid rotates (x,y,z)->(x,-z,y). The front camera at (0,-d,0)
        // then sees camera coordinates (x,y,z-d). Do not apply valid_mask:
        // upstream samples out-of-frame points with border padding as well.
        float depth = distance - p[2] + 1e-8f;
        float x = focal * p[0] / depth + image * 0.5f;
        float y = -focal * p[1] / depth + image * 0.5f;
        xy[2 * i] = (x + 0.5f) / image * 2.f - 1.f;
        xy[2 * i + 1] = (y + 0.5f) / image * 2.f - 1.f;
        if (!std::isfinite(xy[2 * i]) || !std::isfinite(xy[2 * i + 1]))
            return -1;
    }
    return 0;
}

extern "C" int pixal3d_sample_features(const float *hwc, int h, int w, int c, const float *xy, size_t count,
                                       float *out) {
    if (!hwc || !xy || !out || h < 1 || w < 1 || c < 1)
        return -1;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(xy[2 * i]) || !std::isfinite(xy[2 * i + 1]))
            return -1;
        float x = std::clamp((xy[2 * i] + 1.f) * w * 0.5f - 0.5f, 0.f, float(w - 1));
        float y = std::clamp((xy[2 * i + 1] + 1.f) * h * 0.5f - 0.5f, 0.f, float(h - 1));
        int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        float fx = x - x0, fy = y - y0;
        for (int k = 0; k < c; ++k)
            out[i * c + k] =
                ((1 - fx) * hwc[(size_t(y0) * w + x0) * c + k] + fx * hwc[(size_t(y0) * w + x1) * c + k]) *
                    (1 - fy) +
                ((1 - fx) * hwc[(size_t(y1) * w + x0) * c + k] + fx * hwc[(size_t(y1) * w + x1) * c + k]) *
                    fy;
    }
    return 0;
}

extern "C" int64_t pixal3d_cascade_coords(const int32_t *coords, size_t count, int low, int high,
                                          int32_t *out) {
    if (!coords || !out || low <= 0 || high < 32 || high % 16 ||
        count > size_t(std::numeric_limits<int>::max()))
        return -1;
    try {
        std::vector<std::array<int32_t, 4>> unique(count);
        for (size_t i = 0; i < count; ++i) {
            if (coords[i * 4] != 0)
                return -1;
            unique[i][0] = 0;
            for (int a = 1; a < 4; ++a) {
                if (coords[i * 4 + a] < 0 || coords[i * 4 + a] >= low)
                    return -1;
                float value = (coords[i * 4 + a] + 0.5f) / low * (high / 16 - 1);
                // Explicit round-to-even, independent of host rounding mode.
                int base = int(std::floor(value));
                float fraction = value - base;
                unique[i][a] = base + (fraction > 0.5f || (fraction == 0.5f && (base & 1)));
            }
        }
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (size_t i = 0; i < unique.size(); ++i)
            std::copy(unique[i].begin(), unique[i].end(), out + 4 * i);
        return int64_t(unique.size());
    } catch (...) {
        return -1;
    }
}

extern "C" void pixal3d_euler_cfg(float *x, const float *pos, const float *neg, size_t n, float t, float next,
                                  float guidance, float rescale, float sigma) {
    const double a = 1.0 - sigma, b = sigma + (1.0 - sigma) * t;
    double ratio = 1.0;
    if (rescale > 0 && n > 1 && guidance != 1.f && guidance != 0.f) {
        double pm = 0, cm = 0, pv = 0, cv = 0;
        for (size_t i = 0; i < n; ++i) {
            double p = a * x[i] - b * pos[i];
            double c = a * x[i] - b * (guidance * pos[i] + (1 - guidance) * neg[i]);
            double dp = p - pm, dc = c - cm;
            pm += dp / (i + 1);
            cm += dc / (i + 1);
            pv += dp * (p - pm);
            cv += dc * (c - cm);
        }
        if (cv > 0)
            ratio = std::sqrt(pv / cv);
    }
    for (size_t i = 0; i < n; ++i) {
        double v = guidance == 1.f   ? pos[i]
                   : guidance == 0.f ? neg[i]
                                     : guidance * pos[i] + (1 - guidance) * neg[i];
        if (rescale > 0 && guidance != 1.f && guidance != 0.f && b > 0) {
            double x0 = (a * x[i] - b * v) * (rescale * ratio + 1 - rescale);
            v = (a * x[i] - x0) / b;
        }
        x[i] -= float((t - next) * v);
    }
}
