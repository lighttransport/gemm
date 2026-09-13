#include "pipeline.hh"

namespace px {
static double lanczos(double x) {
    if (x == 0)
        return 1;
    if (x <= -3 || x >= 3)
        return 0;
    x *= 3.14159265358979323846;
    return std::sin(x) / x * std::sin(x / 3) / (x / 3);
}
/* Pillow's separable antialiased Lanczos with 22-bit integer coefficients.
 * Each pass clips to uint8, including the intermediate horizontal image. */
static Image resize_axis(const Image &src, int target, bool horizontal) {
    const int source = horizontal ? src.width : src.height, c = src.channels;
    Image dst{{}, horizontal ? target : src.width, horizontal ? src.height : target, c};
    dst.pixels.resize(size_t(dst.width) * dst.height * c);
    double scale = double(source) / target, filter_scale = std::max(1., scale), support = 3 * filter_scale;
    for (int t = 0; t < target; ++t) {
        double center = (t + .5) * scale;
        int low = std::max(0, int(center - support + .5)),
            high = std::min(source, int(center + support + .5));
        std::vector<double> weights(high - low);
        double sum = 0;
        for (int j = low; j < high; ++j) {
            weights[j - low] = lanczos((j - center + .5) / filter_scale);
            sum += weights[j - low];
        }
        std::vector<int> coeff(high - low);
        for (int j = 0; j < high - low; ++j) {
            double v = weights[j] / sum * (1 << 22);
            coeff[j] = int(v + (v < 0 ? -.5 : .5));
        }
        int orth = horizontal ? src.height : src.width;
#pragma omp parallel for schedule(static)
        for (int p = 0; p < orth; ++p)
            for (int ch = 0; ch < c; ++ch) {
                int64_t value = 1 << 21;
                for (int j = low; j < high; ++j) {
                    size_t idx = horizontal ? size_t(p) * source + j : size_t(j) * src.width + p;
                    value += int64_t(src.pixels[idx * c + ch]) * coeff[j - low];
                }
                size_t idx = horizontal ? size_t(p) * target + t : size_t(t) * dst.width + p;
                dst.pixels[idx * c + ch] = uint8_t(std::clamp<int64_t>(value >> 22, 0, 255));
            }
    }
    return dst;
}
Image resize(const Image &src, int width, int height) {
    require(width > 0 && height > 0 && src.width > 0 && src.height > 0, "Invalid resize geometry");
    if (src.width == width && src.height == height)
        return src;
    Image x = src;
    if (x.channels == 4)
        for (size_t i = 0; i < x.pixels.size(); i += 4)
            for (int c = 0; c < 3; ++c) {
                int v = x.pixels[i + c] * x.pixels[i + 3] + 128;
                x.pixels[i + c] = uint8_t((v + (v >> 8)) >> 8);
            }
    if (width != x.width)
        x = resize_axis(x, width, true);
    if (height != x.height)
        x = resize_axis(x, height, false);
    if (x.channels == 4)
        for (size_t i = 0; i < x.pixels.size(); i += 4)
            for (int c = 0; c < 3; ++c) {
                int alpha = x.pixels[i + 3];
                x.pixels[i + c] = uint8_t(alpha ? std::min(255, 255 * x.pixels[i + c] / alpha) : 0);
            }
    return x;
}
Image preprocess(const pixal3d_image &src) {
    require(src.pixels && src.width > 0 && src.height > 0 && (src.channels == 3 || src.channels == 4),
            "Expected RGB/RGBA image");
    require(src.channels == 4 || src.mask, "RGB input requires a mask");
    require(src.width <= 16384 && src.height <= 16384, "Input image exceeds 16384 pixels per axis");
    Image rgba{{}, src.width, src.height, 4};
    rgba.pixels.resize(size_t(src.width) * src.height * 4);
    for (size_t i = 0; i < size_t(src.width) * src.height; ++i) {
        std::copy_n(src.pixels + i * src.channels, 3, rgba.pixels.data() + i * 4);
        rgba.pixels[i * 4 + 3] = src.mask ? src.mask[i] : src.pixels[i * 4 + 3];
    }
    if (std::max(src.width, src.height) > 1024) {
        double scale = 1024. / std::max(src.width, src.height);
        rgba = resize(rgba, std::max(1, int(src.width * scale)), std::max(1, int(src.height * scale)));
    }
    int x0 = rgba.width, y0 = rgba.height, x1 = -1, y1 = -1;
    for (int y = 0; y < rgba.height; ++y)
        for (int x = 0; x < rgba.width; ++x)
            if (rgba.pixels[(size_t(y) * rgba.width + x) * 4 + 3] > 204) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
    require(x1 >= x0 && y1 >= y0, "Mask contains no foreground with alpha > 0.8");
    int extent = int(std::max(x1 - x0, y1 - y0) * 1.1), half = extent / 2, side = 2 * half;
    require(side > 0, "Foreground bounding box is too small");
    int left = int(std::nearbyint((x0 + x1) * .5 - half)), top = int(std::nearbyint((y0 + y1) * .5 - half));
    Image result{{}, side, side, 3};
    result.pixels.resize(size_t(side) * side * 3);
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            int sx = x + left, sy = y + top;
            if (sx < 0 || sx >= rgba.width || sy < 0 || sy >= rgba.height)
                continue;
            const uint8_t *p = rgba.pixels.data() + (size_t(sy) * rgba.width + sx) * 4;
            for (int c = 0; c < 3; ++c)
                result.pixels[(size_t(y) * side + x) * 3 + c] =
                    uint8_t((p[c] / 255.f) * (p[3] / 255.f) * 255.f);
        }
    return result;
}
Vec image_float(const Image &image, bool normalized, bool chw) {
    require(image.channels == 3, "Expected RGB image");
    Vec x(image.pixels.size());
    const float mean[3] = {.485f, .456f, .406f}, std[3] = {.229f, .224f, .225f};
    size_t rows = size_t(image.width) * image.height;
    for (size_t i = 0; i < rows; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = image.pixels[3 * i + c] / 255.f;
            if (normalized)
                v = (v - mean[c]) / std[c];
            x[chw ? c * rows + i : 3 * i + c] = v;
        }
    return x;
}
} // namespace px
