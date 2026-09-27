/* Device operations for the versioned resident protocol. */
#include "pixal3d_mma.inc"

#ifdef PX_HIP
#include "../tmp/pixal3d/wmma_attention.inc"
// RDNA runs wave32; the warp kernels below reduce over 32 lanes.
#define PX_SHFL_XOR(v, s) __shfl_xor(v, s)
#define PX_SHFL_DOWN(v, s) __shfl_down(v, s)
#define PX_SHFL(v, l) __shfl(v, l)
#else
#define PX_SHFL_XOR(v, s) __shfl_xor_sync(0xffffffffu, v, s)
#define PX_SHFL_DOWN(v, s) __shfl_down_sync(0xffffffffu, v, s)
#define PX_SHFL(v, l) __shfl_sync(0xffffffffu, v, l)
#include "../tmp/pixal3d/fa2.inc"
__global__ void px_attention_layout(unsigned short *packed, float *plain, int n, int heads, int precision,
                                    int unpack) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * heads * 128)
        return;
    int d = i % 128, row = i / 128 % n, head = i / (128 * n);
    size_t j = (size_t(row) * heads + head) * 128 + d;
    if (unpack)
        plain[j] = precision == 1 ? __uint_as_float(unsigned(packed[i]) << 16)
                                  : __half2float(__ushort_as_half(packed[i]));
    else
        packed[i] = px_pack(plain[j], precision);
}
// FA2's [heads, n, 128] output straight to a packed BF16 [n, heads*128]
// GEMM input: the value px_attention_layout unpacks, rounded as cast_bfloat.
__global__ void px_attention_unpack_bf16(unsigned short *plain, const unsigned short *packed, int n, int heads,
                                         int precision) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * heads * 128)
        return;
    int d = i % 128, row = i / 128 % n, head = i / (128 * n);
    float value = precision == 1 ? __uint_as_float(unsigned(packed[i]) << 16)
                                 : __half2float(__ushort_as_half(packed[i]));
    plain[(size_t(row) * heads + head) * 128 + d] = __float_as_uint(round_bfloat(value)) >> 16;
}
#endif
static Buffer &px_buffer(void *p) {
    if (!p)
        throw std::runtime_error("Missing device tensor");
    return *static_cast<Buffer *>(p);
}
static float *px_float(void *p) { return p ? px_buffer(p).f() : nullptr; }
static void px_extent(void *p, size_t elements, size_t width = 4) {
    if (elements > SIZE_MAX / width || px_buffer(p).bytes < elements * width)
        throw std::runtime_error("Resident tensor extent exceeds allocation");
}
__global__ void px_element(float *out, const float *x, const float *w, const float *b, int op, size_t n,
                           int c, int precision, int k, int offset) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float value = x[i];
    int ch = i % c;
    if (op == PX_PART)
        value = x[(i / c * k + offset) * c + ch];
    else if (op == PX_GELU)
        value = k ? .5f * value * (1 + tanhf(.7978845608028654f * (value + .044715f * value * value * value)))
                  : .5f * value * (1 + erff(value * .7071067811865475f));
    else if (op == PX_SILU)
        value = value / (1 + expf(-value));
    else if (op == PX_ADD)
        value += w[k ? i : ch];
    else if (op == PX_RESIDUAL)
        value += round_precision(w[i] * (b ? b[offset + ch] : 1), precision);
    else if (op == PX_SCALE_ADD)
        value += w[i] * b[ch];
    else if (op == PX_MODULATE) {
        float shift = w[offset + ch], scale = round_precision(1 + w[offset + c + ch], precision);
        value = round_precision(round_precision(value, precision) * scale, precision) + shift;
    }
    out[i] = round_precision(value, precision);
}
__device__ unsigned short px_bf16_bits(float value) { return __float_as_uint(round_bfloat(value)) >> 16; }
// px_element with one block per row and threads striding over the channels:
// the same per-element arithmetic without a 64-bit modulo per element.
__global__ void px_element_rows(float *out, const float *x, const float *w, const float *b, int op, int c,
                                int precision, int k, int offset, unsigned short *packed) {
    size_t row = blockIdx.x;
    for (int ch = threadIdx.x; ch < c; ch += blockDim.x) {
        size_t i = row * c + ch;
        float value = x[i];
        if (op == PX_PART)
            value = x[(row * k + offset) * c + ch];
        else if (op == PX_GELU)
            value = k ? .5f * value * (1 + tanhf(.7978845608028654f * (value + .044715f * value * value * value)))
                      : .5f * value * (1 + erff(value * .7071067811865475f));
        else if (op == PX_SILU)
            value = value / (1 + expf(-value));
        else if (op == PX_ADD)
            value += w[k ? i : ch];
        else if (op == PX_RESIDUAL)
            value += round_precision(w[i] * (b ? b[offset + ch] : 1), precision);
        else if (op == PX_SCALE_ADD)
            value += w[i] * b[ch];
        else if (op == PX_MODULATE) {
            float shift = w[offset + ch], scale = round_precision(1 + w[offset + c + ch], precision);
            value = round_precision(round_precision(value, precision) * scale, precision) + shift;
        }
        value = round_precision(value, precision);
        if (packed)
            packed[i] = px_bf16_bits(value);
        else
            out[i] = value;
    }
}
__global__ void px_norm(float *out, const float *x, const float *w, const float *b, int c, float eps,
                        int precision, int rms, int heads) {
    int row = blockIdx.x;
    __shared__ double sum[256];
    double mean = 0;
    for (int j = threadIdx.x; j < c; j += 256)
        mean += rms ? double(x[size_t(row) * c + j]) * x[size_t(row) * c + j] : x[size_t(row) * c + j];
    sum[threadIdx.x] = mean;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            sum[threadIdx.x] += sum[threadIdx.x + s];
        __syncthreads();
    }
    mean = rms ? 0 : sum[0] / c;
    double variance = sum[0];
    if (!rms) {
        variance = 0;
        for (int j = threadIdx.x; j < c; j += 256) {
            double v = x[size_t(row) * c + j] - mean;
            variance += v * v;
        }
        __syncthreads();
        sum[threadIdx.x] = variance;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (threadIdx.x < s)
                sum[threadIdx.x] += sum[threadIdx.x + s];
            __syncthreads();
        }
        variance = sum[0];
    }
    float inv =
        rms ? sqrtf(float(c)) / fmaxf(sqrtf(float(variance)), 1e-12f) : 1 / sqrtf(float(variance / c) + eps);
    for (int j = threadIdx.x; j < c; j += 256) {
        float value = x[size_t(row) * c + j];
        value = rms ? value * (inv * w[(row % heads) * c + j])
                    : (value - float(mean)) * inv * (w ? w[j] : 1) + (b ? b[j] : 0);
        out[size_t(row) * c + j] = round_precision(value, precision);
    }
}
// One warp per row: the same double accumulation as px_norm, reduced with warp
// shuffles instead of a 256-thread block, shared memory and barriers. The
// per-head RMS norms (rows of 128 values, heads x tokens rows) left half of
// px_norm's threads idle and spent most of its time in barriers.
__global__ void px_norm_warp(float *out, const float *x, const float *w, const float *b, int c, float eps,
                             int precision, int rms, int heads, int rows, unsigned short *packed) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows)
        return;
    const float *in = x + size_t(row) * c;
    double sum = 0;
    for (int j = lane; j < c; j += 32)
        sum += rms ? double(in[j]) * in[j] : in[j];
    for (int s = 16; s; s /= 2)
        sum += PX_SHFL_XOR(sum, s);
    double mean = rms ? 0 : sum / c, variance = sum;
    if (!rms) {
        variance = 0;
        for (int j = lane; j < c; j += 32) {
            double v = in[j] - mean;
            variance += v * v;
        }
        for (int s = 16; s; s /= 2)
            variance += PX_SHFL_XOR(variance, s);
    }
    float inv =
        rms ? sqrtf(float(c)) / fmaxf(sqrtf(float(variance)), 1e-12f) : 1 / sqrtf(float(variance / c) + eps);
    for (int j = lane; j < c; j += 32) {
        float value = in[j];
        value = rms ? value * (inv * w[(row % heads) * c + j])
                    : (value - float(mean)) * inv * (w ? w[j] : 1) + (b ? b[j] : 0);
        value = round_precision(value, precision);
        if (packed)
            packed[size_t(row) * c + j] = px_bf16_bits(value);
        else
            out[size_t(row) * c + j] = value;
    }
}
__device__ void px_phase(const int *coords, int f, int nf, float &co, float &si) {
    if (f >= 3 * nf) {
        co = 1;
        si = 0;
        return;
    }
    float theta = coords[f / nf + 1] * powf(10000.f, -float(f % nf) / nf);
    co = cosf(theta);
    si = sinf(theta);
}
__global__ void px_rope_phase(float *out, const int *coords, int n, int hd) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * hd / 2)
        return;
    float co, si;
    px_phase(coords + (i / (hd / 2)) * 4, i % (hd / 2), hd / 6, co, si);
    out[2 * i] = co;
    out[2 * i + 1] = si;
}
__global__ void px_rope(float *out, const float *x, const int *coords, size_t count, int hd, int heads,
                        int precision, int cached) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int pairs = hd / 2, nf = hd / 6;
    if (i >= count / 2)
        return;
    size_t row = i / pairs;
    int f = i % pairs;
    float u = x[row * hd + 2 * f], v = x[row * hd + 2 * f + 1];
    if (f < 3 * nf) {
        float co, si;
        if (cached) {
            const float *phase = reinterpret_cast<const float *>(coords) + (row / heads) * hd + 2 * f;
            co = phase[0];
            si = phase[1];
        } else
            px_phase(coords + (row / heads) * 4, f, nf, co, si);
        out[row * hd + 2 * f] = round_precision(u * co - v * si, precision);
        out[row * hd + 2 * f + 1] = round_precision(u * si + v * co, precision);
    } else {
        out[row * hd + 2 * f] = u;
        out[row * hd + 2 * f + 1] = v;
    }
}
__global__ void px_rope2(float *out, const float *x, const float *periods, int grid, int heads, int hd,
                         int prefix) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int half = hd / 2, quarter = hd / 4;
    if (i >= size_t(grid) * grid * heads * half)
        return;
    int j = i % half, h = i / half % heads, p = i / (half * heads);
    float coord = 2 * ((j < quarter ? p / grid : p % grid) + .5f) / grid - 1;
    float period = periods ? periods[j % quarter] : powf(100.f, float(j % quarter) / quarter);
    float angle = 6.283185307179586f * coord / period, co = cosf(angle), si = sinf(angle);
    size_t k = size_t(p + prefix) * heads * hd + h * hd + j;
    float a = x[k], b = x[k + half];
    out[k] = a * co - b * si;
    out[k + half] = b * co + a * si;
}
__global__ void px_cross5(float *out, const float *q, const float *k, const float *v, int n, int heads,
                          int hd, int precision) {
    int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x, c = heads * hd;
    __shared__ float scores[5], tmp[256];
    for (int key = 0; key < 5; ++key) {
        tmp[d] = d < hd ? q[size_t(row) * c + head * hd + d] * k[key * c + head * hd + d] : 0;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (d < s)
                tmp[d] += tmp[d + s];
            __syncthreads();
        }
        if (d == 0)
            scores[key] = tmp[0] / sqrtf(float(hd));
        __syncthreads();
    }
    if (d == 0) {
        float max = scores[0], sum = 0;
        for (int j = 1; j < 5; ++j)
            max = fmaxf(max, scores[j]);
        for (int j = 0; j < 5; ++j) {
            scores[j] = expf(scores[j] - max);
            sum += scores[j];
        }
        for (int j = 0; j < 5; ++j)
            scores[j] /= sum;
    }
    __syncthreads();
    if (d < hd) {
        float value = 0;
        for (int j = 0; j < 5; ++j)
            value += scores[j] * v[j * c + head * hd + d];
        out[size_t(row) * c + head * hd + d] = round_precision(value, precision);
    }
}
// px_cross5 for hd == 128, one warp per (token, head). Each lane holds dims
// l, l+32, l+64, l+96 and reproduces px_cross5's tree exactly: (i, i+64),
// then (i, i+32), then shuffles down 16..1 -- the same pairings, so the same
// sums, without a 256-thread block and ten barriers per key.
__global__ void px_cross5_warp(float *out, const float *q, const float *k, const float *v, int n, int heads,
                               int precision, unsigned short *packed) {
    int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (item >= n * heads)
        return;
    int row = item / heads, head = item % heads, c = heads * 128;
    const float *qr = q + size_t(row) * c + head * 128;
    float x[4];
    for (int j = 0; j < 4; ++j)
        x[j] = qr[lane + 32 * j];
    float scores[5];
    for (int key = 0; key < 5; ++key) {
        const float *kr = k + key * c + head * 128;
        // Products rounded as separate floats, as px_cross5 stores them.
        float a0 = __fadd_rn(__fmul_rn(x[0], kr[lane]), __fmul_rn(x[2], kr[lane + 64]));
        float a1 = __fadd_rn(__fmul_rn(x[1], kr[lane + 32]), __fmul_rn(x[3], kr[lane + 96]));
        float t = __fadd_rn(a0, a1);
        for (int s = 16; s; s /= 2)
            t = __fadd_rn(t, PX_SHFL_DOWN(t, s));
        scores[key] = PX_SHFL(t, 0) / sqrtf(128.f);
    }
    float max = scores[0], sum = 0;
    for (int j = 1; j < 5; ++j)
        max = fmaxf(max, scores[j]);
    for (int j = 0; j < 5; ++j) {
        scores[j] = expf(scores[j] - max);
        sum += scores[j];
    }
    for (int j = 0; j < 5; ++j)
        scores[j] /= sum;
    size_t base = size_t(row) * c + head * 128;
    for (int part = 0; part < 4; ++part) {
        int d = lane + 32 * part;
        float value = 0;
        for (int j = 0; j < 5; ++j)
            value += scores[j] * v[j * c + head * 128 + d];
        value = round_precision(value, precision);
        if (packed)
            packed[base + d] = px_bf16_bits(value);
        else
            out[base + d] = value;
    }
}
__global__ void px_gather(void *output, const float *x, const int *map, int start, int rows, int ci,
                          int dense, int precision) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * 27 * ci)
        return;
    size_t row = i / (27 * ci), j = i % (27 * ci);
    int ch = dense ? j / 27 : j % ci, slot = dense ? j % 27 : j / ci;
    int src = map[(size_t(start) + row) * 27 + slot];
    float value = src >= 0 ? x[size_t(src) * ci + ch] : 0;
    if (precision)
        static_cast<unsigned short *>(output)[i] = px_pack(value, precision);
    else
        static_cast<float *>(output)[i] = value;
}
__global__ void px_c2s(float *out, const float *x, const int *parent, const int *slot, int n, int co, int ci,
                       int skip, int dense, int row0, int parent0) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * co)
        return;
    int r = row0 + int(i / co), ch = i % co, p = parent[r] - parent0, s = slot[r];
    size_t o = size_t(r) * co + ch;
    if (dense) {
        out[o] = x[size_t(p) * co * 8 + ch * 8 + s];
        return;
    }
    out[o] =
        skip ? x[size_t(p) * ci + s * (ci / 8) + ch / (co / (ci / 8))] : x[(size_t(p) * 8 + s) * co + ch];
}
__global__ void px_conv2_gather(float *out, const float *x, int rows, int ci, int kernel, int size,
                                int start) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int kk = ci * kernel * kernel;
    if (i >= size_t(rows) * kk)
        return;
    int r = i / kk + start, ch = i % kk / (kernel * kernel), y = i % (kernel * kernel) / kernel,
        z = i % kernel;
    int py = r / size + y - kernel / 2, px = r % size + z - kernel / 2;
    if (py < 0)
        py = -py;
    if (py >= size)
        py = 2 * size - 2 - py;
    if (px < 0)
        px = -px;
    if (px >= size)
        px = 2 * size - 2 - px;
    out[i] = x[(size_t(py) * size + px) * ci + ch];
}
__global__ void px_group_silu(float *out, const float *x, const float *w, const float *b, int rows, int c) {
    int g = blockIdx.x, cg = c / 8;
    size_t count = size_t(rows) * cg;
    __shared__ double sum[256];
    double value = 0;
    for (size_t i = threadIdx.x; i < count; i += 256)
        value += x[i / cg * c + g * cg + i % cg];
    sum[threadIdx.x] = value;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            sum[threadIdx.x] += sum[threadIdx.x + s];
        __syncthreads();
    }
    double mean = sum[0] / count;
    value = 0;
    for (size_t i = threadIdx.x; i < count; i += 256) {
        double d = x[i / cg * c + g * cg + i % cg] - mean;
        value += d * d;
    }
    __syncthreads();
    sum[threadIdx.x] = value;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            sum[threadIdx.x] += sum[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1 / sqrtf(float(sum[0] / count) + 1e-5f);
    for (size_t i = threadIdx.x; i < count; i += 256) {
        int ch = g * cg + i % cg;
        size_t j = i / cg * c + ch;
        float v = (x[j] - float(mean)) * inv * w[ch] + b[ch];
        out[j] = v / (1 + expf(-v));
    }
}
__global__ void px_average(float *out, const float *x, int source, int target, int c) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(target) * target * c)
        return;
    int p = i / c, ch = i % c, y = p / target, z = p % target;
    int y0 = y * source / target, y1 = ((y + 1) * source + target - 1) / target;
    int x0 = z * source / target, x1 = ((z + 1) * source + target - 1) / target;
    float v = 0;
    for (int a = y0; a < y1; ++a)
        for (int b = x0; b < x1; ++b)
            v += x[(size_t(a) * source + b) * c + ch];
    out[i] = v / ((y1 - y0) * (x1 - x0));
}
__global__ void px_join(float *out, const float *x, int rows, int c, int offset) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(rows) * c)
        out[i / c * (2 * c) + offset + i % c] = x[i];
}
__global__ void px_naf_sample(float *out, const float *guide, const float *keys, const float *patches,
                              const float *xy, int target, int grid) {
    int row = blockIdx.x, head = blockIdx.y, ch = threadIdx.x, dilation = target / grid;
    float x = fminf(fmaxf((xy[2 * row] + 1) * target * .5f - .5f, 0), float(target - 1));
    float y = fminf(fmaxf((xy[2 * row + 1] + 1) * target * .5f - .5f, 0), float(target - 1));
    int x0 = int(x), y0 = int(y);
    x -= x0;
    y -= y0;
    __shared__ float scores[81];
    float corners[4];
    for (int corner = 0; corner < 4; ++corner) {
        int px = min(x0 + (corner % 2), target - 1), py = min(y0 + (corner / 2), target - 1);
        int sy = min(max(py / dilation - 4, 0), grid - 9), sx = min(max(px / dilation - 4, 0), grid - 9);
        if (ch < 81) {
            int k = (sy + ch / 9) * grid + sx + ch % 9;
            float dot = 0;
            for (int d = 0; d < 64; ++d)
                dot += guide[(size_t(py) * target + px) * 256 + head * 64 + d] *
                       keys[size_t(k) * 256 + head * 64 + d];
            scores[ch] = dot * .125f;
        }
        __syncthreads();
        if (ch == 0) {
            float maximum = scores[0], sum = 0;
            for (int j = 1; j < 81; ++j)
                maximum = fmaxf(maximum, scores[j]);
            for (int j = 0; j < 81; ++j) {
                scores[j] = expf(scores[j] - maximum);
                sum += scores[j];
            }
            for (int j = 0; j < 81; ++j)
                scores[j] /= sum;
        }
        __syncthreads();
        float v = 0;
        for (int j = 0; j < 81; ++j) {
            int k = (sy + j / 9) * grid + sx + j % 9;
            v += scores[j] * patches[size_t(k) * 1024 + head * 256 + ch];
        }
        corners[corner] = v;
        __syncthreads();
    }
    out[size_t(row) * 1024 + head * 256 + ch] =
        ((1 - x) * corners[0] + x * corners[1]) * (1 - y) + ((1 - x) * corners[2] + x * corners[3]) * y;
}
extern "C" int px_gpu_device_version() { return PX_DEVICE_ABI; }
extern "C" int px_gpu_configure(void *p, int kernels, int profile) {
    Device &d = *static_cast<Device *>(p);
    if (kernels < 0 || kernels > 2) {
        d.error = "Invalid kernel selection";
        return -1;
    }
    if (kernels == 2 && !d.mma_supported) {
        d.error = "Explicit MMA unsupported on this GPU";
        return -1;
    }
    d.kernels = kernels;
    d.profile = profile;
    return 0;
}
extern "C" void *px_gpu_allocate(void *p, size_t bytes) {
    auto &d = *static_cast<Device *>(p);
    try {
        if (!bytes)
            throw std::runtime_error("Empty resident allocation");
        d.pool_enabled = true;
        return new Buffer(d, bytes);
    } catch (const std::exception &e) {
        d.error = e.what();
        return nullptr;
    }
}
extern "C" void px_gpu_release(void *p) { delete static_cast<Buffer *>(p); }
extern "C" int px_gpu_copy(void *p, void *tensor, void *host, size_t bytes, int download) {
    auto &d = *static_cast<Device *>(p);
    try {
        GPU(gpuSetDevice(d.ordinal));
        auto &b = px_buffer(tensor);
        if (bytes > b.bytes || !host)
            throw std::runtime_error("Invalid resident transfer");
        if (download) {
            GPU(transfer(d, host, b.p, bytes, gpuD2H));
        } else {
            GPU(transfer(d, b.p, host, bytes, gpuH2D));
        }
        return 0;
    } catch (const std::exception &e) {
        d.error = e.what();
        return -1;
    }
}
extern "C" int px_gpu_trim(void *p) {
    auto &d = *static_cast<Device *>(p);
    try {
        GPU(gpuSetDevice(d.ordinal));
        GPU(gpuStreamSynchronize(d.stream));
        while (!d.pool.empty()) {
            auto block = d.pool.back();
            GPU(gpuFree(block.first));
            d.live -= block.second;
            d.pool_bytes -= block.second;
            d.pool.pop_back();
        }
        // A later resident allocation enables pooling again. This also restores
        // original allocation behavior when an Engine switches back to legacy.
        d.pool_enabled = false;
        return 0;
    } catch (const std::exception &e) {
        d.error = e.what();
        return -1;
    }
}
extern "C" int px_gpu_metrics(void *p, px_device_metrics *out) {
    auto &d = *static_cast<Device *>(p);
    if (out) {
        *out = d.metrics;
        out->effective_budget_bytes = d.budget;
        out->active_bytes = d.active;
        out->pooled_bytes = d.pool_bytes;
        out->peak_active_bytes = d.peak_active;
        out->largest_allocation_bytes = d.largest_allocation;
    } else {
        d.metrics = {};
        d.peak = d.live;
        d.peak_active = d.active;
        d.largest_allocation = 0;
    }
    return 0;
}
#ifndef PX_HIP
// D[co x rows] = W^T X + bias in one cuBLASLt call (column-major view, as the
// cublasGemmEx call below): the bias is added in the epilogue in FP32 --
// what finish_gemm does in mixed mode -- instead of a separate pass. False
// when cuBLASLt offers no plan; the caller then takes the GemmEx path.
static bool px_lt_linear(Device &d, float *out, const void *w, const void *x, const float *bias, int rows,
                         int co, int ci) {
    if (d.lt_failed)
        return false;
    if (!d.lt) {
        if (cublasLtCreate(&d.lt) != CUBLAS_STATUS_SUCCESS || gpuMalloc(&d.lt_workspace, Device::kLtWorkspace)) {
            (void)gpuGetLastError();
            d.lt_failed = true;
            return false;
        }
    }
    auto &plan = d.lt_plans[{rows, co, ci}];
    if (!plan.desc) {
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cudaDataType_t bias_type = CUDA_R_32F;
        cublasLtMatmulPreference_t pref{};
        cublasLtMatmulHeuristicResult_t result{};
        int found = 0;
        size_t ws = Device::kLtWorkspace;
        plan.ok = cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == 0 &&
                  cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) == 0 &&
                  cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) == 0 &&
                  cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                 sizeof(epilogue)) == 0 &&
                  cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                 sizeof(bias_type)) == 0 &&
                  cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, ci, co, ci) == 0 &&
                  cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, ci, rows, ci) == 0 &&
                  cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, co, rows, co) == 0 &&
                  cublasLtMatmulPreferenceCreate(&pref) == 0 &&
                  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                                       sizeof(ws)) == 0 &&
                  cublasLtMatmulAlgoGetHeuristic(d.lt, plan.desc, plan.a, plan.b, plan.c, plan.c, pref, 1,
                                                 &result, &found) == 0 &&
                  found > 0;
        if (pref)
            (void)cublasLtMatmulPreferenceDestroy(pref);
        if (plan.ok)
            plan.algo = result.algo;
    }
    if (!plan.ok)
        return false;
    if (cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)) != 0)
        return false;
    float one = 1, zero = 0;
    return cublasLtMatmul(d.lt, plan.desc, &one, w, plan.a, x, plan.b, &zero, out, plan.c, out, plan.c,
                          &plan.algo, d.lt_workspace, Device::kLtWorkspace, d.stream) == CUBLAS_STATUS_SUCCESS;
}
#endif
static void px_resident_linear(Device &d, const px_device_command &op) {
    int n = op.n, co = op.c, ci = op.k;
    bool mixed = op.precision == 3;
    bool packed_input = op.extra != 0;
    int lowp = mixed ? 1 : op.precision;
    px_extent(op.x, size_t(n + op.heads) * ci, packed_input ? 2 : 4);
    px_extent(op.w, size_t(co) * ci, op.precision ? 2 : 4);
    px_extent(op.out, size_t(n + op.offset) * co);
    if (op.b)
        px_extent(op.b, co);
    float one = 1, zero = 0;
    int tile = std::min(n, 2048);
    std::unique_ptr<Buffer> packed;
    if (op.precision && !packed_input)
        packed = std::make_unique<Buffer>(d, size_t(tile) * ci * 2);
    for (int start = 0; start < n; start += tile) {
        int rows = std::min(tile, n - start);
        float *x = px_float(op.x) + size_t(start + op.heads) * ci,
              *out = px_float(op.out) + size_t(start + op.offset) * co;
        if (op.precision) {
            const void *input;
            if (packed_input)
                input = static_cast<const unsigned short *>(px_buffer(op.x).p) + size_t(start + op.heads) * ci;
            else {
                cast_bfloat<<<(size_t(rows) * ci + 255) / 256, 256, 0, d.stream>>>(
                    (unsigned short *)packed->p, x, size_t(rows) * ci, lowp);
                input = packed->p;
            }
            if (d.kernels == 2 && !mixed) {
                px_gemm_mma<<<dim3((co + 15) / 16, (rows + 15) / 16), 32, 0, d.stream>>>(
                    out, (const unsigned short *)input, (const unsigned short *)px_buffer(op.w).p, rows,
                    co, ci, lowp);
                d.metrics.mma_gemms++;
            } else {
#ifndef PX_HIP
                if (mixed && op.b && lowp == 1 &&
                    px_lt_linear(d, out, px_buffer(op.w).p, input, px_float(op.b), rows, co, ci)) {
                    d.metrics.gemms++;
                    continue; // bias added in the epilogue
                }
#endif
                auto type = lowp == 1 ? BF16_TYPE : F16_TYPE;
                GPU(blasGemmEx(d.blas, OP_T, OP_N, co, rows, ci, &one, px_buffer(op.w).p, type, ci, input,
                               type, ci, &zero, out, F32_TYPE, co, COMPUTE_F32, ALGO_DEFAULT));
            }
        } else
            GPU(blasSgemm(d.blas, OP_T, OP_N, co, rows, ci, &one, px_float(op.w), ci, x, ci, &zero, out, co));
        if (co >= 128)
            finish_gemm_rows<<<rows, 256, 0, d.stream>>>(out, px_float(op.b), co, mixed ? 0 : op.precision);
        else
            finish_gemm<<<(size_t(rows) * co + 255) / 256, 256, 0, d.stream>>>(
                out, px_float(op.b), size_t(rows) * co, co, mixed ? 0 : op.precision);
        d.metrics.gemms++;
    }
}
static void px_resident_attention(Device &d, const px_device_command &op) {
    int n = op.n, m = op.k, heads = op.heads, hd = op.c, c = heads * hd;
    bool pack = op.extra & PX_PACK_OUT;
    px_extent(op.x, size_t(n) * c);
    px_extent(op.w, size_t(m) * c);
    px_extent(op.v, size_t(m) * c);
    px_extent(op.out, size_t(n) * c, pack ? 2 : 4);
    d.metrics.attentions++;
    unsigned short *packed_out = pack ? (unsigned short *)px_buffer(op.out).p : nullptr;
    if (m == 5 && hd == 128) {
        px_cross5_warp<<<(size_t(n) * heads + 7) / 8, 256, 0, d.stream>>>(
            pack ? nullptr : px_float(op.out), px_float(op.x), px_float(op.w), px_float(op.v), n, heads,
            op.precision, packed_out);
        return;
    }
    if (pack) {
        // Other paths compute F32 as before, then cast into the packed output.
        Buffer plain(d, size_t(n) * c * 4);
#ifndef PX_HIP
        if (op.precision && hd == 128 && n == m && d.kernels != 1 && d.mma_supported) {
            size_t count = size_t(n) * c;
            Buffer q(d, count * 2), k(d, count * 2), v(d, count * 2), out(d, count * 2);
            px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)q.p,
                                                                           px_float(op.x), n, heads,
                                                                           op.precision, 0);
            px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)k.p,
                                                                           px_float(op.w), n, heads,
                                                                           op.precision, 0);
            px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)v.p,
                                                                           px_float(op.v), n, heads,
                                                                           op.precision, 0);
            if (op.precision == 1)
                px_fa2_bf16::px_fa2_bf16<<<dim3((n + 63) / 64, heads), 128, 34816, d.stream>>>(
                    (unsigned short *)out.p, (unsigned short *)q.p, (unsigned short *)k.p,
                    (unsigned short *)v.p, n, 128, 1 / sqrtf(128.f));
            else
                px_fa2_fp16::px_fa2_fp16<<<dim3((n + 63) / 64, heads), 128, 34816, d.stream>>>(
                    (unsigned short *)out.p, (unsigned short *)q.p, (unsigned short *)k.p,
                    (unsigned short *)v.p, n, 128, 1 / sqrtf(128.f));
            px_attention_unpack_bf16<<<(count + 255) / 256, 256, 0, d.stream>>>(
                packed_out, (unsigned short *)out.p, n, heads, op.precision);
            d.metrics.mma_attentions++;
            return;
        }
#endif
        px_device_command plain_op = op;
        plain_op.extra &= ~PX_PACK_OUT;
        plain_op.out = &plain;
        d.metrics.attentions--; // counted by the call below
        px_resident_attention(d, plain_op);
        cast_bfloat<<<(size_t(n) * c + 255) / 256, 256, 0, d.stream>>>(packed_out, plain.f(), size_t(n) * c, 1);
        return;
    }
    if (m == 5 && hd <= 256) {
        px_cross5<<<dim3(n, heads), 256, 0, d.stream>>>(px_float(op.out), px_float(op.x), px_float(op.w),
                                                        px_float(op.v), n, heads, hd, op.precision);
        return;
    }
#ifndef PX_HIP
    if (op.precision && hd == 128 && n == m && d.kernels != 1 && d.mma_supported) {
        size_t count = size_t(n) * c;
        Buffer q(d, count * 2), k(d, count * 2), v(d, count * 2), out(d, count * 2);
        px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)q.p, px_float(op.x),
                                                                       n, heads, op.precision, 0);
        px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)k.p, px_float(op.w),
                                                                       n, heads, op.precision, 0);
        px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>((unsigned short *)v.p, px_float(op.v),
                                                                       n, heads, op.precision, 0);
        if (op.precision == 1)
            px_fa2_bf16::px_fa2_bf16<<<dim3((n + 63) / 64, heads), 128, 34816, d.stream>>>(
                (unsigned short *)out.p, (unsigned short *)q.p, (unsigned short *)k.p, (unsigned short *)v.p,
                n, 128, 1 / sqrtf(128.f));
        else
            px_fa2_fp16::px_fa2_fp16<<<dim3((n + 63) / 64, heads), 128, 34816, d.stream>>>(
                (unsigned short *)out.p, (unsigned short *)q.p, (unsigned short *)k.p, (unsigned short *)v.p,
                n, 128, 1 / sqrtf(128.f));
        px_attention_layout<<<(count + 255) / 256, 256, 0, d.stream>>>(
            (unsigned short *)out.p, px_float(op.out), n, heads, op.precision, 1);
        d.metrics.mma_attentions++;
        return;
    }
#endif
#ifdef PX_HIP
    if (op.precision == 1 && hd == 128 && n == m && d.kernels != 1 && d.mma_supported) {
        px_trellis_attention<<<dim3(heads, (n + 63) / 64), 128, 0, d.stream>>>(
            px_float(op.out), px_float(op.x), px_float(op.w), px_float(op.v), n, heads, 128, nullptr, 0);
        d.metrics.mma_attentions++;
        return;
    }
#endif
    if (op.precision && hd == 128 && d.mma_supported &&
        (d.kernels == 2
#ifdef PX_HIP
         || (d.kernels == 0 && n <= 4096 && m <= 4096)
#endif
             )) {
        px_flash_mma<<<dim3((n + 15) / 16, heads), 32, 0, d.stream>>>(
            px_float(op.out), px_float(op.x), px_float(op.w), px_float(op.v), n, m, heads, op.precision);
        d.metrics.mma_attentions++;
        return;
    }
    int tile = std::min(n, 128);
    Buffer scores(d, size_t(tile) * m * 4);
    float scale = 1 / sqrtf(float(hd)), one = 1, zero = 0;
    for (int h = 0; h < heads; ++h)
        for (int start = 0; start < n; start += tile) {
            int rows = std::min(tile, n - start);
            GPU(blasSgemm(d.blas, OP_T, OP_N, m, rows, hd, &scale, px_float(op.w) + h * hd, c,
                          px_float(op.x) + size_t(start) * c + h * hd, c, &zero, scores.f(), m));
            softmax_rows<<<rows, 256, 0, d.stream>>>(scores.f(), rows, m);
            GPU(blasSgemm(d.blas, OP_N, OP_N, hd, rows, m, &one, px_float(op.v) + h * hd, c, scores.f(), m,
                          &zero, px_float(op.out) + size_t(start) * c + h * hd, c));
        }
    px_element<<<(size_t(n) * c + 255) / 256, 256, 0, d.stream>>>(
        px_float(op.out), px_float(op.out), nullptr, nullptr, PX_ROUND, size_t(n) * c, c, op.precision, 0, 0);
}
extern "C" int px_gpu_execute(void *p, const px_device_command *command) {
    auto &d = *static_cast<Device *>(p);
    gpuEvent start{}, end{};
    try {
        GPU(gpuSetDevice(d.ordinal));
        if (!command || command->n <= 0 || command->c <= 0 || command->precision < 0 ||
            command->precision > 3)
            throw std::runtime_error("Invalid resident command");
        const auto &o = *command;
        if (o.offset < 0 || o.k < 0 || o.heads < 0)
            throw std::runtime_error("Negative resident geometry");
        if ((o.op == PX_LINEAR && o.k == 0) || (o.op == PX_ATTENTION && (o.k == 0 || o.heads == 0)))
            throw std::runtime_error("Empty matrix extent");
        if (o.op == PX_PART && (o.k == 0 || o.offset >= o.k))
            throw std::runtime_error("Invalid tensor slice");
        if ((o.op == PX_ROPE || o.op == PX_RMS) && (o.k == 0 || o.n % o.k))
            throw std::runtime_error("Invalid head geometry");
        if (o.op == PX_GROUP_SILU && o.c % 8)
            throw std::runtime_error("Invalid group size");
        if (d.profile) {
            GPU(gpuEventCreate(&start));
            GPU(gpuEventCreate(&end));
            GPU(gpuEventRecord(start, d.stream));
        }
        size_t count = size_t(o.n) * o.c;
        // Row-offset forms write rows heads..heads+n of their output.
        bool row_offset = o.op == PX_C2S || o.op == PX_SKIP || (o.op == PX_ADD && o.k);
        size_t shift = row_offset ? size_t(o.heads) * o.c : 0;
        bool pack = (o.extra & PX_PACK_OUT) &&
                    (o.op == PX_NORM || o.op == PX_RMS || o.op == PX_MODULATE || o.op == PX_GELU);
        if (o.op != PX_ATTENTION)
            px_extent(o.out, count + shift, (o.op == PX_GATHER && o.precision) || pack ? 2 : 4);
        unsigned short *packed_out = pack ? (unsigned short *)px_buffer(o.out).p : nullptr;
        if (o.op == PX_PART)
            px_extent(o.x, count * o.k);
        else if ((o.op >= PX_NORM && o.op <= PX_RESIDUAL) || o.op == PX_SCALE_ADD || o.op == PX_GROUP_SILU)
            px_extent(o.x, count);
        if (o.op == PX_NORM || o.op == PX_GROUP_SILU) {
            if (o.w)
                px_extent(o.w, o.c);
            if (o.b)
                px_extent(o.b, o.c);
        }
        if (o.op == PX_RMS)
            px_extent(o.w, size_t(o.k) * o.c);
        if (o.op == PX_ADD || o.op == PX_SCALE_ADD || o.op == PX_RESIDUAL) {
            px_extent(o.w, o.op == PX_ADD && !o.k ? o.c : count + shift);
            if (o.b)
                px_extent(o.b, size_t(o.offset) + o.c);
        }
        if (o.op == PX_MODULATE)
            px_extent(o.w, size_t(o.offset) + 2 * o.c);
        if (o.op == PX_ROPE)
            px_extent(o.w, size_t(o.n / o.k) * (o.extra ? o.c : 4));
        if (o.op == PX_LINEAR)
            px_resident_linear(d, o);
        else if (o.op == PX_ATTENTION)
            px_resident_attention(d, o);
        else if (o.op == PX_NORM || o.op == PX_RMS)
            px_norm_warp<<<(o.n + 7) / 8, 256, 0, d.stream>>>(pack ? nullptr : px_float(o.out), px_float(o.x),
                                                               px_float(o.w), px_float(o.b), o.c, o.epsilon,
                                                               o.precision, o.op == PX_RMS, o.k, o.n,
                                                               packed_out);
        else if (o.op == PX_ROPE)
            px_rope<<<(count / 2 + 255) / 256, 256, 0, d.stream>>>(px_float(o.out), px_float(o.x),
                                                                   (const int *)px_buffer(o.w).p, count, o.c,
                                                                   o.k, o.precision, o.extra);
        else if (o.op == PX_ROPE_PHASE) {
            px_extent(o.w, size_t(o.n) * 4);
            px_rope_phase<<<(count / 2 + 255) / 256, 256, 0, d.stream>>>(
                px_float(o.out), (const int *)px_buffer(o.w).p, o.n, o.c);
        } else if (o.op == PX_NAF_SAMPLE)
            px_naf_sample<<<dim3(o.n, 4), 256, 0, d.stream>>>(px_float(o.out), px_float(o.x), px_float(o.w),
                                                              px_float(o.v), px_float(o.b), o.k, o.extra);
        else if (o.op == PX_CONV2_GATHER)
            px_conv2_gather<<<(count + 255) / 256, 256, 0, d.stream>>>(
                px_float(o.out), px_float(o.x), o.n, o.c / (o.k * o.k), o.k, o.extra, o.offset);
        else if (o.op == PX_GROUP_SILU)
            px_group_silu<<<8, 256, 0, d.stream>>>(px_float(o.out), px_float(o.x), px_float(o.w),
                                                   px_float(o.b), o.n, o.c);
        else if (o.op == PX_AVERAGE)
            px_average<<<(count + 255) / 256, 256, 0, d.stream>>>(px_float(o.out), px_float(o.x), o.k,
                                                                  o.extra, o.c);
        else if (o.op == PX_JOIN)
            px_join<<<(count + 255) / 256, 256, 0, d.stream>>>(px_float(o.out), px_float(o.x), o.n, o.c,
                                                               o.offset);
        else if (o.op == PX_ROPE2)
            px_rope2<<<(size_t(o.n) * o.n * o.heads * o.c / 2 + 255) / 256, 256, 0, d.stream>>>(
                px_float(o.out), px_float(o.x), px_float(o.w), o.n, o.heads, o.c, o.offset);
        else if (o.op == PX_GATHER)
            px_gather<<<(count + 255) / 256, 256, 0, d.stream>>>(px_buffer(o.out).p, px_float(o.x),
                                                                 (const int *)px_buffer(o.w).p, o.offset, o.n,
                                                                 o.c / 27, o.extra, o.precision);
        else if (o.op == PX_C2S || o.op == PX_SKIP)
        {
            px_extent(o.w, size_t(o.heads) + o.n);
            px_extent(o.b, size_t(o.heads) + o.n);
            px_c2s<<<(count + 255) / 256, 256, 0, d.stream>>>(
                px_float(o.out), px_float(o.x), (const int *)px_buffer(o.w).p, (const int *)px_buffer(o.b).p,
                o.n, o.c, o.k, o.op == PX_SKIP, o.extra, o.heads, o.offset);
        }
        else if (((o.op >= PX_PART && o.op <= PX_RESIDUAL) || o.op == PX_SCALE_ADD) && o.c >= 128 &&
                 count % o.c == 0 && count / o.c <= size_t(INT32_MAX))
            px_element_rows<<<unsigned(count / o.c), 256, 0, d.stream>>>(
                pack ? nullptr : px_float(o.out) + shift, px_float(o.x), px_float(o.w) + (o.w ? shift : 0),
                px_float(o.b), o.op, o.c, o.precision, o.k, o.offset, packed_out);
        else if (pack)
            throw std::runtime_error("Packed output needs rows of at least 128 channels");
        else if ((o.op >= PX_PART && o.op <= PX_RESIDUAL) || o.op == PX_SCALE_ADD)
            px_element<<<(count + 255) / 256, 256, 0, d.stream>>>(px_float(o.out) + shift, px_float(o.x),
                                                                  px_float(o.w) + (o.w ? shift : 0),
                                                                  px_float(o.b), o.op, count, o.c, o.precision,
                                                                  o.k, o.offset);
        else
            throw std::runtime_error("Unsupported resident operation");
        GPU(gpuGetLastError());
        if (d.profile) {
            GPU(gpuEventRecord(end, d.stream));
            GPU(gpuEventSynchronize(end));
            float ms = 0;
            GPU(gpuEventElapsedTime(&ms, start, end));
            d.metrics.kernel_ms += ms;
        }
        if (start)
            (void)gpuEventDestroy(start);
        if (end)
            (void)gpuEventDestroy(end);
        return 0;
    } catch (const std::exception &e) {
        if (start)
            (void)gpuEventDestroy(start);
        if (end)
            (void)gpuEventDestroy(end);
        d.error = e.what();
        return -1;
    }
}
