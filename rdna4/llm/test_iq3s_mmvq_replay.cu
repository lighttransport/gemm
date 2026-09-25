/* Template for test_iq3s_mmvq_replay.py; uses the real local llama vecdot. */
#include "vecdotq.cuh"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// RUNNER_KERNELS

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    std::cerr << #call << ": " << hipGetErrorString(e) << "\n"; std::exit(2); } } while (0)

__global__ void pack_q8(block_q8_1 *dst, const signed char *q, const float *s, int groups) {
    int g = blockIdx.x, lane = threadIdx.x;
    if (g >= groups) return;
    dst[g].qs[lane] = q[g * 32 + lane];
    if (!lane) dst[g].ds = __floats2half2_rn(s[g], 0.0f);
}

__global__ void reference(float *dst, const block_iq3_s *w, const block_q8_1 *q,
                          int rows, int cols) {
    const int row = blockIdx.x, token = blockIdx.y, lane = threadIdx.x;
    constexpr int qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ;
    static_assert(qi == 16 && vdr == 2, "Re-audit the RDNA4 MMVQ schedule");
    float sum = 0;
    for (int b = lane / (qi / vdr); b < cols / QK_K; b += vdr * 32 / qi) {
        sum += vec_dot_iq3_s_q8_1(w + row * (cols / QK_K),
            q + token * (cols / 32) + b * 8, b, vdr * (lane % (qi / vdr)));
    }
    sum = warp_reduce_sum<32>(sum);
    if (!lane) dst[token * rows + row] = sum;
}

__global__ void reference_terms(float *dst, const block_iq3_s *w, const block_q8_1 *q,
                                int rows, int cols) {
    int row = blockIdx.x, token = blockIdx.y, lane = threadIdx.x, G = cols / 32;
    for (int g = lane; g < G; g += 32) {
        dst[((size_t)token * rows + row) * G + g] = vec_dot_iq3_s_q8_1(
            w + row * (cols / 256), q + token * G + (g / 8) * 8, g / 8, 2 * (g % 8));
    }
}

static bool finite_bits(float x) {
    uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

template<class T> std::vector<T> read_file(const std::string &path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read((char *)v.data(), n * sizeof(T)) || f.peek() != EOF) {
        std::cerr << "Invalid capture: " << path << "\n"; std::exit(2);
    }
    return v;
}
template<class T> T *alloc(size_t n) {
    T *p; CHECK(hipMalloc(&p, n * sizeof(T))); return p;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    std::string dir = argv[1];
    int rows = std::stoi(argv[2]), cols = std::stoi(argv[3]), M = std::stoi(argv[4]);
    if (rows <= 0 || cols <= 0 || cols % 256 || M <= 0) return 2;
    auto w = read_file<unsigned char>(dir + "/weights.bin", (size_t)rows * cols / 256 * 110);
    auto x = read_file<float>(dir + "/input.bin", (size_t)M * cols);
    auto dw = alloc<unsigned char>(w.size()); auto dx = alloc<float>(x.size());
    auto q = alloc<signed char>(x.size()); auto q1 = alloc<signed char>(x.size());
    auto s = alloc<float>(x.size() / 32); auto s1 = alloc<float>(x.size() / 32);
    auto packed = alloc<block_q8_1>(x.size() / 32);
    auto dy = alloc<float>((size_t)rows * M);
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    quantize_q81_batch_32_exact<<<dim3(cols / 32, M), 32>>>(q, s, q1, s1, dx, cols, M, cols);
    pack_q8<<<x.size() / 32, 32>>>(packed, q, s, x.size() / 32);
    std::vector<float> expected((size_t)rows * M), actual(expected.size());
    reference<<<dim3(rows, M), 32>>>(dy, (const block_iq3_s *)dw, packed, rows, cols);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(expected.data(), dy, expected.size() * sizeof(float), hipMemcpyDeviceToHost));
    int failures = 0;
    for (int mode = 0; mode < 3; ++mode) {
        if (mode == 0) matvec_iq3_s_q81_batch<<<dim3((rows + 7) / 8, M), 256>>>(dy, dw, q, s, rows, cols, M);
        if (mode == 1) candidate<<<dim3((rows + 7) / 8, M), 256>>>(dy, dw, q, s, rows, cols, M);
        if (mode == 2) candidate_fma<<<dim3((rows + 7) / 8, M), 256>>>(dy, dw, q, s, rows, cols, M);
        CHECK(hipGetLastError());
        CHECK(hipMemcpy(actual.data(), dy, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        size_t equal = 0; double err = 0, norm = 0, maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!finite_bits(actual[i]) || !finite_bits(expected[i])) return 2;
            equal += std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0;
            double d = (double)actual[i] - expected[i];
            err += d*d; norm += (double)expected[i]*expected[i];
            maximum = std::max(maximum, std::abs(d));
        }
        const char *names[] = {"production", "separate-mul-add", "mmvq-fma"};
        std::cout << names[mode] << " equal=" << equal << "/" << actual.size()
                  << " rel_l2=" << std::sqrt(err / (norm ? norm : 1.0)) << " max_abs=" << maximum << "\n";
        if (mode == 2 && equal != actual.size()) ++failures;
    }
    size_t nt = (size_t)rows * M * cols / 32;
    auto dt = alloc<float>(nt);
    std::vector<float> rt(nt), lt(nt);
    runner_terms<<<dim3((rows + 7) / 8, M), 256>>>(dt, dw, q, s, rows, cols, M);
    CHECK(hipMemcpy(rt.data(), dt, nt * sizeof(float), hipMemcpyDeviceToHost));
    reference_terms<<<dim3(rows, M), 32>>>(dt, (const block_iq3_s *)dw, packed, rows, cols);
    CHECK(hipMemcpy(lt.data(), dt, nt * sizeof(float), hipMemcpyDeviceToHost));
    size_t term_equal = 0;
    for (size_t i = 0; i < nt; ++i) term_equal += std::memcmp(&rt[i], &lt[i], 4) == 0;
    std::cout << "terms equal=" << term_equal << "/" << nt << "\n";
    if (term_equal != nt) ++failures;
    for (auto item : {std::make_pair("runner-terms.bin", &rt), std::make_pair("llama-terms.bin", &lt)}) {
        std::ofstream f(dir + "/" + item.first, std::ios::binary);
        f.write((const char *)item.second->data(), nt * sizeof(float));
    }
    std::vector<signed char> hq(x.size()); std::vector<float> hs(x.size() / 32);
    CHECK(hipMemcpy(hq.data(), q, hq.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hs.data(), s, hs.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::ofstream(dir + "/q.bin", std::ios::binary).write((const char *)hq.data(), hq.size());
    std::ofstream(dir + "/s.bin", std::ios::binary).write((const char *)hs.data(), hs.size() * sizeof(float));
    CHECK(hipFree(dt));
    for (void *p : { (void *)dw, (void *)dx, (void *)q, (void *)q1,
                    (void *)s, (void *)s1, (void *)packed, (void *)dy }) CHECK(hipFree(p));
    return failures ? 1 : 0;
}
